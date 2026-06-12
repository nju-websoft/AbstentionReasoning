import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

from a_input_path import DATA_PATH_JSON


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR.parent / "data" / "out" / "stage2"
DEFAULT_PROBE_ROOT = SCRIPT_DIR.parent / "model"
DEFAULT_MAX_TOKENS = 7000

WAIT_STOP_WORDS = ["wait", "Wait", "\nWait", "\n\nWait", " Wait", " wait", ".Wait"]
INTERVENTION_PROMPT = r"""Instruction:
 You are not permitted to make assumptions that are not explicitly stated in the problem. There are signs that this problem may lack sufficient information to solve definitively. If you find that any part of your reasoning depends on undefined operations, missing values, or unspecified conditions, you must immediately stop and output: \boxed{I don't know.} Do not attempt to guess, infer, or continue reasoning with incomplete information. This is a strict constraint!!!!
</think>

"""


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return (SCRIPT_DIR / path).resolve()


def read_json_records(path):
    path = resolve_path(path)
    text = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    records = []
    index = 0

    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        record, index = decoder.raw_decode(text, index)
        if isinstance(record, list):
            records.extend(record)
        else:
            records.append(record)

    return records


def write_jsonl(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out_file:
        for record in records:
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_generation_text(record):
    for key in ("cleaned_generated_text", "output", "out_text"):
        if key in record:
            return record[key]
    raise KeyError("Expected one of cleaned_generated_text, output, or out_text")


def to_probe_data(records):
    return [
        {
            "input_text": record["input_text"],
            "output": get_generation_text(record),
        }
        for record in records
    ]


def accumulate_by_stopwords(text, stop_words):
    if not stop_words:
        return [text]

    stop_words = sorted(stop_words, key=len, reverse=True)
    pattern = "(" + "|".join(re.escape(word) for word in stop_words) + ")"

    result = []
    position = 0
    accumulated = ""
    while position < len(text):
        match = re.search(pattern, text[position:])
        if not match:
            accumulated += text[position:]
            result.append(accumulated)
            break

        end_idx = position + match.end()
        accumulated += text[position:end_idx]
        result.append(accumulated)
        position = end_idx

    return result


def filter_single_token_stop_words(tokenizer, stop_words):
    filtered = []
    for token in stop_words:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if len(token_ids) == 1:
            filtered.append(token)
    return filtered


def load_vllm(model_path, max_token):
    from transformers import AutoTokenizer
    from vllm import LLM

    llm_engine = LLM(
        model=model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=1000 + max_token + max_token,
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer, llm_engine


def get_intervention_segment_id(chat_prompt, tokenizer, point, split_text):
    prompt_len = len(tokenizer.encode(chat_prompt, add_special_tokens=False))
    if point == -1:
        return len(split_text) - 1

    for segment_id, segment in enumerate(split_text):
        segment_token_len = len(tokenizer.encode(segment, add_special_tokens=False)) - prompt_len
        if segment_token_len >= point:
            return segment_id
    return len(split_text) - 1


def get_stop_points(tokenizer, prompt):
    from interven import get_input_data

    token_ids = get_input_data(tokenizer, prompt).tolist()
    check_positions = []

    for i in range(1, len(token_ids) + 1):
        if i - 10 < 0:
            continue
        decoded = tokenizer.decode(token_ids[i - 10 : i])
        words = re.findall(r"\b[a-z]+\b", decoded.strip().lower())
        if words and words[-1] == "wait":
            check_positions.append(i)

    return check_positions


def find_first_intervention_point(tokenizer, prompt, probe_scores, threshold):
    check_points = get_stop_points(tokenizer, prompt)
    if not check_points:
        check_points = [-1]

    for point in check_points:
        score_index = point - 1
        if not probe_scores:
            continue
        try:
            score = probe_scores[score_index]
        except IndexError:
            continue
        if score < threshold:
            return score, point

    return None


def detect_intervention_points(
    dataset,
    model_name,
    model_path,
    probe_path,
    probe_layer,
    threshold,
    output_dir,
):
    from interven import load_prob_model, predict_with_linear_probe

    model, tokenizer, probe = load_prob_model(
        model_path=model_path,
        probe_path=probe_path,
        layer=probe_layer,
    )

    output_paths = {}
    for split in ("unsolve", "solve"):
        input_path = DATA_PATH_JSON[dataset][model_name][split]
        records = read_json_records(input_path)
        probe_data = to_probe_data(records)
        probe_results = predict_with_linear_probe(
            datasets=probe_data,
            model=model,
            tokenizer=tokenizer,
            layer_id=probe_layer,
            probe=probe,
            model_name=model_name,
        )

        intervention_points = []
        for idx, record in enumerate(tqdm(records, desc=f"detect {split}")):
            result = find_first_intervention_point(
                tokenizer=tokenizer,
                prompt=probe_data[idx],
                probe_scores=probe_results[idx],
                threshold=threshold,
            )
            if result is None:
                continue
            score, point = result
            intervention_points.append([record["question_index"], score, point])

        output_path = output_dir / f"{split}_inter_q_id_point.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(intervention_points, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        output_paths[split] = output_path
        print(f"Saved {split} intervention points to {output_path}")

    return output_paths


def run_intervention_for_split(
    records,
    intervention_points,
    tokenizer,
    model,
    max_token,
):
    from vllm import SamplingParams

    intervention_by_id = {
        question_id: point for question_id, _, point in intervention_points
    }
    stop_words = filter_single_token_stop_words(tokenizer, WAIT_STOP_WORDS)
    if not stop_words:
        stop_words = WAIT_STOP_WORDS
    stop_tokens = [tokenizer.eos_token] if tokenizer.eos_token is not None else None
    sampling_params = SamplingParams(
        max_tokens=max_token,
        temperature=0.0,
        top_p=1.0,
        seed=42,
        stop=stop_tokens,
    )

    outputs = []
    for record in tqdm(records, desc="intervene"):
        question_index = record["question_index"]
        original_output = get_generation_text(record)
        output_record = {
            "question_index": question_index,
            "input_text": record["input_text"],
            "GT_answer": record["ground_truth"],
        }

        if question_index not in intervention_by_id:
            output_record["output"] = original_output
            outputs.append(output_record)
            continue

        messages = [{"role": "user", "content": record["input_text"]}]
        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_text = chat_prompt + original_output
        split_text = accumulate_by_stopwords(full_text, stop_words)
        segment_id = get_intervention_segment_id(
            chat_prompt=chat_prompt,
            tokenizer=tokenizer,
            point=intervention_by_id[question_index],
            split_text=split_text,
        )
        prompt = split_text[segment_id] + INTERVENTION_PROMPT
        batch_outputs = model.generate(prompt, sampling_params, use_tqdm=False)
        completion_output = batch_outputs[0].outputs[0]
        generated_text = completion_output.text
        generated_ids = completion_output.token_ids

        if generated_text == "":
            output_record["output"] = original_output
        else:
            prefix_without_chat = split_text[segment_id].replace(chat_prompt, "", 1)
            if len(generated_ids) >= max_token:
                output_record["output"] = prefix_without_chat + generated_text
            elif "</think>" not in generated_text and "</think>" not in prefix_without_chat:
                output_record["output"] = prefix_without_chat + "</think>" + generated_text
            else:
                output_record["output"] = prefix_without_chat + generated_text

        outputs.append(output_record)

    return outputs


def run_intervention(dataset, model_name, model_path, max_token, output_dir):
    tokenizer, model = load_vllm(model_path, max_token)

    for split in ("unsolve", "solve"):
        input_path = DATA_PATH_JSON[dataset][model_name][split]
        records = read_json_records(input_path)
        point_path = output_dir / f"{split}_inter_q_id_point.json"
        intervention_points = json.loads(point_path.read_text(encoding="utf-8"))

        outputs = run_intervention_for_split(
            records=records,
            intervention_points=intervention_points,
            tokenizer=tokenizer,
            model=model,
            max_token=max_token,
        )
        output_path = output_dir / f"{split}_intervention_result.jsonl"
        write_jsonl(outputs, output_path)
        print(f"Saved {split} intervention results to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2: detect intervention points and run prompt intervention."
    )
    parser.add_argument("--dataset", default="SUM", choices=sorted(DATA_PATH_JSON.keys()))
    parser.add_argument("--model_name", default="DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument(
        "--model_path",
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="HuggingFace model id or local model path.",
    )
    parser.add_argument(
        "--probe_path",
        default=None,
        help="Path to the linear probe weight. Defaults to ../model/{model_name}_layer_result/layer_{probe_layer}.pt.",
    )
    parser.add_argument("--probe_layer", type=int, default=22)
    parser.add_argument("--therod", type=float, default=0.4)
    parser.add_argument(
        "--output_root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for stage2 outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.model_name not in DATA_PATH_JSON[args.dataset]:
        raise KeyError(f"{args.model_name} is not configured for dataset {args.dataset}.")

    output_root = resolve_path(args.output_root)
    output_dir = output_root / args.dataset / args.model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.probe_path is None:
        probe_path = (
            DEFAULT_PROBE_ROOT
            / f"{args.model_name}_layer_result"
            / f"layer_{args.probe_layer}.pt"
        )
    else:
        probe_path = resolve_path(args.probe_path)

    print(f"Dataset: {args.dataset}")
    print(f"Model name: {args.model_name}")
    print(f"Model path: {args.model_path}")
    print(f"Probe path: {probe_path}")
    print(f"Probe layer: {args.probe_layer}")
    print(f"Threshold: {args.therod}")
    print(f"Output dir: {output_dir}")

    detect_intervention_points(
        dataset=args.dataset,
        model_name=args.model_name,
        model_path=args.model_path,
        probe_path=probe_path,
        probe_layer=args.probe_layer,
        threshold=args.therod,
        output_dir=output_dir,
    )
    run_intervention(
        dataset=args.dataset,
        model_name=args.model_name,
        model_path=args.model_path,
        max_token=DEFAULT_MAX_TOKENS,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
