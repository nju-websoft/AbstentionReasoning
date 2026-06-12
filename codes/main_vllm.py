import argparse
import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm


DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

DATASET_CONFIG = {
    "SUM": {
        "answer": {
            "path": os.path.join(DATA_ROOT, "sum", "sum_answer_data.jsonl"),
        },
        "unanswer": {
            "path": os.path.join(DATA_ROOT, "sum", "sum_unanswer_data.jsonl"),
        },
    },
    "UMWP": {
        "answer": {
            "path": os.path.join(DATA_ROOT, "umwp", "umwp_answer_data2.jsonl"),
        },
        "unanswer": {
            "path": os.path.join(DATA_ROOT, "umwp", "umwp_unanswer_data2.jsonl"),
        },
    },
}


BASE_REASONING_PROMPT = "Let's think step by step and output the final answer within \\boxed{}."
ABSTENTION_PROMPT = (
    "Please solve the problem strictly based on the information provided. "
    "Do not introduce any additional assumptions. If you believe the problem lacks "
    "sufficient information or is unsolvable, first reply with \\boxed{I don't know.}, "
    "and provide your corresponding reason in the format:  Reason{your explanation here}"
)
CONFIDENCE_PROMPT = (
    "Additionally, you are also required to give a score based on how confident you are "
    "of your own answer. The score should be in the range of 1 to 5 where 1 being "
    "'Least Confident' while 5 being 'Extremely Confident'. Please provide the score "
    "in the format: Confidence{your score here}."
)
DEFAULT_MAX_TOKENS = 7000


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_append(record, file_out):
    file_out.write(json.dumps(record, ensure_ascii=False) + "\n")
    file_out.flush()


def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def get_question_index(record):
    if "id" in record:
        return int(record["id"])
    if "question_index" in record:
        return int(record["question_index"])
    raise ValueError("Data must contain either 'id' or 'question_index'.")


def build_input_text(question, with_confidence=False):
    input_text = f"{question} {BASE_REASONING_PROMPT} {ABSTENTION_PROMPT}"
    if with_confidence:
        input_text = f"{input_text} {CONFIDENCE_PROMPT}"
    return input_text


def get_ground_truth(record):
    if record.get("answerable") is False:
        return "I don't know."
    return record["answer"]


def load_dataset_records(dataset, split):
    config = DATASET_CONFIG[dataset][split]
    path = config["path"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return path, read_jsonl(path)


def run_model(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    set_seed(args.seed)

    data_path, data_records = load_dataset_records(args.dataset, args.split)
    model_name = args.model_name
    model_name_pure = model_name.rstrip("/").split("/")[-1]

    print(f"Dataset: {args.dataset} / {args.split}")
    print(f"Data path: {data_path}")
    print(f"Model: {model_name}")

    llm_engine = LLM(
        model=model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len + DEFAULT_MAX_TOKENS,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    batch_prompts = []
    batch_sampling_params = []
    run_info = []

    for record in data_records:
        question_index = get_question_index(record)
        input_text = build_input_text(record["question"], args.with_confidence)
        messages = [{"role": "user", "content": input_text}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        sampling_params = SamplingParams(
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
        )
        batch_prompts.append(formatted_prompt)
        batch_sampling_params.append(sampling_params)
        run_info.append(question_index)

    output_root = os.path.join(DATA_ROOT, "out", "stage1")
    split_dir = "solve" if args.split == "answer" else "unsolve"
    output_dir = os.path.join(
        output_root,
        args.dataset,
        model_name_pure,
        split_dir,
    )
    os.makedirs(output_dir, exist_ok=True)

    generation_log_file = os.path.join(output_dir, "generation_log.jsonl")
    generation_log_formal_file = os.path.join(output_dir, "generation_log_formal.jsonl")
    print(f"Output dir: {output_dir}")

    batch_outputs = llm_engine.generate(batch_prompts, batch_sampling_params, use_tqdm=True)

    with open(generation_log_file, "w", encoding="utf-8") as f_log, open(
        generation_log_formal_file, "w", encoding="utf-8"
    ) as f_log_formal:
        for i, output in enumerate(tqdm(batch_outputs)):
            record = data_records[i]
            question_index = get_question_index(record)
            assert question_index == run_info[i], f"Index mismatch: {question_index} != {run_info[i]}"

            result = {
                "question_index": question_index,
                "input_text": build_input_text(record["question"], args.with_confidence),
                "output": output.outputs[0].text,
                "ground_truth": get_ground_truth(record),
            }

            write_append(result, f_log_formal)
            f_log.write(json.dumps(result, ensure_ascii=False, indent=4) + "\n")
            f_log.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: generate vanilla model outputs.")
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIG), default="SUM")
    parser.add_argument("--split", choices=["answer", "unanswer"], default="answer")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--max_model_len", type=int, default=16384)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with_confidence", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run_model(parse_args())
