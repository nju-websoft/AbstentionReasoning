import argparse
import json
import re
from fractions import Fraction
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_STAGE2_DIR = (
    SCRIPT_DIR.parent
    / "data"
    / "out"
    / "stage2"
    / "SUM"
    / "DeepSeek-R1-Distill-Llama-8B"
)


def resolve_path(path):
    if path is None:
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
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


def get_output(record):
    for key in ("output", "cleaned_generated_text", "out_text", "inter_out"):
        if key in record:
            return record[key]
    raise KeyError("Expected one of output, cleaned_generated_text, out_text, or inter_out")


def get_ground_truth(record):
    for key in ("GT_answer", "ground_truth"):
        if key in record:
            return record[key]
    raise KeyError("Expected one of GT_answer or ground_truth")


def boxed_contains_idk(text):
    pattern = r"boxed\{([^}]*)\}"
    for content in re.findall(pattern, text, re.IGNORECASE):
        lowered = content.lower()
        if "don't" in lowered and "know" in lowered:
            return True
    return False


def classify_model_behavior(output):
    output_lower = output.lower()
    think_count = output.count("</think>")
    if think_count == 2:
        index = output_lower.find("</think>")
        output_lower = output_lower[index + len("</think>") :].strip()
    elif think_count > 2:
        return "other"

    if "</think>" in output_lower:
        post_think = output_lower.split("</think>", 1)[1]
        if (
            "i don't know" in post_think
            or "don't know" in post_think
            or boxed_contains_idk(post_think)
        ):
            return "i_do_not_know"

    if "</think>" in output_lower:
        pre_think, post_think = output_lower.split("</think>", 1)
        if re.search(r"\\boxed\{[^}]+\}", post_think):
            return "return_answer"
        if "final answer" in pre_think:
            return "return_answer"

    if "</think>" not in output_lower:
        return "not_end"

    if "reason" in output_lower or "reasoning" in output_lower:
        return "IDK_reason"

    return "not_end"


def get_reason_after_think(text):
    if "</think>" not in text.lower():
        raise ValueError("No </think> tag found")
    reason_text = text.lower().split("</think>", 1)[1].strip()
    if "reason" not in reason_text and "reasoning" not in reason_text:
        return reason_text
    match = re.search(r"\breason\b[\s:=-]*(.*)", reason_text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else reason_text


def load_reason_data(path):
    data = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "test" in data:
        data = data["test"]

    reason_by_id = {}
    for record in data:
        question_index = int(record.get("id", record.get("question_index")))
        reason_by_id[question_index] = record.get("missing", record.get("reason", ""))
    return reason_by_id


def judge_text_equivalent(gt_text, model_text, judge_mode, ollama_host, ollama_model):
    if judge_mode == "exact":
        return normalize_text(gt_text) == normalize_text(model_text)
    if judge_mode == "contains":
        gt_norm = normalize_text(gt_text)
        model_norm = normalize_text(model_text)
        return gt_norm in model_norm or model_norm in gt_norm
    if judge_mode == "ollama":
        return judge_with_ollama(gt_text, model_text, ollama_host, ollama_model)
    raise ValueError(f"Unknown judge mode: {judge_mode}")


def judge_with_ollama(gt_answer, model_answer, host, model_name):
    from ollama import Client

    client = Client(host=host)
    result_format_true = '{"equivalent": true}'
    result_format_false = '{"equivalent": false}'
    system_content = (
        "You are a mathematics assistant. Decide whether two answers are "
        "mathematically equivalent, ignoring LaTeX formatting differences."
    )
    user_content = (
        "Consider these two answer expressions:\n"
        f"1. {gt_answer}\n"
        f"2. {model_answer}\n"
        "Do they represent the same mathematical object or value? Return only "
        f"{result_format_true} or {result_format_false}."
    )
    response = client.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ],
        format="json",
        options={"temperature": 0, "num_ctx": 8096},
    )
    parsed = json.loads(response["message"]["content"])
    return bool(parsed["equivalent"])


def normalize_text(text):
    text = str(text).strip().lower()
    replacements = {
        "\\dfrac": "\\frac",
        "\\tfrac": "\\frac",
        "\\left": "",
        "\\right": "",
        "\\,": "",
        "$": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    text = text.strip(" .。")
    return text


def parse_numeric_answer(text):
    text = normalize_text(text)
    frac_match = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", text)
    if frac_match:
        sign, numerator, denominator = frac_match.groups()
        value = Fraction(int(numerator), int(denominator))
        return -value if sign == "-" else value

    slash_match = re.fullmatch(r"(-?\d+)/(-?\d+)", text)
    if slash_match:
        numerator, denominator = slash_match.groups()
        return Fraction(int(numerator), int(denominator))

    number_match = re.fullmatch(r"-?\d+(\.\d+)?", text)
    if number_match:
        return Fraction(text)

    return None


def judge_answer(gt_answer, model_answer, judge_mode, ollama_host, ollama_model):
    gt_value = parse_numeric_answer(gt_answer)
    model_value = parse_numeric_answer(model_answer)
    if gt_value is not None and model_value is not None:
        return gt_value == model_value

    return judge_text_equivalent(
        gt_answer,
        model_answer,
        judge_mode=judge_mode,
        ollama_host=ollama_host,
        ollama_model=ollama_model,
    )


def find_matching_brace(text, start_pos):
    brace_level = 1
    for index in range(start_pos + 1, len(text)):
        if text[index] == "{":
            brace_level += 1
        elif text[index] == "}":
            brace_level -= 1
            if brace_level == 0:
                return index
    return -1


def extract_boxed_content(text):
    results = []
    for match in re.finditer(r"\\boxed{", text):
        content_start = match.end()
        brace_start = content_start - 1
        content_end = find_matching_brace(text, brace_start)
        if content_end != -1:
            results.append(text[content_start:content_end])
    return results


def extract_returned_answers(output):
    if "</think>" not in output:
        return []

    post_think = output.split("</think>", 1)[1].strip()
    if post_think == "":
        return []
    if "final answer" in post_think.lower():
        post_think = post_think.lower().split("final answer", 1)[-1].strip()

    answers = extract_boxed_content(post_think)
    if post_think.count("\\boxed") != 0 and post_think.count("\\boxed") != len(answers):
        first_box_index = post_think.find(r"\boxed{")
        return [post_think[first_box_index:]]

    return sorted(set(answers))


def evaluate_unanswerable(records, judge_reason, reason_data, reason_judge, ollama_host, ollama_model):
    behavior = {
        "i_do_not_know": [],
        "return_answer": [],
        "not_end": [],
        "other": [],
        "IDK_reason": [],
        "reason": [],
    }

    reason_by_id = load_reason_data(reason_data) if judge_reason and reason_data else {}

    for record in records:
        question_index = int(record["question_index"])
        output = get_output(record)
        label = classify_model_behavior(output)
        behavior[label].append(question_index)

        if judge_reason and label in {"i_do_not_know", "IDK_reason"}:
            gt_reason = reason_by_id.get(question_index, "")
            model_reason = get_reason_after_think(output)
            if gt_reason and judge_text_equivalent(
                gt_reason,
                model_reason,
                judge_mode=reason_judge,
                ollama_host=ollama_host,
                ollama_model=ollama_model,
            ):
                behavior["reason"].append(question_index)

    total = len(records)
    abstention_count = len(behavior["i_do_not_know"]) + len(behavior["IDK_reason"])
    metrics = {
        "total": total,
        "abstention_count": abstention_count,
        "abstention_rate": round(abstention_count / total * 100, 2) if total else 0.0,
        "i_do_not_know_rate": round(len(behavior["i_do_not_know"]) / total * 100, 2) if total else 0.0,
        "IDK_reason_rate": round(len(behavior["IDK_reason"]) / total * 100, 2) if total else 0.0,
        "return_answer_rate": round(len(behavior["return_answer"]) / total * 100, 2) if total else 0.0,
        "not_end_rate": round(len(behavior["not_end"]) / total * 100, 2) if total else 0.0,
        "other_rate": round(len(behavior["other"]) / total * 100, 2) if total else 0.0,
        "behavior_ids": behavior,
    }
    if judge_reason:
        denominator = abstention_count
        metrics["reason_correct_count"] = len(behavior["reason"])
        metrics["reason_acc_on_abstentions"] = (
            round(len(behavior["reason"]) / denominator * 100, 2) if denominator else 0.0
        )
        metrics["reason_acc_all"] = (
            round(len(behavior["reason"]) / total * 100, 2) if total else 0.0
        )

    return metrics


def evaluate_answerable(records, judge_mode, ollama_host, ollama_model):
    total = len(records)
    parsed_answer_count = 0
    correct_ids = []
    wrong_ids = []
    no_answer_ids = []

    for record in records:
        question_index = int(record["question_index"])
        gt_answer = get_ground_truth(record)
        output = get_output(record)
        answers = extract_returned_answers(output)
        if not answers:
            no_answer_ids.append(question_index)
            wrong_ids.append(question_index)
            continue

        parsed_answer_count += 1
        model_answer = ", ".join(answers)
        if judge_answer(
            gt_answer,
            model_answer,
            judge_mode=judge_mode,
            ollama_host=ollama_host,
            ollama_model=ollama_model,
        ):
            correct_ids.append(question_index)
        else:
            wrong_ids.append(question_index)

    return {
        "total": total,
        "parsed_answer_count": parsed_answer_count,
        "correct_count": len(correct_ids),
        "answer_acc": round(len(correct_ids) / total * 100, 2) if total else 0.0,
        "parsed_answer_rate": round(parsed_answer_count / total * 100, 2) if total else 0.0,
        "correct_ids": correct_ids,
        "wrong_ids": wrong_ids,
        "no_answer_ids": no_answer_ids,
    }


def print_metrics(metrics):
    if "unanswerable" in metrics:
        unans = metrics["unanswerable"]
        print("Unanswerable:")
        print(f"  total: {unans['total']}")
        print(f"  Abstention: {unans['abstention_rate']}% ({unans['abstention_count']}/{unans['total']})")
        print(f"  return_answer: {unans['return_answer_rate']}%")
        print(f"  not_end: {unans['not_end_rate']}%")
        print(f"  other: {unans['other_rate']}%")
        if "reason_acc_on_abstentions" in unans:
            print(f"  Reason Acc: {unans['reason_acc_on_abstentions']}%")

    if "answerable" in metrics:
        ans = metrics["answerable"]
        print("Answerable:")
        print(f"  total: {ans['total']}")
        print(f"  Answer Acc: {ans['answer_acc']}% ({ans['correct_count']}/{ans['total']})")
        print(f"  parsed_answer_rate: {ans['parsed_answer_rate']}%")


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3: evaluate stage2 intervention outputs.")
    parser.add_argument(
        "--unsolve_path",
        default=str(DEFAULT_STAGE2_DIR / "unsolve_intervention_result.jsonl"),
        help="Stage2 output on unanswerable questions.",
    )
    parser.add_argument(
        "--solve_path",
        default=str(DEFAULT_STAGE2_DIR / "solve_intervention_result.jsonl"),
        help="Stage2 output on answerable questions.",
    )
    parser.add_argument(
        "--answer_judge",
        default="exact",
        choices=["exact", "contains", "ollama"],
        help="How to judge answer equivalence for answerable questions.",
    )
    parser.add_argument("--judge_reason", action="store_true")
    parser.add_argument("--reason_data", default=None)
    parser.add_argument(
        "--reason_judge",
        default="contains",
        choices=["exact", "contains", "ollama"],
    )
    parser.add_argument("--ollama_host", default="http://localhost:6161")
    parser.add_argument("--ollama_model", default="qwen3:32b-fp16")
    parser.add_argument("--output_path", default=None, help="Optional JSON file for metrics.")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = {}

    unsolve_path = resolve_path(args.unsolve_path)
    if unsolve_path and unsolve_path.exists():
        metrics["unanswerable"] = evaluate_unanswerable(
            records=read_json_records(unsolve_path),
            judge_reason=args.judge_reason,
            reason_data=args.reason_data,
            reason_judge=args.reason_judge,
            ollama_host=args.ollama_host,
            ollama_model=args.ollama_model,
        )
    else:
        print(f"Skip unanswerable evaluation; file not found: {unsolve_path}")

    solve_path = resolve_path(args.solve_path)
    if solve_path and solve_path.exists():
        metrics["answerable"] = evaluate_answerable(
            records=read_json_records(solve_path),
            judge_mode=args.answer_judge,
            ollama_host=args.ollama_host,
            ollama_model=args.ollama_model,
        )
    else:
        print(f"Skip answerable evaluation; file not found: {solve_path}")

    print_metrics(metrics)

    if args.output_path is not None:
        output_path = resolve_path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
