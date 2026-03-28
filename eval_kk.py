import argparse
import json
import os
import numpy as np
import time

from dataset.kk import KKProcessor
from utils import (
    load_eval_records,
    load_jsonl,
    write_jsonl,
    batch_decode_vllm,
    init_seed,
    load_llm,
)


def _normalize_correct_list(record):
    if "correct_list" in record:
        return [int(x) for x in record["correct_list"]]
    return [int(record.get("correct", 0))]


def _estimate_pass_at_k(n, c, k):
    """Unbiased pass@k estimator from Eq. (1) in arXiv:2107.03374."""
    if n <= 0 or k <= 0:
        return 0.0
    c = max(0, min(int(c), int(n)))
    if c == 0:
        return 0.0

    k = min(int(k), int(n))
    if n - c < k:
        return 1.0

    denom_terms = np.arange(n - c + 1, n + 1, dtype=np.float64)
    return float(1.0 - np.prod(1.0 - (k / denom_terms)))


def _compute_pass_at_k(correct_lists, max_k):
    """Compute pass@k using Eq. (1) estimator over per-problem (n, c)."""
    if not correct_lists:
        return {k: 0.0 for k in range(1, max_k + 1)}

    num_samples = np.array([len(correct_list) for correct_list in correct_lists], dtype=np.int32)
    num_correct = np.array([sum(correct_list) for correct_list in correct_lists], dtype=np.int32)

    pass_at_k = {}
    for k in range(1, max_k + 1):
        estimates = [
            _estimate_pass_at_k(n=int(n), c=int(c), k=k)
            for n, c in zip(num_samples, num_correct)
        ]
        pass_at_k[k] = float(np.mean(estimates)) if estimates else 0.0
    return pass_at_k


def eval_subject(args, subject, llm, test_records, kk_proc, exist_result_records):
    """Evaluate one subject."""
    cors = []
    subject_correct_lists = []

    start_index = len(exist_result_records)
    print(f"Found existing {start_index} records in {subject}")
    for i in range(start_index):
        correct_list = _normalize_correct_list(exist_result_records[i])
        n = len(correct_list)
        c = sum(correct_list)
        subject_correct_lists.append(correct_list)
        cors.append(_estimate_pass_at_k(n=n, c=c, k=1))

    eval_start_time = time.time()

    # Prepare all prompts
    prompts = []
    labels = []
    for i in range(start_index, len(test_records)):
        prompt, label = kk_proc.gen_test_prompt(args.ntrain, test_records, i, args.model)
        prompts.append(prompt)
        if i == start_index:
            print(f"Sample prompt:\n{prompt}")
        labels.append(label)

    # Get responses
    if args.use_vllm:
        responses = batch_decode_vllm(
            llm,
            prompts,
            batch_size=args.batch_size,
            num_generation=args.num_generation,
            temperature=args.temperature if args.num_generation > 1 else 0.0,
        )
        if args.num_generation == 1:
            responses = [[response] for response in responses]
    else:
        responses = []
        do_sample = args.num_generation > 1
        temperature = args.temperature if do_sample else 0.0
        for index, prompt in enumerate(prompts):
            response_list = []
            for _ in range(args.num_generation):
                response = llm.query(
                    prompt,
                    do_sample=do_sample,
                    temperature=temperature,
                )
                response_list.append(response)
            responses.append(response_list)
            if index % 1 == 0:
                print(f"\nResponse {index} (sample 0):\n{response_list[0]}")
                print(f"\nLabel {index}:\n{labels[index]}")

    # Process results
    for i, (prompt, label, response_list) in enumerate(
        zip(prompts, labels, responses), start=start_index
    ):
        correct_list = []
        parsed_pred_list = []
        reformat_gold_conditions = None

        for gen_i, response in enumerate(response_list):
            cor, parsed_pred, reformat_gold_conditions = kk_proc._parse_cot_eval(
                response, label, args.model
            )
            correct_list.append(int(cor))
            parsed_pred_list.append(parsed_pred)

            print(
                f"\nPrompt {i} sample {gen_i}:{prompt}"
                f"\nResponse {i} sample {gen_i}:{response}"
                f"\nPrediction {i} sample {gen_i}:{parsed_pred}"
                f"\nLabel {i}:{reformat_gold_conditions}"
                f"\nCorrect {i} sample {gen_i}:{cor}"
            )

        num_samples = len(correct_list)
        num_correct = sum(correct_list)
        first_correct = int(correct_list[0]) if correct_list else 0
        pass_at_1 = _estimate_pass_at_k(n=num_samples, c=num_correct, k=1)
        pass_at_n = _estimate_pass_at_k(n=num_samples, c=num_correct, k=num_samples)
        cors.append(pass_at_1)
        subject_correct_lists.append(correct_list)

        new_item = {
            "quiz": test_records[i]["quiz"],
            "names": test_records[i]["names"],
            "solution": test_records[i]["solution"],
            "solution_text": test_records[i]["solution_text"],
            "solution_text_format": test_records[i]["solution_text_format"],
            "index": test_records[i]["index"],
            "predicts": parsed_pred_list[0],
            "predicts_list": parsed_pred_list,
            "labels": reformat_gold_conditions,
            "correct": first_correct,
            "correct_list": correct_list,
            "pass_at_1": pass_at_1,
            f"pass_at_{args.num_generation}": pass_at_n,
            "response": response_list[0],
            "response_list": response_list,
            "prompts": prompt,
        }
        exist_result_records.append(new_item)

    eval_end_time = time.time()
    eval_time = eval_end_time - eval_start_time
    pass_at_k = _compute_pass_at_k(subject_correct_lists, args.num_generation)
    acc = pass_at_k[1]
    cors = np.array(cors, dtype=np.float32)

    print("Pass@1 {:.3f} - {}".format(acc, subject))
    for k in range(1, args.num_generation + 1):
        print(f"Pass@{k} {pass_at_k[k]:.3f} - {subject}")
    print(f"Total evaluation time: {eval_time:.2f} seconds")

    return cors, acc, pass_at_k, subject_correct_lists, exist_result_records


def load_limited_test_records(args, subject, exist_result_records):
    """Load limited test records based on given arguments."""
    test_records = load_eval_records(args, subject)

    if args.limit is not None:
        test_records = test_records.select(range(min(args.limit, len(test_records))))
        if args.limit <= len(exist_result_records):
            return None  # have finished exp

    return test_records


def save_final_acc_results(all_cors, all_correct_lists, results, fname, num_generation):
    """Process final results, calculate weighted pass@k, and save to file."""
    if all_cors:
        weighted_acc = float(np.mean(np.concatenate(all_cors)))
        results["weighted_accuracy"] = weighted_acc
        print(f"Weighted Pass@1: {weighted_acc:.3f}")

        weighted_pass_at_k = _compute_pass_at_k(all_correct_lists, num_generation)
        results["weighted_pass_at_k"] = {
            f"pass@{k}": weighted_pass_at_k[k] for k in range(1, num_generation + 1)
        }
        for k in range(1, num_generation + 1):
            print(f"Weighted Pass@{k}: {weighted_pass_at_k[k]:.3f}")

        with open(fname, "w") as f:
            json.dump(results, f)


def load_previous_acc_results(fname):
    """Load previous accuracy results."""
    acc_results = {"subject": {}}
    if os.path.isfile(fname):
        with open(fname, "r", encoding="utf-8") as file:
            acc_results = json.load(file)
        print(f"Previous Results loaded successfully: {acc_results}")
    return acc_results


def get_subjects_to_eval(args):
    """Get subjects to evaluate."""

    subjects = []
    if args.split == "test":
        if args.eval_nppl == 0:
            subjects = [f"people{nppl}_num100" for nppl in range(2, 9)]
        else:
            subjects = [f"people{args.eval_nppl}_num100"]
    elif args.split == "train":
        if args.eval_nppl == 2:
            subjects = ["people2_num200"]
        elif args.eval_nppl > 2:
            subjects = [f"people{args.eval_nppl}_num1000"]
    return subjects


def main(args):
    model_short_name = "/".join(args.model.split("/")[-2:])
    if args.lora_path:
        lora_short_name = "-".join(args.lora_path.strip("/").split("/")[-2:])
        model_short_name = f"{model_short_name}__lora__{lora_short_name}"

    prefix = os.path.join(
        os.path.join(args.save_dir, "{}_{}shot".format(model_short_name, args.ntrain))
    )

    args.config += (
        f"_token{args.max_token}{('_cot' if args.cot else '')}"
        f"_{args.split}{('_' + args.problem_type if args.problem_type != 'clean' else '')}"
        f"_gen{args.num_generation}"
        f"{(f'_temp{args.temperature:g}' if args.num_generation > 1 else '')}"
    )

    output_folder = os.path.join(prefix, args.config)
    acc_fname = os.path.join(prefix, f"result_{args.config}.json")
    os.makedirs(output_folder, exist_ok=True)

    print("args.config", args.config, "\nprefix", prefix, "\noutput_folder", output_folder)

    kk_proc = KKProcessor(cot=args.cot, no_linebreak=args.no_linebreak)

    subjects = get_subjects_to_eval(args)
    acc_results = load_previous_acc_results(acc_fname)

    llm = None
    all_cors = []
    all_correct_lists = []
    for subject in subjects:
        result_outfile = os.path.join(output_folder, "{}.jsonl".format(subject))
        exist_result_records = load_jsonl(result_outfile) if os.path.exists(result_outfile) else []
        test_records = load_limited_test_records(args, subject, exist_result_records)
        if test_records is None:
            continue

        llm = llm or load_llm(args)

        cors, acc, pass_at_k, subject_correct_lists, result_records = eval_subject(
            args, subject, llm, test_records, kk_proc, exist_result_records
        )

        write_jsonl(result_outfile, result_records)
        all_cors.append(cors)
        all_correct_lists.extend(subject_correct_lists)
        acc_results["subject"][subject] = acc
        acc_results.setdefault("subject_pass_at_k", {})
        acc_results["subject_pass_at_k"][subject] = {
            f"pass@{k}": pass_at_k[k] for k in range(1, args.num_generation + 1)
        }

    save_final_acc_results(
        all_cors,
        all_correct_lists,
        acc_results,
        acc_fname,
        args.num_generation,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for KK dataset")
    parser.add_argument("--ntrain", "-k", type=int, default=0, help="Number of training examples")
    parser.add_argument("--data_dir", "-d", type=str, default="data", help="Data directory")
    parser.add_argument("--save_dir", "-s", type=str, default="result_qa", help="Save directory")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model name or path")
    parser.add_argument("--arch", type=str, default=None, help="Model architecture")
    parser.add_argument(
        "--lora_path",
        "--adapter_path",
        dest="lora_path",
        type=str,
        default=None,
        help="Optional LoRA adapter path to load on top of --model",
    )
    parser.add_argument(
        "--no_merge_lora",
        action="store_true",
        help="Keep LoRA adapter unmerged (default merges LoRA into base model)",
    )
    parser.add_argument("--config", "-c", type=str, default="", help="Configuration string")
    parser.add_argument("--max_token", type=int, default=1024, help="Maximum number of tokens")
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Optional vLLM max context length override (e.g., 4096).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples")
    parser.add_argument("--cot", action="store_true", help="Use chain-of-thought prompting")
    parser.add_argument("--no_linebreak", action="store_true", help="Remove line breaks")
    parser.add_argument("--use_vllm", action="store_true", help="Use VLLM for inference")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for VLLM")
    parser.add_argument("--split", type=str, default="test", choices=["test", "train"], help="Data split to use")
    parser.add_argument("--eval_nppl", type=int, default=0, help="Number of people to evaluate")
    parser.add_argument("--problem_type", type=str, default="clean", help="Problem perturbation type")
    parser.add_argument(
        "--num_generation",
        type=int,
        default=1,
        help="Number of samples to generate per problem (used for pass@k).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature when --num_generation > 1.",
    )

    args = parser.parse_args()
    if args.num_generation < 1:
        raise ValueError("--num_generation must be >= 1")
    if args.max_model_len is not None and args.max_model_len < 1:
        raise ValueError("--max_model_len must be >= 1")
    if args.use_vllm:
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    init_seed(seed_cuda=not args.use_vllm)
    main(args)
