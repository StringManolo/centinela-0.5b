import argparse
import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer


LABELS = ["XSS", "INJECTION", "AUTHZ", "SAFE"]


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_answer(text):
    """Extract one of LABELS or VULNERABLE/SAFE from the model output."""
    text = text.upper()
    # 4-class labels
    for label in ["INJECTION", "AUTHZ", "XSS", "SAFE"]:
        if label in text:
            return label
    if "VULNERABLE" in text:
        return "VULNERABLE"
    return "UNKNOWN"


def extract_binary(text):
    """Binary extractor (used in reports to re-score paraphrased runs)."""
    text = text.upper()
    # Check VULNERABLE first (longer, more specific than SAFE)
    if "VULNERABLE" in text:
        return "VULNERABLE"
    if "SAFE" in text:
        return "SAFE"
    return "UNKNOWN"


def ensure_chat_template(tokenizer):
    if getattr(tokenizer, "chat_template", None) is not None:
        return
    name = getattr(tokenizer, "name_or_path", "").lower()
    is_qwen = "qwen" in name
    if is_qwen:
        print("  -> injecting Qwen2.5 ChatML fallback")
        tokenizer.chat_template = (
            "{%- if messages[0]['role'] == 'system' -%}"
            "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' -}}"
            "{%- else -%}"
            "{{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' -}}"
            "{%- endif -%}"
            "{%- for message in messages -%}"
            "{%- if message['role'] != 'system' -%}"
            "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' -}}"
            "{%- endif -%}"
            "{%- endfor -%}"
            "{%- if add_generation_prompt -%}"
            "{{- '<|im_start|>assistant\\n' -}}"
            "{%- endif -%}"
        )
    else:
        print("  -> WARNING: non-Qwen, injecting Llama-style fallback")
        tokenizer.chat_template = (
            "{%- for message in messages -%}"
            "{{- '<|' + message['role'] + '|>\\n' + message['content'] + '\\n' -}}"
            "{%- endfor -%}"
            "{%- if add_generation_prompt -%}"
            "{{- '<|assistant|>\\n' -}}"
            "{%- endif -%}"
        )


def augment_prompt_order(data):
    out = []
    for i, ex in enumerate(data):
        ex2 = dict(ex)
        if i % 2 == 1:
            ex2["prompt"] = ex["prompt"].replace(
                "Respond with exactly one of: XSS, INJECTION, AUTHZ, SAFE.",
                "Respond with exactly one of: SAFE, AUTHZ, INJECTION, XSS.",
            )
        out.append(ex2)
    return out


def format_as_chat(example, tokenizer):
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


def evaluate_model(model, tokenizer, test_data, num_runs=3, max_new_tokens=20):
    model.eval()
    correct = 0
    total = 0
    results = []
    cm = defaultdict(lambda: defaultdict(int))
    t_start = time.time()

    for example in test_data:
        prompt = example["prompt"]
        expected = example["completion"].strip().upper()

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        run_answers = []
        run_raws = []
        for _ in range(num_runs):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            run_answers.append(extract_answer(response))
            run_raws.append(response)

        final_answer = max(set(run_answers), key=run_answers.count)
        is_correct = final_answer == expected
        correct += int(is_correct)
        total += 1
        cm[expected][final_answer] += 1
        results.append({
            "prompt": prompt[:100] + "...",
            "prompt_full": prompt,
            "expected": expected,
            "predicted": final_answer,
            "raw_responses": run_raws,
            "correct": is_correct,
        })

    accuracy = correct / total if total > 0 else 0.0
    recalls = []
    for label in LABELS:
        support = sum(cm[label].values())
        if support > 0:
            recalls.append(cm[label][label] / support)
    balanced_acc = sum(recalls) / len(recalls) if recalls else 0.0

    elapsed = time.time() - t_start
    return accuracy, balanced_acc, results, dict(cm), elapsed


def re_score_binary(results):
    """
    Re-score the same results using the binary extractor on the raw responses.
    Useful for paraphrased binary runs where the 4-class extractor returns UNKNOWN.
    Returns: accuracy, balanced_acc, distribution, confusion
    """
    correct = 0
    total = 0
    cm = defaultdict(lambda: defaultdict(int))
    for r in results:
        expected = r["expected"]
        raw_runs = r.get("raw_responses", [])
        # Apply binary extractor to each raw response
        binary_runs = [extract_binary(raw) for raw in raw_runs]
        final = max(set(binary_runs), key=binary_runs.count) if binary_runs else "UNKNOWN"
        is_correct = final == expected
        correct += int(is_correct)
        total += 1
        cm[expected][final] += 1
    acc = correct / total if total else 0.0
    # Binary balanced accuracy (2 classes)
    recalls = []
    for label in ["VULNERABLE", "SAFE"]:
        support = sum(cm[label].values())
        if support > 0:
            recalls.append(cm[label][label] / support)
    bal = sum(recalls) / len(recalls) if recalls else 0.0
    dist = dict(Counter(
        max(set([extract_binary(raw) for raw in r.get("raw_responses", [])]),
            key=[extract_binary(raw) for raw in r.get("raw_responses", [])].count)
        for r in results
    ))
    return acc, bal, dist, {k: dict(v) for k, v in cm.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./lora_adapter")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_train_examples", type=int, default=None)
    parser.add_argument("--max_test_examples", type=int, default=None)
    parser.add_argument("--alternate-order", action="store_true", default=False)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timings = {}
    t_total_start = time.time()

    # 1. Load
    print(f"[1/7] Loading: {args.base_model}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ensure_chat_template(tokenizer)
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32, trust_remote_code=True)
    model.to("cpu")
    timings["model_load_s"] = round(time.time() - t0, 2)
    print(f"  -> loaded in {timings['model_load_s']}s")

    # 2. Test
    print(f"[2/7] Test: {args.test_file}")
    test_data = load_jsonl(args.test_file)
    if args.max_test_examples:
        test_data = test_data[:args.max_test_examples]
    print(f"  -> {len(test_data)} test examples")

    # 3. PRE eval
    print(f"[3/7] PRE eval (max_new_tokens={args.max_new_tokens})...")
    acc_before, bal_before, results_before, cm_before, eval_before_s = evaluate_model(
        model, tokenizer, test_data, num_runs=3, max_new_tokens=args.max_new_tokens
    )
    timings["eval_before_s"] = round(eval_before_s, 2)
    dist_before = dict(Counter(r["predicted"] for r in results_before))
    # Binary re-score of the same raw responses
    bin_acc_b, bin_bal_b, bin_dist_b, bin_cm_b = re_score_binary(results_before)
    print(f"  -> PRE acc: {acc_before:.2%} | balanced: {bal_before:.2%}")
    print(f"  -> PRE dist: {dist_before}")
    print(f"  -> PRE binary rescore: acc {bin_acc_b:.2%} | bal {bin_bal_b:.2%} | dist {bin_dist_b}")

    # 4. LoRA
    print("[4/7] LoRA...")
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], bias="none",
    ))

    # 5. Train data
    print(f"[5/7] Train: {args.train_file}")
    train_data = load_jsonl(args.train_file)
    if args.max_train_examples:
        train_data = train_data[:args.max_train_examples]
    if args.alternate_order:
        train_data = augment_prompt_order(train_data)
    print(f"  -> {len(train_data)} examples (alt_order={args.alternate_order})")
    train_dataset = Dataset.from_list(train_data).map(lambda ex: format_as_chat(ex, tokenizer))

    # 6. Train
    print(f"[6/7] Training ({args.num_epochs} epochs)...")
    training_args = TrainingArguments(
        output_dir=args.output_dir, num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=2,
        logging_steps=1, save_strategy="no", report_to="none",
        use_cpu=True, fp16=False, bf16=False, dataloader_pin_memory=False,
    )
    trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_dataset, tokenizer=tokenizer, max_seq_length=512)
    t0 = time.time()
    train_result = trainer.train()
    timings["train_s"] = round(time.time() - t0, 2)
    print(f"  -> trained in {timings['train_s']}s")

    # 7. Save
    print(f"[7/7] Saving to {args.output_dir}")
    t0 = time.time()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    timings["save_s"] = round(time.time() - t0, 2)

    # 8. POST eval
    print("POST eval...")
    acc_after, bal_after, results_after, cm_after, eval_after_s = evaluate_model(
        model, tokenizer, test_data, num_runs=3, max_new_tokens=args.max_new_tokens
    )
    timings["eval_after_s"] = round(eval_after_s, 2)
    dist_after = dict(Counter(r["predicted"] for r in results_after))
    bin_acc_a, bin_bal_a, bin_dist_a, bin_cm_a = re_score_binary(results_after)
    print(f"  -> POST acc: {acc_after:.2%} | balanced: {bal_after:.2%}")
    print(f"  -> POST dist: {dist_after}")
    print(f"  -> POST binary rescore: acc {bin_acc_a:.2%} | bal {bin_bal_a:.2%} | dist {bin_dist_a}")

    timings["total_s"] = round(time.time() - t_total_start, 2)

    # 9. Metrics
    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "base_model": args.base_model,
        "num_train_examples": len(train_data),
        "num_test_examples": len(test_data),
        "num_epochs": args.num_epochs,
        "alternate_order": args.alternate_order,
        "max_new_tokens": args.max_new_tokens,
        "accuracy_before": round(acc_before, 4),
        "accuracy_after": round(acc_after, 4),
        "balanced_accuracy_before": round(bal_before, 4),
        "balanced_accuracy_after": round(bal_after, 4),
        "improvement_balanced": round(bal_after - bal_before, 4),
        "prediction_distribution_before": dist_before,
        "prediction_distribution_after": dist_after,
        "confusion_matrix_before": {k: dict(v) for k, v in cm_before.items()},
        "confusion_matrix_after": {k: dict(v) for k, v in cm_after.items()},
        "binary_rescore_before": {
            "accuracy": round(bin_acc_b, 4),
            "balanced_accuracy": round(bin_bal_b, 4),
            "distribution": bin_dist_b,
            "confusion_matrix": bin_cm_b,
        },
        "binary_rescore_after": {
            "accuracy": round(bin_acc_a, 4),
            "balanced_accuracy": round(bin_bal_a, 4),
            "distribution": bin_dist_a,
            "confusion_matrix": bin_cm_a,
        },
        "timings": timings,
        "trainer_metrics": {
            "train_runtime": train_result.metrics.get("train_runtime"),
            "train_loss": train_result.metrics.get("train_loss"),
        },
        "predictions_after": results_after,
    }

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
