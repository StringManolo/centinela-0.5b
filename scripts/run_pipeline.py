import argparse
import json
import os
import time
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


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_answer(text):
    """Extract VULNERABLE or SAFE from the model output."""
    text = text.upper()
    if "VULNERABLE" in text:
        return "VULNERABLE"
    if "SAFE" in text:
        return "SAFE"
    return "UNKNOWN"


def evaluate_model(model, tokenizer, test_data, num_runs=3):
    """Evaluate the model num_runs times per example and take the majority vote (greedy decoding)."""
    model.eval()
    correct = 0
    total = 0
    results = []
    t_start = time.time()

    for example in test_data:
        prompt = example["prompt"]
        expected = example["completion"].strip().upper()

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        run_answers = []
        for _ in range(num_runs):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                )
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
            run_answers.append(extract_answer(response))

        final_answer = max(set(run_answers), key=run_answers.count)
        is_correct = final_answer == expected
        correct += int(is_correct)
        total += 1
        results.append({
            "prompt": prompt[:100] + "...",
            "expected": expected,
            "predicted": final_answer,
            "correct": is_correct,
        })

    accuracy = correct / total if total > 0 else 0.0
    elapsed = time.time() - t_start
    return accuracy, results, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./lora_adapter")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_train_examples", type=int, default=None,
                        help="Limit the number of training examples (for sweeps)")
    parser.add_argument("--max_test_examples", type=int, default=None,
                        help="Limit the number of test examples (for sweeps)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timings = {}
    t_total_start = time.time()

    # 1. Load model
    print(f"[1/7] Loading tokenizer and model: {args.base_model}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.to("cpu")
    timings["model_load_s"] = round(time.time() - t0, 2)
    print(f"  -> loaded in {timings['model_load_s']}s")

    # 2. Load test set
    print(f"[2/7] Loading test set: {args.test_file}")
    test_data = load_jsonl(args.test_file)
    if args.max_test_examples:
        test_data = test_data[:args.max_test_examples]
    print(f"  -> {len(test_data)} test examples")

    # 3. Pre-training evaluation
    print("[3/7] Running PRE-training evaluation...")
    acc_before, results_before, eval_before_s = evaluate_model(
        model, tokenizer, test_data, num_runs=3
    )
    timings["eval_before_s"] = round(eval_before_s, 2)
    print(f"  -> PRE accuracy: {acc_before:.2%} ({timings['eval_before_s']}s)")

    # 4. LoRA config
    print("[4/7] Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. Load training set
    print(f"[5/7] Loading training set: {args.train_file}")
    train_data = load_jsonl(args.train_file)
    if args.max_train_examples:
        train_data = train_data[:args.max_train_examples]
    print(f"  -> {len(train_data)} training examples")

    train_dataset = Dataset.from_list([
        {"text": f"{ex['prompt']}\n\nAnswer: {ex['completion']}"}
        for ex in train_data
    ])

    # 6. Train
    print(f"[6/7] Training ({args.num_epochs} epochs, batch={args.batch_size})...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        use_cpu=True,
        fp16=False,
        bf16=False,
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_seq_length=512,
    )

    t0 = time.time()
    train_result = trainer.train()
    timings["train_s"] = round(time.time() - t0, 2)
    print(f"  -> trained in {timings['train_s']}s")

    # 7. Save adapter
    print(f"[7/7] Saving adapter to {args.output_dir}...")
    t0 = time.time()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    timings["save_s"] = round(time.time() - t0, 2)

    # 8. Post-training evaluation
    print("Running POST-training evaluation...")
    acc_after, results_after, eval_after_s = evaluate_model(
        model, tokenizer, test_data, num_runs=3
    )
    timings["eval_after_s"] = round(eval_after_s, 2)
    print(f"  -> POST accuracy: {acc_after:.2%} ({timings['eval_after_s']}s)")

    timings["total_s"] = round(time.time() - t_total_start, 2)

    # 9. Structured metrics
    improvement = acc_after - acc_before
    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "base_model": args.base_model,
        "num_train_examples": len(train_data),
        "num_test_examples": len(test_data),
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "max_train_examples": args.max_train_examples,
        "max_test_examples": args.max_test_examples,
        "acc_before": round(acc_before, 4),
        "acc_after": round(acc_after, 4),
        "improvement": round(improvement, 4),
        "timings": timings,
        "trainer_metrics": {
            "train_runtime": train_result.metrics.get("train_runtime"),
            "train_loss": train_result.metrics.get("train_loss"),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        },
        "predictions_after": results_after,
    }

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to {metrics_path}")

    # 10. Markdown report for the GitHub Job Summary
    report = f"""
# Centinela-0.5b - Run report

## Configuration
- **Base model**: `{args.base_model}`
- **Train**: {len(train_data)} examples ({args.num_epochs} epochs)
- **Test**: {len(test_data)} examples
- **Batch size**: {args.batch_size}
- **Decoding**: greedy, 3 passes with majority vote

## Results

| Metric | Value |
| :--- | :--- |
| PRE accuracy | **{acc_before:.2%}** |
| POST accuracy | **{acc_after:.2%}** |
| **Improvement** | **{improvement:+.2%}** |

## Timings

| Phase | Seconds |
| :--- | :--- |
| Model load | {timings['model_load_s']} |
| PRE eval | {timings['eval_before_s']} |
| Training | {timings['train_s']} |
| Save | {timings['save_s']} |
| POST eval | {timings['eval_after_s']} |
| **Total** | **{timings['total_s']}** |

## POST predictions

| Prompt (truncated) | Expected | Predicted | OK |
| :--- | :--- | :--- | :--- |
"""
    for r in results_after:
        emoji = "OK" if r["correct"] else "FAIL"
        report += f"| `{r['prompt']}` | {r['expected']} | {r['predicted']} | {emoji} |\n"

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(report)
        print("Report written to GITHUB_STEP_SUMMARY.")


if __name__ == "__main__":
    main()
