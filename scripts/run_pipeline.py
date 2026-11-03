import argparse
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def extract_binary(text: str) -> str:
    t = text.upper()
    if re.search(r'\bNOT\s+VULNERABLE\b', t):
        return "SAFE"
    if re.search(r'\bNOT\s+SAFE\b', t):
        return "VULNERABLE"
    if re.search(r'\b(UNSAFE|INSEGUR[OA]|VULNERABLE|VULN)\b', t):
        return "VULNERABLE"
    if re.search(r'\bSAFE\b', t):
        return "SAFE"
    return "UNKNOWN"


def extract_4class(text: str) -> str:
    t = text.upper()
    pats = {
        "XSS": r'\bXSS\b|\bCROSS[- ]SITE\b',
        "INJECTION": r'\b(INJECTION|SQLI|SQL\s+INJECTION|COMMAND\s+INJECTION|CODE\s+INJECTION|PATH\s+TRAVERSAL|PROTOTYPE\s+POLLUTION)\b',
        "AUTHZ": r'\b(AUTHZ?|AUTHORIZATION|AUTHENTICATION|IDOR|CSRF|ACCESS\s+CONTROL)\b',
        "SAFE": r'\bSAFE\b|\bSECURE\b',
    }
    hits = []
    for label, pat in pats.items():
        m = re.search(pat, t)
        if m:
            hits.append((m.start(), label))
    return sorted(hits)[0][1] if hits else "UNKNOWN"


def detect_test_type(test_data):
    labels = {ex["completion"].strip().upper() for ex in test_data}
    return "binary" if labels <= {"VULNERABLE", "SAFE"} else "4class"


def ensure_chat_template(tok):
    if getattr(tok, "chat_template", None) is not None:
        print("  -> chat_template: native")
        return
    name = getattr(tok, "name_or_path", "").lower()
    if "qwen" in name:
        print("  -> chat_template: injecting Qwen2.5 ChatML fallback")
        tok.chat_template = (
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
        print("  -> chat_template: injecting Llama-style fallback")
        tok.chat_template = (
            "{%- for message in messages -%}"
            "{{- '<|' + message['role'] + '|>\\n' + message['content'] + '\\n' -}}"
            "{%- endfor -%}"
            "{%- if add_generation_prompt -%}"
            "{{- '<|assistant|>\\n' -}}"
            "{%- endif -%}"
        )


def tokenize_with_labels(ex, tok, max_len):
    msgs_user = [{"role": "user", "content": ex["prompt"]}]
    msgs_full = msgs_user + [{"role": "assistant", "content": ex["completion"]}]
    prompt_text = tok.apply_chat_template(msgs_user, tokenize=False, add_generation_prompt=True)
    full_text = tok.apply_chat_template(msgs_full, tokenize=False)
    p_ids = tok(prompt_text, add_special_tokens=False).input_ids
    f_ids = tok(full_text, add_special_tokens=False).input_ids
    if len(f_ids) > max_len:
        raise RuntimeError(f"label truncated: {len(f_ids)} > {max_len}")
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return {"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": labels}


def evaluate(model, tok, test_data, test_type, max_new_tokens=20):
    model.eval()
    extractor = extract_binary if test_type == "binary" else extract_4class
    correct, total = 0, 0
    results = []
    cm = defaultdict(lambda: defaultdict(int))
    t0 = time.time()

    for ex in test_data:
        prompt = ex["prompt"]
        expected = ex["completion"].strip().upper()
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                temperature=None, top_p=None, top_k=None,
            )
        raw = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred = extractor(raw)
        ok = (pred == expected)
        correct += int(ok); total += 1
        cm[expected][pred] += 1
        results.append({"expected": expected, "predicted": pred, "raw": raw, "correct": ok, "prompt_full": prompt})

    acc = correct / total if total else 0.0
    classes = sorted({ex["completion"].strip().upper() for ex in test_data})
    recalls = []
    for c in classes:
        support = sum(cm[c].values())
        if support:
            recalls.append(cm[c][c] / support)
    bal = sum(recalls) / len(recalls) if recalls else 0.0

    f1s = []
    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[k][c] for k in cm if k != c)
        fn = sum(cm[c][k] for k in cm[c] if k != c)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal, 4),
        "macro_f1": round(macro_f1, 4),
        "distribution": dict(Counter(r["predicted"] for r in results)),
        "confusion_matrix": {k: dict(v) for k, v in cm.items()},
        "elapsed_s": round(time.time() - t0, 2),
        "results": results,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--test_file", required=True)
    p.add_argument("--output_dir", default="./lora_adapter")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=20)
    p.add_argument("--max_train_examples", type=int, default=None)
    p.add_argument("--max_test_examples", type=int, default=None)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    t_total = time.time()

    print(f"[1/6] Load: {args.base_model}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ensure_chat_template(tok)
    tok.padding_side = "right"

    set_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float32, trust_remote_code=True,
    ).to("cpu")
    print(f"  -> loaded in {time.time() - t0:.1f}s")

    print(f"[2/6] Test: {args.test_file}")
    test_data = load_jsonl(args.test_file)
    if args.max_test_examples:
        test_data = test_data[:args.max_test_examples]
    test_type = detect_test_type(test_data)
    print(f"  -> {len(test_data)} ex | type={test_type}")

    print("[3/6] PRE eval")
    pre = evaluate(model, tok, test_data, test_type, args.max_new_tokens)
    print(f"  -> acc={pre['accuracy']} bal={pre['balanced_accuracy']} f1={pre['macro_f1']}")

    print(f"[4/6] Train: {args.train_file}")
    train_data = load_jsonl(args.train_file)
    if args.max_train_examples:
        train_data = train_data[:args.max_train_examples]
    print(f"  -> {len(train_data)} examples")
    tokenized = [tokenize_with_labels(ex, tok, args.max_seq_length) for ex in train_data]
    train_ds = Dataset.from_list(tokenized)

    print("[5/6] LoRA + Trainer")
    set_seed(args.seed)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], bias="none",
    ))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        warmup_steps=10,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        use_cpu=True,
        fp16=False, bf16=False,
        dataloader_pin_memory=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    )

    t0 = time.time()
    out = trainer.train()
    train_s = time.time() - t0
    print(f"  -> trained in {train_s:.1f}s | loss={out.metrics.get('train_loss'):.4f}")

    print("[6/6] Save + POST eval")
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    post = evaluate(model, tok, test_data, test_type, args.max_new_tokens)
    print(f"  -> acc={post['accuracy']} bal={post['balanced_accuracy']} f1={post['macro_f1']}")

    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": {
            "base_model": args.base_model, "train_file": args.train_file,
            "test_file": args.test_file, "test_type": test_type,
            "num_train_examples": len(train_data), "num_test_examples": len(test_data),
            "num_epochs": args.num_epochs, "lr": args.lr, "seed": args.seed,
            "max_seq_length": args.max_seq_length, "max_new_tokens": args.max_new_tokens,
        },
        "pre": {k: pre[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "distribution", "confusion_matrix")},
        "post": {k: post[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "distribution", "confusion_matrix")},
        "improvement": {
            "accuracy": round(post["accuracy"] - pre["accuracy"], 4),
            "balanced_accuracy": round(post["balanced_accuracy"] - pre["balanced_accuracy"], 4),
            "macro_f1": round(post["macro_f1"] - pre["macro_f1"], 4),
        },
        "trainer": {
            "train_loss": round(out.metrics.get("train_loss", 0.0), 4),
            "train_runtime": round(out.metrics.get("train_runtime", 0.0), 2),
        },
        "timings": {"train_s": round(train_s, 2), "total_s": round(time.time() - t_total, 2)},
        "predictions_post": post["results"],
    }

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
