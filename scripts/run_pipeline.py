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


def extract_binary(text):
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


def extract_4class(text):
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


def extract_10class(text):
    t = text.upper()
    pats = {
        "SQLI": r'\bSQLI?\b|\bSQL[\s-]*INJECTION\b',
        "XSS": r'\bXSS\b|\bCROSS[- ]?SITE\s+SCRIPT(ING)?\b',
        "RCE": r'\bRCE\b|\bREMOTE\s+CODE\b|\bCOMMAND[\s-]*INJECTION\b|\bOS[\s-]*COMMAND\b',
        "PATHTRAVERSAL": r'\bPATH[\s-]*TRAVERSAL\b|\bDIRECTORY[\s-]*TRAVERSAL\b',
        "SSRF": r'\bSSRF\b|\bSERVER[- ]?SIDE[\s-]*REQUEST\b',
        "PROTOTYPEPOLLUTION": r'\bPROTOTYPE[\s-]*POLLUTION\b',
        "IDOR": r'\bIDOR\b|\bINSECURE\s+DIRECT\s+OBJECT\b',
        "CSRF": r'\bCSRF\b|\bCROSS[- ]?SITE[\s-]*REQUEST\b',
        "DESERIALIZATION": r'\bDESERIALIZATION\b|\bUNSAFE[\s-]*DESERIALIZATION\b',
        "OPENREDIRECT": r'\bOPEN[\s-]*REDIRECT\b|\bUNVALIDATED[\s-]*REDIRECT\b',
    }
    hits = []
    for label, pat in pats.items():
        m = re.search(pat, t)
        if m:
            hits.append((m.start(), label))
    return sorted(hits)[0][1] if hits else "UNKNOWN"


def pick_extractor(test_type):
    if test_type == "binary":
        return extract_binary
    if test_type == "4class":
        return extract_4class
    return extract_10class


def detect_test_type(test_data):
    labels = {ex["completion"].strip().upper() for ex in test_data}
    if labels <= {"VULNERABLE", "SAFE"}:
        return "binary"
    if labels <= {"XSS", "INJECTION", "AUTHZ", "SAFE"}:
        return "4class"
    return "10class"


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
    p_text = tok.apply_chat_template(msgs_user, tokenize=False, add_generation_prompt=True)
    f_text = tok.apply_chat_template(msgs_full, tokenize=False)
    p_ids = tok(p_text, add_special_tokens=False).input_ids
    f_ids = tok(f_text, add_special_tokens=False).input_ids
    if len(f_ids) > max_len:
        raise RuntimeError(f"label truncated: {len(f_ids)} > {max_len}")
    return {
        "input_ids": f_ids,
        "attention_mask": [1] * len(f_ids),
        "labels": [-100] * len(p_ids) + f_ids[len(p_ids):],
    }


def evaluate(model, tok, test_data, test_type, max_new_tokens=30):
    model.eval()
    extractor = pick_extractor(test_type)
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
        correct += int(ok)
        total += 1
        cm[expected][pred] += 1
        results.append({
            "expected": expected,
            "predicted": pred,
            "raw": raw,
            "correct": ok,
            "prompt_full": prompt,
        })

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


def summarize(res):
    return {k: res[k] for k in ("accuracy", "balanced_accuracy", "macro_f1",
                                 "distribution", "confusion_matrix", "elapsed_s")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--test_file_mc", default=None)
    p.add_argument("--test_file_open", default=None)
    p.add_argument("--test_file", default=None)  # legacy single mode
    p.add_argument("--output_dir", default="./lora_adapter")
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_seq_length", type=int, default=1024)
    p.add_argument("--max_new_tokens", type=int, default=30)
    args = p.parse_args()

    if not (args.test_file_mc or args.test_file):
        raise SystemExit("need --test_file_mc or --test_file")

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

    dual_mode = args.test_file_mc is not None

    print("[2/6] Test data")
    if dual_mode:
        test_mc = load_jsonl(args.test_file_mc)
        test_open = load_jsonl(args.test_file_open) if args.test_file_open else None
        print(f"  -> MC: {len(test_mc)} ex" + (f" | open: {len(test_open)} ex" if test_open else ""))
    else:
        test_data = load_jsonl(args.test_file)
        test_type = detect_test_type(test_data)
        print(f"  -> {len(test_data)} ex | type={test_type}")

    print("[3/6] PRE eval")
    if dual_mode:
        pre_mc = evaluate(model, tok, test_mc, "10class", args.max_new_tokens)
        print(f"  -> MC: acc={pre_mc['accuracy']} f1={pre_mc['macro_f1']}")
        if test_open:
            pre_open = evaluate(model, tok, test_open, "10class", args.max_new_tokens)
            print(f"  -> open: acc={pre_open['accuracy']} f1={pre_open['macro_f1']}")
    else:
        pre = evaluate(model, tok, test_data, test_type, args.max_new_tokens)
        print(f"  -> acc={pre['accuracy']} f1={pre['macro_f1']}")

    print(f"[4/6] Train data: {args.train_file}")
    train_data = load_jsonl(args.train_file)
    tokenized = [tokenize_with_labels(ex, tok, args.max_seq_length) for ex in train_data]
    train_ds = Dataset.from_list(tokenized)
    print(f"  -> {len(train_data)} examples")

    train_loss = 0.0
    train_s = 0.0
    if args.num_epochs > 0:
        print(f"[5/6] LoRA + Trainer ({args.num_epochs} epochs)")
        set_seed(args.seed)
        model = get_peft_model(model, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"], bias="none",
        ))
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
            seed=args.seed, data_seed=args.seed,
        )
        trainer = Trainer(
            model=model, args=targs, train_dataset=train_ds,
            data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
        )
        t0 = time.time()
        out = trainer.train()
        train_s = time.time() - t0
        train_loss = out.metrics.get("train_loss", 0.0)
        print(f"  -> loss={train_loss:.4f} in {train_s:.1f}s")
        model.save_pretrained(args.output_dir)
        tok.save_pretrained(args.output_dir)
    else:
        print("[5/6] Skipping training (ep=0)")

    print("[6/6] POST eval")
    if dual_mode:
        post_mc = evaluate(model, tok, test_mc, "10class", args.max_new_tokens)
        print(f"  -> MC: acc={post_mc['accuracy']} f1={post_mc['macro_f1']}")
        post_open = None
        if test_open:
            post_open = evaluate(model, tok, test_open, "10class", args.max_new_tokens)
            print(f"  -> open: acc={post_open['accuracy']} f1={post_open['macro_f1']}")
    else:
        post = evaluate(model, tok, test_data, test_type, args.max_new_tokens)
        print(f"  -> acc={post['accuracy']} f1={post['macro_f1']}")

    metrics = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": {
            "base_model": args.base_model,
            "train_file": args.train_file,
            "num_train_examples": len(train_data),
            "num_epochs": args.num_epochs,
            "lr": args.lr,
            "seed": args.seed,
            "max_seq_length": args.max_seq_length,
            "max_new_tokens": args.max_new_tokens,
        },
        "trainer": {"train_loss": round(train_loss, 4), "train_s": round(train_s, 2)},
        "timings": {"total_s": round(time.time() - t_total, 2)},
    }

    if dual_mode:
        metrics["test_file_mc"] = args.test_file_mc
        metrics["test_file_open"] = args.test_file_open
        metrics["pre_mc"] = summarize(pre_mc)
        metrics["post_mc"] = summarize(post_mc)
        metrics["pre_mc_raw"] = pre_mc["results"]
        metrics["post_mc_raw"] = post_mc["results"]
        if test_open:
            metrics["pre_open"] = summarize(pre_open)
            metrics["post_open"] = summarize(post_open)
            metrics["pre_open_raw"] = pre_open["results"]
            metrics["post_open_raw"] = post_open["results"]
    else:
        metrics["test_file"] = args.test_file
        metrics["test_type"] = test_type
        metrics["pre"] = summarize(pre)
        metrics["post"] = summarize(post)
        metrics["pre_raw"] = pre["results"]
        metrics["post_raw"] = post["results"]

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
