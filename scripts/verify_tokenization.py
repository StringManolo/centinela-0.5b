#!/usr/bin/env python3
"""Verify tokenization: no train example truncates its label at max_seq_length."""
import argparse
import json
import sys
from transformers import AutoTokenizer


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3")
    p.add_argument("--files", nargs="+", required=True)
    p.add_argument("--max_seq_length", type=int, default=1024)
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Fallback ChatML if missing
    if getattr(tok, "chat_template", None) is None:
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

    global_ok = True
    for path in args.files:
        data = load_jsonl(path)
        truncated = 0
        max_full_len = 0
        max_label_len = 0
        for i, ex in enumerate(data):
            msgs_user = [{"role": "user", "content": ex["prompt"]}]
            msgs_full = msgs_user + [{"role": "assistant", "content": ex["completion"]}]
            prompt_text = tok.apply_chat_template(msgs_user, tokenize=False, add_generation_prompt=True)
            full_text = tok.apply_chat_template(msgs_full, tokenize=False)
            p_ids = tok(prompt_text, add_special_tokens=False).input_ids
            f_ids = tok(full_text, add_special_tokens=False).input_ids
            label_len = len(f_ids) - len(p_ids)
            max_full_len = max(max_full_len, len(f_ids))
            max_label_len = max(max_label_len, label_len)
            if len(f_ids) > args.max_seq_length:
                truncated += 1
                if truncated <= 3:
                    print(f"  TRUNCATED example {i}: {len(f_ids)} > {args.max_seq_length} "
                          f"(label would be lost)")
        status = "OK" if truncated == 0 else f"FAIL ({truncated} truncated)"
        print(f"{path}: {len(data)} ex | max_full={max_full_len} | max_label={max_label_len} | {status}")
        if truncated > 0:
            global_ok = False

    sys.exit(0 if global_ok else 1)


if __name__ == "__main__":
    main()
