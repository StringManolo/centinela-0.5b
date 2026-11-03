#!/usr/bin/env python3
"""
Probe an ALREADY TRAINED adapter on training rows, to test the "template confound" hypothesis
without retraining (about a minute on CPU for 40 rows).

For every probe row it scores the label under two prompts:
  T0        the first template (what 'train/seen' used in v2/v3)
  assigned  the template the row had during training under the OLD rule (row index modulo K, K = --old_k)

If the adapter fits its training data but only under the template it was trained with, expect:
  assigned  type_acc_forced close to 1.0 and a spread-out prediction distribution
  T0        type_acc_forced near 0.1 and predictions collapsed to the labels that co-occurred with T0
            (with the old rule and K=5: only SAFE, SSRF, XSS)

Usage:
  python scripts/probe_adapter.py --base_model huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3 \
      --adapter_dir ./lora_adapter --old_k 5 [--n_rows 40] [--score_mode auto|first|sum|mean]
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_pipeline as rp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--train_file", default="data/train.jsonl")
    ap.add_argument("--old_k", type=int, default=5, help="number of templates the adapter was trained with")
    ap.add_argument("--n_rows", type=int, default=40, help="rows to probe (first N of the train file, even)")
    ap.add_argument("--score_mode", choices=["auto", "first", "sum", "mean"], default="auto")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rp.ensure_chat_template(tok)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    model.eval()

    enc = rp.Encoder(tok)
    mode = args.score_mode if args.score_mode != "auto" else ("first" if enc.first_unique else "sum")
    rows = rp.load_jsonl(args.train_file)[:args.n_rows]
    print(f"probing {len(rows)} train rows, score mode {mode}, old rule: template = row index % {args.old_k}")

    for name in ("T0", "assigned"):
        scored = []
        for i, r in enumerate(rows):
            k = 0 if name == "T0" else i % args.old_k
            p_ids = enc.prompt_ids(rp.build_user_text(rp.TRAIN_TEMPLATES[k], r["code"], rp.label_order()))
            scored.append(dict(r, scores=rp.score_prompt(model, enc, p_ids, mode)))
        m = rp.compute_metrics(scored)
        preds = Counter(rp.LABELS[max(range(len(rp.LABELS)), key=lambda j: s["scores"][j])] for s in scored)
        print(f"\n[{name}] type_acc_forced={m['type_acc_forced']}  det_auroc={m['det_auroc']}  "
              f"pair_acc={m['pair_acc']}  mean_logp_correct={m['mean_logp_correct']}")
        print(f"[{name}] predicted labels: {dict(preds)}")


if __name__ == "__main__":
    main()
