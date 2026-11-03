#!/usr/bin/env python3
"""
Centinela pipeline v3.

Cambios vs v2 (los que importan):
  * label_order es CANÓNICO en train y eval. Barajar el orden de la lista de
    labels por ejemplo dejaba que el modelo resolviera la tarea memorizando
    "la respuesta es el token en la posición K" en vez de "feature de código
    -> label". Evidencia en v2: train/seen type_forced se quedó en azar (0.15)
    mientras train_loss caía a 0.003.
  * score_prompt devuelve la MEDIA POR TOKEN del log-prob, no la suma.
    Las sumas crudas están sesgadas hacia labels cortos (SAFE, XSS, SSRF =
    1 token), y eso solo explicaba el pred_dist colapsado
    {SAFE=124, SSRF=55, XSS=41} de v2.
  * Las métricas de detección (det_auroc, det_bal_acc, pair_acc) usan el campo
    ESTRUCTURAL 'role' ('vuln'/'safe'), no el label. Si no, con
    --permute_labels el punto y el bootstrap medían cosas distintas y el
    punto podía caer fuera de su propio CI.
  * --num_epochs 0 es una run válida: solo se hace la evaluación step-0
    (modelo base), se salta el entrenamiento.
"""
import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

try:  # permite importar helpers puros sin torch
    import torch
except ImportError:  # pragma: no cover
    torch = None

VULN_TYPES = [
    "XSS", "SQLI", "CMDI", "CODE_INJECTION", "PATH_TRAVERSAL",
    "SSRF", "PROTO_POLLUTION", "OPEN_REDIRECT", "AUTHZ", "WEAK_CRYPTO",
]
SAFE = "SAFE"
LABELS = VULN_TYPES + [SAFE]
SAFE_IDX = LABELS.index(SAFE)
VULN_IDX = [i for i in range(len(LABELS)) if i != SAFE_IDX]

TRAIN_TEMPLATES = [
    "Analyze the following JavaScript code for security vulnerabilities.\n"
    "If it is vulnerable, answer with the vulnerability type; if it is not, answer SAFE.\n"
    "Answer with exactly one of: {labels}.\n\nCode:\n```javascript\n{code}\n```",
    "You are a security code reviewer. Classify the JavaScript snippet below.\n"
    "Valid answers: {labels}.\nReply with the answer only.\n\n{code}",
    "Does this JavaScript code contain a security flaw? Name the flaw category or reply SAFE.\n"
    "Categories: {labels}\n\n```js\n{code}\n```\nAnswer:",
    "```javascript\n{code}\n```\nClassify the code above as one of {labels}. Output only the label.",
    "Security audit task. Read the code and output a single label from this list: {labels}.\n\n"
    "CODE START\n{code}\nCODE END",
]
HELDOUT_TEMPLATES = [
    "Review this JavaScript for security issues. Which category fits best? Options: {labels}. "
    "Reply with the option only.\n\n{code}",
    "{code}\n\nQuestion: what kind of vulnerability, if any, does the snippet above have? "
    "Choose from [{labels}]. If there is none, choose SAFE.",
    "Act as a static analysis tool. Input is JavaScript. Emit exactly one token from: {labels}.\n"
    "<code>\n{code}\n</code>",
]


# ----------------------------------------------------------------------------- data helpers
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def check_rows(rows, name):
    for r in rows:
        for k in ("id", "pair_id", "set", "label", "code", "role"):
            if k not in r:
                raise ValueError(f"{name}: row {r.get('id')} lacks field '{k}'")
        if r["label"] not in LABELS:
            raise ValueError(f"{name}: row {r['id']} has unknown label {r['label']!r}")
        if r["role"] not in ("vuln", "safe"):
            raise ValueError(f"{name}: row {r['id']} has bad role {r['role']!r}")


def label_order(key=None):
    """Orden CANÓNICO. `key` se acepta por compatibilidad con los call-sites.

    v2 barajaba `LABELS` por ejemplo. Eso crea una tarea de memorización
    posicional: el modelo aprende "la respuesta está en la posición K del
    prompt", no "feature -> label". Evidencia: train/seen type_forced se
    quedó en azar mientras train_loss bajaba a 0.003. Con orden canónico el
    prompt siempre lista los labels en el mismo orden y la posición deja de
    ser informativa.
    """
    return LABELS[:]


def build_user_text(template, code, order):
    return template.replace("{labels}", ", ".join(order)).replace("{code}", code)


def select_pairs(rows, n_pairs):
    order, groups = [], defaultdict(list)
    for r in rows:
        if r["pair_id"] not in groups:
            order.append(r["pair_id"])
        groups[r["pair_id"]].append(r)
    if n_pairs and n_pairs > 0:
        order = order[:n_pairs]
    return [r for pid in order for r in groups[pid]]


# ----------------------------------------------------------------------------- metrics
def lse(a):
    m = a.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=-1, keepdims=True)))[..., 0]


def auroc_arr(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def r4(x):
    if isinstance(x, (float, np.floating)):
        return None if math.isnan(x) else round(float(x), 4)
    return x


def pairs_of(rows):
    """Parejas por `role` ESTRUCTURAL, no por label.

    v2 usaba `r["label"] == SAFE`, lo que con --permute_labels mezclaba
    filas estructuralmente vuln con etiquetas SAFE, invalidando la
    comparación entre el punto y el bootstrap.
    """
    by = defaultdict(dict)
    for i, r in enumerate(rows):
        by[r["pair_id"]][r["role"]] = i
    return [(d["vuln"], d["safe"]) for d in by.values() if "vuln" in d and "safe" in d]


def code_features(code):
    lines = code.split("\n")
    return {"chars": len(code), "lines": len(lines),
            "comment_lines": sum(1 for l in lines if l.lstrip().startswith("//"))}


def shortcut_baselines(rows):
    if not rows:
        return {}
    prs = pairs_of(rows)
    is_vuln = np.asarray([r["role"] == "vuln" for r in rows])
    feats = [code_features(r["code"]) for r in rows]
    out = {}
    for f in ("chars", "lines", "comment_lines"):
        x = np.asarray([ft[f] for ft in feats], dtype=float)
        a = auroc_arr(x[is_vuln], x[~is_vuln])
        pa = float(np.mean([(x[v] > x[s]) + 0.5 * (x[v] == x[s]) for v, s in prs])) if prs else float("nan")
        out[f] = {"det_auroc": r4(max(a, 1 - a)), "pair_acc": r4(max(pa, 1 - pa))}
    out["best"] = {m: max(out[f][m] for f in ("chars", "lines", "comment_lines")) for m in ("det_auroc", "pair_acc")}
    return out


def compute_metrics(rows, ci=False, n_boot=1000):
    n = len(rows)
    if n == 0:
        return {}
    S = np.asarray([r["scores"] for r in rows], dtype=np.float64)
    y = np.asarray([LABELS.index(r["label"]) for r in rows])
    logp = S - lse(S)[:, None]
    pred = S.argmax(1)
    # ROLE estructural, no label: con --permute_labels el label está barajado
    # pero el rol sigue siendo vuln/safe
    is_vuln = np.asarray([r["role"] == "vuln" for r in rows])
    det = lse(S[:, VULN_IDX]) - S[:, SAFE_IDX]
    forced = np.asarray(VULN_IDX)[S[:, VULN_IDX].argmax(1)]
    forced_ok = (forced == y).astype(float)

    recalls = {}
    for c in range(len(LABELS)):
        m = y == c
        if m.any():
            recalls[LABELS[c]] = float((pred[m] == c).mean())
    rv = float((pred[is_vuln] != SAFE_IDX).mean()) if is_vuln.any() else float("nan")
    rs = float((pred[~is_vuln] == SAFE_IDX).mean()) if (~is_vuln).any() else float("nan")
    prs = pairs_of(rows)

    out = {
        "n": n,
        "n_vuln": int(is_vuln.sum()),
        "n_pairs": len(prs),
        "acc": float((pred == y).mean()),
        "macro_recall": float(np.mean(list(recalls.values()))),
        "det_auroc": auroc_arr(det[is_vuln], det[~is_vuln]),
        "det_bal_acc": float(np.nanmean([rv, rs])),
        "det_recall_vuln": rv,
        "det_recall_safe": rs,
        "type_acc_forced": float(forced_ok[is_vuln].mean()) if is_vuln.any() else float("nan"),
        "pair_acc": float(np.mean([(det[v] > det[s]) + 0.5 * (det[v] == det[s]) for v, s in prs]))
        if prs else float("nan"),
        "mean_logp_correct": float(logp[np.arange(n), y].mean()),
        "pred_dist": dict(Counter(LABELS[p] for p in pred)),
        "recall_per_label": {k: r4(v) for k, v in recalls.items()},
    }
    if ci and len(prs) >= 3:
        rng = np.random.default_rng(0)
        P = np.asarray(prs)
        boot = defaultdict(list)
        for _ in range(n_boot):
            pick = P[rng.integers(0, len(P), len(P))]
            iv, js = pick[:, 0], pick[:, 1]
            boot["det_auroc"].append(auroc_arr(det[iv], det[js]))
            boot["pair_acc"].append(float(np.mean((det[iv] > det[js]) + 0.5 * (det[iv] == det[js]))))
            boot["type_acc_forced"].append(float(forced_ok[iv].mean()))
        out["ci95"] = {k: [r4(float(np.percentile(v, 2.5))), r4(float(np.percentile(v, 97.5)))]
                       for k, v in boot.items()}
    return {k: (r4(v) if not isinstance(v, (dict, list)) else v) for k, v in out.items()}


def confusion(rows):
    cm = defaultdict(lambda: defaultdict(int))
    for r in rows:
        cm[r["label"]][LABELS[int(np.argmax(r["scores"]))]] += 1
    return {k: dict(v) for k, v in cm.items()}


# ----------------------------------------------------------------------------- tokenisation
def ensure_chat_template(tokenizer):
    if getattr(tokenizer, "chat_template", None) is not None:
        return "native"
    name = getattr(tokenizer, "name_or_path", "").lower()
    if "qwen" in name:
        print("  -> WARNING: no chat template, injecting Qwen2.5 ChatML fallback")
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
        return "fallback-qwen-chatml"
    print("  -> WARNING: no chat template, non-Qwen model, injecting Llama-style fallback")
    tokenizer.chat_template = (
        "{%- for message in messages -%}"
        "{{- '<|' + message['role'] + '|>\\n' + message['content'] + '\\n' -}}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}"
        "{{- '<|assistant|>\\n' -}}"
        "{%- endif -%}"
    )
    return "fallback-generic"


class Encoder:
    def __init__(self, tok):
        self.tok = tok
        dummy = [{"role": "user", "content": "x"}]
        base = tok.apply_chat_template(dummy, tokenize=False, add_generation_prompt=True)
        self.cand_ids = {}
        for lab in LABELS:
            full = tok.apply_chat_template(
                dummy + [{"role": "assistant", "content": lab}], tokenize=False)
            if not full.startswith(base):
                raise RuntimeError(
                    "chat template: the full conversation does not start with the generation "
                    "prompt; this template needs a custom split.")
            self.cand_ids[lab] = tok(full[len(base):], add_special_tokens=False).input_ids
        firsts = Counter(c[0] for c in self.cand_ids.values())
        if any(v > 1 for v in firsts.values()):
            print("  note: some labels share their first token (fine for sum-logprob scoring)")

    def prompt_ids(self, user_text):
        p = self.tok.apply_chat_template(
            [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True)
        return self.tok(p, add_special_tokens=False).input_ids


def build_train_features(enc, rows, n_templates, seed, max_len):
    feats, skipped, n_prompt, n_comp = [], [], 0, 0
    for i, r in enumerate(rows):
        tpl = TRAIN_TEMPLATES[i % n_templates]
        order = label_order()
        p = enc.prompt_ids(build_user_text(tpl, r["code"], order))
        c = enc.cand_ids[r["label"]]
        if len(p) + len(c) > max_len:
            skipped.append(r["id"])
            continue
        feats.append({
            "input_ids": p + c,
            "attention_mask": [1] * (len(p) + len(c)),
            "labels": [-100] * len(p) + c,
        })
        n_prompt += len(p)
        n_comp += len(c)
    stats = {
        "n": len(feats),
        "mean_prompt_tokens": round(n_prompt / max(1, len(feats)), 1),
        "mean_answer_tokens": round(n_comp / max(1, len(feats)), 2),
        "supervised_fraction": round(n_comp / max(1, n_prompt + n_comp), 4),
    }
    return feats, skipped, stats


# ----------------------------------------------------------------------------- evaluation
def score_prompt(model, enc, p_ids):
    """Media POR TOKEN del log P(candidate | prompt).

    La suma cruda está sesgada hacia labels cortos (SAFE, XSS, SSRF = 1 token)
    porque un label de N tokens suma N términos negativos. Esa sola bias
    explica el pred_dist colapsado que vimos en v2. Dividir por len(c) pone
    todos los labels en la misma escala de "log-prob por token".
    """
    with torch.no_grad():
        dev = next(model.parameters()).device
        ids = torch.tensor([p_ids], dtype=torch.long, device=dev)
        try:
            out = model(input_ids=ids, use_cache=True, num_logits_to_keep=1)
        except TypeError:
            out = model(input_ids=ids, use_cache=True)
        last = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        pkv = out.past_key_values
        scores = []
        for lab in LABELS:
            c = enc.cand_ids[lab]
            s = last[c[0]].item()
            if len(c) > 1:
                cin = torch.tensor([c[:-1]], dtype=torch.long, device=dev)
                o2 = model(input_ids=cin, past_key_values=copy.deepcopy(pkv), use_cache=True)
                lp = torch.log_softmax(o2.logits[0].float(), dim=-1)
                tgt = torch.tensor(c[1:], dtype=torch.long, device=dev)
                s += lp[torch.arange(len(c) - 1, device=dev), tgt].sum().item()
            scores.append(s / len(c))
    return scores


def eval_rows(model, enc, rows, cond):
    pidx, out = {}, []
    for r in rows:
        pidx.setdefault(r["pair_id"], len(pidx))
    for r in rows:
        if cond == "seen":
            tid, tpl = "T0", TRAIN_TEMPLATES[0]
        else:
            k = pidx[r["pair_id"]] % len(HELDOUT_TEMPLATES)
            tid, tpl = f"H{k}", HELDOUT_TEMPLATES[k]
        order = label_order()
        p_ids = enc.prompt_ids(build_user_text(tpl, r["code"], order))
        out.append({"id": r["id"], "pair_id": r["pair_id"], "set": r["set"],
                    "label": r["label"], "role": r["role"],
                    "cond": cond, "template": tid,
                    "scores": score_prompt(model, enc, p_ids)})
    return out


def eval_point(model, enc, test_by_set, probe_rows, shift, ci=False, keep_scores=False):
    was_training = model.training
    model.eval()
    metrics, dump = {}, []
    for sname, rows in test_by_set.items():
        for cond in (["seen", "shift"] if shift else ["seen"]):
            er = eval_rows(model, enc, rows, cond)
            metrics[f"{sname}/{cond}"] = compute_metrics(er, ci=ci)
            if ci:
                metrics[f"{sname}/{cond}"]["confusion"] = confusion(er)
            if keep_scores:
                dump += er
    if probe_rows:
        er = eval_rows(model, enc, probe_rows, "seen")
        metrics["train/seen"] = compute_metrics(er, ci=ci)
        if keep_scores:
            dump += er
    if was_training:
        model.train()
    return metrics, dump


# ----------------------------------------------------------------------------- reporting
KEY_METRICS = ["det_auroc", "pair_acc", "type_acc_forced", "mean_logp_correct"]
CHANCE = {"det_auroc": 0.5, "pair_acc": 0.5, "type_acc_forced": 0.1,
          "mean_logp_correct": round(-math.log(len(LABELS)), 3)}


def write_curve_csv(path, curve):
    keys = sorted({k for pt in curve for k in pt["metrics"]})
    cols = ["step", "epoch", "train_loss"] + [f"{k}:{m}" for k in keys
                                                for m in KEY_METRICS + ["acc", "macro_recall"]]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for pt in curve:
            row = [pt["step"], pt["epoch"], pt["train_loss"]]
            for k in keys:
                m = pt["metrics"].get(k, {})
                row += [m.get(x) for x in KEY_METRICS + ["acc", "macro_recall"]]
            w.writerow(row)


def summary_markdown(cfg, step0, final):
    lines = [f"### Centinela run: `{cfg['base_model']}`", "",
             f"train pairs **{cfg['n_train_pairs']}** (rows {cfg['n_train_rows']}), templates "
             f"**{cfg['train_templates']}**, steps **{cfg['total_steps']}** "
             f"(~{cfg['total_steps'] / max(1, cfg['steps_per_epoch']):.1f} epochs), lr **{cfg['lr']}**, "
             f"seed **{cfg['seed']}**, permute_labels **{cfg['permute_labels']}**", "",
             "step 0 -> final  [95% CI at final]. Chance: det_auroc 0.5, pair_acc 0.5, "
             "type_forced 0.1, mean_logp -2.398.", "",
             "| set/cond | det_auroc | pair_acc | type_forced | mean_logp |", "|---|---|---|---|---|"]
    for k in sorted(final):
        cells = []
        for m in KEY_METRICS:
            a = step0.get(k, {}).get(m) if step0 else None
            b = final[k].get(m)
            ci = final[k].get("ci95", {}).get(m)
            cells.append(f"{a} -> {b}" + (f" {ci}" if ci else ""))
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    sb = cfg.get("shortcut_baselines") or {}
    if sb:
        lines += ["", "Best trivial single-feature baseline (length / lines / comment count):", "",
                  "| set | det_auroc | pair_acc |", "|---|---|---|"]
        lines += [f"| {s} | {b['best']['det_auroc']} | {b['best']['pair_acc']} |" for s, b in sb.items()]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--train_file", default="data/train.jsonl")
    ap.add_argument("--test_file", default="data/test.jsonl")
    ap.add_argument("--output_dir", default="./lora_adapter")
    ap.add_argument("--train_pairs", type=int, default=0)
    ap.add_argument("--train_templates", type=int, default=5)
    ap.add_argument("--num_epochs", type=float, default=8)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_scheduler", default="constant_with_warmup")
    ap.add_argument("--warmup_steps", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_targets", default="q_proj,v_proj")
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_every", type=int, default=0)
    ap.add_argument("--eval_sets", default="")
    ap.add_argument("--eval_pairs_per_set", type=int, default=0)
    ap.add_argument("--train_probe_n", type=int, default=40)
    ap.add_argument("--no_shift_eval", action="store_true")
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--permute_labels", action="store_true")
    ap.add_argument("--skip_step0", action="store_true")
    args = ap.parse_args(argv)
    if not 1 <= args.train_templates <= len(TRAIN_TEMPLATES):
        ap.error(f"--train_templates must be in 1..{len(TRAIN_TEMPLATES)}")
    return args


def main():
    args = parse_args()
    if torch is None:
        sys.exit("torch is required to run the pipeline")
    import peft
    import transformers
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq,
                              Trainer, TrainerCallback, TrainingArguments, set_seed)

    os.makedirs(args.output_dir, exist_ok=True)
    timings, t_all = {}, time.time()
    set_seed(args.seed)

    # 1. model + tokenizer
    print(f"[1/6] Loading {args.base_model}")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    template_source = ensure_chat_template(tok)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float32,
                                                 trust_remote_code=True)
    model.to("cpu")
    timings["model_load_s"] = round(time.time() - t0, 2)
    print(f"  chat template: {template_source}")

    # 2. data
    print("[2/6] Data")
    train_all = load_jsonl(args.train_file)
    test_all = load_jsonl(args.test_file)
    check_rows(train_all, "train")
    check_rows(test_all, "test")
    train_rows = select_pairs(train_all, args.train_pairs)
    if args.permute_labels:
        labs = [r["label"] for r in train_rows]
        random.Random(args.seed + 1).shuffle(labs)
        train_rows = [dict(r, label=l) for r, l in zip(train_rows, labs)]
        print("  !! NULL CONTROL: train labels are shuffled")
    wanted = {s.strip() for s in args.eval_sets.split(",") if s.strip()}
    test_by_set = defaultdict(list)
    for r in test_all:
        if not wanted or r["set"] in wanted:
            test_by_set[r["set"]].append(r)
    test_by_set = {s: select_pairs(rows, args.eval_pairs_per_set) for s, rows in test_by_set.items()}
    probe_rows = train_rows[:args.train_probe_n] if args.train_probe_n > 0 else []
    baselines = {s: shortcut_baselines(rows) for s, rows in test_by_set.items()}
    for s, b in baselines.items():
        print(f"  trivial-feature baseline [{s}]: "
              f"det_auroc<={b['best']['det_auroc']}, pair_acc<={b['best']['pair_acc']}")
    print(f"  train: {len(train_rows)} rows / {len(train_rows) // 2} pairs, "
          f"labels {dict(Counter(r['label'] for r in train_rows))}")
    for s, rows in test_by_set.items():
        print(f"  test[{s}]: {len(rows)} rows / {len(rows) // 2} pairs")

    enc = Encoder(tok)
    feats, skipped, tstats = build_train_features(enc, train_rows, args.train_templates,
                                                  args.seed, args.max_len)
    if skipped:
        print(f"  WARNING: {len(skipped)} train rows longer than --max_len skipped: {skipped[:5]}")
    print(f"  train tokens: {tstats}")

    # 3. LoRA
    print("[3/6] LoRA")
    set_seed(args.seed)
    targets = "all-linear" if args.lora_targets == "all-linear" else \
        [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=2 * args.lora_r,
        lora_dropout=args.lora_dropout, target_modules=targets, bias="none"))
    model.print_trainable_parameters()

    eff_batch = args.batch_size * args.grad_accum
    steps_per_epoch = max(math.ceil(len(feats) / args.batch_size) // args.grad_accum, 1)
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(args.num_epochs * steps_per_epoch)
    eval_every = args.eval_every if args.eval_every > 0 else steps_per_epoch
    shift = not args.no_shift_eval
    print(f"  effective batch {eff_batch}, {steps_per_epoch} steps/epoch, {total_steps} total steps, "
          f"eval every {eval_every} steps, lr {args.lr} ({args.lr_scheduler})")

    curve, scores_step0 = [], []

    # 4. step-0
    if not args.skip_step0:
        print("[4/6] Step-0 evaluation (base model)")
        t0 = time.time()
        m0, scores_step0 = eval_point(model, enc, test_by_set, probe_rows, shift, ci=True, keep_scores=True)
        timings["eval_step0_s"] = round(time.time() - t0, 2)
        curve.append({"step": 0, "epoch": 0.0, "train_loss": None, "metrics": m0})
        for k, m in sorted(m0.items()):
            print(f"  {k:16s} " + " ".join(f"{x}={m.get(x)}" for x in KEY_METRICS))
    step0 = curve[0]["metrics"] if curve else {}

    # 5. training
    if total_steps == 0:
        print("[5/6] Training: SKIPPED (num_epochs=0 -> base model only)")
        mf = step0
        scores_final = scores_step0
        final_step = 0
        result_metrics = {"train_runtime": 0.0}
    else:
        print("[5/6] Training")

        class ListDataset(torch.utils.data.Dataset):
            def __init__(self, items):
                self.items = items
            def __len__(self):
                return len(self.items)
            def __getitem__(self, i):
                return dict(self.items[i])

        class EvalCallback(TrainerCallback):
            def __init__(self):
                self.ptr = 0
            def on_step_end(self, a, state, control, **kw):
                s = state.global_step
                if s % eval_every == 0 and s < state.max_steps:
                    hist = [h["loss"] for h in state.log_history[self.ptr:] if "loss" in h]
                    self.ptr = len(state.log_history)
                    t1 = time.time()
                    m, _ = eval_point(model, enc, test_by_set, probe_rows, shift)
                    curve.append({"step": s, "epoch": round(s / steps_per_epoch, 2),
                                  "train_loss": r4(float(np.mean(hist))) if hist else None,
                                  "metrics": m})
                    print(f"  [step {s}] loss={curve[-1]['train_loss']} ({time.time() - t1:.0f}s eval) "
                          + " | ".join(f"{k}: auroc={v.get('det_auroc')} type={v.get('type_acc_forced')}"
                                       for k, v in sorted(m.items())))

        targs = TrainingArguments(
            output_dir="./trainer_out", num_train_epochs=args.num_epochs,
            max_steps=args.max_steps if args.max_steps > 0 else -1,
            per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr, lr_scheduler_type=args.lr_scheduler, warmup_steps=args.warmup_steps,
            weight_decay=0.0, logging_steps=1, save_strategy="no", report_to="none",
            use_cpu=True, fp16=False, bf16=False, dataloader_pin_memory=False,
            remove_unused_columns=False, seed=args.seed, disable_tqdm=True)
        trainer = Trainer(model=model, args=targs, train_dataset=ListDataset(feats),
                          data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
                          callbacks=[EvalCallback()])
        t0 = time.time()
        result = trainer.train()
        timings["train_s"] = round(time.time() - t0, 2)
        last_hist = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        print(f"  trained in {timings['train_s']}s, final loss (last 5) "
              f"{np.mean(last_hist[-5:]) if last_hist else None}")

        # 6. final eval
        print("[6/6] Final evaluation")
        t0 = time.time()
        mf, scores_final = eval_point(model, enc, test_by_set, probe_rows, shift, ci=True, keep_scores=True)
        timings["eval_final_s"] = round(time.time() - t0, 2)
        final_step = trainer.state.global_step
        curve.append({"step": final_step, "epoch": round(final_step / steps_per_epoch, 2),
                      "train_loss": r4(float(np.mean(last_hist[-steps_per_epoch:]))) if last_hist else None,
                      "metrics": mf})
        result_metrics = result.metrics

    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    timings["total_s"] = round(time.time() - t_all, 2)

    cfg = dict(vars(args), n_train_rows=len(feats), n_train_pairs=len(feats) // 2,
               steps_per_epoch=steps_per_epoch, total_steps=final_step, effective_batch=eff_batch,
               chat_template_source=template_source, skipped_too_long=skipped, train_token_stats=tstats,
               versions={"torch": torch.__version__, "transformers": transformers.__version__,
                         "peft": peft.__version__},
               timestamp=datetime.now(timezone.utc).isoformat(),
               labels=LABELS, chance=CHANCE, shortcut_baselines=baselines,
               train_runtime=result_metrics.get("train_runtime"))
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "timings": timings, "curve": curve}, f, indent=2, ensure_ascii=False)
    write_curve_csv(os.path.join(args.output_dir, "curve.csv"), curve)
    for name, data in (("scores_step0.jsonl", scores_step0), ("scores_final.jsonl", scores_final)):
        if data:
            with open(os.path.join(args.output_dir, name), "w", encoding="utf-8") as f:
                for r in data:
                    f.write(json.dumps(r) + "\n")

    md = summary_markdown(cfg, step0, mf)
    print("\n" + md)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(md + "\n")
    print(f"\nSaved adapter + metrics.json + curve.csv + scores_*.jsonl to {args.output_dir} "
          f"({timings['total_s']}s total)")


if __name__ == "__main__":
    main()