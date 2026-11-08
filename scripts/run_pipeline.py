#!/usr/bin/env python3
"""
Centinela pipeline v2.

What changed vs v1 (and why):
  * Loss is computed ONLY on the answer tokens (prompt/code tokens are masked with -100).
  * Plain HF Trainer + our own tokenisation (no TRL): nothing is truncated silently,
    over-long examples are reported and skipped.
  * Evaluation does not generate text. It scores every candidate label by its
    log-probability given the prompt (11 labels = 10 vulnerability types + SAFE),
    so there is no parsing, no max_new_tokens and no greedy "3 passes".
  * Metrics are hierarchical and prior-free where possible:
      det_auroc        AUROC of "vulnerable vs SAFE"           (chance 0.50)
      pair_acc         vulnerable twin scored above its safe twin (chance 0.50)
      type_acc_forced  type among the 10 vuln labels, vulnerable examples only (chance 0.10)
      mean_logp_correct  mean log P(correct label) over the 11 labels (chance -2.398)
      acc / macro_recall / det_bal_acc   argmax-based, for reference
  * Several test sets (e.g. iid, ood_near) x two prompt conditions:
      seen  = the first training template
      shift = held-out templates never used for training
    plus a "train" probe (can the model even fit its own training data?).
  * The learning curve is measured INSIDE one run (eval every N optimizer steps),
    with a schedule that does not depend on the horizon (constant_with_warmup).
  * --seed controls LoRA init, data order and label-order shuffling.
  * --train_pairs / --train_templates / --max_steps give the grid knobs
    (amount of data, prompt diversity, training budget).
  * --permute_labels is the null control (labels shuffled across train examples).

v4 changes (found while analysing the first full runs):
  * Training template = pair index within its type modulo K (Latin square, twins share it). v2/v3 used
    the row index modulo K, which aliased with the 20-row round-robin order: each template only ever
    co-occurred with 2 of the 10 types, so the model could learn "template -> type".
  * Eval condition 'seen' uses the same template rule as training (v3 evaluated everything with T0).
  * Label order in the prompt is canonical (a per-example shuffle is a unique fingerprint of the example).
  * --score_mode auto: log P(first label token) when every label has a distinct first token. One forward
    pass, no multi-token length effects, and ~10x cheaper than scoring full label sequences.
  * Detection metrics use the structural `role` of a snippet (vuln/safe), not its possibly-permuted label.
  * Step 0 and the final eval always use the FULL test sets; --eval_pairs_per_set only thins mid-run evals.
  * Results are also split by `subtype` (plain vs insufficient mitigation) inside each test set.

v5 changes (continuous-training support):
  * --resume_adapter loads a previous PEFT adapter folder (weights only) before training/eval.
  * --eval_only skips the training loop (requires --resume_adapter); the final eval always runs.
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

try:  # lets the pure helpers (metrics, data selection) be imported without torch
    import torch
except ImportError:  # pragma: no cover
    torch = None

VULN_TYPES = [
    # original 10
    "XSS", "SQLI", "CMDI", "CODE_INJECTION", "PATH_TRAVERSAL",
    "SSRF", "PROTO_POLLUTION", "OPEN_REDIRECT", "AUTHZ", "WEAK_CRYPTO",
    # 15 new code-detectable classes
    "SSTI", "CSRF", "XXE", "LDAP_INJECTION", "HPP",
    "CRLF_INJECTION", "CSSI", "HTMLI", "NOSQLI",
    "HARDCODED_SECRETS", "MASS_ASSIGNMENT", "WEAK_COOKIE",
    "INSECURE_JWT", "RACE_CONDITION", "INFO_DISCLOSURE",
]
SAFE = "SAFE"
LABELS = VULN_TYPES + [SAFE]
SAFE_IDX = LABELS.index(SAFE)
VULN_IDX = [i for i in range(len(LABELS)) if i != SAFE_IDX]

# Templates seen in training (K = --train_templates uses the first K; eval "seen" uses T0).
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
# Never used for training; used by the "shift" condition.
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
        for k in ("id", "pair_id", "set", "label", "code"):
            if k not in r:
                raise ValueError(f"{name}: row {r.get('id')} lacks field '{k}'")
        if r["label"] not in LABELS:
            raise ValueError(f"{name}: row {r['id']} has unknown label {r['label']!r}")


def label_order(key=None):
    """Canonical order, identical in train and eval. (v2 shuffled it per example: the permutation was a
    unique fingerprint of each example, a memorisation channel that bypasses the code.)"""
    return LABELS[:]


def build_user_text(template, code, order):
    return template.replace("{labels}", ", ".join(order)).replace("{code}", code)


def select_pairs(rows, n_pairs):
    """Keep the first n_pairs pairs in file order (files are round-robin by type)."""
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


def is_vuln_row(r):
    """Detection is defined by the structural role of the snippet, not by its (possibly permuted) label."""
    role = r.get("role")
    return (role == "vuln") if role is not None else (r["label"] != SAFE)


def pairs_of(rows):
    by = defaultdict(dict)
    for i, r in enumerate(rows):
        by[r["pair_id"]]["v" if is_vuln_row(r) else "s"] = i
    return [(d["v"], d["s"]) for d in by.values() if "v" in d and "s" in d]


def code_features(code):
    lines = code.split("\n")
    return {"chars": len(code), "lines": len(lines),
            "comment_lines": sum(1 for l in lines if l.lstrip().startswith("//"))}


def shortcut_baselines(rows):
    """What the best SINGLE trivial feature (either direction) scores on the same metrics.
    A model is only interesting if it clearly beats this."""
    if not rows:
        return {}
    prs = pairs_of(rows)
    is_vuln = np.asarray([is_vuln_row(r) for r in rows])
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
    """rows: dicts with 'label', 'pair_id' and 'scores' (sum log-prob per label, LABELS order)."""
    n = len(rows)
    if n == 0:
        return {}
    S = np.asarray([r["scores"] for r in rows], dtype=np.float64)
    y = np.asarray([LABELS.index(r["label"]) for r in rows])
    logp = S - lse(S)[:, None]
    pred = S.argmax(1)
    is_vuln = np.asarray([is_vuln_row(r) for r in rows])
    det = lse(S[:, VULN_IDX]) - S[:, SAFE_IDX]
    forced = np.asarray(VULN_IDX)[S[:, VULN_IDX].argmax(1)]
    forced_ok = (forced == y).astype(float)
    vmask = is_vuln & (y != SAFE_IDX)
    if vmask.any():
        sub = S[vmask][:, VULN_IDX]
        pos = np.asarray([VULN_IDX.index(c) for c in y[vmask]])
        correct = sub[np.arange(len(pos)), pos]
        rank = 1 + (sub > correct[:, None]).sum(1)
        type_mrr, type_top3 = float((1.0 / rank).mean()), float((rank <= 3).mean())
    else:
        type_mrr = type_top3 = float("nan")

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
        "type_mrr": type_mrr,
        "type_top3": type_top3,
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
    """Returns 'native' or the name of the fallback that had to be injected."""
    if getattr(tokenizer, "chat_template", None) is not None:
        return "native"
    name = getattr(tokenizer, "name_or_path", "").lower()
    if "qwen" in name:
        print("  -> WARNING: no chat template in tokenizer, injecting Qwen2.5 ChatML fallback")
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
    """Prompt ids and candidate-completion ids, tokenised separately so train and eval agree."""

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
        # Scoring uses the label tokens only (no trailing <|im_end|>/newline) so the tail cannot dilute anything.
        self.score_ids, fallbacks = {}, 0
        for lab in LABELS:
            only = tok(lab, add_special_tokens=False).input_ids
            if self.cand_ids[lab][:len(only)] == only:
                self.score_ids[lab] = only
            else:
                self.score_ids[lab] = self.cand_ids[lab]
                fallbacks += 1
        self.first_unique = len({self.score_ids[l][0] for l in LABELS}) == len(LABELS)
        self.label_tokens = {l: len(self.score_ids[l]) for l in LABELS}
        if fallbacks:
            print(f"  note: {fallbacks} labels tokenise differently alone vs in context; scoring their full completion")

    def prompt_ids(self, user_text):
        p = self.tok.apply_chat_template(
            [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True)
        return self.tok(p, add_special_tokens=False).input_ids


def pair_number(row):
    """'train-XSS-03' -> 2 (index of the pair inside its type)."""
    try:
        return int(row["pair_id"].rsplit("-", 1)[1]) - 1
    except (ValueError, IndexError):
        return 0


def template_index(row, n_templates):
    """Template of a row = pair index within its type, modulo K. Twins share it, and every type sees every
    template (a Latin square), so template and vulnerability type are NOT correlated. v2 used the row index
    modulo K, which aliased with the 20-row round-robin cycle: each template only ever co-occurred with
    2 of the 10 types."""
    return pair_number(row) % n_templates


def build_train_features(enc, rows, n_templates, seed, max_len):
    feats, skipped, n_prompt, n_comp = [], [], 0, 0
    for i, r in enumerate(rows):
        tpl = TRAIN_TEMPLATES[template_index(r, n_templates)]
        order = label_order()
        p = enc.prompt_ids(build_user_text(tpl, r["code"], order))
        c = enc.cand_ids[r["label"]]
        if len(p) + len(c) > max_len:
            skipped.append(r["id"])
            continue
        feats.append({
            "input_ids": p + c,
            "attention_mask": [1] * (len(p) + len(c)),
            "labels": [-100] * len(p) + c,  # loss ONLY on the answer tokens
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
def score_prompt(model, enc, p_ids, mode):
    """Score every label given the prompt.
      first: log P(first label token)   - one forward pass, no length effects (needs distinct first tokens)
      sum:   sum of log P over the label tokens   (biased against multi-token labels)
      mean:  sum / number of label tokens
    """
    with torch.no_grad():
        dev = next(model.parameters()).device
        ids = torch.tensor([p_ids], dtype=torch.long, device=dev)
        kw = {"use_cache": mode != "first"}
        try:
            out = model(input_ids=ids, num_logits_to_keep=1, **kw)
        except TypeError:  # transformers without this kwarg
            out = model(input_ids=ids, **kw)
        last = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
        if mode == "first":
            return [last[enc.score_ids[lab][0]].item() for lab in LABELS]
        pkv = out.past_key_values
        scores = []
        for lab in LABELS:
            c = enc.score_ids[lab]
            s = last[c[0]].item()
            if len(c) > 1:
                cin = torch.tensor([c[:-1]], dtype=torch.long, device=dev)
                o2 = model(input_ids=cin, past_key_values=copy.deepcopy(pkv), use_cache=True)
                lp = torch.log_softmax(o2.logits[0].float(), dim=-1)
                tgt = torch.tensor(c[1:], dtype=torch.long, device=dev)
                s += lp[torch.arange(len(c) - 1, device=dev), tgt].sum().item()
            scores.append(s / len(c) if mode == "mean" else s)
    return scores


def eval_rows(model, enc, rows, cond, n_templates, mode):
    """cond='seen'  -> the training template this row's pair index maps to (same rule as training);
    cond='shift' -> a held-out template chosen per PAIR (both twins always share template)."""
    pidx, out = {}, []
    for r in rows:
        pidx.setdefault(r["pair_id"], len(pidx))
    for r in rows:
        if cond == "seen":
            k = template_index(r, n_templates)
            tid, tpl = f"T{k}", TRAIN_TEMPLATES[k]
        else:
            k = pidx[r["pair_id"]] % len(HELDOUT_TEMPLATES)
            tid, tpl = f"H{k}", HELDOUT_TEMPLATES[k]
        p_ids = enc.prompt_ids(build_user_text(tpl, r["code"], label_order()))
        out.append({"id": r["id"], "pair_id": r["pair_id"], "set": r["set"], "label": r["label"],
                    "role": r.get("role"), "subtype": r.get("subtype"), "cond": cond, "template": tid,
                    "scores": score_prompt(model, enc, p_ids, mode)})
    return out


def average_rows(rows_a, rows_b):
    """Test-time augmentation at no extra cost: mean log-prob of the same row under two different prompts."""
    by_id = {r["id"]: r for r in rows_b}
    out = []
    for r in rows_a:
        s = [(x + y) / 2.0 for x, y in zip(r["scores"], by_id[r["id"]]["scores"])]
        out.append(dict(r, scores=s, cond="tta2", template=r["template"] + "+" + by_id[r["id"]]["template"]))
    return out


def eval_point(model, enc, test_by_set, probe_rows, shift, n_templates, mode, ci=False, keep_scores=False):
    was_training = model.training
    model.eval()
    metrics, dump = {}, []

    def add(key, er):
        metrics[key] = compute_metrics(er, ci=ci)
        if ci:
            metrics[key]["confusion"] = confusion(er)

    for sname, rows in test_by_set.items():
        ers = {}
        for cond in (["seen", "shift"] if shift else ["seen"]):
            er = eval_rows(model, enc, rows, cond, n_templates, mode)
            ers[cond] = er
            add(f"{sname}/{cond}", er)
            subs = sorted({r["subtype"] for r in er if r.get("subtype")})
            if len(subs) > 1:  # e.g. iid[plain] vs iid[insufficient]: same forward passes, split afterwards
                for sub in subs:
                    part = [r for r in er if r.get("subtype") == sub]
                    if len(pairs_of(part)) >= 3:
                        add(f"{sname}[{sub}]/{cond}", part)
            if keep_scores:
                dump += er
        if shift:
            tta = average_rows(ers["seen"], ers["shift"])
            add(f"{sname}/tta2", tta)
            if keep_scores:
                dump += tta
    if probe_rows:
        er = eval_rows(model, enc, probe_rows, "seen", n_templates, mode)
        add("train/seen", er)
        if keep_scores:
            dump += er
    if was_training:
        model.train()
    return metrics, dump


# ----------------------------------------------------------------------------- reporting
KEY_METRICS = ["det_auroc", "pair_acc", "type_acc_forced", "type_mrr", "mean_logp_correct"]
_H = sum(1.0 / k for k in range(1, len(VULN_IDX) + 1))
CHANCE = {"det_auroc": 0.5, "pair_acc": 0.5, "type_acc_forced": 0.1, "type_mrr": round(_H / len(VULN_IDX), 3),
          "type_top3": 0.3, "mean_logp_correct": round(-math.log(len(LABELS)), 3)}


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
             "type_forced 0.1, type_mrr 0.293, mean_logp -2.398.", "",
             "| set/cond | det_auroc | pair_acc | type_forced | type_mrr | mean_logp |", "|---|---|---|---|---|---|"]
    for k in sorted(final):
        cells = []
        for m in KEY_METRICS:
            a = step0.get(k, {}).get(m) if step0 else None
            b = final[k].get(m)
            ci = final[k].get("ci95", {}).get(m)
            if a is None:
                cells.append(f"{b}" + (f" {ci}" if ci else ""))
            else:
                cells.append(f"{a} -> {b}" + (f" {ci}" if ci else ""))
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    sb = cfg.get("shortcut_baselines") or {}
    if sb:
        lines += ["", "Best trivial single-feature baseline (length / lines / comment count) on the same test sets; "
                      "det_auroc and pair_acc only mean something above these:", "",
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
    ap.add_argument("--train_pairs", type=int, default=0, help="first N pairs of the train file (0 = all)")
    ap.add_argument("--train_templates", type=int, default=1,
                    help="distinct training prompt templates (1-5); template = pair index within type mod K, "
                         "so use K <= pairs per type")
    ap.add_argument("--score_mode", choices=["auto", "first", "sum", "mean"], default="auto",
                    help="auto = first-token log-prob if every label has a distinct first token, else sum")
    ap.add_argument("--num_epochs", type=float, default=8)
    ap.add_argument("--max_steps", type=int, default=0, help="optimizer steps; overrides --num_epochs if > 0")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_scheduler", default="constant_with_warmup")
    ap.add_argument("--warmup_steps", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_targets", default="q_proj,v_proj", help="comma list or 'all-linear'")
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--peft", choices=["lora", "ia3", "loha", "lokr"], default="lora",
                    help="adapter family (ia3 needs a much higher lr, ~3e-3)")
    ap.add_argument("--lora_plus_ratio", type=float, default=0.0, help="LoRA+: lr(B)/lr(A) ratio (0 = off)")
    ap.add_argument("--lora_layers", choices=["all", "upper_half"], default="all",
                    help="adapt all layers or only the upper half (layer freezing; needs an explicit target list)")
    ap.add_argument("--label_smoothing", type=float, default=0.0, help="only with --loss label_ce")
    ap.add_argument("--rslora", action="store_true", help="rank-stabilised LoRA scaling (alpha/sqrt(r))")
    ap.add_argument("--dora", action="store_true", help="DoRA (slower on CPU)")
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--neftune_alpha", type=float, default=0.0, help="NEFTune embedding noise (0 = off)")
    ap.add_argument("--loss", choices=["lm", "label_ce"], default="lm",
                    help="lm = token loss on the answer; label_ce = cross-entropy restricted to the 11 label "
                         "first-tokens at the answer position (matches first-token scoring; needs distinct first tokens)")
    ap.add_argument("--time_budget_min", type=float, default=0,
                    help="stop TRAINING after this many minutes (0 = off); the final eval still runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_every", type=int, default=0, help="optimizer steps between evals (0 = every epoch)")
    ap.add_argument("--eval_sets", default="", help="comma list of test sets to evaluate (default: all in file)")
    ap.add_argument("--eval_pairs_per_set", type=int, default=0, help="limit pairs per test set for INTERMEDIATE evals only; "
                                                                      "step 0 and the final eval always use the full sets")
    ap.add_argument("--train_probe_n", type=int, default=40, help="train rows evaluated as fit-check (0 = off)")
    ap.add_argument("--no_shift_eval", action="store_true", help="skip the held-out-template condition")
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--permute_labels", action="store_true", help="null control: shuffle train labels")
    ap.add_argument("--skip_step0", action="store_true")
    # v5: continuous training
    ap.add_argument("--resume_adapter", default="",
                    help="path to a previous PEFT adapter folder (loads adapter weights before training/eval)")
    ap.add_argument("--eval_only", action="store_true",
                    help="skip training, only run the final evaluation (requires --resume_adapter)")
    args = ap.parse_args(argv)
    if not 1 <= args.train_templates <= len(TRAIN_TEMPLATES):
        ap.error(f"--train_templates must be in 1..{len(TRAIN_TEMPLATES)}")
    if args.eval_only and not args.resume_adapter:
        ap.error("--eval_only requires --resume_adapter")
    return args


def main():
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # progress visible in CI logs
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
    test_full = dict(test_by_set)  # step 0 and final evals
    test_by_set = {s: select_pairs(rows, args.eval_pairs_per_set) for s, rows in test_by_set.items()}  # mid-run evals
    probe_rows = train_rows[:args.train_probe_n] if args.train_probe_n > 0 else []
    baselines = {s: shortcut_baselines(rows) for s, rows in test_full.items()}
    for s, b in baselines.items():
        print(f"  trivial-feature baseline [{s}] (length/lines/comments, best of each): "
              f"det_auroc<={b['best']['det_auroc']}, pair_acc<={b['best']['pair_acc']}")
    print(f"  train: {len(train_rows)} rows / {len(train_rows) // 2} pairs, "
          f"labels {dict(Counter(r['label'] for r in train_rows))}")
    for s, rows in test_full.items():
        print(f"  test[{s}]: {len(rows)} rows / {len(rows) // 2} pairs "
              f"(mid-run evals use {len(test_by_set[s]) // 2})")

    enc = Encoder(tok)
    mode = args.score_mode
    if mode == "auto":
        mode = "first" if enc.first_unique else "mean"  #"sum"
    elif mode == "first" and not enc.first_unique:
        sys.exit("--score_mode first needs a distinct first token per label (see label tokens above)")
    print(f"  label tokens {enc.label_tokens}; distinct first tokens: {enc.first_unique}; score mode: {mode}")
    if args.loss == "label_ce" and mode != "first":
        sys.exit("--loss label_ce trains the first label token only, so it needs --score_mode first "
                 "(distinct first tokens per label)")
    per_type = len(train_rows) / 2 / len(VULN_TYPES)
    if args.train_templates > 1 and per_type < args.train_templates:
        print(f"  WARNING: {per_type:.1f} pairs per type < {args.train_templates} templates: templates cannot be "
              f"spread over all types, so template and type stay correlated. Use --train_templates 1.")
    feats, skipped, tstats = build_train_features(enc, train_rows, args.train_templates,
                                                  args.seed, args.max_len)
    if skipped:
        print(f"  WARNING: {len(skipped)} train rows longer than --max_len skipped: {skipped[:5]}")
    print(f"  train tokens: {tstats}")
    print("  (supervised_fraction = share of tokens that carry loss; v1 trained on 100% incl. the code)")

    # 3. LoRA
    print("[3/6] LoRA")
    set_seed(args.seed)  # LoRA init must be seeded: it happens before the Trainer is built
    targets = "all-linear" if args.lora_targets == "all-linear" else \
        [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    if args.label_smoothing and args.loss != "label_ce":
        sys.exit("--label_smoothing only applies to --loss label_ce")
    if args.peft == "ia3":
        from peft import IA3Config
        peft_cfg = IA3Config(task_type=TaskType.CAUSAL_LM, target_modules=["k_proj", "v_proj", "down_proj"],
                             feedforward_modules=["down_proj"])
    elif args.peft in ("loha", "lokr"):
        from peft import LoHaConfig, LoKrConfig
        peft_cfg = (LoHaConfig if args.peft == "loha" else LoKrConfig)(
            task_type=TaskType.CAUSAL_LM, r=args.lora_r, alpha=2 * args.lora_r, target_modules=targets)
    else:
        extra = {}
        if args.lora_layers == "upper_half":
            if targets == "all-linear":
                sys.exit("--lora_layers upper_half needs an explicit --lora_targets list, not all-linear")
            n_layers = model.config.num_hidden_layers
            extra["layers_to_transform"] = list(range(n_layers // 2, n_layers))
        peft_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=2 * args.lora_r,
                              lora_dropout=args.lora_dropout, target_modules=targets, bias="none",
                              use_rslora=args.rslora, use_dora=args.dora, **extra)
    if args.resume_adapter:
        if not os.path.isdir(args.resume_adapter):
            sys.exit(f"--resume_adapter not found: {args.resume_adapter}")
        from peft import PeftModel
        print(f"  resuming adapter from {args.resume_adapter}")
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    optimizers = (None, None)
    if args.lora_plus_ratio > 0:
        if args.peft != "lora":
            sys.exit("--lora_plus_ratio only applies to --peft lora")
        from peft.optimizers import create_loraplus_optimizer
        optimizers = (create_loraplus_optimizer(model=model, optimizer_cls=torch.optim.AdamW, lr=args.lr,
                                                loraplus_lr_ratio=args.lora_plus_ratio), None)

    eff_batch = args.batch_size * args.grad_accum
    steps_per_epoch = max(math.ceil(len(feats) / args.batch_size) // args.grad_accum, 1)
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(args.num_epochs * steps_per_epoch)
    eval_every = args.eval_every if args.eval_every > 0 else steps_per_epoch
    shift = not args.no_shift_eval
    print(f"  effective batch {eff_batch}, {steps_per_epoch} steps/epoch, {total_steps} total steps, "
          f"eval every {eval_every} steps, lr {args.lr} ({args.lr_scheduler})")

    curve, scores_step0 = [], []

    # 4. step-0 evaluation (LoRA B=0 -> identical to the base model)
    if not args.skip_step0 and not args.eval_only:
        print("[4/6] Step-0 evaluation (base model)")
        t0 = time.time()
        m0, scores_step0 = eval_point(model, enc, test_full, probe_rows, shift, args.train_templates, mode,
                                      ci=True, keep_scores=True)
        timings["eval_step0_s"] = round(time.time() - t0, 2)
        curve.append({"step": 0, "epoch": 0.0, "train_loss": None, "metrics": m0})
        for k, m in sorted(m0.items()):
            print(f"  {k:16s} " + " ".join(f"{x}={m.get(x)}" for x in KEY_METRICS))
    step0 = curve[0]["metrics"] if curve else {}

    # 5. training
    print("[5/6] Training")

    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            # copy: DataCollatorForSeq2Seq re-assigns feature["labels"] when padding (batch_size > 1)
            return dict(self.items[i])

    budget_stop = {}
    t_train0 = time.time()

    class EvalCallback(TrainerCallback):
        def __init__(self):
            self.ptr = 0

        def on_step_end(self, a, state, control, **kw):
            s = state.global_step
            if args.time_budget_min and (time.time() - t_train0) / 60 > args.time_budget_min and not budget_stop:
                budget_stop["stopped_at_step"] = s
                control.should_training_stop = True
                print(f"  [time budget] {args.time_budget_min} min reached at step {s}: stopping training")
            if s % eval_every == 0 and s < state.max_steps:
                hist = [h["loss"] for h in state.log_history[self.ptr:] if "loss" in h]
                self.ptr = len(state.log_history)
                t1 = time.time()
                m, _ = eval_point(model, enc, test_by_set, probe_rows, shift, args.train_templates, mode)
                curve.append({"step": s, "epoch": round(s / steps_per_epoch, 2),
                              "train_loss": r4(float(np.mean(hist))) if hist else None, "metrics": m})
                print(f"  [step {s}] loss={curve[-1]['train_loss']} ({time.time() - t1:.0f}s eval) " +
                      " | ".join(f"{k}: auroc={v.get('det_auroc')} type={v.get('type_acc_forced')}"
                                 for k, v in sorted(m.items())))
            return control

    cand_first = torch.tensor([enc.score_ids[l][0] for l in LABELS], dtype=torch.long)

    class LabelCETrainer(Trainer):
        """Cross-entropy over the 11 label first-tokens at the position that predicts the answer."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            rows = torch.arange(labels.size(0), device=labels.device)
            first = (labels != -100).int().argmax(dim=1)          # first supervised position (first max wins)
            cand = cand_first.to(labels.device)
            match = labels[rows, first][:, None] == cand[None, :]
            if not bool(match.any(dim=1).all()):
                raise RuntimeError("label_ce: first supervised token is not a label first-token")
            target = match.int().argmax(dim=1)
            logits = out.logits[rows, first - 1][:, cand].float()  # position first-1 predicts token `first`
            loss = torch.nn.functional.cross_entropy(logits, target, label_smoothing=args.label_smoothing)
            return (loss, out) if return_outputs else loss

    targs = TrainingArguments(
        output_dir="./trainer_out", num_train_epochs=args.num_epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type=args.lr_scheduler, warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay, neftune_noise_alpha=(args.neftune_alpha or None), logging_steps=1, save_strategy="no", report_to="none",
        use_cpu=True, fp16=False, bf16=False, dataloader_pin_memory=False,
        remove_unused_columns=False, seed=args.seed, disable_tqdm=True)
    TrainerCls = LabelCETrainer if args.loss == "label_ce" else Trainer
    trainer = TrainerCls(model=model, args=targs, train_dataset=ListDataset(feats),
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
                      callbacks=[EvalCallback()], optimizers=optimizers)

    if args.eval_only:
        print("[5/6] Eval only: skipping training loop")
        result, last_hist = None, []
        timings["train_s"] = 0
    else:
        t0 = time.time()
        result = trainer.train()
        timings["train_s"] = round(time.time() - t0, 2)
        last_hist = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        print(f"  trained in {timings['train_s']}s, final loss (last 5 logs) "
              f"{np.mean(last_hist[-5:]) if last_hist else None}")
    diverged = bool(last_hist) and (any(not math.isfinite(x) for x in last_hist) or last_hist[-1] > 25)
    if diverged:
        print("  WARNING: training diverged (non-finite or exploding loss); the evaluation below is of a broken model")

    # 6. final evaluation + outputs
    print("[6/6] Final evaluation")
    t0 = time.time()
    mf, scores_final = eval_point(model, enc, test_full, probe_rows, shift, args.train_templates, mode,
                                  ci=True, keep_scores=True)
    timings["eval_final_s"] = round(time.time() - t0, 2)
    final_step = trainer.state.global_step if not args.eval_only else 0
    curve.append({"step": final_step, "epoch": round(final_step / steps_per_epoch, 2),
                  "train_loss": r4(float(np.mean(last_hist[-steps_per_epoch:]))) if last_hist else None,
                  "metrics": mf})
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    timings["total_s"] = round(time.time() - t_all, 2)

    cfg = dict(vars(args), n_train_rows=len(feats), n_train_pairs=len(feats) // 2,
               steps_per_epoch=steps_per_epoch, total_steps=final_step, effective_batch=eff_batch,
               chat_template_source=template_source, skipped_too_long=skipped, train_token_stats=tstats,
               versions={"torch": torch.__version__, "transformers": transformers.__version__,
                         "peft": peft.__version__},
               timestamp=datetime.now(timezone.utc).isoformat(), labels=LABELS, chance=CHANCE, shortcut_baselines=baselines,
               score_mode_resolved=mode, label_tokens=enc.label_tokens,
               stopped_by_time_budget=budget_stop.get("stopped_at_step"), diverged=diverged,
               final_train_loss=(r4(float(np.mean(last_hist[-5:]))) if last_hist else None),
               train_runtime=(result.metrics.get("train_runtime") if result else None))
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
