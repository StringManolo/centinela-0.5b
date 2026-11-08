#!/usr/bin/env python3
"""
Filter a training file to only NEW examples, then SELECT a subset that fits in the time
budget. Unselected-but-new examples are NOT marked as trained, so they come back next run.

Each example is identified by sha256(code + "|" + label). Renaming `id` or reordering
the file does NOT mark an example as new. Editing code or label DOES (correctly).

Budget model: with batch_size=1, grad_accum=4, one optimizer step consumes 4 rows.
    steps_per_epoch = ceil(rows / 4)
    total_steps     = num_epochs * steps_per_epoch
    total_seconds   = total_steps * sec_per_step
So the maximum rows that fit in `target_seconds` is:
    rows <= (target_seconds / sec_per_step / num_epochs) * 4

Usage:
    python scripts/continue_training.py \
        --all_train_file data/train.jsonl \
        --trained_file ./prev/trained.json \
        --out ./train_new.jsonl \
        --out_hashes ./new_hashes.json \
        --deferred_file ./deferred.json \
        --time_budget_min 300 \
        --num_epochs 6 \
        --sec_per_step 17.0 \
        --safety_factor 0.85 \
        --select_strategy stratified \
        --seed 42 \
        --min_new_rows 2
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict


def sha256(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def key_of(row):
    return sha256(row["code"] + "|" + row["label"])


def load_hashes(path):
    if not path or not os.path.isfile(path):
        return set()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return set(data.get("hashes", []))
        if isinstance(data, list):
            return set(data)
    except Exception as e:
        print(f"WARNING: could not read {path}: {e}", file=sys.stderr)
    return set()


def group_key(row):
    """Stratification key: type + subtype. Falls back gracefully if fields are missing."""
    return (row.get("vuln_type") or row.get("label") or "UNK",
            row.get("subtype") or "plain")


def select_balanced(rows, target_rows, rng):
    """Pick target_rows spread across (vuln_type, subtype) groups.

    Quota per group = target // n_groups, with a floor of 1 if a group has any rows.
    Leftover goes to groups with unmet demand (round-robin). If all groups are
    saturated before target is reached, returns everything (cannot over-sample).
    """
    if target_rows <= 0 or len(rows) <= target_rows:
        return list(rows)

    groups = defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)

    quotas = {}
    selected = []
    base = max(1, target_rows // len(groups))
    for k, g in groups.items():
        q = min(base, len(g))
        quotas[k] = q
        selected.extend(rng.sample(g, q))

    remaining = target_rows - len(selected)
    while remaining > 0:
        pool = [(k, len(g) - quotas[k]) for k, g in groups.items() if len(g) > quotas[k]]
        if not pool:
            break
        rng.shuffle(pool)
        for k, _ in pool[:remaining]:
            quotas[k] += 1
            remaining -= 1

    selected = []
    for k, g in groups.items():
        selected.extend(rng.sample(g, quotas[k]))
    rng.shuffle(selected)
    return selected


def compute_target_rows(time_budget_min, num_epochs, sec_per_step, safety_factor, rows_per_step):
    if time_budget_min <= 0 or num_epochs <= 0 or sec_per_step <= 0:
        return 0  # 0 = no limit
    budget_s = time_budget_min * 60.0 * safety_factor
    rows = int((budget_s / sec_per_step / num_epochs) * rows_per_step)
    rows = max(rows_per_step, (rows // rows_per_step) * rows_per_step)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all_train_file", required=True)
    ap.add_argument("--trained_file", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out_hashes", required=True)
    ap.add_argument("--deferred_file", default="",
                    help="where to record new-but-not-selected hashes (informational)")
    ap.add_argument("--time_budget_min", type=float, default=0,
                    help="training budget in minutes (0 = no selection, use all new rows)")
    ap.add_argument("--num_epochs", type=float, default=6)
    ap.add_argument("--sec_per_step", type=float, default=17.0,
                    help="observed seconds per optimizer step (from a previous run's metrics.json)")
    ap.add_argument("--safety_factor", type=float, default=0.85,
                    help="fraction of the budget to actually plan for (leaves room for eval)")
    ap.add_argument("--rows_per_step", type=int, default=4,
                    help="batch_size * grad_accum (from run_pipeline defaults)")
    ap.add_argument("--select_strategy", choices=["stratified", "random", "all"], default="stratified")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_new_rows", type=int, default=2)
    ap.add_argument("--max_rows", type=int, default=0,
                    help="hard cap on selected rows (0 = use budget-derived target)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    trained = load_hashes(args.trained_file)

    rows = []
    with open(args.all_train_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    new_rows = []
    for r in rows:
        h = key_of(r)
        if h not in trained:
            new_rows.append((h, r))

    n_total = len(rows)
    n_new = len(new_rows)
    print(f"total={n_total}  previously_trained={n_total - n_new}  new={n_new}")

    if n_new < args.min_new_rows:
        print(f"NOTHING_TO_TRAIN: only {n_new} new rows (< {args.min_new_rows})", file=sys.stderr)
        sys.exit(78)

    # --- target rows ---
    target = compute_target_rows(args.time_budget_min, args.num_epochs,
                                 args.sec_per_step, args.safety_factor, args.rows_per_step)
    if args.max_rows and (target == 0 or args.max_rows < target):
        target = args.max_rows
    if args.select_strategy == "all":
        target = 0  # no selection

    if target == 0:
        selected_pairs = new_rows
        print(f"selection=all ({n_new} rows, no budget cap)")
    elif n_new <= target:
        selected_pairs = new_rows
        print(f"selection=all ({n_new} <= {target} fit, no deferral)")
    else:
        only_rows = [r for _, r in new_rows]
        if args.select_strategy == "random":
            picked = rng.sample(only_rows, target)
        else:
            picked = select_balanced(only_rows, target, rng)
        picked_ids = {id(r) for r in picked}
        selected_pairs = [(h, r) for h, r in new_rows if id(r) in picked_ids]
        print(f"selection={args.select_strategy} ({len(selected_pairs)} of {n_new} new rows selected, "
              f"{n_new - len(selected_pairs)} deferred to next run)")

    if not selected_pairs:
        print("NOTHING_SELECTED", file=sys.stderr)
        sys.exit(78)

    with open(args.out, "w", encoding="utf-8") as f:
        for _, r in selected_pairs:
            f.write(json.dumps(r) + "\n")

    with open(args.out_hashes, "w", encoding="utf-8") as f:
        json.dump({"hashes": [h for h, _ in selected_pairs], "count": len(selected_pairs)}, f)

    deferred = [(h, r) for h, r in new_rows if h not in {x for x, _ in selected_pairs}]
    if args.deferred_file:
        by_type = defaultdict(int)
        for _, r in deferred:
            by_type[group_key(r)] += 1
        with open(args.deferred_file, "w", encoding="utf-8") as f:
            json.dump({"count": len(deferred),
                       "by_group": {f"{t}/{s}": n for (t, s), n in sorted(by_type.items())}}, f, indent=2)
        print(f"deferred: {len(deferred)} rows -> {args.deferred_file}")

    print(f"selected {len(selected_pairs)} rows -> {args.out}")
    if len(selected_pairs) >= 2:
        rows_per_epoch = math.ceil(len(selected_pairs) / args.rows_per_step)
        steps = int(args.num_epochs * rows_per_epoch)
        est_min = steps * args.sec_per_step / 60.0
        print(f"estimated training: ~{steps} steps, ~{est_min:.0f} min "
              f"(budget {args.time_budget_min:.0f} min, safety {args.safety_factor:.2f})")


if __name__ == "__main__":
    main()
