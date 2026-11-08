#!/usr/bin/env python3
"""
Filter a training file to only NEW examples (not trained on in previous releases),
and emit the hashes that should be marked as trained after this run.

Each example is identified by sha256(code + "|" + label). Renaming `id` or reordering
the file does NOT mark an example as new. Editing the code or the label DOES (correctly).

Usage:
    python scripts/continue_training.py \
        --all_train_file data/train.jsonl \
        --trained_file ./prev/trained.json \
        --out ./train_new.jsonl \
        --out_hashes ./new_hashes.json \
        --min_new_rows 2
"""
import argparse
import hashlib
import json
import os
import sys


def sha256(s: str) -> str:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all_train_file", required=True)
    ap.add_argument("--trained_file", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out_hashes", required=True)
    ap.add_argument("--min_new_rows", type=int, default=2)
    args = ap.parse_args()

    trained = load_hashes(args.trained_file)

    rows = []
    with open(args.all_train_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    new_rows, new_hashes = [], []
    for r in rows:
        h = key_of(r)
        if h not in trained:
            new_rows.append(r)
            new_hashes.append(h)

    n_total = len(rows)
    n_prev = n_total - len(new_rows)
    n_new = len(new_rows)
    print(f"total={n_total}  previously_trained={n_prev}  new={n_new}")

    if n_new < args.min_new_rows:
        print(f"NOTHING_TO_TRAIN: only {n_new} new rows (< {args.min_new_rows})",
              file=sys.stderr)
        sys.exit(78)  # special exit code: workflow catches it

    with open(args.out, "w", encoding="utf-8") as f:
        for r in new_rows:
            f.write(json.dumps(r) + "\n")

    with open(args.out_hashes, "w", encoding="utf-8") as f:
        json.dump({"hashes": new_hashes, "count": n_new}, f)

    print(f"wrote {args.out} ({n_new} rows) and {args.out_hashes}")


if __name__ == "__main__":
    main()
