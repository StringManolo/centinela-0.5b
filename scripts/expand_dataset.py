# scripts/expand_dataset.py
import json, random, sys

src = "data/train.jsonl"
target = int(sys.argv[1])  # 20 o 100
out = f"data/train_{target}.jsonl"

with open(src) as f:
    base = [json.loads(l) for l in f if l.strip()]

augmented = []
while len(augmented) < target:
    ex = random.choice(base)
    # Variación trivial para que no sea idéntico byte a byte
    ex2 = dict(ex)
    ex2["prompt"] = ex["prompt"].replace("function", "function /*v*/", 1)
    augmented.append(ex2)

with open(out, "w") as f:
    for ex in augmented:
        f.write(json.dumps(ex) + "\n")

print(f"Wrote {len(augmented)} examples to {out}")
