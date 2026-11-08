# Centinela

A **0.5B-parameter security classifier** for JavaScript. Given a code snippet, it predicts
one of **25 vulnerability classes** or `SAFE`. It does not generate text — it scores
the 26 candidate labels by their log-probability and returns the argmax.

```
┌──────────────────────────────────────────────────────────────┐
│                    snippet of JavaScript                     │
└────────────────────────────┬─────────────────────────────────┘
                             │
                    ┌────────▼─────────┐
                    │   Qwen2.5-0.5B   │
                    │  + rsLoRA r=32   │
                    └────────┬─────────┘
                             │
                   ┌─────────▼──────────┐
                   │  argmax over 26    │
                   │  label log-probs   │
                   └─────────┬──────────┘
                             │
              ┌──────────────▼───────────────┐
              │   XSS / SQLI / SSTI / … /SAFE│
              └──────────────────────────────┘
```

- **Base model:** [`huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3`](https://huggingface.co/huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3)
- **Adapter:** rsLoRA, rank 32, targets `q_proj` + `v_proj`, α = 2r = 64
- **Training budget:** 150–850 optimizer steps per run, CPU-only (`ubuntu-latest` runner)
- **Runs in ~2–4 hours on a GitHub-hosted runner** (free tier compatible)

---

## What it is / What it is not

**It is**

- A **triage tool**: rank chunks of code by how likely they are to contain a vulnerability,
  so a human reviews the top-K instead of everything.
- A **research artifact** for studying LoRA fine-tuning recipes, quantization effects, and
  continuous training on tiny budgets.
- A **baseline** for comparing adaptation techniques (rsLoRA, IA3, LoRA+, DoRA, target
  module selection, rank, learning rate).

**It is not**

- An **auto-alerting scanner**. It has a known bias toward `SAFE`-falsely-flagged-as-vulnerable
  on generic code (see [Limitations](#limitations)).
- A **chat model**. It will not explain CVEs or propose pentest plans. Ask it for one label.
- **Multi-language out of the box**. It was trained on JavaScript. It partially transfers
  to PHP/Java/Go for pattern-level bugs (`SQLI`, `CMDI`, `CODE_INJECTION`) but not for
  `SAFE` — a `function add(a, b) { return a + b; }` in any language can trigger a false positive.
- A **line locator**. It classifies a whole chunk; you need to chunk the file yourself.

---

## Quick start

### 1. Use a pre-built GGUF

Grab a quantized model from the [Releases](../../releases) page:

| Quant | Size (Qwen 0.5B) | Speed (P40 Lite, CPU) | Accuracy |
|-------|------------------|-----------------------|----------|
| F32   | ~2.0 GB | slow | reference |
| F16   | ~1.0 GB | slow | reference |
| Q8_0  | ~530 MB | ~7.5 tok/s | **highest (recommended for quality)** |
| Q6_K  | ~470 MB | ~7.5 tok/s | same as Q8 |
| **Q4_K_M** | **~380 MB** | **~7.5 tok/s** | **same as Q8 (recommended for size)** |
| Q3_K_M | ~280 MB | ~7.5 tok/s | −1 to −2 pp |
| Q2_K  | ~324 MB | ~7.5 tok/s | −3.3 pp (not recommended) |

Q4_K_M and Q8_0 give identical in-distribution accuracy (26/30 on our test set). Q8 is
33 % larger for zero benefit. **Use Q4_K_M.**

### 2. Serve it with `llama-server`

```bash
llama-server \
  -m Centinela-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8192 \
  --threads 3 --threads-batch 2 \
  -c 8192 -b 512 -ub 256 \
  --parallel 1 \
  --cache-type-k q8_0 --cache-type-v f16
```

### 3. Query it

The model expects the **exact** training prompt. Do not change a comma, do not add a
system message.

```python
import json, urllib.request

LABELS = [
    "XSS", "SQLI", "CMDI", "CODE_INJECTION", "PATH_TRAVERSAL", "SSRF",
    "PROTO_POLLUTION", "OPEN_REDIRECT", "AUTHZ", "WEAK_CRYPTO",
    "SSTI", "CSRF", "XXE", "LDAP_INJECTION", "HPP", "CRLF_INJECTION",
    "CSSI", "HTMLI", "NOSQLI", "HARDCODED_SECRETS", "MASS_ASSIGNMENT",
    "WEAK_COOKIE", "INSECURE_JWT", "RACE_CONDITION", "INFO_DISCLOSURE", "SAFE",
]

PROMPT = """Analyze the following JavaScript code for security vulnerabilities.
If it is vulnerable, answer with the vulnerability type; if it is not, answer SAFE.
Answer with exactly one of: {labels}.

Code:
```javascript
{code}
```"""

code = """app.get("/user", (req, res) => {
  const sql = "SELECT * FROM users WHERE id = " + req.query.id;
  db.query(sql, (err, rows) => res.json(rows));
});"""

body = json.dumps({
    "messages": [{"role": "user", "content": PROMPT.format(labels=", ".join(LABELS), code=code)}],
    "max_tokens": 8,
    "temperature": 0,
}).encode()

req = urllib.request.Request("http://127.0.0.1:8192/v1/chat/completions",
                             data=body, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as r:
    resp = json.load(r)
print(resp["choices"][0]["message"]["content"])   # -> SQLI
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Training (GitHub Actions)                    │
│                                                                      │
│  data/train.jsonl ──► continue_training.py ──► train_new.jsonl      │
│        │                    │                                        │
│        │                    ├─► filter unseen (sha256 of code+label)│
│        │                    ├─► stratified selection (fits budget)  │
│        │                    └─► defer rest to next run              │
│        │                                                             │
│        └───────────────► run_pipeline.py ──► LoRA adapter + metrics │
│                                                      │               │
│                                                      ▼               │
│                                         merge + convert to GGUF       │
│                                                      │               │
│                                                      ▼               │
│                                    GitHub Release (assets + JSON)    │
└──────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │   Continuous training loop   │
              │   (drip-feed, once per day)  │
              └──────────────────────────────┘
```

### Design decisions

| Decision | Why |
|---|---|
| **Classification, not generation** | No parsing, no `max_new_tokens`, no greedy decoding. One forward pass per snippet. ~10× cheaper than generating text. |
| **First-token or mean log-prob scoring** | With 26 labels, several share a first token (`SQLI`/`SSRF`/`SSTI`). The pipeline auto-falls back to `mean` (average log-prob of every label token) when the first token is not unique. |
| **Loss only on answer tokens** | Prompt/code tokens are masked with `-100`. `supervised_fraction` is ~5 %. |
| **Canonical label order** | Always the same 26 labels, in the same order, in train and eval. Per-example shuffling is a memorization channel. |
| **Template = pair index mod K** | Templates are distributed evenly across types (Latin square). `train_templates = 1` for all default runs. |
| **Seeds control adapter init + data order** | Reproducible across runs. |
| **No optimizer state across runs** | Continuous training uses a fresh AdamW per run with the same constant LR. Works well for LoRA at small scale; fragile at large scale. |

---

## Training pipeline

### The dataset

Each training example is a **pair**: one vulnerable snippet + one fixed twin.
The two snippets share everything except the line(s) that cause (or fix) the vulnerability.

```jsonl
{"pair_id": "train-XSS-01", "set": "train", "vuln_type": "XSS", "subtype": "plain",
 "id": "train-XSS-01-v", "label": "XSS", "role": "vuln",
 "code": "function showWelcome() { ... document.getElementById('welcome').innerHTML = 'Hello, ' + name; ... }"}

{"pair_id": "train-XSS-01", "set": "train", "vuln_type": "XSS", "subtype": "plain",
 "id": "train-XSS-01-s", "label": "SAFE", "role": "safe",
 "code": "function showWelcome() { ... document.getElementById('welcome').textContent = 'Hello, ' + name; ... }"}
```

**Fields:**

| Field | Required | Description |
|---|---|---|
| `pair_id` | yes | Groups the vuln + safe twins |
| `set` | yes | `train`, `iid`, `ood_near`, `ood_hard`, `easy` |
| `vuln_type` | no | Same as `label` for vuln examples, `SAFE` for safe examples |
| `subtype` | no | `plain` or `insufficient` (vulnerability *looks* mitigated but isn't) |
| `id` | yes | Unique identifier |
| `label` | yes | One of 26: 25 vuln classes + `SAFE` |
| `role` | yes | `vuln` or `safe` (structural, independent of the possibly-permuted label) |
| `code` | yes | The snippet (JavaScript in the current dataset) |

**Label set (26 labels, `LABELS` in `run_pipeline.py`):**

```
XSS, SQLI, CMDI, CODE_INJECTION, PATH_TRAVERSAL,
SSRF, PROTO_POLLUTION, OPEN_REDIRECT, AUTHZ, WEAK_CRYPTO,
SSTI, CSRF, XXE, LDAP_INJECTION, HPP,
CRLF_INJECTION, CSSI, HTMLI, NOSQLI, HARDCODED_SECRETS,
MASS_ASSIGNMENT, WEAK_COOKIE, INSECURE_JWT, RACE_CONDITION,
INFO_DISCLOSURE, SAFE
```

Adding a class requires editing `VULN_TYPES` in `scripts/run_pipeline.py` **and** regenerating
the test set (otherwise metrics for the new class are `NaN`).

### The training run

```bash
# One-shot (from zero)
python scripts/run_pipeline.py \
  --base_model huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3 \
  --train_file data/train.jsonl \
  --test_file data/test.jsonl \
  --output_dir ./lora_adapter \
  --num_epochs 6 \
  --time_budget_min 280 \
  --lr 1e-4 \
  --seed 42 \
  --train_templates 1 \
  --rslora \
  --lora_r 32 \
  --lora_targets q_proj,v_proj \
  --lora_dropout 0.05 \
  --lr_scheduler constant_with_warmup \
  --grad_accum 4 \
  --train_probe_n 20 \
  --skip_step0 \
  --eval_every 1000000
```

**Key flags:**

- `--rslora` — rank-stabilized LoRA scaling (`alpha/sqrt(r)` instead of `alpha/r`). Improves
  stability at r ≥ 16.
- `--lora_r 32` — higher rank than the default 8. Required for this model.
- `--lora_targets q_proj,v_proj` — only 2 modules per layer, matching the training recipe
  used in the winner of our combinatorial study.
- `--time_budget_min` — hard cap on training time. Final eval always runs.
- `--skip_step0` — skip the step-0 evaluation. Only relevant for continuous training,
  where the initial model is already a fine-tuned adapter.
- `--eval_every 1000000` — disable intermediate evals; only final metrics. Halves runtime
  on tiny datasets.

### Interpreting `metrics.json`

```json
{
  "config": { "labels": [...], "total_steps": 510, "diverged": false, ... },
  "timings": { "train_s": 8400, "eval_final_s": 320 },
  "curve": [ { "step": 510, "metrics": { "iid/seen": { ... }, ... } } ]
}
```

For each `set/condition`:

| Metric | Chance | Meaning |
|---|---|---|
| `type_acc_forced` | 0.10 | Accuracy among the 25 vuln classes, vulnerable examples only |
| `type_mrr` | 0.293 | Mean reciprocal rank of the correct type |
| `det_auroc` | 0.50 | AUROC of "vulnerable vs SAFE" |
| `pair_acc` | 0.50 | Vulnerable twin scored above its safe twin |
| `det_bal_acc` | 0.50 | Balanced accuracy of the binary detection |

`diverged: true` → training produced non-finite or exploding loss; the release is
**not eligible** for continuous training (no `elegible.txt` will be created).

---

## Continuous training (drip-feed)

The repo ships a **fully automated continuous-training loop**. You add new examples,
run the workflow, and it trains only the **unseen** ones from the latest eligible adapter.

### How it works

1. **`trained.json`** (per release) stores a SHA-256 hash of every `(code, label)` pair
   ever used in training. Renaming `id` does **not** mark an example as new. Editing the
   code or the label **does** (correctly — you want to retrain on the fix).
2. **`continue_training.py`** filters `data/train.jsonl` to unseen examples, then selects
   a **stratified subset** that fits the training budget (`time_budget_min × safety_factor
   / sec_per_step / num_epochs`). The rest goes to **`deferred.json`** — they are **not**
   marked as trained and come back next run.
3. **`elegible.txt`** (per release) is the sentinel. A release without it is **skipped**
   by `resume_from=latest`, so failed/diverged runs do not break the chain.

### The daily loop

```bash
# 1. Add new examples (append to data/train.jsonl)
python scripts/gen_extended_data.py --n-pairs 40 --append data/train.jsonl

# 2. Commit and push
git add data/train.jsonl
git commit -m "day 2: add 600 new training pairs"
git push

# 3. Trigger training (finds the latest eligible release automatically)
gh workflow run centinela-optimized-training.yml -f resume_from=latest
```

The workflow:

1. Finds the latest release with `elegible.txt` + `*-lora.safetensors` + `trained.json`
2. Downloads the adapter, `trained.json`, and previous metrics
3. Filters unseen examples, selects a budget-fitting subset, writes `deferred.json`
4. Trains from `--resume_adapter ./prev` on the new subset
5. Merges selected hashes into `trained.json`
6. Publishes 7 quantizations + LoRA + metrics to a new Release
7. Dispatches `centinela-optimized-test.yml` for the new release

**Cero intervention** beyond step 1–3. If you add 10,000 examples at once, it trains
~600 per day for 16 days, keeping the rest in `deferred.json`.

### Rollback

```bash
gh workflow run centinela-optimized-training.yml \
  -f resume_from=centinela-2026-11-08-42438926600-0.5B-rslora+rank32 \
  -f num_epochs 4
```

Uses the `trained.json` from that release. Examples that were trained on the bad run
come back as new and get re-trained.

---

## GitHub Actions workflows

| Workflow | Purpose | Cost (free tier) |
|---|---|---|
| `centinela-optimized-training.yml` | Continuous training: filter → select → train → quantize → release → dispatch test | ~50–250 min / run |
| `centinela-optimized-test.yml` | Evaluate a released model on the test set, smoke-test the GGUF | ~10–20 min / run |
| `centinela-combo.yml` | Full factorial over 5 LoRA-compatible techniques (32 combinations) | ~30 h / experiment |
| `centinela-oat.yml` | One-at-a-time technique comparison vs baseline | ~19 h / experiment |
| `centinela-train.yml` | Manual single-shot training with all knobs exposed | ~1 h / run |

**`centinela-optimized-training.yml` inputs:**

| Input | Default | Notes |
|---|---|---|
| `base_model` | `huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3` | |
| `model_size` | `0.5B` | Used in release tag |
| `techniques` | `rslora+rank32` | Used in release tag |
| `all_train_file` | `data/train.jsonl` | Filtered to unseen by default |
| `num_epochs` | `6` | Epochs on the selected subset |
| `time_budget_min` | `250` | Training-time cap. Job timeout = 360 min |
| `safety_factor` | `0.85` | Fraction of budget to plan for |
| `sec_per_step` | `0` | 0 = auto-detect from previous `metrics.json` |
| `lr` | `1e-4` | |
| `seed` | `42` | |
| `resume_from` | `''` (fresh) | `latest` for drip-feed, or explicit tag |
| `min_new_rows` | `2` | Abort if fewer unseen examples than this |

### Release naming

```
centinela-{YYYY-MM-DD}-{RUN_ID}-{SIZE}-{TECHNIQUES}
```

Example:

```
centinela-2026-11-08-42438926600-0.5B-rslora+rank32
  ├── centinela-…-F32.gguf
  ├── centinela-…-F16.gguf
  ├── centinela-…-Q8_0.gguf
  ├── centinela-…-Q6_K.gguf
  ├── centinela-…-Q4_K_M.gguf
  ├── centinela-…-Q3_K_M.gguf
  ├── centinela-…-Q2_K.gguf
  ├── centinela-…-lora.safetensors
  ├── centinela-…-adapter_config.json
  ├── centinela-…-metrics.json
  ├── trained.json
  ├── deferred.json
  └── elegible.txt
```

`elegible.txt` example:

```yaml
tag: centinela-2026-11-08-42438926600-0.5B-rslora+rank32
run_id: 42438926600
timestamp: 2026-11-08T06:22:10Z
base_model: huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3
techniques: rslora+rank32
num_epochs: 6
time_budget_min: 250
safety_factor: 0.85
sec_per_step: 27.3
resumed_from:
new_rows_selected_this_run: 340
deferred_rows_remaining: 0
total_rows_trained: 340
diverged: false
stopped_by_time_budget: False
```

---

## Inference

### Batch classification (recommended)

```python
from pathlib import Path
import json, urllib.request

SERVER = "http://127.0.0.1:8192"
LABELS = "XSS SQLI CMDI CODE_INJECTION PATH_TRAVERSAL SSRF PROTO_POLLUTION OPEN_REDIRECT AUTHZ WEAK_CRYPTO SSTI CSRF XXE LDAP_INJECTION HPP CRLF_INJECTION CSSI HTMLI NOSQLI HARDCODED_SECRETS MASS_ASSIGNMENT WEAK_COOKIE INSECURE_JWT RACE_CONDITION INFO_DISCLOSURE SAFE".split()

PROMPT = """Analyze the following JavaScript code for security vulnerabilities.
If it is vulnerable, answer with the vulnerability type; if it is not, answer SAFE.
Answer with exactly one of: {labels}.

Code:
```javascript
{code}
```"""

def classify(code: str) -> str:
    body = json.dumps({
        "messages": [{"role": "user", "content": PROMPT.format(labels=", ".join(LABELS), code=code)}],
        "max_tokens": 8,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(f"{SERVER}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = json.load(r)["choices"][0]["message"]["content"].strip()
    for tok in raw.replace(",", " ").replace(".", " ").split():
        tok = tok.strip("`'\"()[]{}:;")
        if tok in LABELS:
            return tok
    return "UNKNOWN"

# Chunk large files (no line-level localization)
def chunks(code, max_lines=120, overlap=10):
    lines = code.split("\n")
    if len(lines) <= max_lines:
        yield 1, "\n".join(lines)
        return
    i = 0
    while i < len(lines):
        yield i + 1, "\n".join(lines[i:i + max_lines])
        if i + max_lines >= len(lines):
            break
        i += max_lines - overlap

for path in Path("./src").rglob("*.js"):
    for line, chunk in chunks(path.read_text(errors="replace")):
        label = classify(chunk)
        if label != "SAFE":
            print(f"{label:20s} {path}:{line}")
```

### Recommended serving parameters

```bash
llama-server \
  -m Centinela-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8192 \
  --threads 3 --threads-batch 2 \
  -c 8192 -b 512 -ub 256 \
  --parallel 1 \
  --cache-type-k q8_0 --cache-type-v f16 \
  --cache-ram 256 \
  --no-context-shift \
  --temp 0.0 --top-k 1 --top-p 1.0 \
  --repeat-penalty 1.0 --repeat-last-n 0 \
  --offline
```

**Always override per request** with `temperature: 0, max_tokens: 8`. The server-level
sampling flags are for chat use; classification needs greedy decoding.

### Mobile (Termux / P40 Lite)

The repo includes no mobile script (kept out of scope). On a Huawei P40 Lite (Kirin 810),
Q4_K_M runs at **~7.5 tokens/s** for prompt processing, ~2–4 s per classification after
the first request. First request is ~10–15 s (cold cache). 100-chunk file → ~5–8 min.

---

## Experimental results

### Factorial study (5 LoRA-compatible techniques, 32 combinations)

Best single technique in the OAT study: **IA3**, SCORE 0.736 (SCORE = mean of TYPE,
TYPE_SHIFT, DET). Best observed combination: `targets_all + lr3e4 + rank32`,
SCORE 0.816. **Selection-inflated** — confirmed with 5 new seeds:

| Config | 5-seed SCORE | Δ vs IA3 | Significant? |
|---|---|---|---|
| `rslora + rank32` | 0.740 ± 0.041 | **+0.059** | `*` only (not `**`) |
| `targets_all + lr3e4 + rslora` | 0.732 ± 0.070 | +0.051 | no |
| `targets_all + lr3e4 + rank32` | 0.716 ± 0.074 | +0.035 | no |
| **IA3** (best single) | 0.682 ± 0.021 | — | — |
| Baseline | 0.510 ± 0.014 | — | — |

**Takeaway:** all recipes beat baseline by **+0.17 to +0.23 SCORE**; no combo reliably
beats IA3 with 5 seeds. `rslora + rank32` is our working recipe because it is combinable
(IA3 replaces the adapter family and cannot be mixed).

### Quantization (25 classes, Q4_K_M run)

| Quant | Hand-crafted | In-distribution (30 samples) | Notes |
|---|---|---|---|
| Q2_K | 3/4 | 25/30 (83.3 %) | Loses `PATH_TRAVERSAL`, `SUMO` hallucination |
| **Q4_K_M** | **3/4** | **26/30 (86.7 %)** | **Recommended** |
| Q8_0 | 3/4 | 26/30 (86.7 %) | +33 % size for zero gain |

Fails are **systematic**, not random:

- `AUTHZ`, `CODE_INJECTION`, `PROTO_POLLUTION`: classified `SAFE` by all three quants
- Generic code (`function add(a, b) { return a + b; }`) classified as `XSS` by all three

These are **data issues**, not quantization issues. See [Limitations](#limitations).

---

## Repo structure

```
.
├── .github/workflows/
│   ├── centinela-optimized-training.yml   # continuous training + quantize + release
│   ├── centinela-optimized-test.yml       # evaluation + smoke test
│   ├── centinela-combo.yml                # factorial over LoRA techniques
│   ├── centinela-oat.yml                  # one-at-a-time comparison
│   └── centinela-train.yml                # manual single-shot training
├── data/
│   ├── train.jsonl                        # training pairs (vuln + safe twins)
│   └── test.jsonl                         # held-out test set
├── scripts/
│   ├── run_pipeline.py                    # core training + evaluation
│   ├── continue_training.py               # filter unseen + budget-aware selection
│   ├── gen_extended_data.py               # synthetic data generator
│   ├── oat.py                             # OAT workflow planner/reporter
│   └── combo.py                           # factorial workflow planner/reporter
├── requirements.txt
└── README.md
```

---

## Limitations

The current model has **five known issues**, all rooted in the data, not the quantization:

1. **SAFE bias on generic code.** The training set's `SAFE` examples are *fixed twins* of
   vulnerable snippets — they have a "fixed" look. The model never saw a plain utility
   function. As a result, `function add(a, b) { return a + b; }` is classified `XSS`.
   **Workaround:** add ~200 generic `SAFE` examples to `data/train.jsonl` (see
   `add-safe.js.py` in the companion `~/OpenMobile/` folder, or regenerate with
   `gen_extended_data.py` after extending its `SAFE` templates). Retrain with `resume_from=latest`.

2. **Three systematically missing classes.** `AUTHZ`, `CODE_INJECTION`, `PROTO_POLLUTION`
   are classified `SAFE` by every quantization tested. They have 5–8 pairs/clase — likely
   undertrained. **Workaround:** more data for those specific classes.

3. **JavaScript-only in practice.** Partial transfer to PHP/Java/Go for `SQLI`, `CMDI`,
   `CODE_INJECTION`, `WEAK_CRYPTO`. Poor transfer for `XSS` (JS-specific `innerHTML`),
   `PROTO_POLLUTION` (JS-specific `__proto__`), `SSTI`. **Workaround:** restrict the
   scanner to `.js/.ts/.jsx/.tsx` for high-precision use, or add per-language training
   data.

4. **No line-level localization.** The classifier sees a whole chunk. A file with a
   vulnerability on line 200 will flag the whole 120-line chunk that contains it. **Workaround:**
   chunk with overlap and let the reviewer inspect.

5. **Multi-vulnerability collapses to one label.** A snippet with XSS and SQLI gets one
   label. **Workaround:** chunk small enough that most chunks contain at most one bug.

**Operational implication:** use Centinela for **triage**, not alerting. Ranking a repo's
chunks from "most likely vulnerable" to "least likely" is useful. Reporting every non-`SAFE`
chunk as a finding is not — expect 40–100 % false positives on generic code, depending on
how much the input resembles the synthetic training distribution.

---

## Roadmap

- [ ] Add ~200 generic `SAFE` examples (in progress — fixes issue 1)
- [ ] Add 20+ examples per class for `AUTHZ`, `CODE_INJECTION`, `PROTO_POLLUTION` (issue 2)
- [ ] Expand to PHP / Java / Go with balanced training data (issue 3)
- [ ] Line-level localization (chunk scoring with offset aggregation)
- [ ] Two-stage classifier: "is this security-relevant code?" → 25-way classification
- [ ] Proper calibration (temperature scaling, isotonic regression on validation set)
- [ ] Server-side logprob scoring (`/completion` endpoint with `echo`), avoiding the
      first-token collision between `SSTI` / `SQLI` / `SSRF`

---

## Requirements

- Python 3.10+
- `transformers==4.46.3`, `peft==0.13.2`, `torch==2.5.1` (CPU), `datasets==3.1.0`
- `llama.cpp` for GGUF conversion and quantization (cloned at build time inside the workflow)
- For inference: `llama-server` (from `llama.cpp`), or any OpenAI-compatible server

Install:

```bash
pip install -r requirements.txt
```

---

## Acknowledgments

- Base model: [`huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3`](https://huggingface.co/huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3)
- LoRA + PEFT: [`huggingface/peft`](https://github.com/huggingface/peft)
- GGUF quantization: [`ggerganov/llama.cpp`](https://github.com/ggerganov/llama.cpp)
- Trained entirely on GitHub-hosted runners. No GPU was used at any point.

---

## License

Code: MIT (see `LICENSE`). Model weights inherit the base model's license.