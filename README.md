# centinela-0.5b

> *A scalpel, not a hammer: a 0.5B security specialist trained by drip-feed.*

**Vision**: a tiny, offline, local security agent capable of auditing **code, web servers and infrastructure** — eventually wrapping tools such as `nmap`, `nuclei`, `feroxbuster`, `nikto` and similar through an agent wrapper.

**Current state: MVP (Phase 1)**. This phase is not a product. Its goal is to **validate the methodology** on a 0.5B model:

- How much LoRA improves accuracy over a minimal dataset
- How long training takes as a function of number of examples and epochs
- What the optimal number of epochs is for this task
- Whether the "drip-feed" cycle on GitHub Actions (no GPU) is time-viable

## Why JavaScript in the MVP

JavaScript is the ideal sandbox for *this phase*:

- Ubiquitous (every web stack uses it)
- Rich in classic vulnerability patterns (XSS, `eval`, prototype pollution, command injection, string-concatenated SQLi)
- Easy to generate synthetic "safe vs vulnerable" pairs that are short (fast to tokenize)
- The *skill* being trained is "detect dangerous code", not "know JavaScript". That skill transfers to other languages in later phases.

## Roadmap

| Phase | Status | Description |
| :--- | :--- | :--- |
| **Phase 1** | In progress | MVP: JS vulnerability detection, LoRA on 0.5B, GitHub Actions pipeline |
| **Phase 2** | Planned | Multi-language code: Python, C, PHP, Go, SQL, Kotlin |
| **Phase 3** | Planned | Web recon: headers, cookies, CORS, misconfigs, server responses |
| **Phase 4** | Planned | Server/infra agent: wrapper over `nmap`, `nuclei`, output parsing |
| **Phase 5** | Planned | Continuous drip-feed with replay buffer + auto-rollback |

## How the pipeline works (Phase 1)

1. `data/train.jsonl` — vulnerable/safe code examples (JS in the MVP)
2. `data/test.jsonl` — **different vulnerability classes** than train (held-out) to measure real generalization
3. `scripts/run_pipeline.py`:
   - loads the base model
   - runs PRE evaluation (greedy, 3 passes, majority vote)
   - trains LoRA (`r=8`, `q_proj`/`v_proj`, 3 epochs)
   - runs POST evaluation
   - writes `metrics.json` with timings, improvement and predictions
4. The workflow uploads the adapter and `metrics.json` as an artifact

## Repository layout

```
.
├── .github/workflows/centinela-train.yml   # GitHub Actions workflow
├── data/
│   ├── train.jsonl                         # Training examples
│   └── test.jsonl                          # Held-out test examples
├── scripts/
│   └── run_pipeline.py                     # Eval -> train -> eval pipeline
├── requirements.txt
└── README.md
```

## Run a training job

```bash
gh workflow run centinela-train.yml
gh run watch
```

## Sweeps (find optimal epochs and dataset size)

The workflow accepts `num_epochs`, `max_train_examples` and `max_test_examples` as inputs. Sweep from the CLI:

```bash
# Epoch sweep
for e in 1 3 5 10; do
  gh workflow run centinela-train.yml -f num_epochs=$e
done

# Dataset size x epochs sweep
for e in 1 3 5; do
  for n in 10 50 100; do
    gh workflow run centinela-train.yml -f num_epochs=$e -f max_train_examples=$n
  done
done
```

## Aggregate sweep results

Each run uploads a `metrics.json`. To aggregate them:

```bash
mkdir -p sweep && cd sweep
for run_id in $(gh run list --workflow=centinela-train.yml --limit 20 --json databaseId --jq '.[].databaseId'); do
  gh run download "$run_id" -n "centinela-run-$run_id" -D "run_$run_id" 2>/dev/null || true
done

# Tabular summary
jq -r '[.timestamp, .num_train_examples, .num_epochs, .acc_before, .acc_after, .improvement, .timings.train_s, .timings.total_s] | @tsv' \
  $(find . -name metrics.json) | column -t
```

## Download a trained adapter

```bash
RUN_ID=$(gh run list --workflow=centinela-train.yml --limit 1 --json databaseId --jq '.[0].databaseId')
gh run download "$RUN_ID" -n "centinela-run-$RUN_ID" -D ./local_run
ls -la ./local_run
cat ./local_run/metrics.json | jq .
```

## Metrics format

Every run produces a `metrics.json` next to the LoRA adapter:

```json
{
  "timestamp": "2026-11-02T12:00:00Z",
  "base_model": "huihui-ai/Qwen2.5-0.5B-Instruct-abliterated",
  "num_train_examples": 10,
  "num_test_examples": 10,
  "num_epochs": 3,
  "batch_size": 1,
  "acc_before": 0.5,
  "acc_after": 0.7,
  "improvement": 0.2,
  "timings": {
    "model_load_s": 12.3,
    "eval_before_s": 45.1,
    "train_s": 320.5,
    "save_s": 1.2,
    "eval_after_s": 44.8,
    "total_s": 423.9
  },
  "trainer_metrics": {
    "train_runtime": 320.1,
    "train_loss": 0.42,
    "train_samples_per_second": 0.09
  },
  "predictions_after": []
}
```

## Limitations (Phase 1)

- **10 examples per class** only validate the pipeline; they don't teach real detection
- **Small test set**: high variance in accuracy
- **Abliterated base model**: no refusal filters, but not a "security expert" either
- **CPU on GitHub Actions**: 4 vCPUs, ~10x slower than a GPU T4
- **Disjoint vulnerabilities between train and test** on purpose: if it improves, it's generalization, not memorization

