# Estimated Training Performance — centinela-0.5b

Reference document for planning training runs on **GitHub Actions (ubuntu-latest, 4 vCPU, 16 GB RAM, no GPU)**.

All times are measured in seconds unless stated otherwise. Figures include:
- The **linear model** fitted from real runs (see §2)
- A **+20% safety margin** to account for shared-runner noise (see §10)

---

## 1. Reference data (measured)

| Examples | Epochs | Steps | train_s | total_s | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 10 | 3 | 30 | 28.40 | 75.34 | First run; runner faster than average |
| 10 | 3 | 30 | 36.32 | 81.94 | Second run; used as **reference point** |
| 20 | 3 | 60 | 78.63 | 125.91 | |
| 100 | 3 | 300 | 352.33 | 397.48 | |

**Overhead (load + eval_pre + eval_post + save) ≈ 46 s**, assuming a test set of 10 examples.

---

## 2. Linear model

```
train_s ≈ 1.17 × steps + 1.2
total_s ≈ train_s + 46
```

- **steps** = `num_examples × num_epochs`
- **1.17 s/step** is the cost per forward+backward pass, for an example of **~200 tokens**
- **1.2 s** is a fixed per-training overhead (optimizer init, checkpoint write, etc.)
- **46 s** is the run-level overhead (model load + pre-eval + post-eval + save)

Per-token cost, assuming **200 tokens per example**:

```
τ = 1.17 / 200 = 0.00585 s / token / step
```

> ⚠️ This is a **linear approximation**. Attention adds a quadratic term that becomes noticeable above ~1024 tokens. The current pipeline caps sequences at `max_seq_length=512`, so the linear model holds for most realistic inputs.

---

## 3. Length conversions

For **source code**, the empirical averages are:

| Unit | Approx. value |
| :--- | :--- |
| 1 character | 0.286 tokens (≈ 3.5 chars/token) |
| 1 line of code | ~40 characters ≈ 11.4 tokens |
| 5-line snippet | ~200 characters ≈ 57 tokens |
| 20-line snippet | ~800 characters ≈ 229 tokens |
| 50-line file | ~2000 characters ≈ 571 tokens |
| 500-line file | ~20000 characters ≈ 5714 tokens ⚠️ (truncated to 512 in the pipeline) |

---

## 4. Single-input statistics (one example)

| Input | Chars | Tokens | Time / step | Time / epoch | Full run (1 epoch, +46 s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1 character** | 1 | 0.3 | 0.0017 s | 0.0017 s | ~46.0 s |
| **100 characters** | 100 | 28.6 | 0.167 s | 0.167 s | ~46.2 s |
| **1 line of code** | 40 | 11.4 | 0.067 s | 0.067 s | ~46.1 s |
| **5-line snippet** | 200 | 57 | 0.334 s | 0.334 s | ~46.3 s |
| **20-line snippet** | 800 | 229 | 1.34 s | 1.34 s | ~47.3 s |
| **50-line file** | 2000 | 571 | 3.34 s | 3.34 s | ~49.3 s |
| **100-line file** | 4000 | 1143 | 6.69 s | 6.69 s | ~52.7 s |
| **500-line file** ⚠️ | 20000 | 5714→512 | 3.00 s | 3.00 s | ~49.0 s |

**Key insight**: for inputs smaller than ~200 characters, **the run overhead (46 s) dominates completely**. Training time is negligible; you are paying for model load and evaluation.

> ⚠️ **500-line files are truncated to 512 tokens** by the current `max_seq_length` setting. If you need full files, raise it to 1024 or 2048 (at the cost of slower training and more RAM).

---

## 5. Dataset statistics — 5-line snippets (~57 tokens / step)

| Examples | Epochs | Steps | Train | **Full run** | **+20% margin** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 1 | 1 | 0.3 s | 46 s | **56 s** |
| 10 | 1 | 10 | 3.3 s | 49 s | **59 s** |
| 10 | 3 | 30 | 10 s | 56 s | **67 s** |
| 100 | 1 | 100 | 33 s | 79 s | **95 s** |
| 100 | 3 | 300 | 100 s | 146 s | **175 s** (2.9 min) |
| 1000 | 1 | 1000 | 334 s | 380 s | **456 s** (7.6 min) |
| 1000 | 3 | 3000 | 1002 s | 1048 s | **1258 s** (21 min) |
| 10000 | 1 | 10000 | 3340 s | 3386 s | **4063 s** (68 min) |
| 10000 | 3 | 30000 | 10020 s | 10066 s | **12079 s** (3.35 h) |

---

## 6. Dataset statistics — 20-line snippets (~229 tokens / step)

| Examples | Epochs | Steps | Train | **Full run** | **+20% margin** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 1 | 1 | 1.3 s | 47 s | **57 s** |
| 10 | 1 | 10 | 13 s | 59 s | **71 s** |
| 10 | 3 | 30 | 40 s | 86 s | **103 s** |
| 100 | 1 | 100 | 134 s | 180 s | **216 s** (3.6 min) |
| 100 | 3 | 300 | 402 s | 448 s | **538 s** (9 min) |
| 1000 | 1 | 1000 | 1340 s | 1386 s | **1663 s** (27.7 min) |
| 1000 | 3 | 3000 | 4020 s | 4066 s | **4879 s** (81 min) |
| 5000 | 1 | 5000 | 6700 s | 6746 s | **8095 s** (2.25 h) |
| 5000 | 3 | 15000 | 20100 s | 20146 s | **24175 s** (6.7 h) ❌ |

---

## 7. Dataset statistics — 50-line files (~571 tokens / step)

| Examples | Epochs | Steps | Train | **Full run** | **+20% margin** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 1 | 1 | 3.3 s | 49 s | **59 s** |
| 10 | 1 | 10 | 33 s | 79 s | **95 s** |
| 10 | 3 | 30 | 100 s | 146 s | **175 s** (2.9 min) |
| 100 | 1 | 100 | 334 s | 380 s | **456 s** (7.6 min) |
| **100** | **3** | **300** | **1002 s** | **1048 s** | **1258 s** (21 min) |
| 500 | 1 | 500 | 1670 s | 1716 s | **2059 s** (34 min) |
| 500 | 3 | 1500 | 5010 s | 5056 s | **6067 s** (101 min) |
| 1000 | 1 | 1000 | 3340 s | 3386 s | **4063 s** (68 min) |
| 1000 | 3 | 3000 | 10020 s | 10066 s | **12079 s** (3.35 h) |
| 2000 | 1 | 2000 | 6680 s | 6726 s | **8071 s** (2.24 h) |
| 2000 | 3 | 6000 | 20040 s | 20086 s | **24103 s** (6.7 h) ❌ |

---

## 8. Dataset statistics — 500-line files (truncated to 512 tokens / step)

> ⚠️ Truncation: only the first ~512 tokens of each file are seen by the model.

| Examples | Epochs | Steps | Train | **Full run** | **+20% margin** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 10 | 3 | 30 | 90 s | 136 s | **163 s** (2.7 min) |
| 100 | 3 | 300 | 900 s | 946 s | **1135 s** (19 min) |
| 500 | 3 | 1500 | 4500 s | 4546 s | **5455 s** (91 min) |
| 1000 | 3 | 3000 | 9000 s | 9046 s | **10855 s** (3 h) |
| 2000 | 3 | 6000 | 18000 s | 18046 s | **21655 s** (6.0 h) ❌ borderline |

---

## 9. Budget ceiling — what fits in the 6-hour GitHub Actions limit

The hard limit is `timeout-minutes: 360` = **21,600 s**. With the +20% safety margin applied to the whole run:

**Safe budget = (21600 − 46) / 1.2 ≈ 17,962 s of training time**

| Input type | Time / example × 3 epochs | **Max safe examples** |
| :--- | :--- | :--- |
| 1-line snippets | 0.2 s | ~90,000 |
| 5-line snippets | 1.0 s | ~17,900 |
| 20-line snippets | 4.0 s | ~4,490 |
| 50-line files | 10.0 s | **~1,796** |
| 100-line files | 20.1 s | ~893 |
| 500-line files (truncated) | 9.0 s | ~1,995 |

**Practical planning rule**: for 50-line files, **~1,500 examples × 3 epochs** is a comfortable target that leaves 15–20 % of headroom. Going to 1,800+ starts to be risky on a bad runner day.

If you need more examples:
- **Reduce epochs** (3 → 2 → 1)
- **Reduce max_seq_length** (512 → 256 for snippets)
- **Split across multiple runs** (drip-feed style)

---

## 10. The +20% safety margin

**Rationale**: shared GitHub runners are not perfectly consistent. In your own measurements:

| Configuration | Run A | Run B | Δ |
| :--- | :--- | :--- | :--- |
| 10 examples × 3 epochs | 28.40 s | 36.32 s | **+28 %** |

The 28 s run was ~22 % faster than the 36 s run because the runner was less contended at that moment. The +20 % margin is a **conservative lower bound** for that variance; on a bad day, expect up to +30 %.

**How to use it**: every table in this document has a **+20% margin** column that represents a "safe planning" value. If you size a run using that column and the runner happens to be fast, you'll finish early. If it's slow, you're covered.

---

## 11. Caveats and limits

1. **Eval time is treated as constant (≈46 s overhead)**. This assumes a test set of 10 examples. If you grow the test set to 100, overhead grows by ~5–7 min (linear in test size). Budget accordingly.
2. **Sequence-length cost is linear here, but quadratic in reality.** Attention scales `O(L²)`. For `L ≤ 512` the quadratic term is <10 % of total FLOPs and can be ignored. At `L = 2048` it becomes 30–40 % of the cost.
3. **Token counts are approximate.** Code tokenizes at ~3.5 chars/token on average, but dense code (e.g. minified JS) can be 2 chars/token, inflating step times by ~40 %.
4. **`OMP_NUM_THREADS=4`** is currently set. If you move to a runner with more cores, adjust it or the linear model breaks (parallelism changes the per-step cost).
5. **The 500-line row in §8 assumes truncation to 512 tokens.** Without truncation, a 20,000-char file would be ~5,700 tokens and the cost per step would be ~33 s, **10× higher** than what's listed.

---

## 12. Quick reference — planning formulas

```
steps        = num_examples × num_epochs
train_s      ≈ 1.17 × steps_for_200_token_inputs
train_s(L)   ≈ 0.00585 × L × num_examples × num_epochs   # L = tokens per example
total_s      ≈ train_s + 46                              # + ~5-7 min if test set > 100
safe_total_s = total_s × 1.2                             # +20% margin
```

**Example**: 500 examples of 50-line files × 3 epochs

```
L        = 571 tokens
train_s  = 0.00585 × 571 × 500 × 3 ≈ 5,010 s
total_s  = 5,010 + 46        ≈ 5,056 s (84 min)
safe     = 5,056 × 1.2       ≈ 6,067 s (101 min)
```

Well within the 6-hour limit. You could go up to **~1,500 examples** before hitting the ceiling.
