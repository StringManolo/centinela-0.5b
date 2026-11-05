#!/usr/bin/env python3
"""
One-at-a-time (OAT) technique comparison for the GitHub workflow centinela-oat.yml.

  plan    prints the GitHub matrix ({"include": [...]}) = baseline x N seeds + every technique x its seeds.
          Each cell is the SAME baseline recipe with ONE change.
  report  reads the downloaded artifacts and compares every technique with the baseline, using the
          seed-to-seed spread of the baseline runs as the noise level (Student t, uncorrected and
          Bonferroni-corrected), plus two "summed results" rows (ensembles from the saved label scores).

Fairness notes, printed again in the report:
  * every run has the same optimizer-step budget (small_batch keeps the same number of forward/backward
    passes); runs stopped by --time_budget_min are marked,
  * ia3 uses its literature learning rate (3e-3): at 1e-4 it would simply not train, which would measure
    the learning rate, not the technique.
"""
import argparse
import json
import math
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)  # nanmean of all-NaN columns is expected
HERE = os.path.dirname(os.path.abspath(__file__))


def _rp():
    """run_pipeline is only needed by `report` (metrics for the ensembles); `plan` must work without it."""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import run_pipeline
    return run_pipeline


def es_load(d):
    rows = {}
    with open(os.path.join(d, "scores_final.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows[(r["id"], r["cond"])] = r
    return rows


def es_groups(rows):
    g = {}
    for r in rows:
        g.setdefault(f"{r['set']}/{r['cond']}", []).append(r)
    return g


def es_combine(runs, how):
    """Average the per-row label scores of several runs: logp = mean of log-probs, prob = log of mean prob."""
    out = []
    for k in sorted(runs[0]):
        S = np.array([run[k]["scores"] for run in runs], dtype=np.float64)
        if how == "logp":
            s = S.mean(0)
        else:
            mx = S.max(0)
            s = mx + np.log(np.exp(S - mx).mean(0))
        out.append(dict(runs[0][k], scores=s.tolist()))
    return out

BASELINE = {
    "--lr": "1e-4", "--lora_targets": "q_proj,v_proj", "--lora_r": "8", "--lora_dropout": "0.05",
    "--train_templates": "1", "--lr_scheduler": "constant_with_warmup", "--grad_accum": "4",
    "--skip_step0": None, "--train_probe_n": "20", "--eval_every": "1000000",
}
# label -> (description, overrides). "x4" for --max_steps means 4x the base budget.
TECHNIQUES = {
    "targets_all": ("LoRA en todos los lineales", {"--lora_targets": "all-linear"}),
    "rank32": ("LoRA rango 32", {"--lora_r": "32"}),
    "rslora": ("rsLoRA (escala alpha/sqrt(r))", {"--rslora": None}),
    "dora": ("DoRA", {"--dora": None}),
    "lr3e4": ("LR 3e-4", {"--lr": "3e-4"}),
    "lora_plus": ("LoRA+ (lr de B x16)", {"--lora_plus_ratio": "16"}),
    "label_ce": ("loss CE restringida a las 11 etiquetas", {"--loss": "label_ce"}),
    "label_ce_smooth": ("label_ce + label smoothing 0.1", {"--loss": "label_ce", "--label_smoothing": "0.1"}),
    "neftune": ("NEFTune alpha 5", {"--neftune_alpha": "5"}),
    "dropout02": ("LoRA dropout 0.2", {"--lora_dropout": "0.2"}),
    "wd01": ("weight decay 0.1", {"--weight_decay": "0.1"}),
    "templates5": ("5 plantillas de entrenamiento", {"--train_templates": "5"}),
    "cosine": ("schedule coseno", {"--lr_scheduler": "cosine"}),
    "small_batch": ("batch efectivo 1 (mismo computo)", {"--grad_accum": "1", "--max_steps": "x4"}),
    "upper_half": ("LoRA solo en la mitad superior", {"--lora_layers": "upper_half"}),
    "ia3": ("IA3 (lr 3e-3)", {"--peft": "ia3", "--lr": "3e-3"}),
    "loha": ("LoHa", {"--peft": "loha"}),
    "lokr": ("LoKr", {"--peft": "lokr"}),
}

RESP = {
    "TYPE": [(f"{s}/seen", "type_acc_forced") for s in ("iid", "ood_near", "ood_hard")],
    "TYPE_SHIFT": [(f"{s}/shift", "type_acc_forced") for s in ("iid", "ood_near", "ood_hard")],
    "MRR": [(f"{s}/seen", "type_mrr") for s in ("iid", "ood_near", "ood_hard")],
    "DET": [(f"{s}/seen", "det_auroc") for s in ("iid", "ood_near", "ood_hard")],
    "EASY_TYPE": [("easy/seen", "type_acc_forced")],
    "EASY_DET": [("easy/seen", "det_auroc")],
    "FIT": [("train/seen", "type_acc_forced")],
    "TYPE_TTA": [(f"{s}/tta2", "type_acc_forced") for s in ("iid", "ood_near", "ood_hard")],
    "DET_TTA": [(f"{s}/tta2", "det_auroc") for s in ("iid", "ood_near", "ood_hard")],
}
TABLE_COLS = ["TYPE", "TYPE_SHIFT", "MRR", "DET", "EASY_TYPE", "EASY_DET", "FIT"]


# --------------------------------------------------------------------------- statistics
def t_ppf(p, df):
    """Student-t quantile by numerical integration (no scipy needed)."""
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)

    def cdf(x):
        n = max(4001, int(x * 400))
        xs = np.linspace(0.0, x, n)
        pdf = c * (1 + xs ** 2 / df) ** (-(df + 1) / 2)
        return 0.5 + float(np.sum((pdf[1:] + pdf[:-1]) / 2 * np.diff(xs)))

    lo, hi = 0.0, 400.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) < p else (lo, mid)
    return (lo + hi) / 2


def resp_value(metrics, pairs):
    vals = [metrics.get(k, {}).get(m) for k, m in pairs]
    vals = [v for v in vals if isinstance(v, (int, float))]
    return float(np.mean(vals)) if vals else float("nan")


# --------------------------------------------------------------------------- plan
def build_args(label, max_steps, time_budget):
    flags = dict(BASELINE)
    flags["--max_steps"] = str(max_steps)
    flags["--time_budget_min"] = str(time_budget)
    if label != "baseline":
        for k, v in TECHNIQUES[label][1].items():
            flags[k] = str(max_steps * 4) if (k == "--max_steps" and v == "x4") else v
    return " ".join(k if v is None else f"{k} {v}" for k, v in flags.items())


def cmd_plan(a):
    seeds_b = [int(s) for s in a.baseline_seeds.split(",") if s.strip()]
    seeds_t = [int(s) for s in a.technique_seeds.split(",") if s.strip()]
    names = list(TECHNIQUES) if a.techniques.strip() in ("", "all") else [t.strip() for t in a.techniques.split(",")]
    unknown = [n for n in names if n not in TECHNIQUES]
    if unknown:
        sys.exit(f"unknown techniques {unknown}; available: {list(TECHNIQUES)}")
    include = [{"label": "baseline", "seed": s, "args": build_args("baseline", a.max_steps, a.time_budget)} for s in seeds_b]
    include += [{"label": n, "seed": s, "args": build_args(n, a.max_steps, a.time_budget)} for n in names for s in seeds_t]
    print(f"{len(include)} jobs ({len(seeds_b)} baseline + {len(names) * len(seeds_t)} technique)", file=sys.stderr)
    print(json.dumps({"include": include}, separators=(",", ":")))


# --------------------------------------------------------------------------- report
def parse_dir(name):
    core = name[len("centinela-oat-"):]
    label, seed = core.rsplit("-s", 1)
    return label, int(seed)


def load_runs(root):
    runs = []
    for d in sorted(os.listdir(root)):
        path = os.path.join(root, d)
        if not (d.startswith("centinela-oat-") and os.path.isfile(os.path.join(path, "metrics.json"))):
            continue
        m = json.load(open(os.path.join(path, "metrics.json")))
        label, seed = parse_dir(d)
        final = m["curve"][-1]["metrics"]
        runs.append({"label": label, "seed": seed, "dir": path, "final": final, "cfg": m["config"],
                     "timings": m["timings"], "resp": {r: resp_value(final, p) for r, p in RESP.items()}})
    return runs


def fmt(x, signed=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:+.3f}" if signed else f"{x:.3f}"


def ensemble_row(runs, how="logp"):
    loaded = [es_load(r["dir"]) for r in runs if os.path.isfile(os.path.join(r["dir"], "scores_final.jsonl"))]
    if len(loaded) < 2:
        return None
    keys = set(loaded[0])
    loaded = [x for x in loaded if set(x) == keys]
    if len(loaded) < 2:
        return None
    groups = es_groups(es_combine(loaded, how))
    met = {k: _rp().compute_metrics(v) for k, v in groups.items()}
    return len(loaded), {r: resp_value(met, p) for r, p in RESP.items()}


def cmd_report(a):
    runs = load_runs(a.dir)
    base = [r for r in runs if r["label"] == "baseline"]
    techs = sorted({r["label"] for r in runs if r["label"] != "baseline"})
    lines = ["## Centinela: comparacion de tecnicas (una a una)", ""]
    if a.plan_json:
        planned = {(c["label"], c["seed"]) for c in json.loads(a.plan_json)["include"]}
        done = {(r["label"], r["seed"]) for r in runs}
        miss = sorted(planned - done)
        if miss:
            lines += [f"**Runs sin resultado ({len(miss)}):** " + ", ".join(f"{l}/s{s}" for l, s in miss), ""]
    if len(base) < 2:
        lines.append(f"Solo {len(base)} runs de baseline: sin ruido medible, no hay umbrales de significacion.")
    n_b = len(base)
    mean_b = {r: float(np.nanmean([x["resp"][r] for x in base])) if base else float("nan") for r in RESP}
    sd_b = {r: float(np.nanstd([x["resp"][r] for x in base], ddof=1)) if n_b > 1 else float("nan") for r in RESP}
    lines += ["", f"### Baseline ({n_b} seeds): media +- desviacion entre seeds", "",
              "| " + " | ".join(TABLE_COLS) + " |", "|" + "---|" * len(TABLE_COLS),
              "| " + " | ".join(f"{fmt(mean_b[c])} +- {fmt(sd_b[c])}" for c in TABLE_COLS) + " |", "",
              "azar: TYPE 0.10, MRR 0.29, DET 0.50. TYPE/MRR/DET = media de iid, ood_near y ood_hard (plantilla vista); "
              "TYPE_SHIFT = con plantillas no vistas; EASY = set sencillo; FIT = ajuste a sus propios datos de entrenamiento.", ""]

    def thresholds(c, m):
        if n_b < 3 or math.isnan(sd_b[c]):
            return None
        se = sd_b[c] * math.sqrt(1.0 / m + 1.0 / n_b)
        return t_ppf(0.975, n_b - 1) * se, t_ppf(1 - 0.05 / (2 * max(1, len(techs))), n_b - 1) * se

    rows = []
    for t in techs:
        rs = [r for r in runs if r["label"] == t]
        row = {"label": t, "n": len(rs), "steps": int(np.mean([r["cfg"]["total_steps"] for r in rs])),
               "min": float(np.mean([r["timings"].get("train_s", 0) for r in rs])) / 60,
               "early": any(r["cfg"].get("stopped_by_time_budget") for r in rs), "cells": {}}
        for c in TABLE_COLS:
            mean_t = float(np.nanmean([r["resp"][c] for r in rs]))
            d = mean_t - mean_b[c]
            th = thresholds(c, len(rs))
            star = "" if th is None or math.isnan(d) else ("**" if abs(d) > th[1] else ("*" if abs(d) > th[0] else ""))
            row["cells"][c] = (d, star)
        rows.append(row)
    rows.sort(key=lambda r: -(r["cells"]["TYPE"][0] if not math.isnan(r["cells"]["TYPE"][0]) else -9))
    lines += ["### Diferencia frente a la media del baseline", "",
              "`*` supera el 95% sin corregir; `**` supera el umbral con correccion de Bonferroni por "
              f"{len(techs)} tecnicas (por columna). En toda esta tabla ({len(techs)} tecnicas x {len(TABLE_COLS)} columnas) "
              f"se esperan ~{0.05 * len(techs) * len(TABLE_COLS):.1f} `*` y ~{0.05 * len(TABLE_COLS):.1f} `**` solo por azar: "
              "fiate de `**` y de las mejoras que aparecen en varias columnas a la vez.", "",
              "| tecnica | n | pasos | min | " + " | ".join("d" + c for c in TABLE_COLS) + " |",
              "|---|---|---|---|" + "---|" * len(TABLE_COLS)]
    for r in rows:
        desc = TECHNIQUES.get(r["label"], ("", {}))[0]
        cells = " | ".join(fmt(r["cells"][c][0], True) + r["cells"][c][1] for c in TABLE_COLS)
        lines.append(f"| {r['label']} ({desc}) | {r['n']} | {r['steps']}{' (tope de tiempo)' if r['early'] else ''} "
                     f"| {r['min']:.0f} | {cells} |")

    lines += ["", "### Sumar resultados (ensamblado de los scores guardados)", "",
              "| ensamblado | runs | " + " | ".join(TABLE_COLS) + " |", "|---|---|" + "---|" * len(TABLE_COLS)]
    for name, subset in (("baseline (todas las seeds)", base), ("TODOS los runs", runs)):
        e = ensemble_row(subset)
        if e:
            lines.append(f"| {name} | {e[0]} | " + " | ".join(f"{fmt(e[1][c])} ({fmt(e[1][c] - mean_b[c], True)})" for c in TABLE_COLS) + " |")
    lines += ["", "(entre parentesis: diferencia con la media del baseline. El ensamblado cuesta N pasadas en el movil.)"]

    gain = [(x["resp"]["TYPE_TTA"] - x["resp"]["TYPE"], x["resp"]["DET_TTA"] - x["resp"]["DET"]) for x in runs]
    gain = [g for g in gain if not any(math.isnan(v) for v in g)]
    if gain:
        g = np.array(gain)
        lines += ["", "### TTA gratuito (promedio de la plantilla vista y la no vista, dentro de cada run)", "",
                  f"efecto medio sobre TYPE: {g[:, 0].mean():+.3f} (sd {g[:, 0].std(ddof=1) if len(g) > 1 else float('nan'):.3f}), "
                  f"sobre DET: {g[:, 1].mean():+.3f} (sd {g[:, 1].std(ddof=1) if len(g) > 1 else float('nan'):.3f}), en {len(g)} runs."]

    lines += ["", "### Como leer esto", "",
              "- El ruido sale solo de las seeds del baseline; cada tecnica tiene pocas seeds, asi que una ganancia "
              "aislada es una hipotesis. Repite las ganadoras con seeds nuevas antes de combinarlas.",
              "- La mejor de 18 tecnicas esta inflada por seleccion (mismo test para elegir y medir).",
              "- Un `(tope de tiempo)` significa que el entrenamiento se detuvo antes de los pasos previstos.",
              "- Las ganancias no se suman: combina las ganadoras en un run nuevo y verificalo."]
    md = "\n".join(lines)
    print(md)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(md + "\n")
    sp = os.environ.get("GITHUB_STEP_SUMMARY")
    if sp:
        open(sp, "a", encoding="utf-8").write(md + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--baseline_seeds", default="42,7,123,11,99")
    p.add_argument("--technique_seeds", default="42")
    p.add_argument("--techniques", default="all")
    p.add_argument("--max_steps", type=int, default=150)
    p.add_argument("--time_budget", type=float, default=42)
    p.set_defaults(fn=cmd_plan)
    r = sub.add_parser("report")
    r.add_argument("--dir", required=True)
    r.add_argument("--plan-json", default="")
    r.add_argument("--out", default="")
    r.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
