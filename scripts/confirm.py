#!/usr/bin/env python3
"""
Confirmation experiment: re-run the best combinations from the factorial experiment
with NEW seeds and compare them against ia3 and baseline with proper t-tests.

plan    fixed list of configurations (e.g. "targets_all+lr3e4+rank32") x seeds,
        plus baseline seeds and reference techniques (ia3) x seeds.
report  mean +- sd per config, dSCORE vs ia3 and vs baseline, with uncorrected
        and Bonferroni-corrected significance (Welch t-test, df by Satterthwaite).

Why: in the factorial each cell is ONE run; the best of 26 cells is inflated by
selection. Re-running the winners with independent seeds removes that inflation.
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oat    # noqa: E402
import combo  # noqa: E402


# --------------------------------------------------------------------------- plan
def cmd_plan(a):
    configs = [c.strip() for c in a.configs.split(",") if c.strip()]
    refs    = [r.strip() for r in a.refs.split(",") if r.strip()]
    seeds   = sorted({int(s) for s in a.seeds.split(",") if s.strip()})
    seeds_b = sorted({int(s) for s in a.baseline_seeds.split(",") if s.strip()} | set(seeds))

    for cfg in configs:
        for n in cfg.split("+"):
            if n not in oat.TECHNIQUES:
                sys.exit(f"unknown technique '{n}' in config '{cfg}'; available: {list(oat.TECHNIQUES)}")
    for r in refs:
        if r not in oat.TECHNIQUES:
            sys.exit(f"unknown reference '{r}'")

    include = []
    # baseline: seeds_b
    for s in seeds_b:
        include.append({"label": "baseline", "slug": "baseline", "seed": s,
                        "args": combo.build_args([], a.max_steps, a.time_budget)})
    # each config x seeds
    for cfg in configs:
        slug = cfg.replace("+", "_x_")
        args = combo.build_args(cfg.split("+"), a.max_steps, a.time_budget)
        for s in seeds:
            include.append({"label": cfg, "slug": slug, "seed": s, "args": args})
    # each ref x seeds
    for r in refs:
        args = combo.build_args([r], a.max_steps, a.time_budget)
        for s in seeds:
            include.append({"label": r, "slug": r, "seed": s, "args": args})

    print(f"{len(include)} jobs: {len(seeds_b)} baseline + "
          f"{len(configs)}x{len(seeds)} configs + {len(refs)}x{len(seeds)} refs", file=sys.stderr)
    print(json.dumps({"include": include}, separators=(",", ":")))


# --------------------------------------------------------------------------- stats
def welch(xs, ys):
    """Return (mean_diff, t, df) for Welch's two-sample t-test (xs vs ys)."""
    nx, ny = len(xs), len(ys)
    if nx < 2 or ny < 2:
        return float("nan"), float("nan"), float("nan")
    mx, my = float(np.mean(xs)), float(np.mean(ys))
    vx, vy = float(np.var(xs, ddof=1)), float(np.var(ys, ddof=1))
    se = math.sqrt(vx / nx + vy / ny)
    if se == 0:
        return float("nan"), float("nan"), float("nan")
    t = (mx - my) / se
    df = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    return mx - my, float(t), float(df)


def stars(t, df, n_tests):
    if math.isnan(t) or df < 1:
        return ""
    tc  = oat.t_ppf(0.975, df)
    tcb = oat.t_ppf(1 - 0.05 / (2 * max(1, n_tests)), df)
    return "**" if abs(t) > tcb else ("*" if abs(t) > tc else "")


# --------------------------------------------------------------------------- report
def cmd_report(a):
    refs = [r.strip() for r in a.refs.split(",") if r.strip()]
    ref = refs[0] if refs else None

    runs = combo.load(a.dir)              # label -> unslugged, with SCORE added
    by = defaultdict(list)
    for r in runs:
        by[r["label"]].append(r["resp"])

    lines = ["## Centinela: confirmación con semillas nuevas", ""]

    if a.plan_json:
        planned = {(c["label"], c["seed"]) for c in json.loads(a.plan_json)["include"]}
        done    = {(r["label"], r["seed"]) for r in runs}
        miss = sorted(planned - done)
        if miss:
            lines += [f"**Runs sin resultado ({len(miss)}):** " + ", ".join(f"{l}/s{s}" for l, s in miss), ""]

    base = by.get("baseline", [])
    if not base:
        lines.append("No hay baseline: no se puede analizar.")
        combo._emit(lines, a)
        return
    bs = [b["SCORE"] for b in base if not math.isnan(b["SCORE"])]
    mean_b = float(np.mean(bs)) if bs else float("nan")
    sd_b   = float(np.std(bs, ddof=1)) if len(bs) > 1 else float("nan")

    # configs = labels that are neither baseline nor refs; sort by mean SCORE desc
    def mean_of(lab):
        xs = [r["SCORE"] for r in by[lab] if not math.isnan(r["SCORE"])]
        return float(np.mean(xs)) if xs else float("nan")

    configs = sorted([l for l in by if l not in refs and l != "baseline"],
                     key=lambda l: -(mean_of(l) if not math.isnan(mean_of(l)) else -9))
    n_tests = max(1, len(configs))

    ref_xs = [r["SCORE"] for r in by.get(ref, []) if not math.isnan(r["SCORE"])] if ref else []

    lines += [f"Baseline: {len(bs)} semillas, media {mean_b:.3f} ± {sd_b:.3f} (sd entre semillas).", "",
              f"Contraste frente a **{ref or '—'}** con t de Welch (df Satterthwaite). "
              f"`*` p<0.05 sin corregir, `**` p<0.05 con Bonferroni por {n_tests} configuraciones.", "",
              "| configuración | n | SCORE | sd | dSCORE vs baseline | dSCORE vs " + (ref or "—") + " |",
              "|---|---|---|---|---|---|"]

    order = configs + [r for r in refs if r in by]
    for lab in order:
        xs = [r["SCORE"] for r in by[lab] if not math.isnan(r["SCORE"])]
        if not xs:
            continue
        m = float(np.mean(xs))
        s = float(np.std(xs, ddof=1)) if len(xs) > 1 else float("nan")
        db = m - mean_b
        if ref and ref in by and lab != ref:
            d, t, df = welch(xs, ref_xs)
            ds = f"{d:+.3f}{stars(t, df, n_tests)}" if not math.isnan(t) else "—"
        else:
            ds = "—"
        sd_txt = f"{s:.3f}" if not math.isnan(s) else "—"
        lines.append(f"| {lab} | {len(xs)} | {m:.3f} | {sd_txt} | {db:+.3f} | {ds} |")

    # verdict
    lines += ["", "### Veredicto", ""]
    if ref and ref in by and len(ref_xs) >= 2:
        confirmed, not_confirmed = [], []
        for lab in configs:
            xs = [r["SCORE"] for r in by[lab] if not math.isnan(r["SCORE"])]
            if len(xs) < 2:
                continue
            d, t, df = welch(xs, ref_xs)
            if math.isnan(t):
                continue
            s = stars(t, df, n_tests)
            (confirmed if (d > 0 and s) else not_confirmed).append((lab, d, s))
        if confirmed:
            lines.append(f"**Confirmadas frente a {ref}** (Bonferroni, {n_tests} comparaciones): "
                         + ", ".join(f"**{l}** ({d:+.3f}{s})" for l, d, s in
                                     sorted(confirmed, key=lambda x: -x[1])))
        else:
            lines.append(f"**Ninguna** configuración supera a **{ref}** de forma fiable con estas semillas. "
                         "Las ganancias del factorial no se confirman: eran, al menos en parte, selección.")
        if not_confirmed:
            lines.append("")
            lines.append("No confirmadas: " + ", ".join(f"{l} ({d:+.3f})" for l, d, _ in not_confirmed) + ".")
    else:
        lines.append(f"Faltan seeds de la referencia **{ref}** (o solo hay 1): no se puede contrastar.")

    lines += ["", "### Notas", "",
              "- Semillas **nuevas** (no usadas en el factorial): esta ronda no está inflada por selección.",
              "- El baseline de esta ronda fija el ruido de referencia; si su sd difiere mucho del factorial "
              "(0.015), usa la más conservadora para juzgar diferencias sueltas.",
              "- El contraste es de medias con varianzas posiblemente distintas (Welch), no un z-test.",
              "- Para que una combinación se considere **confirmada** debe superar a ia3 con `**`."]
    combo._emit(lines, a)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "report"):
        p = sub.add_parser(name)
        p.add_argument("--pool", default="lora_plus,targets_all,lr3e4,rslora,rank32")
        p.add_argument("--refs", default="ia3")
        p.add_argument("--configs",
                       default="targets_all+lr3e4+rank32,rslora+rank32,targets_all+lr3e4+rslora")
    sub.choices["plan"].add_argument("--seeds", default="1,2,3,4,5",
                                     help="NEW seeds; NOT used before: 42,7,123,11,99,2024,2025,2026,2027")
    sub.choices["plan"].add_argument("--baseline_seeds", default="1,2,3,4,5")
    sub.choices["plan"].add_argument("--max_steps", type=int, default=150)
    sub.choices["plan"].add_argument("--time_budget", type=float, default=42)
    sub.choices["plan"].set_defaults(fn=cmd_plan)
    sub.choices["report"].add_argument("--dir", required=True)
    sub.choices["report"].add_argument("--plan-json", default="")
    sub.choices["report"].add_argument("--out", default="")
    sub.choices["report"].set_defaults(fn=cmd_report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
