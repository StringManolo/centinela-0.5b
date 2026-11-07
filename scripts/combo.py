#!/usr/bin/env python3
"""
Combination experiment for the GitHub workflow centinela-combo.yml.

plan    every SUBSET of a pool of k techniques (2^k cells, run in parallel) + extra baseline seeds + optional
        single "reference" techniques that cannot be combined (e.g. ia3, a different adapter family).
report  compares every combination with the techniques on their own and with the best single technique, and
        analyses the whole design as a FULL FACTORIAL: main effect of each technique and its interactions
        (negative interaction = the gains overlap; positive = they reinforce each other), using all runs
        for each estimate, which is far more powerful than comparing runs one by one.

Why a factorial and not just "compare each combination": with one run per cell, comparing two cells is noisy,
but the effect of a technique averaged over the other 2^(k-1) settings uses every run.
"""
import argparse
import itertools
import json
import math
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import oat  # noqa: E402

# Seed-to-seed spread of the baseline measured in the previous 23-run comparison (5 seeds, df=4);
# pooled with the spread of the baseline seeds of this run.
HIST_SD = {"TYPE": 0.031, "TYPE_SHIFT": 0.023, "MRR": 0.027, "DET": 0.013, "EASY_TYPE": 0.055, "EASY_DET": 0.021, "FIT": 0.055}
HIST_DF = 4
EFFECT_RESP = ["TYPE", "TYPE_SHIFT", "MRR", "DET", "SCORE"]
COLS = ["TYPE", "TYPE_SHIFT", "MRR", "DET", "SCORE"]


def slug(label):
    return label.replace("+", "_x_")


def unslug(s):
    return s.replace("_x_", "+")


def merged_overrides(names):
    out = {}
    for n in names:
        for k, v in oat.TECHNIQUES[n][1].items():
            if k in out and out[k] != v:
                raise SystemExit(f"conflict: '{n}' sets {k}={v} but another technique already set {k}={out[k]}")
            out[k] = v
    return out


def build_args(names, max_steps, time_budget):
    flags = dict(oat.BASELINE)
    flags["--max_steps"] = str(max_steps)
    flags["--time_budget_min"] = str(time_budget)
    for k, v in merged_overrides(names).items():
        flags[k] = str(max_steps * 4) if (k == "--max_steps" and v == "x4") else v
    return " ".join(k if v is None else f"{k} {v}" for k, v in flags.items())


def label_of(names):
    return "baseline" if not names else "+".join(names)


# --------------------------------------------------------------------------- plan
def cmd_plan(a):
    pool = [t.strip() for t in a.pool.split(",") if t.strip()]
    refs = [t.strip() for t in a.refs.split(",") if t.strip()]
    for t in pool + refs:
        if t not in oat.TECHNIQUES:
            sys.exit(f"unknown technique {t}; available: {list(oat.TECHNIQUES)}")
    for t in pool:
        if "--peft" in oat.TECHNIQUES[t][1]:
            sys.exit(f"'{t}' replaces the adapter family and cannot be combined with LoRA techniques; put it in --refs")
    seeds_b = sorted({int(s) for s in a.baseline_seeds.split(",") if s.strip()} | {a.seed})
    include = []
    for r in range(len(pool) + 1):
        for subset in itertools.combinations(pool, r):
            args = build_args(subset, a.max_steps, a.time_budget)        # raises on any key conflict
            label = label_of(subset)
            for s in (seeds_b if r == 0 else [a.seed]):
                include.append({"label": label, "slug": slug(label), "seed": s, "args": args})
    for t in refs:
        include.append({"label": t, "slug": slug(t), "seed": a.seed, "args": build_args([t], a.max_steps, a.time_budget)})
    n_cells = 2 ** len(pool)
    print(f"{len(include)} jobs: {n_cells} cells of the 2^{len(pool)} design + {len(seeds_b) - 1} extra baseline seeds "
          f"+ {len(refs)} reference(s)", file=sys.stderr)
    print(json.dumps({"include": include}, separators=(",", ":")))


# --------------------------------------------------------------------------- report helpers
def load(dir_):
    runs = oat.load_runs(dir_)
    for r in runs:
        r["label"] = unslug(r["label"])
        resp = r["resp"]
        resp["SCORE"] = float(np.mean([resp["TYPE"], resp["TYPE_SHIFT"], resp["DET"]]))
    return runs


def noise(resp, base_runs):
    vals = [r["resp"][resp] for r in base_runs if not math.isnan(r["resp"][resp])]
    n = len(vals)
    var_new, df_new = (float(np.var(vals, ddof=1)), n - 1) if n >= 2 else (0.0, 0)
    if resp in HIST_SD:
        var = (df_new * var_new + HIST_DF * HIST_SD[resp] ** 2) / (df_new + HIST_DF)
        return math.sqrt(var), df_new + HIST_DF
    if df_new == 0:
        return float("nan"), 0
    return math.sqrt(var_new), df_new


def lenth(effects):
    e = np.abs(np.asarray(effects, dtype=float))
    s0 = 1.5 * np.median(e)
    pse = 1.5 * np.median(e[e < 2.5 * s0]) if (e < 2.5 * s0).any() else s0
    d = max(1, int(round(len(e) / 3)))
    return float(pse), oat.t_ppf(0.975, d) * float(pse)


def signs(mask, subset_mask, k):
    s = 1
    for i in range(k):
        if subset_mask >> i & 1:
            s *= 1 if (mask >> i & 1) else -1
    return s


def effect_name(subset_mask, pool):
    return " x ".join(pool[i] for i in range(len(pool)) if subset_mask >> i & 1)


def flag(d, thr95, thrb):
    if thr95 is None or math.isnan(d):
        return ""
    return "**" if abs(d) > thrb else ("*" if abs(d) > thr95 else "")


def cmd_report(a):
    pool = [t.strip() for t in a.pool.split(",") if t.strip()]
    refs = [t.strip() for t in a.refs.split(",") if t.strip()]
    k = len(pool)
    runs = load(a.dir)
    base_runs = [r for r in runs if r["label"] == "baseline"]
    cells = {r["label"]: r for r in runs if r["seed"] == a.seed}
    lines = ["## Centinela: combinaciones de las mejores tecnicas", ""]
    if a.plan_json:
        planned = {(c["label"], c["seed"]) for c in json.loads(a.plan_json)["include"]}
        done = {(r["label"], r["seed"]) for r in runs}
        miss = sorted(planned - done)
        if miss:
            lines += [f"**Runs sin resultado ({len(miss)}):** " + ", ".join(f"{l}/s{s}" for l, s in miss), ""]
    if "baseline" not in cells and base_runs:
        cells["baseline"] = {"resp": {c: float(np.nanmean([b["resp"][c] for b in base_runs])) for c in base_runs[0]["resp"]},
                             "cfg": {}, "label": "baseline"}
    if "baseline" not in cells:
        lines.append("No hay ningun run de baseline: no se puede analizar.")
        _emit(lines, a)
        return
    sd = {c: noise(c, base_runs) for c in COLS}
    mean_b = {c: float(np.nanmean([b["resp"][c] for b in base_runs])) for c in COLS}
    sd_txt = ", ".join(f"{c} {sd[c][0]:.3f}" for c in COLS if not math.isnan(sd[c][0]))
    lines += [f"Baseline: {len(base_runs)} semillas en este experimento. Ruido entre semillas (desviacion): {sd_txt}.", "",
              "SCORE = media de TYPE, TYPE_SHIFT y DET (acierto de tipo, robustez ante frases nuevas, distinguir vulnerable de seguro). "
              "Una comparacion entre dos ejecuciones sueltas necesita una diferencia mayor que la del ruido x raiz de 2.", ""]

    singles = [t for t in pool + refs if t in cells]
    combos = [label_of(s) for r in range(2, k + 1) for s in itertools.combinations(pool, r) if label_of(s) in cells]
    if not singles:
        lines.append("No hay ninguna tecnica individual con resultado.")
        _emit(lines, a)
        return
    best_single = max(singles, key=lambda t: cells[t]["resp"]["SCORE"])
    bs = cells[best_single]["resp"]

    def thr(c, n_tests):
        s, df = sd[c]
        if math.isnan(s) or df < 2:
            return None, None
        return oat.t_ppf(0.975, df) * s * math.sqrt(2), oat.t_ppf(1 - 0.05 / (2 * max(1, n_tests)), df) * s * math.sqrt(2)

    rows = []
    for lab in singles + combos:
        r = cells[lab]
        notes = []
        if r["cfg"].get("diverged"):
            notes.append("divergio")
        if r["cfg"].get("stopped_by_time_budget"):
            notes.append("tope de tiempo")
        rows.append((lab, r["resp"], ", ".join(notes)))
    rows.sort(key=lambda x: -(x[1]["SCORE"] if not math.isnan(x[1]["SCORE"]) else -9))

    lines += [f"### Todas las configuraciones (mejor tecnica individual: **{best_single}**, SCORE {bs['SCORE']:.3f})", "",
              "`*` / `**`: la diferencia con la mejor individual supera el ruido (sin corregir / con Bonferroni por "
              f"{max(1, len(combos))} combinaciones).", "",
              "| configuracion | TYPE | TYPE_SHIFT | MRR | DET | SCORE | dSCORE vs mejor individual | notas |",
              "|---|---|---|---|---|---|---|---|"]
    lines.append(f"| baseline (media de {len(base_runs)}) | {mean_b['TYPE']:.3f} | {mean_b['TYPE_SHIFT']:.3f} | {mean_b['MRR']:.3f} "
                 f"| {mean_b['DET']:.3f} | {mean_b['SCORE']:.3f} | | |")
    for lab, resp, notes in rows:
        t95, tb = thr("SCORE", len(combos))
        d = resp["SCORE"] - bs["SCORE"]
        kind = "individual" if lab in singles else "combinacion"
        dtxt = "(referencia)" if lab == best_single else f"{d:+.3f}{flag(d, t95, tb)}"
        lines.append(f"| {lab} ({kind}) | {resp['TYPE']:.3f} | {resp['TYPE_SHIFT']:.3f} | {resp['MRR']:.3f} | {resp['DET']:.3f} "
                     f"| {resp['SCORE']:.3f} | {dtxt} | {notes} |")

    winners = []
    for lab in combos:
        r = cells[lab]["resp"]
        ds = {c: r[c] - bs[c] for c in ("TYPE", "TYPE_SHIFT", "DET", "SCORE")}
        flags_ = {c: flag(ds[c], *thr(c, len(combos))) for c in ds}
        if flags_["SCORE"] and ds["SCORE"] > 0:
            winners.append((lab, ds, flags_))
    lines += ["", "### ¿Alguna combinacion supera a la mejor tecnica individual?", ""]
    if winners:
        lines += ["| combinacion | dTYPE | dTYPE_SHIFT | dDET | dSCORE |", "|---|---|---|---|---|"]
        for lab, ds, fl in sorted(winners, key=lambda w: -w[1]["SCORE"]):
            lines.append(f"| {lab} | " + " | ".join(f"{ds[c]:+.3f}{fl[c]}" for c in ("TYPE", "TYPE_SHIFT", "DET", "SCORE")) + " |")
        lines += ["", f"La mejor de {len(combos)} combinaciones esta inflada por seleccion: confirma la ganadora con semillas nuevas."]
    else:
        lines.append(f"**Ninguna** de las {len(combos)} combinaciones supera de forma fiable a **{best_single}** sola. "
                     "Combinar no ha mejorado lo que ya daba la mejor tecnica individual.")

    # ----------------------------------------------------------------- full factorial
    lines += ["", "### Analisis factorial: efecto de cada tecnica y sus interacciones", ""]
    masks = list(range(2 ** k))
    names_of = {m: label_of([pool[i] for i in range(k) if m >> i & 1]) for m in masks}
    missing = [names_of[m] for m in masks if names_of[m] not in cells]
    if missing:
        lines.append(f"No se puede hacer el analisis factorial completo: faltan {len(missing)} celdas ({', '.join(missing[:6])}...).")
        _emit(lines, a)
        return
    Y = {c: np.array([cells[names_of[m]]["resp"][c] for m in masks], dtype=float) for c in EFFECT_RESP}
    subsets = [s for s in range(1, 2 ** k)]
    eff, me, active = {}, {}, {}
    for c in EFFECT_RESP:
        e = np.array([sum(signs(m, s, k) * Y[c][m] for m in masks) / (2 ** (k - 1)) for s in subsets])
        eff[c] = dict(zip(subsets, e))
        pse, me[c] = lenth(e)
        active[c] = [s for s in subsets if abs(eff[c][s]) > me[c]]
    lines += ["**Efecto principal** = cuanto cambia la medida, en promedio sobre todas las combinaciones de las demas tecnicas, "
              "al activar esa tecnica. `*` = supera el margen de error (Lenth).", "",
              "| tecnica | " + " | ".join(f"efecto en {c}" for c in EFFECT_RESP) + " |", "|---|" + "---|" * len(EFFECT_RESP)]
    for i, t in enumerate(pool):
        s = 1 << i
        lines.append(f"| {t} | " + " | ".join(f"{eff[c][s]:+.3f}{'*' if s in active[c] else ''}" for c in EFFECT_RESP) + " |")
    lines.append("| margen de error | " + " | ".join(f"{me[c]:.3f}" for c in EFFECT_RESP) + " |")

    inter = [s for s in subsets if bin(s).count("1") >= 2 and (s in active["SCORE"] or s in active["TYPE"])]
    inter.sort(key=lambda s: -abs(eff["SCORE"][s]))
    lines += ["", "**Interacciones activas** (combinaciones de tecnicas cuyo efecto conjunto no es la suma de los individuales):", ""]
    if inter:
        lines += ["| interaccion | efecto en SCORE | efecto en TYPE | lectura |", "|---|---|---|---|"]
        for s in inter[:12]:
            es, et = eff["SCORE"][s], eff["TYPE"][s]
            reading = ("se solapan: juntas ayudan menos que la suma de lo que ayudan por separado"
                       if es < 0 else "se refuerzan: juntas ayudan mas que la suma de las partes")
            lines.append(f"| {effect_name(s, pool)} | {es:+.3f}{'*' if s in active['SCORE'] else ''} | "
                         f"{et:+.3f}{'*' if s in active['TYPE'] else ''} | {reading} |")
    else:
        lines.append("Ninguna: dentro del ruido, los efectos de las tecnicas se suman sin interferir.")

    gm = float(Y["SCORE"].mean())
    pred = {m: gm + sum(eff["SCORE"][s] / 2 * signs(m, s, k) for s in active["SCORE"]) for m in masks}
    top_pred = sorted(masks, key=lambda m: -pred[m])[:5]
    top_obs = sorted(masks, key=lambda m: -Y["SCORE"][m])[:5]
    lines += ["", "**Mejores configuraciones** segun el SCORE observado y segun el modelo que solo conserva los efectos activos "
              "(suaviza el ruido de cada celda):", "",
              "| puesto | observado | SCORE | predicho por el modelo | SCORE predicho |", "|---|---|---|---|---|"]
    for i in range(5):
        lines.append(f"| {i + 1} | {names_of[top_obs[i]]} | {Y['SCORE'][top_obs[i]]:.3f} | {names_of[top_pred[i]]} | {pred[top_pred[i]]:.3f} |")
    lines += ["", "### Como leer esto", "",
              "- Cada celda del diseno es una sola ejecucion; los efectos factoriales usan las 2^k ejecuciones, por eso son mas fiables que cada celda.",
              "- La mejor configuracion observada esta inflada por seleccion; verificala con semillas nuevas antes de darla por buena.",
              "- Una combinacion que `divergio` combina ritmos de aprendizaje demasiado altos: no es un fallo del experimento sino un resultado.",
              "- IA3 y las demas tecnicas de referencia no se combinan (cambian el tipo de adaptador); se miden solo por separado."]
    _emit(lines, a)


def _emit(lines, a):
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
    for name in ("plan", "report"):
        p = sub.add_parser(name)
        p.add_argument("--pool", default="lora_plus,targets_all,lr3e4,rslora,rank32")
        p.add_argument("--refs", default="ia3")
        p.add_argument("--seed", type=int, default=2024)
    sub.choices["plan"].add_argument("--baseline_seeds", default="2024,2025,2026,2027")
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
