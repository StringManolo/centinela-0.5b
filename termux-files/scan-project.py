#!/usr/bin/env python3
"""
scan-project.py — Escanea un proyecto y rankea por probabilidad de vulnerabilidad.

Uso:
    python3 scan-project.py /ruta/al/repo
    python3 scan-project.py /ruta/al/repo --ext .php,.py,.js
    python3 scan-project.py /ruta/al/repo --top 50 --min-score 0.7
"""
import argparse, json, math, os, sys, time, urllib.request, urllib.error
from pathlib import Path
from collections import Counter, defaultdict

SCRIPT_DIR = Path(__file__).parent


def _labels_path():
    return Path(os.environ.get("CENTINELA_LABELS_JSON") or SCRIPT_DIR / "labels.json")


def load_labels():
    p = _labels_path()
    if not p.is_file():
        sys.exit(f"ERROR: labels.json not found at {p}")
    with open(p) as f:
        return json.load(f)["labels"]


LABELS = load_labels()
LABELS_SET = set(LABELS)

SERVER_DEFAULT = os.environ.get("CENTINELA_SERVER", "http://127.0.0.1:8192")

SUPPORTED_EXT = {
    ".js", ".mjs", ".cjs", ".jsx", ".tsx", ".ts",           # training domain (best)
    ".php", ".py", ".rb", ".java", ".go", ".rs",           # web
    ".c", ".cpp", ".cc", ".h", ".hpp", ".cs",              # C/C++/C# (memory bugs)
}
SKIP_DIRS = {
    "node_modules", ".git", "dist", "build", "vendor", "venv", ".venv",
    "__pycache__", "target", ".next", "coverage", ".cache", "out", "bin",
    "obj", "packages", ".idea", ".vscode", "test", "tests", "__tests__",
    "spec", "specs", "fixtures", "mocks", "docs",
}

MAX_CHUNK_LINES = 120
OVERLAP_LINES = 10

PROMPT_TMPL = """Analyze the following JavaScript code for security vulnerabilities.
If it is vulnerable, answer with the vulnerability type; if it is not, answer SAFE.
Answer with exactly one of: {labels}.

Code:
```javascript
{code}
```"""


def should_skip(rel_path):
    return any(part.lower() in SKIP_DIRS for part in rel_path.parts)


def chunk_lines(lines):
    if len(lines) <= MAX_CHUNK_LINES:
        yield 1, lines
        return
    i = 0
    while i < len(lines):
        yield i + 1, lines[i:i + MAX_CHUNK_LINES]
        if i + MAX_CHUNK_LINES >= len(lines):
            break
        i += MAX_CHUNK_LINES - OVERLAP_LINES


def classify(server, code):
    prompt = PROMPT_TMPL.format(labels=", ".join(LABELS), code=code)
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 1,
    }).encode()
    req = urllib.request.Request(f"{server}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            resp = json.load(r)
    except Exception as e:
        return "ERROR", 0.0, str(e)[:40]

    choice = resp["choices"][0]
    raw = choice["message"]["content"].strip()
    label = "UNKNOWN"
    for tok in raw.replace(",", " ").replace(".", " ").split():
        tok = tok.strip("`'\"()[]{}:;")
        if tok in LABELS_SET:
            label = tok
            break

    conf = 0.5
    try:
        content_lp = choice["logprobs"]["content"]
        if content_lp and len(content_lp) > 0:
            conf = math.exp(float(content_lp[0]["logprob"]))
    except (KeyError, TypeError, IndexError):
        pass

    return label, conf, raw[:40]


def score_for(label, conf):
    if label in ("SAFE", "UNKNOWN", "ERROR"):
        return 0.0
    return round(conf, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="Directorio o fichero a escanear")
    ap.add_argument("--server", default=SERVER_DEFAULT)
    ap.add_argument("--out", default="scan-results.json")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--ext", default="", help="Extensiones separadas por coma (default: todas)")
    ap.add_argument("--skip-safe", action="store_true", help="No mostrar chunks SAFE en el ranking")
    args = ap.parse_args()

    target = Path(args.target)
    if not target.exists():
        print(f"ERROR: {target} no existe", file=sys.stderr)
        sys.exit(1)

    exts = {("." + e.strip(". ")) for e in args.ext.split(",")} if args.ext else SUPPORTED_EXT

    files = [target] if target.is_file() else [
        p for p in target.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
        and not should_skip(p.relative_to(target))
    ]

    if not files:
        print(f"No hay ficheros que escanear en {target} con extensiones {exts}", file=sys.stderr)
        sys.exit(1)

    print(f"Escaneando {len(files)} ficheros en {target}")
    print(f"Server: {args.server}  |  Labels: {len(LABELS)}")
    print(f"Chunk: {MAX_CHUNK_LINES} líneas, overlap {OVERLAP_LINES}")
    print()

    stats = {"total_chunks": 0, "vuln_chunks": 0,
             "labels": Counter(), "files_scanned": 0}
    all_results = []
    t0 = time.time()

    for i, f in enumerate(files, 1):
        if i % 20 == 0 or i == len(files):
            elapsed = time.time() - t0
            print(f"  [{i}/{len(files)}] {elapsed:.0f}s — {stats['vuln_chunks']} vuln chunks", flush=True)
        try:
            text = f.read_text(errors="replace")
        except Exception:
            continue
        lines = text.split("\n")
        for start_line, chunk in chunk_lines(lines):
            label, conf, raw = classify(args.server, "\n".join(chunk))
            stats["total_chunks"] += 1
            stats["labels"][label] += 1
            score = score_for(label, conf)
            if score > 0:
                stats["vuln_chunks"] += 1
            all_results.append({
                "file": str(f), "line": start_line, "lines": len(chunk),
                "label": label, "confidence": round(conf, 3),
                "score": score, "raw": raw,
            })
        stats["files_scanned"] += 1

    elapsed = time.time() - t0
    all_results.sort(key=lambda r: -r["score"])

    with open(args.out, "w") as fp:
        json.dump({"target": str(target),
                   "stats": {**{k: v for k, v in stats.items() if k != "labels"},
                             "elapsed_seconds": round(elapsed, 1),
                             "labels": dict(stats["labels"])},
                   "results": all_results}, fp, indent=2)
    print(f"\nGuardado: {args.out}")

    print(f"\n{'='*75}")
    print(f"  TOP {args.top} CHUNKS MÁS SOSPECHOSOS")
    print(f"{'='*75}")
    print(f"{'Score':6s} {'Label':24s} {'Fichero':32s} {'Línea':6s}")
    print("-" * 75)
    shown = 0
    for r in all_results:
        if r["score"] < args.min_score or r["score"] == 0:
            continue
        if args.skip_safe and r["label"] == "SAFE":
            continue
        fname = Path(r["file"]).name
        if len(fname) > 30:
            fname = "..." + fname[-27:]
        print(f"{r['score']:6.2f} {r['label']:24s} {fname:32s} L{r['line']:<5d}")
        shown += 1
        if shown >= args.top:
            break
    if shown == 0:
        print("(ninguno por encima del umbral)")

    print(f"\n{'='*75}")
    print("  DISTRIBUCIÓN DE LABELS")
    print(f"{'='*75}")
    for label, count in stats["labels"].most_common():
        pct = 100 * count / max(1, stats["total_chunks"])
        print(f"  {label:24s} {count:5d} ({pct:5.1f}%)")

    by_file = defaultdict(int)
    for r in all_results:
        if r["score"] > 0:
            by_file[r["file"]] += 1
    if by_file:
        print(f"\n{'='*75}")
        print("  FICHEROS CON MÁS CHUNKS SOSPECHOSOS")
        print(f"{'='*75}")
        for f, n in sorted(by_file.items(), key=lambda x: -x[1])[:15]:
            print(f"  {n:3d}  {f}")

    print(f"\nTotal: {elapsed:.0f}s, {stats['total_chunks']} chunks, "
          f"{stats['total_chunks']/max(elapsed,1):.1f} chunks/s")


if __name__ == "__main__":
    main()
