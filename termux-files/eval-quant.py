#!/usr/bin/env python3
"""
Evalúa un llama-server corriendo (Centinela) sobre un set fijo.
Guarda resultados en eval-{tag}.json para comparar.
"""
import argparse, json, os, sys, time, urllib.request, urllib.error
from pathlib import Path

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

PROMPT = """Analyze the following JavaScript code for security vulnerabilities.
If it is vulnerable, answer with the vulnerability type; if it is not, answer SAFE.
Answer with exactly one of: {labels}.

Code:
```javascript
{code}
```"""

HAND_CRAFTED = [
    {"id": "SQLI_concat", "expected": "SQLI", "code": 'app.get("/user", (req, res) => {\n  const sql = "SELECT * FROM users WHERE id = " + req.query.id;\n  db.query(sql, (err, rows) => res.json(rows));\n});'},
    {"id": "SQLI_template", "expected": "SQLI", "code": 'app.post("/login", async (req, res) => {\n  const { email } = req.body;\n  const rows = await db.query(`SELECT * FROM users WHERE email = \'${email}\'`);\n  res.json(rows);\n});'},
    {"id": "XSS_innerHTML", "expected": "XSS", "code": 'function showName(name) {\n  document.getElementById("welcome").innerHTML = "Hello, " + name;\n}'},
    {"id": "SAFE_trivial", "expected": "SAFE", "code": 'function add(a, b) {\n  return a + b;\n}'},
]

DEFAULT_SAMPLES = [
    {"id": "XSS_01", "label": "XSS", "code": '// returns JSON\napp.get("/search", (req, res) => {\n  res.send("<h2>Results for " + req.query.q + "</h2>");\n});\n// covered by the integration tests'},
    {"id": "SQLI_01", "label": "SQLI", "code": '// reviewed in the last sprint\napp.get("/users", (req, res) => {\n  const q = "SELECT id, email FROM users WHERE name = \'" + req.query.name + "\'";\n  connection.query(q, (err, rows) => res.json(rows));\n});\n// see ticket #4821 for context'},
    {"id": "CMDI_01", "label": "CMDI", "code": '// owner: platform team\nconst { exec } = require("child_process");\napp.get("/ping", (req, res) => {\n  exec("ping -c 1 " + req.query.host, (err, stdout) => res.send(stdout));\n});\n// kept simple on purpose'},
    {"id": "SAFE_01", "label": "SAFE", "code": '// returns JSON\napp.get("/users", (req, res) => {\n  connection.query("SELECT id, email FROM users WHERE name = ?", [req.query.name], (err, rows) => res.json(rows));\n});\n// see ticket #4821 for context'},
    {"id": "PATH_TRAV_01", "label": "PATH_TRAVERSAL", "code": '// part of the public API\napp.get("/download", (req, res) => {\n  const file = path.join(__dirname, "uploads", req.query.name);\n  fs.readFile(file, (err, data) => err ? res.sendStatus(404) : res.send(data));\n});\n// used by the dashboard'},
    {"id": "SSRF_01", "label": "SSRF", "code": '// legacy endpoint, kept for mobile clients\napp.get("/preview", async (req, res) => {\n  const response = await axios.get(req.query.url);\n  res.send(response.data);\n});\n// metrics are collected elsewhere'},
    {"id": "WEAK_CRYPTO_01", "label": "WEAK_CRYPTO", "code": '// reviewed in the last sprint\nconst crypto = require("crypto");\nfunction hashPassword(password) {\n  return crypto.createHash("md5").update(password).digest("hex");\n}\n// TODO: add pagination later'},
    {"id": "PROTO_POLL_01", "label": "PROTO_POLLUTION", "code": '// first version, may change\nfunction merge(target, source) {\n  for (const key in source) {\n    if (typeof source[key] === "object" && source[key] !== null) {\n      if (!target[key]) target[key] = {};\n      merge(target[key], source[key]);\n    } else {\n      target[key] = source[key];\n    }\n  }\n  return target;\n}\n// refactor when we drop v1'},
    {"id": "OPEN_REDIR_01", "label": "OPEN_REDIRECT", "code": '// returns JSON\napp.get("/login/callback", (req, res) => {\n  req.session.user = authenticate(req);\n  res.redirect(req.query.next || "/");\n});\n// refactor when we drop v1'},
    {"id": "AUTHZ_01", "label": "AUTHZ", "code": '// TODO: add pagination later\napp.get("/api/orders/:id", requireLogin, async (req, res) => {\n  const order = await Order.findById(req.params.id);\n  if (!order) return res.sendStatus(404);\n  res.json(order);\n});\n// returns JSON'},
    {"id": "CODE_INJ_01", "label": "CODE_INJECTION", "code": '// also used by the dashboard\napp.post("/calc", (req, res) => {\n  const result = eval(req.body.expression);\n  res.json({ result });\n});\n// feature flag lives in the config'},
    {"id": "SAFE_02", "label": "SAFE", "code": '// documented in the wiki\napp.post("/calc", (req, res) => {\n  const m = /^\\s*(-?\\d+(?:\\.\\d+)?)\\s*([+\\-*/])\\s*(-?\\d+(?:\\.\\d+)?)\\s*$/.exec(String(req.body.expression));\n  if (!m) return res.status(400).json({ error: "invalid" });\n  res.json({ result: Number(m[1]) + Number(m[3]) });\n});\n// feature flag lives in the config'},
]


def classify(url, code):
    prompt = PROMPT.format(labels=", ".join(LABELS), code=code)
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8, "temperature": 0,
    }).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            resp = json.load(r)
    except Exception as e:
        return f"ERROR({e})"
    out = resp["choices"][0]["message"]["content"].strip()
    for tok in out.replace(",", " ").replace(".", " ").split():
        tok = tok.strip("`'\"()[]{}:;")
        if tok in LABELS_SET:
            return tok
    return f"UNKNOWN({out!r})"


def load_samples(path):
    if not path:
        return DEFAULT_SAMPLES
    try:
        with open(path) as f:
            return [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        return DEFAULT_SAMPLES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("CENTINELA_SERVER", "http://127.0.0.1:8192"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--samples", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = args.out or f"eval-{args.tag}.json"
    samples = load_samples(args.samples)
    results = {"tag": args.tag, "url": args.url, "labels_count": len(LABELS),
               "hand_crafted": [], "test_samples": [], "n_samples": len(samples)}

    def run_case(tc, expected_key="expected"):
        t0 = time.time()
        pred = classify(args.url, tc["code"])
        dt = time.time() - t0
        return {"id": tc["id"], "expected": tc[expected_key], "predicted": pred,
                "ok": pred == tc[expected_key], "seconds": round(dt, 2)}

    print(f"=== {args.tag} — hand-crafted ({len(HAND_CRAFTED)} tests) ===", flush=True)
    print("(el primer request tarda ~10-15s en CPU móvil, paciencia)", flush=True)
    for tc in HAND_CRAFTED:
        r = run_case(tc)
        results["hand_crafted"].append(r)
        mark = "✅" if r["ok"] else "❌"
        print(f"{mark} {r['id']:18s} exp={r['expected']:22s} got={r['predicted']:22s} ({r['seconds']:.1f}s)", flush=True)

    print(f"\n=== {args.tag} — test samples ({len(samples)} tests) ===", flush=True)
    per_class_ok, per_class_total = {}, {}
    for s in samples:
        r = run_case(s, "label")
        results["test_samples"].append(r)
        per_class_total[r["expected"]] = per_class_total.get(r["expected"], 0) + 1
        if r["ok"]:
            per_class_ok[r["expected"]] = per_class_ok.get(r["expected"], 0) + 1
        mark = "✅" if r["ok"] else "❌"
        print(f"{mark} {r['id']:22s} exp={r['expected']:22s} got={r['predicted']:22s}", flush=True)

    hc_ok = sum(1 for r in results["hand_crafted"] if r["ok"])
    hc_total = len(results["hand_crafted"])
    ts_ok = sum(1 for r in results["test_samples"] if r["ok"])
    ts_total = len(results["test_samples"])

    results["summary"] = {
        "hand_crafted_acc": round(hc_ok / hc_total, 3) if hc_total else None,
        "test_samples_acc": round(ts_ok / ts_total, 3) if ts_total else None,
        "per_class_acc": {k: round(per_class_ok.get(k, 0) / per_class_total[k], 3) for k in per_class_total},
        "total_seconds": round(sum(r["seconds"] for r in results["hand_crafted"] + results["test_samples"]), 1),
    }

    print(f"\n=== Summary {args.tag} ===")
    print(f"Hand-crafted:  {hc_ok}/{hc_total} ({results['summary']['hand_crafted_acc']*100:.1f}%)")
    print(f"Test samples:  {ts_ok}/{ts_total} ({results['summary']['test_samples_acc']*100:.1f}%)")
    print(f"Total time:    {results['summary']['total_seconds']:.0f}s")

    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
