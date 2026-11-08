#!/usr/bin/env python3
"""
Centinela batch classifier. Uso:
    python3 classify.py fichero.js
    python3 classify.py directorio/         # .js recursivo
    cat fichero.js | python3 classify.py -  # stdin
"""
import json, sys, urllib.request, urllib.error
from pathlib import Path

SERVER = "http://127.0.0.1:8192"
LABELS = ("XSS SQLI CMDI CODE_INJECTION PATH_TRAVERSAL SSRF PROTO_POLLUTION "
          "OPEN_REDIRECT AUTHZ WEAK_CRYPTO SSTI CSRF XXE LDAP_INJECTION HPP "
          "CRLF_INJECTION CSSI HTMLI NOSQLI HARDCODED_SECRETS MASS_ASSIGNMENT "
          "WEAK_COOKIE INSECURE_JWT RACE_CONDITION INFO_DISCLOSURE SAFE").split()
LABEL_LIST = ", ".join(LABELS)

PROMPT = """Analyze the following JavaScript code for security vulnerabilities.
If it is vulnerable, answer with the vulnerability type; if it is not, answer SAFE.
Answer with exactly one of: {labels}.

Code:
```javascript
{code}
```"""

MAX_CHUNK_LINES = 150
OVERLAP_LINES = 10


def chunk(code, max_lines=MAX_CHUNK_LINES, overlap=OVERLAP_LINES):
    lines = code.split("\n")
    if len(lines) <= max_lines:
        return [code]
    out, i = [], 0
    while i < len(lines):
        out.append("\n".join(lines[i:i + max_lines]))
        if i + max_lines >= len(lines):
            break
        i += max_lines - overlap
    return out


def classify(code):
    prompt = PROMPT.format(labels=LABEL_LIST, code=code)
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4,
        "temperature": 0,
        "stop": ["\n", "<|im_end|>"],
    }).encode()
    req = urllib.request.Request(f"{SERVER}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.load(r)
    except urllib.error.URLError as e:
        return f"ERROR({e})"
    out = resp["choices"][0]["message"]["content"].strip()
    for tok in out.replace(",", " ").replace(".", " ").split():
        tok = tok.strip("`'\"()[]{}:;")
        if tok in LABELS:
            return tok
    return f"UNKNOWN({out!r})"


def process_file(path):
    try:
        code = path.read_text(errors="replace")
    except Exception as e:
        return [(path, f"READ_ERROR({e})", 0)]
    results = []
    for idx, ch in enumerate(chunk(code)):
        label = classify(ch)
        results.append((path, label, idx))
    return results


def main():
    if len(sys.argv) < 2:
        print("Uso: classify.py <fichero|directorio|->", file=sys.stderr)
        sys.exit(1)
    target = sys.argv[1]

    if target == "-":
        code = sys.stdin.read()
        for idx, ch in enumerate(chunk(code)):
            print(f"{classify(ch):20s}  stdin[chunk {idx}]")
        return

    p = Path(target)
    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = sorted(p.rglob("*.js"))
    else:
        print(f"no existe: {target}", file=sys.stderr); sys.exit(1)

    for f in files:
        for (path, label, idx) in process_file(f):
            suffix = f" [chunk {idx}]" if idx > 0 else ""
            print(f"{label:20s}  {path}{suffix}")


if __name__ == "__main__":
    main()
