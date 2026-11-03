#!/usr/bin/env python3
import json

BASE_CODE = """const express = require('express');
const { execFile, exec } = require('child_process');
const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');
const fetch = require('node-fetch');
const escapeHtml = require('escape-html');
const app = express();

app.get('/data', async (req, res) => {
  const id = req.query.id;
  const rows = await db.query('SELECT * FROM data WHERE id = ?', [id]);

  const content = rows[0].content;
  res.write(escapeHtml(content));

  const cmd = req.query.cmd;
  execFile('ls', [cmd], (err, stdout) => res.write(stdout));

  const file = req.query.file;
  const safe = path.basename(file);
  const fcontent = fs.readFileSync(path.join('/var/data', safe), 'utf8');

  const url = req.query.url;
  if (!url.startsWith('https://api.example.com/')) return res.status(400).end();
  const urlResp = await fetch(url);

  const input = req.body;
  const safe_keys = ['theme', 'lang'];
  const merged = Object.fromEntries(safe_keys.map(k => [k, input[k]]));

  const userId = req.query.userId;
  const profile = await User.findOne({ _id: userId, ownerId: req.user.id });

  if (req.method === 'POST' && req.body.csrf === req.session.csrf) {
    await db.query('UPDATE users SET last = ? WHERE id = ?', [req.body.act, req.user.id]);
  }

  const yamlInput = req.body.yaml;
  const parsed = yaml.load(yamlInput, { schema: yaml.FAILSAFE_SCHEMA });

  const target = req.query.target;
  if (target.startsWith('/')) res.redirect(target);

  res.end();
});"""

REPLACEMENTS = {
    "SQLI": (
        "const rows = await db.query('SELECT * FROM data WHERE id = ?', [id]);",
        "const rows = await db.query('SELECT * FROM data WHERE id = ' + id);",
    ),
    "XSS": (
        "res.write(escapeHtml(content));",
        "res.write(content);",
    ),
    "RCE": (
        "execFile('ls', [cmd], (err, stdout) => res.write(stdout));",
        "exec(cmd, (err, stdout) => res.write(stdout));",
    ),
    "PATHTRAVERSAL": (
        "const fcontent = fs.readFileSync(path.join('/var/data', safe), 'utf8');",
        "const fcontent = fs.readFileSync('/var/data/' + file, 'utf8');",
    ),
    "SSRF": (
        "if (!url.startsWith('https://api.example.com/')) return res.status(400).end();\n  const urlResp = await fetch(url);",
        "const urlResp = await fetch(url);",
    ),
    "PROTOTYPEPOLLUTION": (
        "const merged = Object.fromEntries(safe_keys.map(k => [k, input[k]]));",
        "const merged = Object.assign({}, input);",
    ),
    "IDOR": (
        "const profile = await User.findOne({ _id: userId, ownerId: req.user.id });",
        "const profile = await User.findOne({ _id: userId });",
    ),
    "CSRF": (
        "if (req.method === 'POST' && req.body.csrf === req.session.csrf) {\n    await db.query('UPDATE users SET last = ? WHERE id = ?', [req.body.act, req.user.id]);\n  }",
        "if (req.method === 'GET') {\n    await db.query('UPDATE users SET last = ? WHERE id = ?', [req.query.act, req.user.id]);\n  }",
    ),
    "DESERIALIZATION": (
        "const parsed = yaml.load(yamlInput, { schema: yaml.FAILSAFE_SCHEMA });",
        "const parsed = yaml.load(yamlInput);",
    ),
    "OPENREDIRECT": (
        "if (target.startsWith('/')) res.redirect(target);",
        "res.redirect(target);",
    ),
}

MC_PROMPT = """Classify the vulnerability type in the following JavaScript code. Respond with exactly one of: SQLI, XSS, RCE, PATHTRAVERSAL, SSRF, PROTOTYPEPOLLUTION, IDOR, CSRF, DESERIALIZATION, OPENREDIRECT.

```js
{code}
```"""

OPEN_PROMPT = """What type of vulnerability is in the following JavaScript code? Respond briefly with the vulnerability name.

```js
{code}
```"""

def main():
    for cat, (safe, vuln) in REPLACEMENTS.items():
        if safe not in BASE_CODE:
            raise SystemExit(f"safe line not found for {cat}")

    with open("data/test_vuln_types_mc.jsonl", "w") as fmc, \
         open("data/test_vuln_types_open.jsonl", "w") as fopen:
        for cat, (safe, vuln) in REPLACEMENTS.items():
            code = BASE_CODE.replace(safe, vuln)
            assert code != BASE_CODE, f"replacement failed for {cat}"
            fmc.write(json.dumps({"prompt": MC_PROMPT.format(code=code), "completion": cat}) + "\n")
            fopen.write(json.dumps({"prompt": OPEN_PROMPT.format(code=code), "completion": cat}) + "\n")
    print(f"Wrote 10 MC + 10 open entries")
    print(f"Classes: {list(REPLACEMENTS.keys())}")

if __name__ == "__main__":
    main()
