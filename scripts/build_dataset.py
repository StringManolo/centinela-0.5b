#!/usr/bin/env python3
"""
Builds data/train.jsonl and data/test.jsonl for Centinela.

Design (see the chat discussion):
  * 10 vulnerability types + SAFE  -> 11 labels.
  * Every vulnerable snippet has a SAFE "twin" (same function, same surrounding
    code, the fix applied). Twins always live in the same split.
  * Per type: 5 train pairs, 2 iid-test pairs, 3 ood_near-test pairs.
      train     sinks/APIs "set A"
      iid       same sinks/APIs as train, new identifiers/contexts
      ood_near  same vulnerability pattern, DIFFERENT sinks/APIs/frameworks ("set B")
      easy      1 minimal canonical pair per type: a sanity check of general competence
      ood_hard  1 pair per type on set-B sinks where the VULNERABLE twin has a mitigation-looking but
                bypassable check: a detector of "mitigation idioms" cannot solve it
  * Some vulnerable snippets contain an *insufficient* mitigation (bypassable
    filter) and some safe ones use a dangerous sink with constant input, so
    "has validation code" / "uses innerHTML" are not reliable shortcuts.
  * Neutral comment lines are added to each twin (independently, seed chosen on cue statistics) so
    that snippet length / line count are weak cues for the label (see "shortcut control").
  * Files are ordered round-robin by type, pair by pair (vuln, safe adjacent),
    so `--train_pairs N` (first N pairs) is class-balanced when N is a
    multiple of 10.

Run:  python scripts/build_dataset.py            (writes data/*.jsonl and validates)
      python scripts/build_dataset.py --check-js (also runs `node --check` on every snippet)
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections import Counter

VULN_TYPES = [
    "XSS", "SQLI", "CMDI", "CODE_INJECTION", "PATH_TRAVERSAL",
    "SSRF", "PROTO_POLLUTION", "OPEN_REDIRECT", "AUTHZ", "WEAK_CRYPTO",
]
SAFE = "SAFE"
N_TRAIN, N_IID, N_OOD, N_HARD, N_EASY = 5, 2, 3, 1, 1

# Pairs whose VULNERABLE twin carries an insufficient (bypassable) mitigation, keyed (set, type, pair index).
# Everything in ood_hard is insufficient by construction. Used for the `subtype` field.
_INSUFF_TRAIN = {"XSS": [3], "SQLI": [2], "CMDI": [2], "CODE_INJECTION": [3], "PATH_TRAVERSAL": [2, 3],
                 "SSRF": [2], "PROTO_POLLUTION": [2], "OPEN_REDIRECT": [2, 3], "AUTHZ": [3], "WEAK_CRYPTO": [3]}
_INSUFF_IID = ["XSS", "SQLI", "CMDI", "CODE_INJECTION", "PATH_TRAVERSAL", "SSRF", "PROTO_POLLUTION",
               "OPEN_REDIRECT", "AUTHZ"]
INSUFFICIENT = {("train", t, j) for t, js in _INSUFF_TRAIN.items() for j in js} | {("iid", t, 1) for t in _INSUFF_IID}

# DATA[type] = {"train": [(vulnerable, safe), ...], "iid": [...], "ood_near": [...]}
DATA = {}


def P(s):
    return s.strip("\n")

# ---------------------------------------------------------------- XSS
DATA["XSS"] = {
    "train": [
        (P(r'''
function showWelcome() {
  const params = new URLSearchParams(window.location.search);
  const name = params.get('name');
  document.getElementById('welcome').innerHTML = 'Hello, ' + name + '!';
}
'''), P(r'''
function showWelcome() {
  const params = new URLSearchParams(window.location.search);
  const name = params.get('name');
  document.getElementById('welcome').textContent = 'Hello, ' + name + '!';
}
''')),
        (P(r'''
app.get('/search', (req, res) => {
  const q = req.query.q || '';
  res.send('<h2>Results for ' + q + '</h2><ul id="results"></ul>');
});
'''), P(r'''
const escapeHtml = require('escape-html');

app.get('/search', (req, res) => {
  const q = req.query.q || '';
  res.send('<h2>Results for ' + escapeHtml(q) + '</h2><ul id="results"></ul>');
});
''')),
        (P(r'''
async function renderComments(postId) {
  const res = await fetch('/api/posts/' + postId + '/comments');
  const comments = await res.json();
  const list = document.getElementById('comments');
  list.innerHTML = comments.map(c => `<li><b>${c.author}</b>: ${c.body}</li>`).join('');
}
'''), P(r'''
async function renderComments(postId) {
  const res = await fetch('/api/posts/' + postId + '/comments');
  const comments = await res.json();
  const list = document.getElementById('comments');
  list.replaceChildren(...comments.map(c => {
    const li = document.createElement('li');
    const b = document.createElement('b');
    b.textContent = c.author;
    li.append(b, ': ' + c.body);
    return li;
  }));
}
''')),
        (P(r'''
function renderBio(user, container) {
  const cleaned = user.bio.replace(/<script>/gi, '').replace(/<\/script>/gi, '');
  container.innerHTML = '<p class="bio">' + cleaned + '</p>';
}
'''), P(r'''
function renderBio(user, container) {
  const p = document.createElement('p');
  p.className = 'bio';
  p.textContent = user.bio;
  container.appendChild(p);
}
''')),
        (P(r'''
app.get('/profile', (req, res) => {
  const username = req.query.user;
  res.send(`<html><body><input type="text" value="${username}"><div>Profile page</div></body></html>`);
});
'''), P(r'''
function esc(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

app.get('/profile', (req, res) => {
  const username = req.query.user;
  res.send(`<html><body><input type="text" value="${esc(username)}"><div>Profile page</div></body></html>`);
});
''')),
    ],
    "iid": [
        (P(r'''
window.addEventListener('load', () => {
  const msg = decodeURIComponent(location.hash.slice(1));
  document.querySelector('#banner').innerHTML = '<span class="alert">' + msg + '</span>';
});
'''), P(r'''
window.addEventListener('load', () => {
  const msg = decodeURIComponent(location.hash.slice(1));
  const span = document.createElement('span');
  span.className = 'alert';
  span.textContent = msg;
  document.querySelector('#banner').appendChild(span);
});
''')),
        (P(r'''
app.get('/greet', (req, res) => {
  const name = String(req.query.name || '').replace(/<script>|<\/script>/gi, '');
  res.send(`<div class="greeting">Welcome back, ${name}</div>`);
});
'''), P(r'''
app.get('/greet', (req, res) => {
  const name = String(req.query.name || '').replace(/[&<>"']/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[ch]));
  res.send(`<div class="greeting">Welcome back, ${name}</div>`);
});
''')),
    ],
    "ood_near": [
        (P(r'''
const params = new URLSearchParams(location.search);
const lang = params.get('lang');
document.write('<script src="/i18n/' + lang + '.js"><\/script>');
'''), P(r'''
const params = new URLSearchParams(location.search);
const allowed = ['en', 'es', 'fr', 'de'];
const lang = allowed.includes(params.get('lang')) ? params.get('lang') : 'en';
const s = document.createElement('script');
s.src = '/i18n/' + lang + '.js';
document.head.appendChild(s);
''')),
        (P(r'''
$(function () {
  const term = new URL(window.location.href).searchParams.get('q');
  $('#panel').html('<h3>Results for ' + term + '</h3>');
});
'''), P(r'''
$(function () {
  const term = new URL(window.location.href).searchParams.get('q');
  $('#panel').empty().append($('<h3>').text('Results for ' + term));
});
''')),
        (P(r'''
socket.on('chat', (msg) => {
  const log = document.getElementById('log');
  log.insertAdjacentHTML('beforeend', '<div class="line"><b>' + msg.user + '</b> ' + msg.text + '</div>');
});
'''), P(r'''
socket.on('chat', (msg) => {
  const log = document.getElementById('log');
  const line = document.createElement('div');
  line.className = 'line';
  const b = document.createElement('b');
  b.textContent = msg.user;
  line.append(b, ' ' + msg.text);
  log.appendChild(line);
});
''')),
    ],
}

# ---------------------------------------------------------------- SQLI
DATA["SQLI"] = {
    "train": [
        (P(r'''
app.get('/users', (req, res) => {
  const sql = "SELECT id, email FROM users WHERE name = '" + req.query.name + "'";
  connection.query(sql, (err, rows) => {
    if (err) return res.status(500).end();
    res.json(rows);
  });
});
'''), P(r'''
app.get('/users', (req, res) => {
  const sql = 'SELECT id, email FROM users WHERE name = ?';
  connection.query(sql, [req.query.name], (err, rows) => {
    if (err) return res.status(500).end();
    res.json(rows);
  });
});
''')),
        (P(r'''
app.post('/login', async (req, res) => {
  const { email, password } = req.body;
  const rows = await db.query(`SELECT * FROM accounts WHERE email = '${email}' AND pw_hash = '${hash(password)}'`);
  res.json(rows.length > 0 ? rows[0] : null);
});
'''), P(r'''
app.post('/login', async (req, res) => {
  const { email, password } = req.body;
  const rows = await db.query('SELECT * FROM accounts WHERE email = ? AND pw_hash = ?', [email, hash(password)]);
  res.json(rows.length > 0 ? rows[0] : null);
});
''')),
        (P(r'''
app.get('/orders/:id', (req, res) => {
  const id = req.params.id.replace(/'/g, '');
  pool.query('SELECT * FROM orders WHERE id = ' + id, (err, rows) => res.json(rows));
});
'''), P(r'''
app.get('/orders/:id', (req, res) => {
  const id = parseInt(req.params.id, 10);
  if (Number.isNaN(id)) return res.status(400).end();
  pool.query('SELECT * FROM orders WHERE id = ?', [id], (err, rows) => res.json(rows));
});
''')),
        (P(r'''
app.get('/products', (req, res) => {
  const query = "SELECT * FROM products WHERE title LIKE '%" + req.query.term + "%' ORDER BY price";
  connection.query(query, (err, rows) => res.json(rows));
});
'''), P(r'''
app.get('/products', (req, res) => {
  const query = 'SELECT * FROM products WHERE title LIKE ? ORDER BY price';
  connection.query(query, ['%' + req.query.term + '%'], (err, rows) => res.json(rows));
});
''')),
        (P(r'''
app.get('/articles', (req, res) => {
  const sort = req.query.sort || 'created_at';
  db.query('SELECT id, title FROM articles ORDER BY ' + sort, (err, rows) => res.json(rows));
});
'''), P(r'''
app.get('/articles', (req, res) => {
  const columns = ['created_at', 'title', 'view_count'];
  const sort = columns.includes(req.query.sort) ? req.query.sort : 'created_at';
  db.query('SELECT id, title FROM articles ORDER BY ' + sort, (err, rows) => res.json(rows));
});
''')),
    ],
    "iid": [
        (P(r'''
app.post('/logout', (req, res) => {
  connection.query("DELETE FROM sessions WHERE token = '" + req.cookies.token + "'", () => res.sendStatus(204));
});
'''), P(r'''
app.post('/logout', (req, res) => {
  connection.query('DELETE FROM sessions WHERE token = ?', [req.cookies.token], () => res.sendStatus(204));
});
''')),
        (P(r'''
app.post('/comments', (req, res) => {
  const body = req.body.text.replace(/"/g, '');
  connection.query("INSERT INTO comments (post_id, text) VALUES (" + req.body.postId + ", '" + body + "')", () => res.sendStatus(201));
});
'''), P(r'''
app.post('/comments', (req, res) => {
  connection.query('INSERT INTO comments (post_id, text) VALUES (?, ?)', [req.body.postId, req.body.text], () => res.sendStatus(201));
});
''')),
    ],
    "ood_near": [
        (P(r'''
app.post('/login', async (req, res) => {
  const [results] = await sequelize.query(`SELECT * FROM "Users" WHERE email = '${req.body.email}'`);
  res.json(results[0] || null);
});
'''), P(r'''
app.post('/login', async (req, res) => {
  const [results] = await sequelize.query('SELECT * FROM "Users" WHERE email = :email', { replacements: { email: req.body.email } });
  res.json(results[0] || null);
});
''')),
        (P(r'''
router.get('/events', async (req, res) => {
  const rows = await knex.raw("SELECT * FROM events WHERE city = '" + req.query.city + "' AND start_date > now()");
  res.json(rows.rows);
});
'''), P(r'''
router.get('/events', async (req, res) => {
  const rows = await knex.raw('SELECT * FROM events WHERE city = ? AND start_date > now()', [req.query.city]);
  res.json(rows.rows);
});
''')),
        (P(r'''
const Database = require('better-sqlite3');
const db = new Database('app.db');

app.post('/notes', (req, res) => {
  db.exec(`INSERT INTO notes (user_id, text) VALUES (${req.user.id}, '${req.body.text}')`);
  res.sendStatus(201);
});
'''), P(r'''
const Database = require('better-sqlite3');
const db = new Database('app.db');

app.post('/notes', (req, res) => {
  db.prepare('INSERT INTO notes (user_id, text) VALUES (?, ?)').run(req.user.id, req.body.text);
  res.sendStatus(201);
});
''')),
    ],
}

# ---------------------------------------------------------------- CMDI
DATA["CMDI"] = {
    "train": [
        (P(r'''
const { exec } = require('child_process');

app.get('/ping', (req, res) => {
  exec('ping -c 1 ' + req.query.host, (err, stdout) => res.send(stdout));
});
'''), P(r'''
const { execFile } = require('child_process');

app.get('/ping', (req, res) => {
  const host = String(req.query.host);
  if (!/^[a-z0-9][a-z0-9.-]{0,252}$/i.test(host)) return res.status(400).send('bad host');
  execFile('ping', ['-c', '1', host], (err, stdout) => res.send(stdout));
});
''')),
        (P(r'''
const { exec } = require('child_process');

app.post('/thumbnail', (req, res) => {
  exec(`convert uploads/${req.body.filename} -resize ${req.body.size} thumbs/${req.body.filename}`, (err) => res.json({ ok: !err }));
});
'''), P(r'''
const { execFile } = require('child_process');
const path = require('path');

app.post('/thumbnail', (req, res) => {
  const { filename, size } = req.body;
  if (!/^\d{2,4}x\d{2,4}$/.test(size)) return res.status(400).json({ ok: false });
  const name = path.basename(String(filename));
  execFile('convert', [path.join('uploads', name), '-resize', size, path.join('thumbs', name)], (err) => res.json({ ok: !err }));
});
''')),
        (P(r'''
app.get('/lookup', (req, res) => {
  const domain = req.query.domain.replace(/;/g, '');
  exec('nslookup ' + domain, (err, out) => res.type('text').send(out));
});
'''), P(r'''
app.get('/lookup', (req, res) => {
  const domain = String(req.query.domain);
  if (!/^[a-z0-9.-]+$/i.test(domain) || domain.startsWith('-')) return res.sendStatus(400);
  execFile('nslookup', [domain], (err, out) => res.type('text').send(out));
});
''')),
        (P(r'''
app.post('/deploy/checkout', (req, res) => {
  exec('git -C /srv/app checkout ' + req.body.branch, (err, out) => res.json({ ok: !err, out }));
});
'''), P(r'''
app.post('/deploy/checkout', (req, res) => {
  const branch = String(req.body.branch);
  if (!/^[\w./-]+$/.test(branch) || branch.startsWith('-')) return res.status(400).json({ ok: false });
  execFile('git', ['-C', '/srv/app', 'checkout', branch], (err, out) => res.json({ ok: !err, out }));
});
''')),
        (P(r'''
app.post('/backup', (req, res) => {
  const target = req.body.target;
  exec(`tar czf /backups/${target}.tgz /var/data`, (err) => res.json({ ok: !err }));
});
'''), P(r'''
app.post('/backup', (req, res) => {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  exec(`tar czf /backups/${stamp}.tgz /var/data`, (err) => res.json({ ok: !err }));
});
''')),
    ],
    "iid": [
        (P(r'''
router.get('/whois/:domain', (req, res) => {
  exec('whois ' + req.params.domain, { timeout: 5000 }, (err, stdout) => res.send(stdout));
});
'''), P(r'''
router.get('/whois/:domain', (req, res) => {
  if (!/^[a-z0-9.-]+$/i.test(req.params.domain) || req.params.domain.startsWith('-')) return res.sendStatus(400);
  execFile('whois', [req.params.domain], { timeout: 5000 }, (err, stdout) => res.send(stdout));
});
''')),
        (P(r'''
app.post('/logs/archive', (req, res) => {
  const safeName = req.body.name.replace(/[;&]/g, '');
  exec('zip -r /tmp/' + safeName + '.zip /var/log/app', () => res.sendStatus(202));
});
'''), P(r'''
app.post('/logs/archive', (req, res) => {
  const name = String(req.body.name);
  if (!/^\w+$/.test(name)) return res.sendStatus(400);
  execFile('zip', ['-r', '/tmp/' + name + '.zip', '/var/log/app'], () => res.sendStatus(202));
});
''')),
    ],
    "ood_near": [
        (P(r'''
const { execSync } = require('child_process');

app.get('/git/log', (req, res) => {
  const out = execSync(`git -C /srv/repos/${req.query.repo} log --author="${req.query.author}" --oneline`).toString();
  res.type('text').send(out);
});
'''), P(r'''
const { execFileSync } = require('child_process');

app.get('/git/log', (req, res) => {
  const repo = String(req.query.repo);
  if (!/^[\w-]+$/.test(repo)) return res.sendStatus(400);
  const out = execFileSync('git', ['-C', '/srv/repos/' + repo, 'log', '--author=' + req.query.author, '--oneline']).toString();
  res.type('text').send(out);
});
''')),
        (P(r'''
const { spawn } = require('child_process');

app.post('/convert', (req, res) => {
  const proc = spawn('ffmpeg -i ' + req.body.input + ' -vn out.mp3', { shell: true });
  proc.on('close', (code) => res.json({ code }));
});
'''), P(r'''
const { spawn } = require('child_process');
const path = require('path');

app.post('/convert', (req, res) => {
  const input = path.join('media', path.basename(String(req.body.input)));
  const proc = spawn('ffmpeg', ['-i', input, '-vn', 'out.mp3']);
  proc.on('close', (code) => res.json({ code }));
});
''')),
        (P(r'''
const { spawnSync } = require('child_process');

app.post('/scripts/run', (req, res) => {
  const r = spawnSync('sh', ['-c', `./scripts/${req.body.name}.sh ${req.body.args.join(' ')}`], { encoding: 'utf8' });
  res.json({ status: r.status, stdout: r.stdout });
});
'''), P(r'''
const { spawnSync } = require('child_process');
const ALLOWED_SCRIPTS = new Set(['cleanup', 'rotate-logs', 'healthcheck']);

app.post('/scripts/run', (req, res) => {
  if (!ALLOWED_SCRIPTS.has(req.body.name)) return res.sendStatus(404);
  const r = spawnSync(`./scripts/${req.body.name}.sh`, req.body.args.map(String), { encoding: 'utf8' });
  res.json({ status: r.status, stdout: r.stdout });
});
''')),
    ],
}

# ---------------------------------------------------------------- CODE_INJECTION
DATA["CODE_INJECTION"] = {
    "train": [
        (P(r'''
app.post('/calc', (req, res) => {
  const result = eval(req.body.expression);
  res.json({ result });
});
'''), P(r'''
app.post('/calc', (req, res) => {
  const m = /^\s*(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)\s*$/.exec(String(req.body.expression));
  if (!m) return res.status(400).json({ error: 'invalid expression' });
  const [, a, op, b] = m;
  const ops = { '+': (x, y) => x + y, '-': (x, y) => x - y, '*': (x, y) => x * y, '/': (x, y) => x / y };
  res.json({ result: ops[op](Number(a), Number(b)) });
});
''')),
        (P(r'''
app.post('/import', (req, res) => {
  const data = eval('(' + req.body.payload + ')');
  res.json({ imported: Object.keys(data).length });
});
'''), P(r'''
app.post('/import', (req, res) => {
  const data = JSON.parse(req.body.payload);
  res.json({ imported: Object.keys(data).length });
});
''')),
        (P(r'''
app.get('/records/:id/field', async (req, res) => {
  const record = await Record.findById(req.params.id);
  res.json({ value: eval('record.' + req.query.field) });
});
'''), P(r'''
app.get('/records/:id/field', async (req, res) => {
  const record = await Record.findById(req.params.id);
  const field = String(req.query.field);
  res.json({ value: Object.prototype.hasOwnProperty.call(record.toObject(), field) ? record[field] : null });
});
''')),
        (P(r'''
app.post('/formula', (req, res) => {
  const f = req.body.formula;
  if (f.includes('require') || f.includes('process')) return res.status(400).end();
  res.json({ value: eval(f) });
});
'''), P(r'''
const FORMULAS = {
  area: (w, h) => w * h,
  perimeter: (w, h) => 2 * (w + h),
};

app.post('/formula', (req, res) => {
  if (!Object.prototype.hasOwnProperty.call(FORMULAS, req.body.formula)) return res.status(400).end();
  res.json({ value: FORMULAS[req.body.formula](Number(req.body.w), Number(req.body.h)) });
});
''')),
        (P(r'''
function renderTemplate(template, vars) {
  const { name, plan } = vars;
  return eval('`' + template + '`');
}

app.post('/render', (req, res) => {
  res.type('text').send(renderTemplate(req.body.template, req.body.vars));
});
'''), P(r'''
function renderTemplate(template, vars) {
  return template.replace(/\{\{(\w+)\}\}/g, (_, key) =>
    Object.prototype.hasOwnProperty.call(vars, key) ? String(vars[key]) : '');
}

app.post('/render', (req, res) => {
  res.type('text').send(renderTemplate(req.body.template, req.body.vars));
});
''')),
    ],
    "iid": [
        (P(r'''
app.get('/filter', (req, res) => {
  const predicate = eval('(item) => ' + req.query.cond);
  res.json(items.filter(predicate));
});
'''), P(r'''
const FILTERS = {
  inStock: (item) => item.stock > 0,
  cheap: (item) => item.price < 20,
};

app.get('/filter', (req, res) => {
  const predicate = Object.prototype.hasOwnProperty.call(FILTERS, req.query.cond) ? FILTERS[req.query.cond] : () => true;
  res.json(items.filter(predicate));
});
''')),
        (P(r'''
function runMacro(code) {
  if (/(require|import|child_process)/.test(code)) {
    throw new Error('forbidden');
  }
  return eval(code);
}

app.post('/macros/run', (req, res) => {
  res.json({ result: runMacro(req.body.code) });
});
'''), P(r'''
const MACROS = {
  today: () => new Date().toDateString(),
  count: () => items.length,
};

function runMacro(name) {
  if (!Object.prototype.hasOwnProperty.call(MACROS, name)) throw new Error('unknown macro');
  return MACROS[name]();
}

app.post('/macros/run', (req, res) => {
  res.json({ result: runMacro(req.body.name) });
});
''')),
    ],
    "ood_near": [
        (P(r'''
function makeValidator(rule) {
  return new Function('value', 'return ' + rule + ';');
}

app.post('/validate', (req, res) => {
  const ok = makeValidator(req.body.rule)(req.body.value);
  res.json({ ok });
});
'''), P(r'''
const RULES = {
  nonEmpty: (v) => String(v).length > 0,
  isNumber: (v) => !Number.isNaN(Number(v)),
};

app.post('/validate', (req, res) => {
  const rule = Object.prototype.hasOwnProperty.call(RULES, req.body.rule) ? RULES[req.body.rule] : null;
  res.json({ ok: rule ? rule(req.body.value) : false });
});
''')),
        (P(r'''
const vm = require('vm');

app.post('/plugins/run', (req, res) => {
  const output = vm.runInNewContext(req.body.script, { console, data: req.body.data });
  res.json({ output });
});
'''), P(r'''
const PLUGINS = {
  uppercase: (data) => String(data).toUpperCase(),
  reverse: (data) => String(data).split('').reverse().join(''),
};

app.post('/plugins/run', (req, res) => {
  if (!Object.prototype.hasOwnProperty.call(PLUGINS, req.body.plugin)) return res.sendStatus(404);
  res.json({ output: PLUGINS[req.body.plugin](req.body.data) });
});
''')),
        (P(r'''
const ejs = require('ejs');

app.post('/preview', (req, res) => {
  const html = ejs.render(req.body.template, { user: req.user });
  res.send(html);
});
'''), P(r'''
const ejs = require('ejs');
const TEMPLATE = '<h1>Hello <%= user.name %></h1>';

app.post('/preview', (req, res) => {
  const html = ejs.render(TEMPLATE, { user: { name: req.body.name } });
  res.send(html);
});
''')),
    ],
}

# ---------------------------------------------------------------- PATH_TRAVERSAL
DATA["PATH_TRAVERSAL"] = {
    "train": [
        (P(r'''
app.get('/download', (req, res) => {
  const file = path.join(__dirname, 'uploads', req.query.name);
  fs.readFile(file, (err, data) => err ? res.sendStatus(404) : res.send(data));
});
'''), P(r'''
app.get('/download', (req, res) => {
  const file = path.join(__dirname, 'uploads', path.basename(String(req.query.name)));
  fs.readFile(file, (err, data) => err ? res.sendStatus(404) : res.send(data));
});
''')),
        (P(r'''
app.get('/page/:name', (req, res) => {
  const html = fs.readFileSync('./pages/' + req.params.name, 'utf8');
  res.send(html);
});
'''), P(r'''
app.get('/page/:name', (req, res) => {
  if (!/^[\w-]+$/.test(req.params.name)) return res.sendStatus(400);
  const html = fs.readFileSync('./pages/' + req.params.name, 'utf8');
  res.send(html);
});
''')),
        (P(r'''
app.get('/files', (req, res) => {
  const name = req.query.path.replace('../', '');
  res.send(fs.readFileSync(path.join('/srv/public', name), 'utf8'));
});
'''), P(r'''
const BASE = '/srv/public';

app.get('/files', (req, res) => {
  const full = path.resolve(BASE, String(req.query.path));
  if (!full.startsWith(BASE + path.sep)) return res.sendStatus(403);
  res.send(fs.readFileSync(full, 'utf8'));
});
''')),
        (P(r'''
app.get('/avatar', (req, res) => {
  const p = req.query.p;
  if (!p.startsWith('avatars/')) return res.sendStatus(400);
  fs.readFile(path.join('/srv/data', p), (err, buf) => res.send(buf));
});
'''), P(r'''
app.get('/avatar', (req, res) => {
  const full = path.resolve('/srv/data/avatars', path.basename(String(req.query.p)));
  fs.readFile(full, (err, buf) => err ? res.sendStatus(404) : res.send(buf));
});
''')),
        (P(r'''
app.post('/upload', (req, res) => {
  const dest = path.join(UPLOAD_DIR, req.body.filename);
  fs.writeFile(dest, req.body.content, () => res.sendStatus(201));
});
'''), P(r'''
app.post('/upload', (req, res) => {
  const dest = path.join(UPLOAD_DIR, path.basename(String(req.body.filename)));
  fs.writeFile(dest, req.body.content, () => res.sendStatus(201));
});
''')),
    ],
    "iid": [
        (P(r'''
router.get('/logs/:name', (req, res) => {
  fs.readFile('/var/log/app/' + req.params.name, 'utf8', (err, text) => res.type('text').send(text));
});
'''), P(r'''
router.get('/logs/:name', (req, res) => {
  const name = path.basename(req.params.name);
  fs.readFile(path.join('/var/log/app', name), 'utf8', (err, text) => res.type('text').send(text));
});
''')),
        (P(r'''
function readNote(noteName) {
  const cleaned = noteName.replace('..', '');
  return fs.readFileSync(path.join('/data/notes', cleaned), 'utf8');
}

app.get('/notes/:name', (req, res) => {
  res.type('text').send(readNote(req.params.name));
});
'''), P(r'''
const ROOT = '/data/notes';

function readNote(noteName) {
  const full = path.resolve(ROOT, noteName);
  if (!full.startsWith(ROOT + path.sep)) throw new Error('forbidden');
  return fs.readFileSync(full, 'utf8');
}

app.get('/notes/:name', (req, res) => {
  res.type('text').send(readNote(req.params.name));
});
''')),
    ],
    "ood_near": [
        (P(r'''
app.get('/media/:file', (req, res) => {
  const stream = fs.createReadStream(path.join(MEDIA_DIR, req.params.file));
  stream.on('error', () => res.sendStatus(404));
  stream.pipe(res);
});
'''), P(r'''
app.get('/media/:file', (req, res) => {
  const full = path.resolve(MEDIA_DIR, req.params.file);
  if (!full.startsWith(path.resolve(MEDIA_DIR) + path.sep)) return res.sendStatus(403);
  const stream = fs.createReadStream(full);
  stream.on('error', () => res.sendStatus(404));
  stream.pipe(res);
});
''')),
        (P(r'''
const unzipper = require('unzipper');

function extractArchive(zipPath, destDir) {
  fs.createReadStream(zipPath)
    .pipe(unzipper.Parse())
    .on('entry', (entry) => {
      entry.pipe(fs.createWriteStream(path.join(destDir, entry.path)));
    });
}
'''), P(r'''
const unzipper = require('unzipper');

function extractArchive(zipPath, destDir) {
  const root = path.resolve(destDir);
  fs.createReadStream(zipPath)
    .pipe(unzipper.Parse())
    .on('entry', (entry) => {
      const target = path.resolve(root, entry.path);
      if (!target.startsWith(root + path.sep)) return entry.autodrain();
      entry.pipe(fs.createWriteStream(target));
    });
}
''')),
        (P(r'''
app.delete('/attachments/:name', (req, res) => {
  fs.unlink(ATTACH_DIR + '/' + req.params.name, (err) => res.sendStatus(err ? 500 : 204));
});
'''), P(r'''
app.delete('/attachments/:name', (req, res) => {
  fs.unlink(path.join(ATTACH_DIR, path.basename(req.params.name)), (err) => res.sendStatus(err ? 500 : 204));
});
''')),
    ],
}

# ---------------------------------------------------------------- SSRF
DATA["SSRF"] = {
    "train": [
        (P(r'''
app.get('/preview', async (req, res) => {
  const response = await axios.get(req.query.url);
  res.send(response.data);
});
'''), P(r'''
const ALLOWED_HOSTS = new Set(['api.github.com', 'api.example.com']);

app.get('/preview', async (req, res) => {
  const target = new URL(String(req.query.url));
  if (target.protocol !== 'https:' || !ALLOWED_HOSTS.has(target.hostname)) return res.sendStatus(400);
  const response = await axios.get(target.href, { maxRedirects: 0 });
  res.send(response.data);
});
''')),
        (P(r'''
app.post('/profile/avatar', (req, res) => {
  http.get(req.body.avatarUrl, (upstream) => {
    upstream.pipe(fs.createWriteStream('avatars/' + req.user.id + '.img'));
    res.sendStatus(204);
  });
});
'''), P(r'''
const CDN_HOST = 'cdn.example.com';

app.post('/profile/avatar', (req, res) => {
  const u = new URL(String(req.body.avatarUrl));
  if (u.protocol !== 'https:' || u.hostname !== CDN_HOST) return res.sendStatus(400);
  https.get(u, (upstream) => {
    upstream.pipe(fs.createWriteStream('avatars/' + req.user.id + '.img'));
    res.sendStatus(204);
  });
});
''')),
        (P(r'''
app.get('/proxy', async (req, res) => {
  const url = req.query.url;
  if (!url.startsWith('https://api.example.com')) return res.sendStatus(403);
  const r = await axios.get(url);
  res.json(r.data);
});
'''), P(r'''
app.get('/proxy', async (req, res) => {
  const u = new URL(String(req.query.url));
  if (u.protocol !== 'https:' || u.hostname !== 'api.example.com') return res.sendStatus(403);
  const r = await axios.get(u.href, { maxRedirects: 0 });
  res.json(r.data);
});
''')),
        (P(r'''
app.post('/webhooks/test', async (req, res) => {
  const status = await axios.post(req.body.callbackUrl, { event: 'ping' }).then(r => r.status).catch(() => 0);
  res.json({ status });
});
'''), P(r'''
const PING_URL = 'https://hooks.example.com/ping';

app.post('/webhooks/test', async (req, res) => {
  const status = await axios.post(PING_URL, { event: 'ping', id: String(req.body.webhookId) }).then(r => r.status).catch(() => 0);
  res.json({ status });
});
''')),
        (P(r'''
async function importFromUrl(feedUrl) {
  const res = await axios.get(feedUrl, { timeout: 5000 });
  return parseFeed(res.data);
}

app.post('/feeds', async (req, res) => {
  res.json(await importFromUrl(req.body.feedUrl));
});
'''), P(r'''
const FEED_HOSTS = ['news.example.com', 'blog.example.com'];

async function importFromUrl(feedUrl) {
  const u = new URL(feedUrl);
  if (u.protocol !== 'https:' || !FEED_HOSTS.includes(u.hostname)) throw new Error('feed host not allowed');
  const res = await axios.get(u.href, { timeout: 5000, maxRedirects: 0 });
  return parseFeed(res.data);
}

app.post('/feeds', async (req, res) => {
  res.json(await importFromUrl(req.body.feedUrl));
});
''')),
    ],
    "iid": [
        (P(r'''
router.get('/thumbnail', async (req, res) => {
  const img = await axios.get(req.query.src, { responseType: 'arraybuffer' });
  res.type('image/jpeg').send(img.data);
});
'''), P(r'''
const IMAGE_HOSTS = new Set(['images.example.com']);

router.get('/thumbnail', async (req, res) => {
  const src = new URL(String(req.query.src));
  if (src.protocol !== 'https:' || !IMAGE_HOSTS.has(src.hostname)) return res.sendStatus(400);
  const img = await axios.get(src.href, { responseType: 'arraybuffer', maxRedirects: 0 });
  res.type('image/jpeg').send(img.data);
});
''')),
        (P(r'''
app.post('/import', async (req, res) => {
  const link = req.body.link;
  if (!link.includes('example.com')) return res.sendStatus(400);
  const doc = await axios.get(link);
  res.json({ title: doc.data.title });
});
'''), P(r'''
app.post('/import', async (req, res) => {
  const link = new URL(String(req.body.link));
  const okHost = link.hostname === 'example.com' || link.hostname.endsWith('.example.com');
  if (link.protocol !== 'https:' || !okHost) return res.sendStatus(400);
  const doc = await axios.get(link.href, { maxRedirects: 0 });
  res.json({ title: doc.data.title });
});
''')),
    ],
    "ood_near": [
        (P(r'''
app.post('/page-meta', async (req, res) => {
  const resp = await fetch(req.body.pageUrl);
  const html = await resp.text();
  res.json({ title: (html.match(/<title>(.*?)<\/title>/i) || [])[1] || null });
});
'''), P(r'''
const PAGE_HOSTS = new Set(['example.org', 'www.example.org']);

app.post('/page-meta', async (req, res) => {
  const u = new URL(String(req.body.pageUrl));
  if (u.protocol !== 'https:' || !PAGE_HOSTS.has(u.hostname)) return res.sendStatus(400);
  const resp = await fetch(u.href, { redirect: 'error' });
  const html = await resp.text();
  res.json({ title: (html.match(/<title>(.*?)<\/title>/i) || [])[1] || null });
});
''')),
        (P(r'''
const got = require('got');

exports.handler = async (event) => {
  const body = JSON.parse(event.body);
  const upstream = await got(body.endpoint, { method: 'POST', json: body.payload }).json();
  return { statusCode: 200, body: JSON.stringify(upstream) };
};
'''), P(r'''
const got = require('got');
const ENDPOINTS = {
  billing: 'https://billing.internal.example.com/api',
  search: 'https://search.internal.example.com/api',
};

exports.handler = async (event) => {
  const body = JSON.parse(event.body);
  const endpoint = Object.prototype.hasOwnProperty.call(ENDPOINTS, body.service) ? ENDPOINTS[body.service] : null;
  if (!endpoint) return { statusCode: 400, body: 'unknown service' };
  const upstream = await got(endpoint, { method: 'POST', json: body.payload }).json();
  return { statusCode: 200, body: JSON.stringify(upstream) };
};
''')),
        (P(r'''
app.get('/pdf', async (req, res) => {
  const page = await browser.newPage();
  await page.goto(req.query.url);
  res.type('application/pdf').send(await page.pdf());
});
'''), P(r'''
app.get('/pdf', async (req, res) => {
  const u = new URL(String(req.query.url));
  if (u.protocol !== 'https:' || u.hostname !== 'reports.example.com') return res.sendStatus(400);
  const page = await browser.newPage();
  await page.goto(u.href);
  res.type('application/pdf').send(await page.pdf());
});
''')),
    ],
}

# ---------------------------------------------------------------- PROTO_POLLUTION
DATA["PROTO_POLLUTION"] = {
    "train": [
        (P(r'''
function merge(target, source) {
  for (const key in source) {
    if (typeof source[key] === 'object' && source[key] !== null) {
      if (!target[key]) target[key] = {};
      merge(target[key], source[key]);
    } else {
      target[key] = source[key];
    }
  }
  return target;
}

app.post('/settings', (req, res) => {
  res.json(merge(defaultSettings, req.body));
});
'''), P(r'''
function merge(target, source) {
  for (const key of Object.keys(source)) {
    if (key === '__proto__' || key === 'constructor' || key === 'prototype') continue;
    if (typeof source[key] === 'object' && source[key] !== null) {
      if (!Object.prototype.hasOwnProperty.call(target, key)) target[key] = {};
      merge(target[key], source[key]);
    } else {
      target[key] = source[key];
    }
  }
  return target;
}

app.post('/settings', (req, res) => {
  res.json(merge(defaultSettings, req.body));
});
''')),
        (P(r'''
function deepExtend(dest, src) {
  Object.keys(src).forEach((k) => {
    if (src[k] && typeof src[k] === 'object') {
      dest[k] = deepExtend(dest[k] || {}, src[k]);
    } else {
      dest[k] = src[k];
    }
  });
  return dest;
}

app.post('/theme', (req, res) => {
  res.json(deepExtend(themeSettings, req.body));
});
'''), P(r'''
function deepExtend(dest, src) {
  Object.keys(src).forEach((k) => {
    if (k === '__proto__' || k === 'constructor' || k === 'prototype') return;
    if (src[k] && typeof src[k] === 'object') {
      dest[k] = deepExtend(dest[k] || {}, src[k]);
    } else {
      dest[k] = src[k];
    }
  });
  return dest;
}

app.post('/theme', (req, res) => {
  res.json(deepExtend(themeSettings, req.body));
});
''')),
        (P(r'''
function setDefaults(obj, patch) {
  for (const k of Object.keys(patch)) {
    if (k === '__proto__') continue;
    if (patch[k] && typeof patch[k] === 'object') {
      obj[k] = obj[k] || {};
      setDefaults(obj[k], patch[k]);
    } else {
      obj[k] = patch[k];
    }
  }
}

app.put('/defaults', (req, res) => {
  setDefaults(globalDefaults, req.body);
  res.sendStatus(204);
});
'''), P(r'''
function setDefaults(obj, patch) {
  for (const k of Object.keys(patch)) {
    if (k === '__proto__' || k === 'constructor' || k === 'prototype') continue;
    if (patch[k] && typeof patch[k] === 'object') {
      obj[k] = obj[k] || {};
      setDefaults(obj[k], patch[k]);
    } else {
      obj[k] = patch[k];
    }
  }
}

app.put('/defaults', (req, res) => {
  setDefaults(globalDefaults, req.body);
  res.sendStatus(204);
});
''')),
        (P(r'''
app.post('/prefs', (req, res) => {
  const { section, key, value } = req.body;
  prefs[section][key] = value;
  res.sendStatus(204);
});
'''), P(r'''
const SECTIONS = new Set(['ui', 'notifications', 'privacy']);

app.post('/prefs', (req, res) => {
  const { section, key, value } = req.body;
  if (!SECTIONS.has(section) || key === '__proto__' || key === 'constructor' || key === 'prototype') return res.sendStatus(400);
  prefs[section][key] = value;
  res.sendStatus(204);
});
''')),
        (P(r'''
function applyPatch(doc, patch) {
  Object.entries(patch).forEach(([field, value]) => {
    if (value !== null && typeof value === 'object') {
      doc[field] = doc[field] || {};
      applyPatch(doc[field], value);
    } else {
      doc[field] = value;
    }
  });
}

app.patch('/profile', (req, res) => {
  applyPatch(req.session.profile, req.body);
  res.sendStatus(204);
});
'''), P(r'''
const FORBIDDEN = new Set(['__proto__', 'constructor', 'prototype']);

function applyPatch(doc, patch) {
  Object.entries(patch).forEach(([field, value]) => {
    if (FORBIDDEN.has(field)) return;
    if (value !== null && typeof value === 'object') {
      doc[field] = doc[field] || {};
      applyPatch(doc[field], value);
    } else {
      doc[field] = value;
    }
  });
}

app.patch('/profile', (req, res) => {
  applyPatch(req.session.profile, req.body);
  res.sendStatus(204);
});
''')),
    ],
    "iid": [
        (P(r'''
function mergeInto(target, source) {
  for (const key in source) {
    if (source[key] && typeof source[key] === 'object') {
      target[key] = target[key] || {};
      mergeInto(target[key], source[key]);
    } else {
      target[key] = source[key];
    }
  }
}

app.post('/config/overrides', (req, res) => {
  mergeInto(appConfig, JSON.parse(req.body.overrides));
  res.sendStatus(200);
});
'''), P(r'''
function mergeInto(target, source) {
  for (const key of Object.keys(source)) {
    if (key === '__proto__' || key === 'constructor' || key === 'prototype') continue;
    if (source[key] && typeof source[key] === 'object') {
      target[key] = target[key] || {};
      mergeInto(target[key], source[key]);
    } else {
      target[key] = source[key];
    }
  }
}

app.post('/config/overrides', (req, res) => {
  mergeInto(appConfig, JSON.parse(req.body.overrides));
  res.sendStatus(200);
});
''')),
        (P(r'''
function assignDeep(dst, src) {
  for (const k of Object.keys(src)) {
    if (k.startsWith('__')) continue;
    if (src[k] && typeof src[k] === 'object') {
      dst[k] = dst[k] || {};
      assignDeep(dst[k], src[k]);
    } else {
      dst[k] = src[k];
    }
  }
  return dst;
}

app.post('/widgets/config', (req, res) => {
  res.json(assignDeep(widgetConfig, req.body));
});
'''), P(r'''
function assignDeep(dst, src) {
  for (const k of Object.keys(src)) {
    if (k === '__proto__' || k === 'constructor' || k === 'prototype') continue;
    if (src[k] && typeof src[k] === 'object') {
      dst[k] = dst[k] || {};
      assignDeep(dst[k], src[k]);
    } else {
      dst[k] = src[k];
    }
  }
  return dst;
}

app.post('/widgets/config', (req, res) => {
  res.json(assignDeep(widgetConfig, req.body));
});
''')),
    ],
    "ood_near": [
        (P(r'''
function setByPath(obj, path, value) {
  const parts = path.split('.');
  let cur = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    if (cur[parts[i]] === undefined) cur[parts[i]] = {};
    cur = cur[parts[i]];
  }
  cur[parts[parts.length - 1]] = value;
}

app.post('/flags', (req, res) => {
  setByPath(flags, req.body.path, req.body.value);
  res.sendStatus(204);
});
'''), P(r'''
function setByPath(obj, path, value) {
  const parts = path.split('.');
  if (parts.some((p) => p === '__proto__' || p === 'constructor' || p === 'prototype')) {
    throw new Error('invalid path');
  }
  let cur = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    if (cur[parts[i]] === undefined) cur[parts[i]] = {};
    cur = cur[parts[i]];
  }
  cur[parts[parts.length - 1]] = value;
}

app.post('/flags', (req, res) => {
  setByPath(flags, req.body.path, req.body.value);
  res.sendStatus(204);
});
''')),
        (P(r'''
function parseNested(qs) {
  const result = {};
  qs.split('&').forEach((pair) => {
    const [rawKey, value] = pair.split('=');
    const keys = rawKey.replace(/\]/g, '').split('[');
    let node = result;
    keys.forEach((k, i) => {
      if (i === keys.length - 1) node[k] = decodeURIComponent(value);
      else node = node[k] = node[k] || {};
    });
  });
  return result;
}

app.get('/filters', (req, res) => {
  res.json(parseNested(req.url.split('?')[1] || ''));
});
'''), P(r'''
function parseNested(qs) {
  const result = Object.create(null);
  qs.split('&').forEach((pair) => {
    const [rawKey, value] = pair.split('=');
    const keys = rawKey.replace(/\]/g, '').split('[');
    if (keys.some((k) => k === '__proto__' || k === 'constructor' || k === 'prototype')) return;
    let node = result;
    keys.forEach((k, i) => {
      if (i === keys.length - 1) node[k] = decodeURIComponent(value);
      else node = node[k] = node[k] || Object.create(null);
    });
  });
  return result;
}

app.get('/filters', (req, res) => {
  res.json(parseNested(req.url.split('?')[1] || ''));
});
''')),
        (P(r'''
function applyPatchOps(doc, ops) {
  for (const { path, value } of ops) {
    const keys = path.split('/').filter(Boolean);
    const last = keys.pop();
    const target = keys.reduce((o, k) => (o[k] = o[k] || {}), doc);
    target[last] = value;
  }
  return doc;
}

app.patch('/documents/:id', (req, res) => {
  res.json(applyPatchOps(documents[req.params.id], req.body.ops));
});
'''), P(r'''
function applyPatchOps(doc, ops) {
  for (const { path, value } of ops) {
    const keys = path.split('/').filter(Boolean);
    if (keys.some((k) => k === '__proto__' || k === 'constructor' || k === 'prototype')) continue;
    const last = keys.pop();
    const target = keys.reduce((o, k) => (o[k] = o[k] || {}), doc);
    target[last] = value;
  }
  return doc;
}

app.patch('/documents/:id', (req, res) => {
  res.json(applyPatchOps(documents[req.params.id], req.body.ops));
});
''')),
    ],
}

# ---------------------------------------------------------------- OPEN_REDIRECT
DATA["OPEN_REDIRECT"] = {
    "train": [
        (P(r'''
app.get('/login/callback', (req, res) => {
  req.session.user = authenticate(req);
  res.redirect(req.query.next || '/');
});
'''), P(r'''
const SAFE_DESTINATIONS = { home: '/', dashboard: '/dashboard', settings: '/settings' };

app.get('/login/callback', (req, res) => {
  req.session.user = authenticate(req);
  const dest = Object.prototype.hasOwnProperty.call(SAFE_DESTINATIONS, req.query.next) ? SAFE_DESTINATIONS[req.query.next] : '/';
  res.redirect(dest);
});
''')),
        (P(r'''
app.post('/logout', (req, res) => {
  req.session.destroy(() => {
    res.redirect(req.body.returnUrl);
  });
});
'''), P(r'''
function isLocalPath(p) {
  return typeof p === 'string' && p.startsWith('/') && !p.startsWith('//') && !p.startsWith('/\\');
}

app.post('/logout', (req, res) => {
  req.session.destroy(() => {
    res.redirect(isLocalPath(req.body.returnUrl) ? req.body.returnUrl : '/');
  });
});
''')),
        (P(r'''
app.get('/go', (req, res) => {
  const to = req.query.to;
  if (!to.startsWith('/')) return res.sendStatus(400);
  res.redirect(to);
});
'''), P(r'''
app.get('/go', (req, res) => {
  const to = String(req.query.to);
  if (!to.startsWith('/') || to.startsWith('//') || to.startsWith('/\\')) return res.sendStatus(400);
  res.redirect(to);
});
''')),
        (P(r'''
app.get('/sso/return', (req, res) => {
  const target = req.query.redirect_uri;
  if (target.includes('example.com')) return res.redirect(target);
  res.redirect('/');
});
'''), P(r'''
app.get('/sso/return', (req, res) => {
  let target;
  try { target = new URL(String(req.query.redirect_uri)); } catch (e) { return res.redirect('/'); }
  if (target.protocol === 'https:' && target.hostname === 'app.example.com') return res.redirect(target.href);
  res.redirect('/');
});
''')),
        (P(r'''
app.get('/language/:code', (req, res) => {
  res.cookie('lang', req.params.code);
  res.redirect(req.query.back || '/');
});
'''), P(r'''
app.get('/language/:code', (req, res) => {
  res.cookie('lang', req.params.code);
  const back = String(req.query.back || '/');
  res.redirect(back.startsWith('/') && !back.startsWith('//') && !back.startsWith('/\\') ? back : '/');
});
''')),
    ],
    "iid": [
        (P(r'''
router.get('/confirm-email', async (req, res) => {
  await verifyToken(req.query.token);
  res.redirect(req.query.continue || '/welcome');
});
'''), P(r'''
router.get('/confirm-email', async (req, res) => {
  await verifyToken(req.query.token);
  res.redirect('/welcome');
});
''')),
        (P(r'''
app.get('/r', (req, res) => {
  const url = req.query.url;
  if (url.startsWith('http://') || url.startsWith('https://')) return res.sendStatus(400);
  res.redirect(url);
});
'''), P(r'''
app.get('/r', (req, res) => {
  const url = String(req.query.url);
  if (!url.startsWith('/') || url.startsWith('//') || url.startsWith('/\\')) return res.sendStatus(400);
  res.redirect(url);
});
''')),
    ],
    "ood_near": [
        (P(r'''
const http = require('http');

http.createServer((req, res) => {
  const target = new URL(req.url, 'http://localhost').searchParams.get('to');
  res.writeHead(302, { Location: target });
  res.end();
}).listen(8080);
'''), P(r'''
const http = require('http');
const ALLOWED = new Set(['https://docs.example.com/', 'https://status.example.com/']);

http.createServer((req, res) => {
  const target = new URL(req.url, 'http://localhost').searchParams.get('to');
  res.writeHead(302, { Location: ALLOWED.has(target) ? target : '/' });
  res.end();
}).listen(8080);
''')),
        (P(r'''
app.get('/auth/google/done', (req, res) => {
  const state = JSON.parse(Buffer.from(req.query.state, 'base64').toString());
  res.status(302).set('Location', state.returnTo).end();
});
'''), P(r'''
app.get('/auth/google/done', (req, res) => {
  const state = JSON.parse(Buffer.from(req.query.state, 'base64').toString());
  const returnTo = typeof state.returnTo === 'string' && /^\/(?![/\\])/.test(state.returnTo) ? state.returnTo : '/';
  res.status(302).set('Location', returnTo).end();
});
''')),
        (P(r'''
fastify.get('/out', async (request, reply) => {
  const { url } = request.query;
  await recordClick(url);
  return reply.redirect(url);
});
'''), P(r'''
const PARTNERS = new Set(['https://partner-a.example.com', 'https://partner-b.example.com']);

fastify.get('/out', async (request, reply) => {
  const { url } = request.query;
  if (!PARTNERS.has(url)) return reply.code(400).send({ error: 'unknown destination' });
  await recordClick(url);
  return reply.redirect(url);
});
''')),
    ],
}

# ---------------------------------------------------------------- AUTHZ
DATA["AUTHZ"] = {
    "train": [
        (P(r'''
app.get('/api/orders/:id', requireLogin, async (req, res) => {
  const order = await Order.findById(req.params.id);
  if (!order) return res.sendStatus(404);
  res.json(order);
});
'''), P(r'''
app.get('/api/orders/:id', requireLogin, async (req, res) => {
  const order = await Order.findById(req.params.id);
  if (!order) return res.sendStatus(404);
  if (order.userId.toString() !== req.user.id) return res.sendStatus(403);
  res.json(order);
});
''')),
        (P(r'''
app.delete('/api/users/:id', requireLogin, async (req, res) => {
  await User.deleteOne({ _id: req.params.id });
  res.sendStatus(204);
});
'''), P(r'''
app.delete('/api/users/:id', requireLogin, async (req, res) => {
  if (req.user.role !== 'admin') return res.sendStatus(403);
  await User.deleteOne({ _id: req.params.id });
  res.sendStatus(204);
});
''')),
        (P(r'''
app.put('/api/me', requireLogin, async (req, res) => {
  const user = await User.findByIdAndUpdate(req.user.id, req.body, { new: true });
  res.json(user);
});
'''), P(r'''
app.put('/api/me', requireLogin, async (req, res) => {
  const { displayName, bio, avatarUrl } = req.body;
  const user = await User.findByIdAndUpdate(req.user.id, { displayName, bio, avatarUrl }, { new: true });
  res.json(user);
});
''')),
        (P(r'''
function requireAdmin(req, res, next) {
  if (req.headers['x-user-role'] === 'admin') return next();
  res.sendStatus(403);
}

app.get('/admin/export', requireAdmin, (req, res) => res.json(exportAllUsers()));
'''), P(r'''
function requireAdmin(req, res, next) {
  if (req.user && req.user.role === 'admin') return next();
  res.sendStatus(403);
}

app.get('/admin/export', requireLogin, requireAdmin, (req, res) => res.json(exportAllUsers()));
''')),
        (P(r'''
app.put('/api/posts/:id', requireLogin, async (req, res) => {
  const post = await Post.findByIdAndUpdate(req.params.id, { body: req.body.body }, { new: true });
  res.json(post);
});
'''), P(r'''
app.put('/api/posts/:id', requireLogin, async (req, res) => {
  const post = await Post.findOneAndUpdate({ _id: req.params.id, authorId: req.user.id }, { body: req.body.body }, { new: true });
  if (!post) return res.sendStatus(404);
  res.json(post);
});
''')),
    ],
    "iid": [
        (P(r'''
router.get('/invoices/:invoiceId/pdf', ensureAuth, async (req, res) => {
  const invoice = await Invoice.findByPk(req.params.invoiceId);
  res.type('pdf').send(await renderInvoice(invoice));
});
'''), P(r'''
router.get('/invoices/:invoiceId/pdf', ensureAuth, async (req, res) => {
  const invoice = await Invoice.findOne({ where: { id: req.params.invoiceId, customerId: req.user.id } });
  if (!invoice) return res.sendStatus(404);
  res.type('pdf').send(await renderInvoice(invoice));
});
''')),
        (P(r'''
app.post('/api/settings/global', requireLogin, async (req, res) => {
  if (!req.body.isAdmin) return res.sendStatus(403);
  await Settings.update(req.body.values);
  res.sendStatus(204);
});
'''), P(r'''
app.post('/api/settings/global', requireLogin, async (req, res) => {
  if (req.user.role !== 'admin') return res.sendStatus(403);
  await Settings.update(req.body.values);
  res.sendStatus(204);
});
''')),
    ],
    "ood_near": [
        (P(r'''
const resolvers = {
  Mutation: {
    deleteComment: async (_, { id }, { user }) => {
      if (!user) throw new AuthenticationError('login required');
      await prisma.comment.delete({ where: { id } });
      return true;
    },
  },
};
'''), P(r'''
const resolvers = {
  Mutation: {
    deleteComment: async (_, { id }, { user }) => {
      if (!user) throw new AuthenticationError('login required');
      const comment = await prisma.comment.findUnique({ where: { id } });
      if (!comment || (comment.authorId !== user.id && user.role !== 'moderator')) throw new ForbiddenError('not allowed');
      await prisma.comment.delete({ where: { id } });
      return true;
    },
  },
};
''')),
        (P(r'''
app.patch('/users/:id/email', requireLogin, async (req, res) => {
  const updated = await prisma.user.update({ where: { id: Number(req.params.id) }, data: { email: req.body.email } });
  res.json({ id: updated.id, email: updated.email });
});
'''), P(r'''
app.patch('/users/:id/email', requireLogin, async (req, res) => {
  if (Number(req.params.id) !== req.user.id) return res.sendStatus(403);
  const updated = await prisma.user.update({ where: { id: req.user.id }, data: { email: req.body.email } });
  res.json({ id: updated.id, email: updated.email });
});
''')),
        (P(r'''
app.get('/documents/:id/download', requireLogin, async (req, res) => {
  const doc = await knex('documents').where({ id: req.params.id }).first();
  if (!doc) return res.sendStatus(404);
  res.download(doc.storage_path, doc.filename);
});
'''), P(r'''
app.get('/documents/:id/download', requireLogin, async (req, res) => {
  const doc = await knex('documents').where({ id: req.params.id, owner_id: req.user.id }).first();
  if (!doc) return res.sendStatus(404);
  res.download(doc.storage_path, doc.filename);
});
''')),
    ],
}

# ---------------------------------------------------------------- WEAK_CRYPTO
DATA["WEAK_CRYPTO"] = {
    "train": [
        (P(r'''
const crypto = require('crypto');

function hashPassword(password) {
  return crypto.createHash('md5').update(password).digest('hex');
}
'''), P(r'''
const crypto = require('crypto');

function hashPassword(password) {
  const salt = crypto.randomBytes(16);
  const hash = crypto.scryptSync(password, salt, 64);
  return salt.toString('hex') + ':' + hash.toString('hex');
}
''')),
        (P(r'''
function createResetToken() {
  return Math.random().toString(36).substring(2) + Math.random().toString(36).substring(2);
}
'''), P(r'''
const crypto = require('crypto');

function createResetToken() {
  return crypto.randomBytes(32).toString('hex');
}
''')),
        (P(r'''
function newSessionId(userId) {
  return userId + '-' + Date.now() + '-' + Math.floor(Math.random() * 1000);
}
'''), P(r'''
const crypto = require('crypto');

function newSessionId(userId) {
  return userId + '-' + crypto.randomUUID();
}
''')),
        (P(r'''
function storePassword(user, password) {
  const salt = 'static-site-salt';
  user.passwordHash = crypto.createHash('sha1').update(salt + password).digest('hex');
}
'''), P(r'''
function storePassword(user, password) {
  const salt = crypto.randomBytes(16);
  user.passwordSalt = salt.toString('hex');
  user.passwordHash = crypto.pbkdf2Sync(password, salt, 310000, 32, 'sha256').toString('hex');
}
''')),
        (P(r'''
function generateOtp() {
  let otp = '';
  for (let i = 0; i < 6; i++) otp += Math.floor(Math.random() * 10);
  return otp;
}
'''), P(r'''
const crypto = require('crypto');

function generateOtp() {
  let otp = '';
  for (let i = 0; i < 6; i++) otp += crypto.randomInt(0, 10);
  return otp;
}
''')),
    ],
    "iid": [
        (P(r'''
const crypto = require('crypto');

async function verifyLogin(db, email, password) {
  const user = await db.users.findOne({ email });
  const digest = crypto.createHash('md5').update(password).digest('hex');
  return user && user.passwordHash === digest ? user : null;
}
'''), P(r'''
const bcrypt = require('bcrypt');

async function verifyLogin(db, email, password) {
  const user = await db.users.findOne({ email });
  return user && (await bcrypt.compare(password, user.passwordHash)) ? user : null;
}
''')),
        (P(r'''
function generateApiKey() {
  return 'sk_' + Math.random().toString(16).slice(2) + Math.random().toString(16).slice(2);
}
'''), P(r'''
const crypto = require('crypto');

function generateApiKey() {
  return 'sk_' + crypto.randomBytes(24).toString('hex');
}
''')),
    ],
    "ood_near": [
        (P(r'''
const crypto = require('crypto');

function encryptField(plaintext, key) {
  const cipher = crypto.createCipheriv('aes-128-ecb', key, null);
  return Buffer.concat([cipher.update(plaintext, 'utf8'), cipher.final()]).toString('base64');
}
'''), P(r'''
const crypto = require('crypto');

function encryptField(plaintext, key) {
  const iv = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv('aes-256-gcm', key, iv);
  const enc = Buffer.concat([cipher.update(plaintext, 'utf8'), cipher.final()]);
  return Buffer.concat([iv, cipher.getAuthTag(), enc]).toString('base64');
}
''')),
        (P(r'''
const crypto = require('crypto');
const IV = Buffer.alloc(16, 0);

function encryptToken(data, key) {
  const cipher = crypto.createCipheriv('aes-256-cbc', key, IV);
  return cipher.update(data, 'utf8', 'hex') + cipher.final('hex');
}
'''), P(r'''
const crypto = require('crypto');

function encryptToken(data, key) {
  const iv = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv('aes-256-gcm', key, iv);
  const enc = Buffer.concat([cipher.update(data, 'utf8'), cipher.final()]);
  return iv.toString('hex') + cipher.getAuthTag().toString('hex') + enc.toString('hex');
}
''')),
        (P(r'''
const seedrandom = require('seedrandom');

function newPasswordResetCode() {
  const rng = seedrandom(String(Date.now()));
  return String(Math.floor(rng() * 1e6)).padStart(6, '0');
}
'''), P(r'''
const crypto = require('crypto');

function newPasswordResetCode() {
  return String(crypto.randomInt(0, 1e6)).padStart(6, '0');
}
''')),
    ],
}

# ---------------------------------------------------------------- ood_hard
# One pair per type. The VULNERABLE twin contains a mitigation-looking but bypassable check
# (so "mitigation idiom present -> SAFE" fails), on sinks/APIs not used in training.
# The SAFE twin keeps a similar-looking line but is genuinely safe.
DATA["XSS"]["ood_hard"] = [
    (P(r'''
$(function () {
  const note = new URL(window.location.href).searchParams.get('note') || '';
  const cleaned = note.replace(/<script[^>]*>/gi, '');
  $('#note').html('<p>' + cleaned + '</p>');
});
'''), P(r'''
$(function () {
  const note = new URL(window.location.href).searchParams.get('note') || '';
  const cleaned = note.replace(/[<>]/g, '');
  $('#note').empty().append($('<p>').text(cleaned));
});
''')),
]

DATA["SQLI"]["ood_hard"] = [
    (P(r'''
router.get('/items/:id', async (req, res) => {
  const id = String(req.params.id).replace(/;/g, '');
  const rows = await knex.raw('SELECT * FROM items WHERE id = ' + id);
  res.json(rows.rows);
});
'''), P(r'''
router.get('/items/:id', async (req, res) => {
  const id = String(req.params.id).replace(/;/g, '');
  const rows = await knex.raw('SELECT * FROM items WHERE id = ?', [Number(id)]);
  res.json(rows.rows);
});
''')),
]

DATA["CMDI"]["ood_hard"] = [
    (P(r'''
const { execSync } = require('child_process');

app.get('/compress', (req, res) => {
  const file = String(req.query.file).replace(/[;&|]/g, '');
  execSync(`gzip -k uploads/${file}`);
  res.sendStatus(200);
});
'''), P(r'''
const { execFileSync } = require('child_process');

app.get('/compress', (req, res) => {
  const file = String(req.query.file).replace(/[;&|]/g, '');
  execFileSync('gzip', ['-k', path.join('uploads', path.basename(file))]);
  res.sendStatus(200);
});
''')),
]

DATA["CODE_INJECTION"]["ood_hard"] = [
    (P(r'''
const vm = require('vm');

app.post('/script/eval', (req, res) => {
  if (/require|process|child_process/.test(req.body.script)) return res.sendStatus(400);
  res.json({ result: vm.runInNewContext(req.body.script) });
});
'''), P(r'''
const vm = require('vm');
const SCRIPTS = { total: 'items.reduce((a, b) => a + b.price, 0)', count: 'items.length' };

app.post('/script/eval', (req, res) => {
  if (/require|process|child_process/.test(req.body.script)) return res.sendStatus(400);
  const source = Object.prototype.hasOwnProperty.call(SCRIPTS, req.body.script) ? SCRIPTS[req.body.script] : null;
  if (!source) return res.sendStatus(404);
  res.json({ result: vm.runInNewContext(source, { items: ITEMS.map((i) => ({ price: i.price })) }) });
});
''')),
]

DATA["PATH_TRAVERSAL"]["ood_hard"] = [
    (P(r'''
app.get('/reports/:name', (req, res) => {
  const name = req.params.name.replace(/\.\.\//g, '');
  fs.createReadStream(path.join(REPORT_DIR, name)).pipe(res);
});
'''), P(r'''
app.get('/reports/:name', (req, res) => {
  const name = req.params.name.replace(/\.\.\//g, '');
  const full = path.resolve(REPORT_DIR, name);
  if (!full.startsWith(path.resolve(REPORT_DIR) + path.sep)) return res.sendStatus(403);
  fs.createReadStream(full).pipe(res);
});
''')),
]

DATA["SSRF"]["ood_hard"] = [
    (P(r'''
app.post('/link-preview', async (req, res) => {
  const u = new URL(String(req.body.url));
  if (!u.hostname.endsWith('example.com')) return res.sendStatus(400);
  const r = await fetch(u.href);
  res.json({ title: (await r.text()).slice(0, 80) });
});
'''), P(r'''
app.post('/link-preview', async (req, res) => {
  const u = new URL(String(req.body.url));
  if (u.protocol !== 'https:' || (u.hostname !== 'example.com' && !u.hostname.endsWith('.example.com'))) return res.sendStatus(400);
  const r = await fetch(u.href, { redirect: 'error' });
  res.json({ title: (await r.text()).slice(0, 80) });
});
''')),
]

DATA["PROTO_POLLUTION"]["ood_hard"] = [
    (P(r'''
function setByPath(obj, path, value) {
  const parts = path.split('.');
  if (parts.includes('__proto__')) throw new Error('invalid path');
  let cur = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    if (cur[parts[i]] === undefined) cur[parts[i]] = {};
    cur = cur[parts[i]];
  }
  cur[parts[parts.length - 1]] = value;
}

app.post('/options', (req, res) => {
  setByPath(options, req.body.path, req.body.value);
  res.sendStatus(204);
});
'''), P(r'''
function setByPath(obj, path, value) {
  const parts = path.split('.');
  if (parts.some((p) => p === '__proto__' || p === 'constructor' || p === 'prototype')) throw new Error('invalid path');
  let cur = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    if (cur[parts[i]] === undefined) cur[parts[i]] = {};
    cur = cur[parts[i]];
  }
  cur[parts[parts.length - 1]] = value;
}

app.post('/options', (req, res) => {
  setByPath(options, req.body.path, req.body.value);
  res.sendStatus(204);
});
''')),
]

DATA["OPEN_REDIRECT"]["ood_hard"] = [
    (P(r'''
fastify.get('/exit', async (request, reply) => {
  const { to } = request.query;
  if (!to.startsWith('https://example.com')) return reply.code(400).send({ error: 'blocked' });
  return reply.redirect(to);
});
'''), P(r'''
fastify.get('/exit', async (request, reply) => {
  const { to } = request.query;
  let target;
  try { target = new URL(to); } catch (e) { return reply.code(400).send({ error: 'blocked' }); }
  if (target.protocol !== 'https:' || target.hostname !== 'example.com') return reply.code(400).send({ error: 'blocked' });
  return reply.redirect(target.href);
});
''')),
]

DATA["AUTHZ"]["ood_hard"] = [
    (P(r'''
app.put('/projects/:id', requireLogin, async (req, res) => {
  if (req.body.ownerId !== req.user.id) return res.sendStatus(403);
  const project = await prisma.project.update({ where: { id: Number(req.params.id) }, data: { name: req.body.name } });
  res.json(project);
});
'''), P(r'''
app.put('/projects/:id', requireLogin, async (req, res) => {
  const existing = await prisma.project.findUnique({ where: { id: Number(req.params.id) } });
  if (!existing || existing.ownerId !== req.user.id) return res.sendStatus(403);
  const project = await prisma.project.update({ where: { id: existing.id }, data: { name: req.body.name } });
  res.json(project);
});
''')),
]

DATA["WEAK_CRYPTO"]["ood_hard"] = [
    (P(r'''
const crypto = require('crypto');

function encryptNote(text, key) {
  const iv = Buffer.from(String(Date.now()).padEnd(16, '0'));
  const cipher = crypto.createCipheriv('aes-256-cbc', key, iv);
  return iv.toString('hex') + ':' + cipher.update(text, 'utf8', 'hex') + cipher.final('hex');
}
'''), P(r'''
const crypto = require('crypto');

function encryptNote(text, key) {
  const iv = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv('aes-256-gcm', key, iv);
  const enc = Buffer.concat([cipher.update(text, 'utf8'), cipher.final()]);
  return iv.toString('hex') + ':' + cipher.getAuthTag().toString('hex') + ':' + enc.toString('hex');
}
''')),
]

# ---------------------------------------------------------------- easy
# One minimal, canonical pair per type (textbook form, no look-alike mitigations). A sanity check of
# general competence: if a technique hurts here, it is breaking basic skills.
DATA["XSS"]["easy"] = [(
    P(r'''
document.getElementById('out').innerHTML = location.hash.slice(1);
'''), P(r'''
document.getElementById('out').textContent = location.hash.slice(1);
'''))]

DATA["SQLI"]["easy"] = [(
    P(r'''
db.query("SELECT * FROM users WHERE id = " + req.query.id);
'''), P(r'''
db.query('SELECT * FROM users WHERE id = ?', [req.query.id]);
'''))]

DATA["CMDI"]["easy"] = [(
    P(r'''
exec('ls ' + req.query.dir, (err, out) => res.send(out));
'''), P(r'''
execFile('ls', ['--', req.query.dir], (err, out) => res.send(out));
'''))]

DATA["CODE_INJECTION"]["easy"] = [(
    P(r'''
res.send(String(eval(req.query.expr)));
'''), P(r'''
res.send(String(JSON.parse(req.query.expr)));
'''))]

DATA["PATH_TRAVERSAL"]["easy"] = [(
    P(r'''
fs.readFile('./public/' + req.query.file, (err, data) => res.send(data));
'''), P(r'''
fs.readFile('./public/' + path.basename(req.query.file), (err, data) => res.send(data));
'''))]

DATA["SSRF"]["easy"] = [(
    P(r'''
axios.get(req.query.url).then((r) => res.send(r.data));
'''), P(r'''
axios.get('https://api.example.com/items/' + encodeURIComponent(req.query.id)).then((r) => res.send(r.data));
'''))]

DATA["PROTO_POLLUTION"]["easy"] = [(
    P(r'''
Object.keys(req.body).forEach((k) => { target[k] = Object.assign(target[k] || {}, req.body[k]); });
'''), P(r'''
Object.keys(req.body).forEach((k) => {
  if (k === '__proto__' || k === 'constructor' || k === 'prototype') return;
  target[k] = Object.assign(target[k] || {}, req.body[k]);
});
'''))]

DATA["OPEN_REDIRECT"]["easy"] = [(
    P(r'''
res.redirect(req.query.next);
'''), P(r'''
res.redirect(req.query.next === 'dashboard' ? '/dashboard' : '/');
'''))]

DATA["AUTHZ"]["easy"] = [(
    P(r'''
app.delete('/posts/:id', requireLogin, (req, res) => Post.deleteOne({ _id: req.params.id }).then(() => res.sendStatus(204)));
'''), P(r'''
app.delete('/posts/:id', requireLogin, (req, res) => Post.deleteOne({ _id: req.params.id, authorId: req.user.id }).then(() => res.sendStatus(204)));
'''))]

DATA["WEAK_CRYPTO"]["easy"] = [(
    P(r'''
const token = Math.random().toString(36).slice(2);
'''), P(r'''
const token = crypto.randomBytes(16).toString('hex');
'''))]


# ====================================================================== shortcut control
# Safe twins naturally contain more code (the fix), so "longer = SAFE" would score ~0.9 on pair
# accuracy without any understanding. We add neutral comment lines, independently for each twin
# (vulnerable twins get 0-8, safe twins 1-4), so length / line count stop being reliable cues.
# The padding seed is chosen on cue statistics only (never on model results). Some residual cue
# is unavoidable: run_pipeline.py reports the best trivial-feature baseline next to the model.
PAD_VULN = (0, 8)
PAD_SAFE = (1, 4)
NEUTRAL_COMMENTS = [
    "// handler for the main page", "// TODO: add pagination later", "// legacy endpoint, kept for mobile clients",
    "// see ticket #4821 for context", "// returns JSON", "// keep in sync with the client",
    "// cached for a short time", "// also used by the dashboard", "// runs on every request",
    "// refactor when we drop v1", "// default timeout is fine here", "// owner: platform team",
    "// part of the public API", "// called from the settings screen", "// reviewed in the last sprint",
    "// metrics are collected elsewhere", "// added during the March release", "// ask the backend team before changing",
    "// order of arguments matters here", "// covered by the integration tests", "// kept simple on purpose",
    "// feature flag lives in the config", "// logging happens in the middleware", "// used by two other services",
    "// temporary until the migration ends", "// documented in the wiki", "// response shape is stable",
    "// first version, may change", "// shared with the mobile app", "// nightly job calls this too",
]


def _feats(code):
    lines = code.split("\n")
    return {"chars": len(code), "lines": len(lines),
            "comment_lines": sum(1 for l in lines if l.lstrip().startswith("//"))}


def _pad_pair(vuln, safe, rng):
    def pad(code, n):
        if n == 0:
            return code
        picks = rng.sample(NEUTRAL_COMMENTS, n)
        top = [c for c in picks if rng.random() < 0.5]
        bottom = [c for c in picks if c not in top]
        return "\n".join(top + [code] + bottom)
    return pad(vuln, rng.randint(*PAD_VULN)), pad(safe, rng.randint(*PAD_SAFE))


def trivial_cues(pairs):
    """Best a single trivial feature can do at 'which twin is SAFE' (pair accuracy), per feature."""
    out = {}
    for f in ("chars", "lines", "comment_lines"):
        wins = 0.0
        for v, s in pairs:
            a, b = _feats(v)[f], _feats(s)[f]
            wins += (b > a) + 0.5 * (a == b)
        p = wins / len(pairs)
        out[f] = max(p, 1 - p)
    return out


def _padded_data(seed, orig=None, sets=("train", "iid", "ood_near", "ood_hard", "easy")):
    orig = orig or DATA
    new = {}
    for t in VULN_TYPES:
        new[t] = {}
        for s in sets:
            new[t][s] = [_pad_pair(v, sf, random.Random(f"{seed}|{s}|{t}|{j}"))
                         for j, (v, sf) in enumerate(orig[t][s])]
    return new


def _cues_by_set(data, sets=("train", "iid", "ood_near", "ood_hard", "easy")):
    return {s: trivial_cues([p for t in VULN_TYPES for p in data[t][s]]) for s in sets}


def choose_and_apply_padding(n_seeds=300, select_on=("train", "iid", "ood_near")):
    """The global seed is chosen on the original three sets only (so adding new sets never changes the code
    of existing snippets, keeping earlier runs comparable). ood_hard (already shipped) uses that global seed.
    Sets added later (`easy`) get their OWN seed, chosen on their own cue statistics."""
    orig = {t: {s: list(DATA[t][s]) for s in DATA[t]} for t in VULN_TYPES}
    best = None
    for seed in range(n_seeds):
        cues = _cues_by_set(_padded_data(seed, orig, select_on), select_on)
        score = sum((c - 0.5) ** 2 for s in select_on for c in cues[s].values())
        if best is None or score < best[0]:
            best = (score, seed)
    seed = best[1]
    DATA.update(_padded_data(seed, orig, ("train", "iid", "ood_near", "ood_hard")))
    cues = _cues_by_set(DATA, ("train", "iid", "ood_near", "ood_hard"))
    best_easy = None
    for es in range(n_seeds):
        c = _cues_by_set(_padded_data(es, orig, ("easy",)), ("easy",))["easy"]
        score = sum((v - 0.5) ** 2 for v in c.values())
        if best_easy is None or score < best_easy[0]:
            best_easy = (score, es)
    easy = _padded_data(best_easy[1], orig, ("easy",))
    for t in VULN_TYPES:
        DATA[t]["easy"] = easy[t]["easy"]
    cues["easy"] = _cues_by_set(DATA, ("easy",))["easy"]
    return seed, cues


# ====================================================================== build
def interleave(set_name):
    """Round-robin over types, pair by pair; each pair = (vuln, safe) adjacent."""
    rows = []
    n = max(len(DATA[t][set_name]) for t in VULN_TYPES)
    for j in range(n):
        for t in VULN_TYPES:
            pairs = DATA[t][set_name]
            if j >= len(pairs):
                continue
            vuln, safe = pairs[j]
            pid = f"{set_name}-{t}-{j + 1:02d}"
            sub = "insufficient" if (set_name == "ood_hard" or (set_name, t, j) in INSUFFICIENT) else "plain"
            base = {"pair_id": pid, "set": set_name, "vuln_type": t, "subtype": sub}
            rows.append(dict(base, id=pid + "-v", label=t, role="vuln", code=vuln))
            rows.append(dict(base, id=pid + "-s", label=SAFE, role="safe", code=safe))
    return rows


def idents(code):
    import re
    code = "\n".join(l for l in code.split("\n") if not l.lstrip().startswith("//"))  # ignore padding comments
    return set(re.findall(r"[A-Za-z_$][A-Za-z0-9_$]*", code))


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def validate(train, test):
    errors = []
    expected = {"train": N_TRAIN, "iid": N_IID, "ood_near": N_OOD, "ood_hard": N_HARD, "easy": N_EASY}
    for t in VULN_TYPES:
        for s, n in expected.items():
            if len(DATA[t][s]) != n:
                errors.append(f"{t}/{s}: expected {n} pairs, got {len(DATA[t][s])}")
    allrows = train + test
    ids = [r["id"] for r in allrows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate ids")
    codes = [r["code"] for r in allrows]
    if len(codes) != len(set(codes)):
        dup = [c for c, k in Counter(codes).items() if k > 1]
        errors.append(f"{len(dup)} duplicated snippets (first: {dup[0][:60]!r})")
    for r in allrows:
        if "'''" in r["code"]:
            errors.append(f"{r['id']}: contains triple quote")
    for i in range(0, len(allrows), 2):
        if allrows[i]["code"] == allrows[i + 1]["code"]:
            errors.append(f"{allrows[i]['pair_id']}: twins identical")
    return errors


def similarity_report(train, test):
    tr = [idents(r["code"]) for r in train]
    out = {}
    for s in ("iid", "ood_near", "ood_hard", "easy"):
        sims = []
        for r in test:
            if r["set"] != s:
                continue
            ids = idents(r["code"])
            sims.append(max(jaccard(ids, x) for x in tr))
        out[s] = (sum(sims) / len(sims), max(sims))
    return out


def check_js(rows):
    node = shutil.which("node")
    if not node:
        print("  node not found: skipping JS syntax check")
        return []
    bad = []
    with tempfile.TemporaryDirectory() as d:
        for r in rows:
            fp = os.path.join(d, r["id"] + ".js")
            with open(fp, "w", encoding="utf-8") as f:
                f.write(r["code"] + "\n")
            p = subprocess.run([node, "--check", fp], capture_output=True, text=True)
            if p.returncode != 0:
                bad.append((r["id"], p.stderr.strip().splitlines()[:4]))
    return bad


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    ap.add_argument("--check-js", action="store_true", help="run `node --check` on every snippet")
    args = ap.parse_args()

    pad_seed, cues = choose_and_apply_padding()
    train = interleave("train")
    test = interleave("iid") + interleave("ood_near") + interleave("ood_hard") + interleave("easy")

    errors = validate(train, test)
    if args.check_js:
        for rid, msg in check_js(train + test):
            errors.append(f"JS syntax error in {rid}: {msg}")
    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print("  -", e)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
    write_jsonl(os.path.join(args.out_dir, "test.jsonl"), test)

    print(f"train.jsonl: {len(train)} snippets = {len(train) // 2} pairs")
    print("  labels:", dict(Counter(r["label"] for r in train)))
    for s in ("iid", "ood_near", "ood_hard", "easy"):
        rows = [r for r in test if r["set"] == s]
        print(f"test.jsonl [{s}]: {len(rows)} snippets = {len(rows) // 2} pairs; "
              f"{dict(Counter(r['label'] for r in rows))}")
    print("identifier-set Jaccard to nearest train snippet (mean, max):")
    for s, (m, mx) in similarity_report(train, test).items():
        print(f"  {s:9s} mean={m:.3f} max={mx:.3f}")
    print(f"padding seed {pad_seed}; best trivial-feature pair accuracy (0.5 = no cue):")
    for s, per in cues.items():
        print(f"  {s:9s} " + "  ".join(f"{k}={v:.2f}" for k, v in per.items()))
    worst = max(v for per in cues.values() for v in per.values())
    if worst > 0.72:
        print(f"WARNING: a trivial feature still reaches {worst:.2f} pair accuracy")
    lens = [len(r["code"]) for r in train + test]
    print(f"snippet length chars: mean={sum(lens) / len(lens):.0f} max={max(lens)}")


if __name__ == "__main__":
    main()
