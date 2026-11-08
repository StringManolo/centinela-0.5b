#!/usr/bin/env python3
"""
Generate synthetic training data for the 15 new code-detectable classes.

Each class has multiple (vuln, safe, subtype) templates. The generator:
- randomly picks a template
- adds BALANCED comments (same count on vuln and safe side)
  so the comment-count feature cannot act as a shortcut
- produces N pairs total per class

Usage:
    python scripts/gen_extended_data.py --n-pairs 20 --out /tmp/new.jsonl
    python scripts/gen_extended_data.py --n-pairs 20 --append data/train.jsonl
"""
import argparse
import json
import random
import os

COMMENTS = [
    "// feature flag lives in the config",
    "// TODO: add pagination later",
    "// keep in sync with the client",
    "// returns JSON",
    "// reviewed in the last sprint",
    "// owner: platform team",
    "// default timeout is fine here",
    "// metrics are collected elsewhere",
    "// used by two other services",
    "// see ticket #4821 for context",
    "// also used by the dashboard",
    "// legacy endpoint, kept for mobile clients",
    "// nightly job calls this too",
    "// temporary until the migration ends",
    "// shared with the mobile app",
    "// order of arguments matters here",
    "// refactor when we drop v1",
    "// documented in the wiki",
    "// called from the settings screen",
    "// first version, may change",
    "// part of the public API",
    "// ask the backend team before changing",
    "// cached for a short time",
    "// runs on every request",
    "// logging happens in the middleware",
    "// handler for the main page",
    "// covered by the integration tests",
    "// response shape is stable",
]

# Each entry: (vuln_code, safe_code, subtype)
TEMPLATES = {
    "SSTI": [
        ("app.post('/render', (req, res) => {\n  res.type('html').send(ejs.render(req.body.template, req.body.data));\n});",
         "const TEMPLATE = '<h1><%= title %></h1>';\napp.post('/render', (req, res) => {\n  res.type('html').send(ejs.render(TEMPLATE, { title: String(req.body.title) }));\n});",
         "plain"),
        ("const hbs = require('handlebars');\napp.post('/hbs', (req, res) => {\n  const tpl = hbs.compile(req.body.tpl);\n  res.send(tpl(req.body.ctx));\n});",
         "const hbs = require('handlebars');\nconst FIXED = hbs.compile('<p>{{name}}</p>');\napp.post('/hbs', (req, res) => {\n  res.send(FIXED({ name: String(req.body.name) }));\n});",
         "plain"),
        ("const _ = require('lodash');\napp.post('/tpl', (req, res) => {\n  const fn = _.template(req.body.tpl);\n  res.send(fn(req.body.ctx));\n});",
         "const _ = require('lodash');\nconst fn = _.template('<%= name %>', { escape: '' });\napp.post('/tpl', (req, res) => {\n  res.send(_.escape(fn({ name: String(req.body.name) })));\n});",
         "insufficient"),
    ],
    "CSRF": [
        ("app.post('/transfer', requireLogin, (req, res) => {\n  transfer(req.user.id, req.body.to, req.body.amount);\n  res.json({ ok: true });\n});",
         "app.post('/transfer', requireLogin, csrfProtection, (req, res) => {\n  if (!req.session.csrf || req.body.csrfToken !== req.session.csrf) return res.sendStatus(403);\n  transfer(req.user.id, req.body.to, req.body.amount);\n  res.json({ ok: true });\n});",
         "plain"),
        ("app.delete('/api/posts/:id', requireLogin, async (req, res) => {\n  await Post.findByIdAndDelete(req.params.id);\n  res.sendStatus(204);\n});",
         "app.delete('/api/posts/:id', requireLogin, csrfProtection, async (req, res) => {\n  if (req.headers['x-csrf-token'] !== req.session.csrf) return res.sendStatus(403);\n  await Post.findByIdAndDelete(req.params.id);\n  res.sendStatus(204);\n});",
         "plain"),
        ("app.post('/settings', requireLogin, (req, res) => {\n  if (!req.body.csrfToken) return res.sendStatus(400);\n  updateSettings(req.user.id, req.body);\n  res.json({ ok: true });\n});",
         "app.post('/settings', requireLogin, (req, res) => {\n  if (req.body.csrfToken !== req.session.csrf) return res.sendStatus(403);\n  updateSettings(req.user.id, req.body);\n  res.json({ ok: true });\n});",
         "insufficient"),
    ],
    "XXE": [
        ("const libxml = require('libxmljs');\napp.post('/parse', (req, res) => {\n  const doc = libxml.parseXml(req.body.xml, { resolveEntities: true });\n  res.json({ root: doc.root().name() });\n});",
         "const libxml = require('libxmljs');\napp.post('/parse', (req, res) => {\n  const doc = libxml.parseXml(req.body.xml, { resolveEntities: false, noent: false, noblanks: true });\n  res.json({ root: doc.root().name() });\n});",
         "plain"),
        ("const sax = require('saxes');\napp.post('/import', (req, res) => {\n  const p = new sax.SaxesParser({ xmlns: true, externalEntities: true });\n  p.write(req.body.xml).close();\n  res.json({ ok: true });\n});",
         "const sax = require('saxes');\napp.post('/import', (req, res) => {\n  const p = new sax.SaxesParser({ xmlns: true, externalEntities: false });\n  p.write(req.body.xml).close();\n  res.json({ ok: true });\n});",
         "plain"),
        ("const libxml = require('libxmljs');\napp.post('/parse', (req, res) => {\n  const xml = req.body.xml.replace(/<!DOCTYPE[^>]+>/gi, '');\n  const doc = libxml.parseXml(xml, { resolveEntities: true });\n  res.json({ root: doc.root().name() });\n});",
         "const libxml = require('libxmljs');\napp.post('/parse', (req, res) => {\n  const doc = libxml.parseXml(req.body.xml, { resolveEntities: false, noent: false });\n  res.json({ root: doc.root().name() });\n});",
         "insufficient"),
    ],
    "LDAP_INJECTION": [
        ("const ldap = require('ldapjs');\napp.post('/login', (req, res) => {\n  const filter = '(&(uid=' + req.body.user + ')(userPassword=' + req.body.pw + '))';\n  ldapClient.search('dc=example,dc=com', { filter }, (err, r) => {\n    r.on('searchEntry', () => res.json({ ok: true }));\n    r.on('end', () => res.json({ ok: false }));\n  });\n});",
         "const ldap = require('ldapjs');\nconst { escape } = require('ldapjs');\napp.post('/login', (req, res) => {\n  const filter = '(&(uid=' + escape(String(req.body.user)) + ')(userPassword=' + escape(String(req.body.pw)) + '))';\n  ldapClient.search('dc=example,dc=com', { filter }, (err, r) => {\n    r.on('searchEntry', () => res.json({ ok: true }));\n    r.on('end', () => res.json({ ok: false }));\n  });\n});",
         "plain"),
        ("app.get('/users/search', (req, res) => {\n  const filter = '(cn=*' + req.query.q + '*)';\n  ldapClient.search('ou=people,dc=example,dc=com', { filter }, (err, r) => r.on('end', () => res.json({ done: true })));\n});",
         "const { escape } = require('ldapjs');\napp.get('/users/search', (req, res) => {\n  const filter = '(cn=*' + escape(String(req.query.q)) + '*)';\n  ldapClient.search('ou=people,dc=example,dc=com', { filter }, (err, r) => r.on('end', () => res.json({ done: true })));\n});",
         "plain"),
        ("app.post('/lookup', (req, res) => {\n  const sanitized = String(req.body.uid).replace(/[*()\\\\]/g, '');\n  const filter = '(uid=' + sanitized + ')';\n  ldapClient.search('dc=example,dc=com', { filter }, (err, r) => r.on('end', () => res.json({ done: true })));\n});",
         "const { escape } = require('ldapjs');\napp.post('/lookup', (req, res) => {\n  const filter = '(uid=' + escape(String(req.body.uid)) + ')';\n  ldapClient.search('dc=example,dc=com', { filter }, (err, r) => r.on('end', () => res.json({ done: true })));\n});",
         "insufficient"),
    ],
    "HPP": [
        ("app.post('/checkout', requireLogin, (req, res) => {\n  const amount = req.query.amount;\n  charge(req.user.id, amount);\n  res.json({ ok: true });\n});",
         "app.post('/checkout', requireLogin, (req, res) => {\n  const raw = req.query.amount;\n  const amount = Array.isArray(raw) ? Number(raw[0]) : Number(raw);\n  if (!Number.isFinite(amount) || amount <= 0) return res.sendStatus(400);\n  charge(req.user.id, amount);\n  res.json({ ok: true });\n});",
         "plain"),
        ("app.post('/set-role', requireLogin, (req, res) => {\n  const target = req.body.target;\n  const role = req.body.role;\n  updateRole(req.user.id, target, role);\n  res.json({ ok: true });\n});",
         "app.post('/set-role', requireLogin, (req, res) => {\n  const target = Array.isArray(req.body.target) ? req.body.target[0] : req.body.target;\n  const role = Array.isArray(req.body.role) ? req.body.role[0] : req.body.role;\n  if (!['user', 'mod'].includes(role)) return res.sendStatus(400);\n  updateRole(req.user.id, target, role);\n  res.json({ ok: true });\n});",
         "plain"),
        ("app.post('/like', (req, res) => {\n  const id = typeof req.query.id === 'string' ? req.query.id : req.query.id[0];\n  like(req.user.id, id);\n  res.json({ ok: true });\n});",
         "app.post('/like', (req, res) => {\n  if (Array.isArray(req.query.id)) return res.sendStatus(400);\n  like(req.user.id, String(req.query.id));\n  res.json({ ok: true });\n});",
         "insufficient"),
    ],
    "CRLF_INJECTION": [
        ("app.get('/share', (req, res) => {\n  res.setHeader('X-Share-URL', req.query.url);\n  res.json({ ok: true });\n});",
         "app.get('/share', (req, res) => {\n  if (/[\\r\\n]/.test(String(req.query.url))) return res.sendStatus(400);\n  res.setHeader('X-Share-URL', encodeURI(String(req.query.url)));\n  res.json({ ok: true });\n});",
         "plain"),
        ("app.get('/login', (req, res) => {\n  res.setHeader('Set-Cookie', 'session=' + req.query.sid + '; Path=/');\n  res.json({ ok: true });\n});",
         "app.get('/login', (req, res) => {\n  const sid = String(req.query.sid).replace(/[^A-Za-z0-9_-]/g, '');\n  res.cookie('session', sid, { httpOnly: true, secure: true, sameSite: 'strict' });\n  res.json({ ok: true });\n});",
         "plain"),
        ("app.get('/redirect', (req, res) => {\n  const loc = String(req.query.to).replace(/\\n/g, '');\n  res.setHeader('Location', loc);\n  res.status(302).end();\n});",
         "app.get('/redirect', (req, res) => {\n  const loc = String(req.query.to);\n  if (/[\\r\\n]/.test(loc)) return res.sendStatus(400);\n  res.setHeader('Location', loc);\n  res.status(302).end();\n});",
         "insufficient"),
    ],
    "CSSI": [
        ("function applyTheme(url) {\n  document.getElementById('bg').style.backgroundImage = \"url('\" + url + \"')\";\n}\n// theme picker",
         "function applyTheme(url) {\n  const u = new URL(url, location.href);\n  if (u.protocol !== 'https:' || u.origin !== 'https://cdn.example.com') return;\n  document.getElementById('bg').style.backgroundImage = `url('${encodeURI(u.href)}')`;\n}\n// theme picker",
         "plain"),
        ("app.get('/theme', (req, res) => {\n  res.type('text/css').send(req.query.css);\n});\n// serves user css",
         "const PROPERTIES = new Set(['color', 'background-color', 'font-size']);\napp.get('/theme', (req, res) => {\n  const rules = [];\n  for (const [k, v] of Object.entries(JSON.parse(String(req.query.css || '{}')))) {\n    if (PROPERTIES.has(k) && /^[#\\w(),.\\s%-]+$/.test(String(v))) rules.push(`${k}: ${v}`);\n  }\n  res.type('text/css').send(`body { ${rules.join(';')} }`);\n});\n// serves user css",
         "plain"),
        ("document.getElementById('theme').innerHTML = '<style>' + localStorage.getItem('css') + '</style>';\n// loads saved theme",
         "const css = localStorage.getItem('css') || '';\nif (/[{}<>]/.test(css)) {\n  console.warn('invalid saved css');\n} else {\n  document.getElementById('theme').textContent = css;\n}\n// loads saved theme",
         "insufficient"),
    ],
    "HTMLI": [
        ("app.get('/card', (req, res) => {\n  res.send('<div class=\"card\">' + req.query.html + '</div>');\n});\n// profile card renderer",
         "const DOMPurify = require('dompurify');\nconst { JSDOM } = require('jsdom');\nconst w = new JSDOM('').window;\nconst purify = DOMPurify(w);\napp.get('/card', (req, res) => {\n  res.send('<div class=\"card\">' + purify.sanitize(req.query.html, { ALLOWED_TAGS: ['b','i','p'] }) + '</div>');\n});\n// profile card renderer",
         "plain"),
        ("function showEmbed(html) {\n  document.getElementById('embed').innerHTML = html;\n}\n// embedded media",
         "function showEmbed(html) {\n  const safe = html.replace(/<(?!\\/?(iframe|video|img)\\b)[^>]+>/gi, '');\n  document.getElementById('embed').innerHTML = safe;\n}\n// embedded media",
         "insufficient"),
        ("app.get('/bio', (req, res) => {\n  res.send(`<section><p>${req.query.bio}</p></section>`);\n});\n// public bio endpoint",
         "const he = require('he');\napp.get('/bio', (req, res) => {\n  res.send(`<section><p>${he.encode(String(req.query.bio))}</p></section>`);\n});\n// public bio endpoint",
         "plain"),
    ],
    "NOSQLI": [
        ("app.post('/login', async (req, res) => {\n  const user = await User.findOne({ email: req.body.email, password: req.body.password });\n  res.json({ ok: !!user });\n});\n// login endpoint",
         "app.post('/login', async (req, res) => {\n  const email = String(req.body.email || '');\n  const password = String(req.body.password || '');\n  const user = await User.findOne({ email, password });\n  res.json({ ok: !!user });\n});\n// login endpoint",
         "plain"),
        ("app.get('/accounts', async (req, res) => {\n  const filter = { role: req.query.role };\n  const list = await Account.find(filter);\n  res.json(list);\n});\n// admin listing",
         "const ROLES = new Set(['admin', 'mod', 'user']);\napp.get('/accounts', async (req, res) => {\n  if (!ROLES.has(String(req.query.role))) return res.sendStatus(400);\n  const list = await Account.find({ role: String(req.query.role) });\n  res.json(list);\n});\n// admin listing",
         "plain"),
        ("app.get('/search', async (req, res) => {\n  const docs = await Doc.find({ $where: 'this.title.match(' + JSON.stringify(req.query.q) + ')' });\n  res.json(docs);\n});\n// full-text search",
         "app.get('/search', async (req, res) => {\n  const q = String(req.query.q || '').replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&');\n  const docs = await Doc.find({ title: { $regex: q, $options: 'i' } });\n  res.json(docs);\n});\n// full-text search",
         "insufficient"),
    ],
    "HARDCODED_SECRETS": [
        ("const STRIPE_KEY = 'sk_live_51H8xAbCdEfGhIjKlMnOpQrStUvWxYz0123456789';\nconst stripe = require('stripe')(STRIPE_KEY);\n// payment client",
         "const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);\n// payment client",
         "plain"),
        ("const AWS_ACCESS_KEY = 'AKIAIOSFODNN7EXAMPLE';\nconst AWS_SECRET = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY';\nconst s3 = new AWS.S3({ accessKeyId: AWS_ACCESS_KEY, secretAccessKey: AWS_SECRET });\n// s3 client",
         "const s3 = new AWS.S3({ credentials: new AWS.EnvironmentCredentials('AWS') });\n// s3 client",
         "plain"),
        ("const DB_PASSWORD = Buffer.from('c3VwZXJzZWNyZXQ=', 'base64').toString();\nconst pool = mysql.createPool({ host: 'db', user: 'app', password: DB_PASSWORD });\n// database pool",
         "const pool = mysql.createPool({ host: process.env.DB_HOST, user: process.env.DB_USER, password: process.env.DB_PASSWORD });\n// database pool",
         "insufficient"),
    ],
    "MASS_ASSIGNMENT": [
        ("app.post('/api/users', async (req, res) => {\n  const user = await User.create(req.body);\n  res.status(201).json(user);\n});\n// registration",
         "app.post('/api/users', async (req, res) => {\n  const { email, name, password } = req.body;\n  const user = await User.create({ email, name, password, role: 'user' });\n  res.status(201).json(user);\n});\n// registration",
         "plain"),
        ("app.put('/api/users/:id', requireLogin, async (req, res) => {\n  const user = await User.findByIdAndUpdate(req.params.id, req.body, { new: true });\n  res.json(user);\n});\n// profile update",
         "app.put('/api/users/:id', requireLogin, async (req, res) => {\n  const ALLOWED = ['name', 'bio', 'avatarUrl'];\n  const patch = {};\n  for (const k of ALLOWED) if (req.body[k] !== undefined) patch[k] = req.body[k];\n  const user = await User.findByIdAndUpdate(req.params.id, patch, { new: true });\n  res.json(user);\n});\n// profile update",
         "plain"),
        ("app.patch('/api/me', requireLogin, async (req, res) => {\n  const ALLOWED = ['name', 'bio'];\n  const patch = {};\n  for (const k of Object.keys(req.body)) if (ALLOWED.includes(k)) patch[k] = req.body[k];\n  const user = await User.findByIdAndUpdate(req.user.id, patch, { new: true });\n  res.json(user);\n});\n// profile patch",
         "app.patch('/api/me', requireLogin, async (req, res) => {\n  const { name, bio } = req.body;\n  const user = await User.findByIdAndUpdate(req.user.id, { name, bio }, { new: true });\n  res.json(user);\n});\n// profile patch",
         "insufficient"),
    ],
    "WEAK_COOKIE": [
        ("app.post('/login', (req, res) => {\n  const token = signToken(req.user);\n  res.cookie('session', token);\n  res.json({ ok: true });\n});\n// login",
         "app.post('/login', (req, res) => {\n  const token = signToken(req.user);\n  res.cookie('session', token, { httpOnly: true, secure: true, sameSite: 'strict', maxAge: 3600000 });\n  res.json({ ok: true });\n});\n// login",
         "plain"),
        ("app.get('/auth', (req, res) => {\n  res.cookie('auth', req.query.token, { httpOnly: true });\n  res.json({ ok: true });\n});\n// oauth callback",
         "app.get('/auth', (req, res) => {\n  res.cookie('auth', String(req.query.token), { httpOnly: true, secure: true, sameSite: 'lax' });\n  res.json({ ok: true });\n});\n// oauth callback",
         "insufficient"),
        ("app.post('/set-theme', (req, res) => {\n  res.cookie('theme', req.body.theme, { maxAge: 86400000 });\n  res.json({ ok: true });\n});\n// theme preference",
         "app.post('/set-theme', (req, res) => {\n  const t = ['light', 'dark'].includes(req.body.theme) ? req.body.theme : 'light';\n  res.cookie('theme', t, { httpOnly: true, secure: true, sameSite: 'lax', maxAge: 86400000 });\n  res.json({ ok: true });\n});\n// theme preference",
         "plain"),
    ],
    "INSECURE_JWT": [
        ("const jwt = require('jsonwebtoken');\napp.post('/verify', (req, res) => {\n  const payload = jwt.verify(req.body.token, '', { algorithms: ['none'] });\n  res.json(payload);\n});\n// auth check",
         "const jwt = require('jsonwebtoken');\napp.post('/verify', (req, res) => {\n  try {\n    const payload = jwt.verify(req.body.token, process.env.JWT_SECRET, { algorithms: ['HS256'] });\n    res.json(payload);\n  } catch (e) { res.sendStatus(401); }\n});\n// auth check",
         "plain"),
        ("const jwt = require('jsonwebtoken');\napp.get('/me', (req, res) => {\n  const payload = jwt.decode(req.query.token);\n  res.json(payload);\n});\n// current user",
         "const jwt = require('jsonwebtoken');\napp.get('/me', (req, res) => {\n  try {\n    const payload = jwt.verify(req.query.token, process.env.JWT_SECRET, { algorithms: ['HS256'] });\n    res.json(payload);\n  } catch (e) { res.sendStatus(401); }\n});\n// current user",
         "plain"),
        ("const jwt = require('jsonwebtoken');\napp.post('/auth', (req, res) => {\n  const payload = jwt.verify(req.body.token, process.env.JWT_SECRET);\n  res.json(payload);\n});\n// insecure default algorithms",
         "const jwt = require('jsonwebtoken');\napp.post('/auth', (req, res) => {\n  try {\n    const payload = jwt.verify(req.body.token, process.env.JWT_SECRET, { algorithms: ['HS256'], ignoreExpiration: false });\n    res.json(payload);\n  } catch (e) { res.sendStatus(401); }\n});\n// strict algorithms",
         "insufficient"),
    ],
    "RACE_CONDITION": [
        ("app.post('/redeem', requireLogin, async (req, res) => {\n  const coupon = await Coupon.findById(req.body.couponId);\n  if (coupon.used) return res.sendStatus(400);\n  coupon.used = true;\n  await coupon.save();\n  await credit(req.user.id, coupon.value);\n  res.json({ ok: true });\n});\n// coupon redemption",
         "app.post('/redeem', requireLogin, async (req, res) => {\n  const result = await Coupon.findOneAndUpdate(\n    { _id: req.body.couponId, used: false },\n    { $set: { used: true, usedBy: req.user.id, usedAt: new Date() } },\n    { new: true }\n  );\n  if (!result) return res.sendStatus(400);\n  await credit(req.user.id, result.value);\n  res.json({ ok: true });\n});\n// coupon redemption",
         "plain"),
        ("app.post('/withdraw', requireLogin, async (req, res) => {\n  const acc = await Account.findById(req.user.accountId);\n  if (acc.balance < req.body.amount) return res.sendStatus(400);\n  acc.balance -= req.body.amount;\n  await acc.save();\n  await transfer(req.body.amount);\n  res.json({ ok: true });\n});\n// balance withdraw",
         "app.post('/withdraw', requireLogin, async (req, res) => {\n  const updated = await Account.findOneAndUpdate(\n    { _id: req.user.accountId, balance: { $gte: req.body.amount } },\n    { $inc: { balance: -req.body.amount } },\n    { new: true }\n  );\n  if (!updated) return res.sendStatus(400);\n  await transfer(req.body.amount);\n  res.json({ ok: true });\n});\n// balance withdraw",
         "plain"),
        ("app.post('/vote', requireLogin, async (req, res) => {\n  const post = await Post.findById(req.body.postId);\n  if (post.voters.includes(req.user.id)) return res.sendStatus(400);\n  post.voters.push(req.user.id);\n  post.score += 1;\n  await post.save();\n  res.json({ ok: true });\n});\n// vote endpoint",
         "app.post('/vote', requireLogin, async (req, res) => {\n  const r = await Post.updateOne({ _id: req.body.postId, voters: { $ne: req.user.id } }, { $inc: { score: 1 }, $push: { voters: req.user.id } });\n  if (r.modifiedCount === 0) return res.sendStatus(400);\n  res.json({ ok: true });\n});\n// vote endpoint",
         "insufficient"),
    ],
    "INFO_DISCLOSURE": [
        ("app.use((err, req, res, next) => {\n  res.status(500).json({ message: err.message, stack: err.stack, code: err.code });\n});\n// error handler",
         "app.use((err, req, res, next) => {\n  console.error(err);\n  res.status(500).json({ message: 'internal error', requestId: req.id });\n});\n// error handler",
         "plain"),
        ("app.get('/_debug', (req, res) => {\n  res.json({ env: process.env, uptime: process.uptime(), memory: process.memoryUsage() });\n});\n// debug info",
         "app.get('/_debug', requireAdmin, (req, res) => {\n  res.json({ uptime: process.uptime(), memory: process.memoryUsage() });\n});\n// debug info",
         "plain"),
        ("app.get('/version', (req, res) => {\n  res.json({ version: process.env.npm_package_version, git: process.env.GIT_SHA, node: process.version, db: process.env.DATABASE_URL });\n});\n// version endpoint",
         "app.get('/version', (req, res) => {\n  res.json({ version: process.env.npm_package_version });\n});\n// version endpoint",
         "insufficient"),
    ],
}


def add_comments(code, n_top, n_bottom, rng):
    lines = code.split("\n")
    top = [rng.choice(COMMENTS) for _ in range(n_top)]
    bottom = [rng.choice(COMMENTS) for _ in range(n_bottom)]
    return "\n".join(top + lines + bottom)


def gen_pairs(cls, n_pairs, rng, start_idx=1):
    tpls = TEMPLATES[cls]
    out = []
    for i in range(n_pairs):
        tpl_v, tpl_s, subtype = rng.choice(tpls)
        n_top = rng.randint(1, 2)
        n_bot = rng.randint(0, 2)
        v = add_comments(tpl_v, n_top, n_bot, rng)
        s = add_comments(tpl_s, n_top, n_bot, rng)
        pid = f"train-{cls}-{start_idx + i:02d}"
        out.append({"pair_id": pid, "set": "train", "vuln_type": cls, "subtype": subtype,
                    "id": f"{pid}-v", "label": cls, "role": "vuln", "code": v})
        out.append({"pair_id": pid, "set": "train", "vuln_type": cls, "subtype": subtype,
                    "id": f"{pid}-s", "label": "SAFE", "role": "safe", "code": s})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=20, help="pairs per class")
    ap.add_argument("--classes", default="all", help="comma list or 'all'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="", help="output file (empty = stdout)")
    ap.add_argument("--append", default="", help="append to existing file")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    classes = list(TEMPLATES) if args.classes == "all" else [c.strip() for c in args.classes.split(",")]
    all_rows = []
    for cls in classes:
        rows = gen_pairs(cls, args.n_pairs, rng)
        all_rows.extend(rows)
        print(f"{cls}: {len(rows)} rows", flush=True)
    print(f"TOTAL: {len(all_rows)} rows ({len(all_rows)//2} pairs)")
    if args.append:
        if os.path.exists(args.append):
            with open(args.append, "rb+") as f:
                f.seek(0, 2)
                if f.tell() > 0:
                    f.seek(-1, 2)
                    if f.read(1) != b"\n":
                        f.seek(0, 2)
                        f.write(b"\n")
        with open(args.append, "a", encoding="utf-8") as f:
            for r in all_rows:
                f.write(json.dumps(r) + "\n")
        print(f"appended to {args.append}")
    elif args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in all_rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {args.out}")
    else:
        for r in all_rows:
            print(json.dumps(r))


if __name__ == "__main__":
    main()
