#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABELS_JSON="${CENTINELA_LABELS_JSON:-$SCRIPT_DIR/labels.json}"
SERVER="${CENTINELA_SERVER:-http://127.0.0.1:8192}"

if [ ! -f "$LABELS_JSON" ]; then
    echo "ERROR: labels.json not found at $LABELS_JSON" >&2
    exit 1
fi

# Extraer labels como CSV
LABELS_CSV=$(python3 -c "import json,sys; print(', '.join(json.load(open(sys.argv[1]))['labels']))" "$LABELS_JSON")
LABELS_COUNT=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['labels']))" "$LABELS_JSON")

DEBUG=false
for arg in "$@"; do
  case "$arg" in
    --debug) DEBUG=true ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
  esac
done

FAILED=0
PASSED=0

run_test() {
  local name="$1" code="$2" expect="$3"
  echo "--- $name (esperado: $expect) ---"

  REQUEST=$(python3 - "$code" "$LABELS_CSV" <<'PY'
import json, sys
code, labels_csv = sys.argv[1], sys.argv[2]
prompt = f"""Analyze the following JavaScript code for security vulnerabilities.
If it is vulnerable, answer with the vulnerability type; if it is not, answer SAFE.
Answer with exactly one of: {labels_csv}.

Code:
```javascript
{code}
```"""
print(json.dumps({
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 8,
    "temperature": 0,
}))
PY
)

  if [ "$DEBUG" = true ]; then
    echo "REQUEST BODY:"
    echo "$REQUEST" | python3 -m json.tool
    echo
  fi

  RESPONSE=$(curl -sS -X POST "$SERVER/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$REQUEST")

  if [ "$DEBUG" = true ]; then
    echo "RESPONSE (full):"
    echo "$RESPONSE" | python3 -m json.tool
    echo
  fi

  OUTPUT=$(echo "$RESPONSE" | python3 -c "
import json, sys
try:
    r = json.load(sys.stdin)
    if 'choices' in r:
        print(repr(r['choices'][0]['message']['content']))
    elif 'error' in r:
        print('ERROR: ' + str(r['error']))
    else:
        print('UNEXPECTED: ' + repr(r))
except Exception as e:
    print('PARSE_FAIL: ' + str(e))
")
  echo "OUTPUT:  $OUTPUT"

  # Match exacto de palabra (no substring: NOSQLI ≠ SQLI)
  if echo "$OUTPUT" | grep -qiw "$expect"; then
    echo "✅ OK"
    PASSED=$((PASSED + 1))
  else
    echo "❌ FALLO (esperaba $expect)"
    FAILED=$((FAILED + 1))
  fi
  echo
}

SQLI_CODE='app.get("/user", (req, res) => {
  const sql = "SELECT * FROM users WHERE id = " + req.query.id;
  db.query(sql, (err, rows) => res.json(rows));
});'

SQLI_CODE2='app.post("/login", async (req, res) => {
  const { email } = req.body;
  const rows = await db.query(`SELECT * FROM users WHERE email = '"'"'${email}'"'"'`);
  res.json(rows);
});'

XSS_CODE='function showName(name) {
  document.getElementById("welcome").innerHTML = "Hello, " + name;
}'

SAFE_CODE='function add(a, b) {
  return a + b;
}'

echo "=== Test Centinela — server: $SERVER, labels: $LABELS_COUNT ==="
echo

run_test "SQLI concat" "$SQLI_CODE"  "SQLI"
run_test "SQLI template" "$SQLI_CODE2" "SQLI"
run_test "XSS innerHTML" "$XSS_CODE"  "XSS"
run_test "SAFE (suma)" "$SAFE_CODE"  "SAFE"

echo "=== Resumen: $PASSED passed, $FAILED failed ==="
[ "$FAILED" -eq 0 ] || exit 1
