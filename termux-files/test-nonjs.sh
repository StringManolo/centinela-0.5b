#!/usr/bin/env bash
set -e
SERVER="http://127.0.0.1:8192"
LABELS_CSV="XSS, SQLI, CMDI, CODE_INJECTION, PATH_TRAVERSAL, SSRF, PROTO_POLLUTION, OPEN_REDIRECT, AUTHZ, WEAK_CRYPTO, SSTI, CSRF, XXE, LDAP_INJECTION, HPP, CRLF_INJECTION, CSSI, HTMLI, NOSQLI, HARDCODED_SECRETS, MASS_ASSIGNMENT, WEAK_COOKIE, INSECURE_JWT, RACE_CONDITION, INFO_DISCLOSURE, SAFE"

test_case() {
  local id="$1" lang="$2" expected="$3" code="$4"
  local request output
  request=$(python3 - "$code" "$LABELS_CSV" <<'PY'
import json, sys
code, labels = sys.argv[1], sys.argv[2]
prompt = f"""Analyze the following JavaScript code for security vulnerabilities.
If it is vulnerable, answer with the vulnerability type; if it is not, answer SAFE.
Answer with exactly one of: {labels}.

Code:
```javascript
{code}
```"""
print(json.dumps({"messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 8, "temperature": 0}))
PY
)
  output=$(curl -sS -X POST "$SERVER/v1/chat/completions" -H 'Content-Type: application/json' -d "$request" \
    | python3 -c "import json,sys; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'].strip())")
  if echo "$output" | grep -qi "$expected"; then
    echo "✅ [$lang] $id — got $output"
  elif echo "$output" | grep -qi "SAFE"; then
    echo "❌ [$lang] $id — got SAFE (missed)"
  else
    echo "⚠️  [$lang] $id — got '$output' (expected $expected)"
  fi
}

test_case "php_sqli_concat" "php" "SQLI" '<?php
$id = $_GET["id"];
$sql = "SELECT * FROM users WHERE id = " . $id;
$result = mysqli_query($conn, $sql);
?>'

test_case "php_cmdi" "php" "CMDI" '<?php
$host = $_GET["host"];
system("ping -c 1 " . $host);
?>'

test_case "python_pickle" "python" "CODE_INJECTION" 'import pickle
from flask import Flask, request
app = Flask(__name__)

@app.route("/load", methods=["POST"])
def load():
    return pickle.loads(request.data)
'

test_case "java_sqli" "java" "SQLI" 'public class UserDao {
    public User find(String id) throws SQLException {
        String sql = "SELECT * FROM users WHERE id = " + id;
        Statement st = conn.createStatement();
        ResultSet rs = st.executeQuery(sql);
        return mapUser(rs);
    }
}'

test_case "go_pathtraversal" "go" "PATH_TRAVERSAL" 'package main
import ("net/http"; "os"; "path/filepath")

func handler(w http.ResponseWriter, r *http.Request) {
    name := r.URL.Query().Get("file")
    data, _ := os.ReadFile(filepath.Join("/srv/uploads", name))
    w.Write(data)
}'

test_case "c_strcpy" "c" "CODE_INJECTION" '#include <string.h>
int main(int argc, char **argv) {
    char buf[64];
    strcpy(buf, argv[1]);
    return 0;
}'

test_case "php_safe" "php" "SAFE" '<?php
function add($a, $b) {
    return $a + $b;
}
$conn->query("SELECT * FROM users WHERE id = ?", [$id]);
?>'

test_case "python_safe" "python" "SAFE" 'def add(a, b):
    return a + b

def is_valid_email(s):
    return "@" in s and "." in s
'

test_case "java_safe" "java" "SAFE" 'public class MathHelper {
    public int add(int a, int b) { return a + b; }
    public int multiply(int a, int b) { return a * b; }
}'

test_case "go_safe" "go" "SAFE" 'package main
func add(a, b int) int {
    return a + b
}
'
