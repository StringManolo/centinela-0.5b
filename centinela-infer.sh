#!/usr/bin/env bash
# ============================================================================
# centinela-infer.sh — Classify code with the Centinela model.
#
# Uses the exact prompt template the model was trained on (ChatML wrapped,
# greedy sampling). Ships with the project so users don't need to know
# about the prompt format, the label list, or the sampler params.
#
# Usage:
#   ./centinela-infer.sh --file src/auth.js
#   ./centinela-infer.sh --code 'const q = "SELECT ..." + req.query.id'
#   echo 'const x = req.query.id' | ./centinela-infer.sh
#   ./centinela-infer.sh --files a.js b.js c.js
#   ./centinela-infer.sh serve          # long-running server for other tools
#
# Modes:
#   --backend server   (default) spawn a temporary llama-server
#   --backend cli      use llama-cli directly, no server
#
# Options:
#   --model PATH       Path to GGUF file
#   --model-dir DIR    Directory to auto-detect GGUF (default: ./centinela-models)
#   --quant Q          Preferred quantization (default: any .gguf found)
#   --server-url URL   Use an existing server instead of spawning
#   --port N           Server port (default: 8192)
#   --threads N        CPU threads (default: 3)
#   --threads-batch N  Batch threads (default: 3)
#   --ctx N            Context length (default: 4096)
#   --labels PATH      labels.json (default: ./labels.json)
#   --json             Emit JSON instead of TSV
#   --quiet            Only print the label
#   --help, -h         Show this help
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Defaults (override with env vars or CLI flags)
# ---------------------------------------------------------------------------
LABELS_JSON="${CENTINELA_LABELS:-$SCRIPT_DIR/labels.json}"
MODEL_DIR="${CENTINELA_MODEL_DIR:-$SCRIPT_DIR/centinela-models}"
MODEL=""
QUANT=""
BACKEND="server"
SERVER_URL=""
PORT="${CENTINELA_PORT:-8192}"
THREADS="${CENTINELA_THREADS:-3}"
THREADS_BATCH="${CENTINELA_THREADS_BATCH:-3}"
CTX="${CENTINELA_CTX:-4096}"
FORMAT="tsv"       # tsv | json | quiet
INPUT_MODE=""      # file | files | code | stdin | serve
INPUT_ARG=""
FILES=()
SERVER_PID=""
SPAWNED_SERVER=false

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
print_help() {
    sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        serve|--serve)  INPUT_MODE="serve"; shift ;;
        --file)         INPUT_MODE="file";  INPUT_ARG="$2"; shift 2 ;;
        --code)         INPUT_MODE="code";  INPUT_ARG="$2"; shift 2 ;;
        --files)        INPUT_MODE="files"; shift
                        while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                            FILES+=("$1"); shift
                        done ;;
        --model)        MODEL="$2"; shift 2 ;;
        --model-dir)    MODEL_DIR="$2"; shift 2 ;;
        --quant)        QUANT="$2"; shift 2 ;;
        --backend|--mode) BACKEND="$2"; shift 2 ;;
        --server-url)   SERVER_URL="$2"; shift 2 ;;
        --port)         PORT="$2"; shift 2 ;;
        --threads)      THREADS="$2"; shift 2 ;;
        --threads-batch) THREADS_BATCH="$2"; shift 2 ;;
        --ctx)          CTX="$2"; shift 2 ;;
        --labels)       LABELS_JSON="$2"; shift 2 ;;
        --json)         FORMAT="json"; shift ;;
        --quiet)        FORMAT="quiet"; shift ;;
        --help|-h)      print_help; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; echo "try: $0 --help" >&2; exit 1 ;;
    esac
done

# Default input: stdin (if something is piped), else error
if [[ -z "$INPUT_MODE" ]]; then
    if [[ ! -t 0 ]]; then
        INPUT_MODE="stdin"
    else
        echo "error: no input. Use --file, --code, --files, or pipe code via stdin." >&2
        echo "try: $0 --help" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Model detection
# ---------------------------------------------------------------------------
detect_model() {
    if [[ -n "$MODEL" ]]; then
        if [[ ! -f "$MODEL" ]]; then
            echo "error: model not found: $MODEL" >&2
            exit 1
        fi
        echo "$MODEL"
        return
    fi

    local dirs=("$MODEL_DIR" "$SCRIPT_DIR/centinela-models" "$SCRIPT_DIR/models" ".")
    local pattern="*.gguf"
    [[ -n "$QUANT" ]] && pattern="*${QUANT}*.gguf"

    for d in "${dirs[@]}"; do
        [[ -d "$d" ]] || continue
        local found
        found=$(find "$d" -maxdepth 1 -name "$pattern" -type f 2>/dev/null | head -1)
        if [[ -n "$found" ]]; then
            echo "$found"
            return
        fi
    done

    echo "error: no .gguf model found." >&2
    echo "  Looked in: ${dirs[*]}" >&2
    echo "  Use --model /path/to/model.gguf to specify explicitly." >&2
    exit 1
}

MODEL_PATH=$(detect_model)
echo "model: $MODEL_PATH" >&2

# ---------------------------------------------------------------------------
# Prompt builder (exact training template + ChatML)
# ---------------------------------------------------------------------------
build_prompt() {
    local code="$1"
    python3 - "$LABELS_JSON" "$code" <<'PYEOF'
import json, sys
labels = json.load(open(sys.argv[1]))['labels']
code = sys.argv[2]
inner = (
    "Analyze the following JavaScript code for security vulnerabilities.\n"
    "If it is vulnerable, answer with the vulnerability type; if it is not, answer SAFE.\n"
    f"Answer with exactly one of: {', '.join(labels)}.\n\n"
    "Code:\n"
    "```javascript\n"
    f"{code}\n"
    "```"
)
print(
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n" + inner + "<|im_end|>\n"
    "<|im_start|>assistant\n",
    end="",
)
PYEOF
}

# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------
server_is_up() {
    curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1
}

spawn_server() {
    if server_is_up; then
        echo "using existing server on port $PORT" >&2
        SPAWNED_SERVER=false
        return 0
    fi

    echo "spawning llama-server on port $PORT (model: $(basename "$MODEL_PATH"))" >&2
    llama-server \
        -m "$MODEL_PATH" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --threads "$THREADS" \
        --threads-batch "$THREADS_BATCH" \
        -c "$CTX" \
        -b 1024 \
        -ub 512 \
        -n 24 \
        --parallel 1 \
        --cache-ram 512 \
        --cache-type-k q4_0 \
        --cache-type-v f16 \
        --no-context-shift \
        --no-mmproj \
        --no-webui \
        --offline \
        --log-colors off \
        --temp 0.0 \
        --top-k 1 \
        --top-p 1.0 \
        --min-p 0.0 \
        --repeat-penalty 1.0 \
        --repeat-last-n 0 \
        >/dev/null 2>&1 &
    SERVER_PID=$!
    SPAWNED_SERVER=true

    local t0=$SECONDS
    while ! server_is_up; do
        if (( SECONDS - t0 > 180 )); then
            echo "error: server failed to start within 180s" >&2
            kill "$SERVER_PID" 2>/dev/null || true
            exit 1
        fi
        sleep 1
    done
    echo "server ready in $((SECONDS - t0))s" >&2
}

kill_server() {
    if [[ "$SPAWNED_SERVER" == "true" && -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap kill_server EXIT

# ---------------------------------------------------------------------------
# Classification — server backend
# ---------------------------------------------------------------------------
classify_server() {
    local code="$1"
    local prompt
    prompt=$(build_prompt "$code")

    local payload
    payload=$(python3 - "$prompt" <<'PYEOF'
import json, sys
print(json.dumps({
    "prompt": sys.argv[1],
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "min_p": 0.0,
    "n_predict": 24,
    "repeat_penalty": 1.0,
    "repeat_last_n": 0,
    "stop": ["<|im_end|>"],
    "logprobs": True,
    "top_logprobs": 10,
}))
PYEOF
)

    local url="${SERVER_URL:-http://127.0.0.1:$PORT/completion}"
    curl -sf -X POST "$url" \
        -H 'Content-Type: application/json' \
        -d "$payload" \
        -o /tmp/centinela-infer-resp.json \
        || { echo "error: server request failed" >&2; return 1; }

    python3 - "$LABELS_JSON" /tmp/centinela-infer-resp.json <<'PYEOF'
import json, sys, math
labels = json.load(open(sys.argv[1]))['labels']
resp = json.load(open(sys.argv[2]))
content = (resp.get("content") or "").strip()
cprobs = resp.get("completion_probabilities") or []

# Parse emitted label
norm = content.upper().replace(" ", "_").replace("-", "_")
label = None
for L in labels:
    if norm == L.upper() or norm.startswith(L.upper()):
        label = L
        break
if label is None:
    label = "UNKNOWN"

# Confidence = geometric mean of the emitted label's per-token probs
conf = 0.0
if cprobs and label != "UNKNOWN":
    cumulative = ""
    sum_lp = 0.0
    n = 0
    for cp in cprobs:
        piece = cp.get("content", "")
        if "logprob" in cp:
            lp = float(cp["logprob"])
        elif "prob" in cp:
            lp = math.log(max(float(cp["prob"]), 1e-10))
        else:
            lp = 0.0
        cumulative += piece
        sum_lp += lp
        n += 1
        cnorm = cumulative.upper().replace(" ", "_").replace("-", "_")
        if cnorm == label.upper() or cnorm.startswith(label.upper()):
            break
    if n > 0:
        conf = math.exp(sum_lp / n)

print(f"{label}\t{conf:.4f}")
PYEOF
}

# ---------------------------------------------------------------------------
# Classification — CLI backend
# ---------------------------------------------------------------------------
classify_cli() {
    local code="$1"
    local prompt
    prompt=$(build_prompt "$code")

    local out
    out=$(printf '%s' "$prompt" | llama-cli \
        -m "$MODEL_PATH" \
        -n 24 \
        --temp 0.0 \
        --top-k 1 \
        --top-p 1.0 \
        --min-p 0.0 \
        --repeat-penalty 1.0 \
        --repeat-last-n 0 \
        --no-display-prompt \
        --log-disable \
        -no-cnv \
        2>/dev/null) || true

    python3 - "$LABELS_JSON" "$out" <<'PYEOF'
import json, sys
labels = json.load(open(sys.argv[1]))['labels']
content = sys.argv[2].strip()
norm = content.upper().replace(" ", "_").replace("-", "_")
label = "UNKNOWN"
for L in labels:
    if norm == L.upper() or norm.startswith(L.upper()):
        label = L
        break
# CLI mode has no logprobs: report 0.0 as "unknown confidence"
print(f"{label}\t0.0000")
PYEOF
}

# ---------------------------------------------------------------------------
# Dispatch + output
# ---------------------------------------------------------------------------
run_one() {
    local code="$1"
    local path_label="$2"

    local result
    if [[ "$BACKEND" == "cli" ]]; then
        result=$(classify_cli "$code") || result="ERROR\t0.0000"
    else
        result=$(classify_server "$code") || result="ERROR\t0.0000"
    fi

    local label conf
    label=$(echo "$result" | cut -f1)
    conf=$(echo "$result" | cut -f2)

    case "$FORMAT" in
        json)
            python3 -c "
import json
print(json.dumps({'path': '$path_label', 'label': '$label', 'confidence': float('$conf')}))
"
            ;;
        quiet)
            echo "$label"
            ;;
        *)
            printf '%s\t%s\t%s\n' "$path_label" "$label" "$conf"
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Serve (long-running foreground server)
# ---------------------------------------------------------------------------
if [[ "$INPUT_MODE" == "serve" ]]; then
    echo "Starting Centinela server on http://127.0.0.1:$PORT (Ctrl+C to stop)" >&2
    exec llama-server \
        -m "$MODEL_PATH" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --threads "$THREADS" \
        --threads-batch "$THREADS_BATCH" \
        -c "$CTX" \
        -b 1024 \
        -ub 512 \
        -n 24 \
        --parallel 1 \
        --cache-ram 512 \
        --cache-type-k q4_0 \
        --cache-type-v f16 \
        --no-context-shift \
        --no-mmproj \
        --no-webui \
        --offline \
        --log-colors off \
        --temp 0.0 \
        --top-k 1 \
        --top-p 1.0 \
        --min-p 0.0 \
        --repeat-penalty 1.0 \
        --repeat-last-n 0
fi

# ---------------------------------------------------------------------------
# Prepare backend
# ---------------------------------------------------------------------------
if [[ "$BACKEND" == "server" ]]; then
    spawn_server
fi

# ---------------------------------------------------------------------------
# Classify input
# ---------------------------------------------------------------------------
if [[ "$FORMAT" == "json" ]]; then
    echo "["
    FIRST=true
fi

emit_json_item() {
    local item="$1"
    if [[ "$FORMAT" == "json" ]]; then
        if [[ "$FIRST" == "true" ]]; then FIRST=false; else echo ","; fi
        printf '  %s' "$item"
    fi
}

case "$INPUT_MODE" in
    file)
        if [[ ! -f "$INPUT_ARG" ]]; then
            echo "error: file not found: $INPUT_ARG" >&2
            exit 1
        fi
        code=$(cat "$INPUT_ARG")
        out=$(run_one "$code" "$INPUT_ARG")
        if [[ "$FORMAT" == "json" ]]; then emit_json_item "$out"; else echo "$out"; fi
        ;;

    files)
        for f in "${FILES[@]}"; do
            if [[ ! -f "$f" ]]; then
                echo "warning: skipping missing file: $f" >&2
                continue
            fi
            code=$(cat "$f")
            out=$(run_one "$code" "$f")
            if [[ "$FORMAT" == "json" ]]; then emit_json_item "$out"; else echo "$out"; fi
        done
        ;;

    code)
        out=$(run_one "$INPUT_ARG" "<code>")
        if [[ "$FORMAT" == "json" ]]; then emit_json_item "$out"; else echo "$out"; fi
        ;;

    stdin)
        code=$(cat)
        out=$(run_one "$code" "<stdin>")
        if [[ "$FORMAT" == "json" ]]; then emit_json_item "$out"; else echo "$out"; fi
        ;;
esac

if [[ "$FORMAT" == "json" ]]; then
    echo
    echo "]"
fi
