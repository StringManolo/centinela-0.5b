#!/usr/bin/env bash
set -euo pipefail

# Lista de modelos de <= 0.5B (abliterated / heretic)
MODELS=(
  "huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3"
  "saidutta69/Qwen2.5-Coder-0.5B-Instruct-heretic"
  "vlx1/Qwen2.5-0.5B-Instruct-heretic"
  "dalatexcoder/Qwen2.5-0.5B-Instruct-heretic-test-v2"
  "mradermacher/Qwen2.5-0.5B-Instruct-abliterated-GGUF"
  "mradermacher/Qwen2.5-Coder-0.5B-Instruct-abliterated-GGUF"
  "mradermacher/Qwen2-0.5B-Instruct-heretic-GGUF"
  "isotnek/SmolLM2-360M-Instruct-heretic"
  "TiGa-RCE/SmolLM2-360M-OBLITERATED"
  "mradermacher/SmolLM2-360M-OBLITERATED-GGUF"
  "usermma/Atomight-V2.2-UltraThink-0.5B-abliterated"
  "mradermacher/Atomight-V2.2-UltraThink-0.5B-abliterated-GGUF"
  "mradermacher/Ektome-Qwen1.5-0.5B-Chat-PristinelyUncensored-GGUF"
)

# Opcional: exporta HF_TOKEN si necesitas comprobar modelos gated
HF_TOKEN="${HF_TOKEN:-}"

echo "Validating ${#MODELS[@]} model(s) on Hugging Face (<= 0.5B)..."
echo ""

for model in "${MODELS[@]}"; do
    url="https://huggingface.co/api/models/${model}"
    if [ -n "$HF_TOKEN" ]; then
        code=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $HF_TOKEN" "$url")
    else
        code=$(curl -s -o /dev/null -w "%{http_code}" "$url")
    fi

    case "$code" in
        200) status="✅ EXISTS" ;;
        401|403) status="🔒 GATED/PRIVATE" ;;
        404) status="❌ NOT FOUND" ;;
        *) status="⚠️  HTTP $code" ;;
    esac

    printf "%-65s %s\n" "$model" "$status"
done
