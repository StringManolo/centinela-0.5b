#!/usr/bin/env bash
# Convert a PEFT LoRA adapter + base model into a GGUF file.
# Runs only when the workflow is dispatched with convert_to_gguf=true.
set -euo pipefail

BASE_MODEL="huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3"
ADAPTER_DIR="./lora_adapter"
MERGED_DIR="./merged_model"
GGUF_F16="./centinela-0.5b-f16.gguf"
GGUF_Q4="./centinela-0.5b-Q4_K_M.gguf"
LLAMA_CPP_DIR="/tmp/llama.cpp"

echo "[1/5] Cloning llama.cpp (shallow)..."
rm -rf "$LLAMA_CPP_DIR"
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"

echo "[2/5] Installing conversion dependencies..."
pip install -r "$LLAMA_CPP_DIR/requirements.txt"

echo "[3/5] Merging LoRA adapter into the base model..."
python3 - <<'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = "huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3"
adapter = "./lora_adapter"
out = "./merged_model"

print(f"Loading base model: {base}")
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(base)

print(f"Loading LoRA adapter: {adapter}")
model = PeftModel.from_pretrained(model, adapter)

print("Merging weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {out}...")
model.save_pretrained(out)
tokenizer.save_pretrained(out)
print("Merge done.")
PY

echo "[4/5] Converting merged model to GGUF (F16)..."
python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MERGED_DIR" \
  --outfile "$GGUF_F16" --outtype f16
echo "F16 GGUF written: $GGUF_F16"

echo "[5/5] Building llama-quantize and producing Q4_K_M..."
cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$LLAMA_CPP_DIR/build" --target llama-quantize -j4 >/dev/null
"$LLAMA_CPP_DIR/build/bin/llama-quantize" "$GGUF_F16" "$GGUF_Q4" Q4_K_M
echo "Q4_K_M GGUF written: $GGUF_Q4"

echo "Done. Files:"
ls -lh "$GGUF_F16" "$GGUF_Q4"
