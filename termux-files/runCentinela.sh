#!/usr/bin/env bash
# runCentinela.sh — Lanza Centinela 0.5B con cuantización seleccionable por flag
# Optimizado para Huawei P40 Lite (Kirin 810, 4–6 GB RAM, CPU-only).
#
# Uso:
#   ./runCentinela.sh                 # Q4_K_M por defecto
#   ./runCentinela.sh --Q8            # cuantización Q8_0
#   ./runCentinela.sh --F16           # modelo sin cuantizar
#   ./runCentinela.sh --Q6 --threads 6
#   ./runCentinela.sh --help
#
# Cuantizaciones disponibles:
#   --Q2 → Q2_K   (mínima, ~200 MB)     --Q6 → Q6_K   (~470 MB)
#   --Q3 → Q3_K_M (~280 MB)             --Q8 → Q8_0   (~530 MB)
#   --Q4 → Q4_K_M (~350 MB) ← default   --F16 → F16   (~1.0 GB)

set -euo pipefail

# ---------- Defaults ----------
MODEL_BASENAME="Centinela0.1-0.5B-Instruct-cybersec-abliterated"
MODEL_DIR="$HOME/OpenMobile/models"
QUANT="Q4_K_M"
THREADS=3            # 2× A76 + 1× A55: mejor que 8 (los A55 pequeños saturan)
# 2 === 4.5 t/s
# 3 === 7.5 t/s
# 4 === 7.0 t/s
# 5 === 6.5 t/s
THREADS_BATCH=2      # solo A76 para prefill (menos throttling térmico)
# 1 === 7.0 t/s
# 2 === 7.5 t/s
# 3 === 7.1 t/s 
CTX=8192             # 2k: sobra para clasificación de un snippet + sistema
PORT=8192
HOST="127.0.0.1"
NP=1                 # un solo slot: 1 usuario en móvil
BATCH=512
UBATCH=256
NPREDICT=-1           # un label de 1–4 tokens; 8 es margen de seguridad
CACHE_RAM=256        # MiB de caché de KV slots (móvil: no queremos 8 GiB)
CACHE_K=q8_0
# q8_0 === 7.5 t/s
# f16 === 7.2 t/s
CACHE_V=f16
CONFIG="./cyber.json"
SAMPLING="--temp 0.85 --top-p 0.95 --top-k 40 --min-p 0.05 --repeat-penalty 1.1 --repeat-last-n 256"
# q8_0 === 7.5 t/s
# f16 === 7.6 t/s

# ---------- Parseo de flags ----------
while [ $# -gt 0 ]; do
  case "$1" in
    --Q2)  QUANT="Q2_K";   shift ;;
    --Q3)  QUANT="Q3_K_M"; shift ;;
    --Q4)  QUANT="Q4_K_M"; shift ;;
    --Q5)  QUANT="Q5_K_M"; shift ;;
    --Q6)  QUANT="Q6_K";   shift ;;
    --Q8)  QUANT="Q8_0";   shift ;;
    --F16) QUANT="F16";    shift ;;
    --threads)       THREADS="$2";       shift 2 ;;
    --threads-batch) THREADS_BATCH="$2"; shift 2 ;;
    --ctx)           CTX="$2";           shift 2 ;;
    --port)          PORT="$2";          shift 2 ;;
    --host)          HOST="$2";          shift 2 ;;
    --classify)      CONFIG="./centinela-classify.json"
                     SAMPLING="--temp 0.0 --top-k 1 --top-p 1.0 --min-p 0.0 --repeat-penalty 1.0 --repeat-last-n 0"
                     shift ;;
    --np)            NP="$2";            shift 2 ;;
    -h|--help)
      sed -n '2,18p' "$0"; exit 0 ;;
    *)
      echo "Flag desconocido: $1 (usa --help)" >&2; exit 1 ;;
  esac
done

MODEL_PATH="${MODEL_DIR}/${MODEL_BASENAME}-${QUANT}.gguf"
if [ ! -f "$MODEL_PATH" ]; then
  echo "ERROR: no se encuentra $MODEL_PATH" >&2
  echo "Descarga el GGUF desde el release y colócalo en ${MODEL_DIR}/" >&2
  exit 1
fi

SIZE=$(du -h "$MODEL_PATH" | cut -f1)
echo "Centinela 0.5B — ${QUANT} (${SIZE})"
echo "  modelo:  ${MODEL_PATH}"
echo "  threads: ${THREADS}  batch: ${THREADS_BATCH}"
echo "  ctx:     ${CTX}      server: http://${HOST}:${PORT}"
echo

exec llama-server \
  -m "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --threads "${THREADS}" \
  --threads-batch "${THREADS_BATCH}" \
  -c "${CTX}" \
  -b "${BATCH}" \
  -ub "${UBATCH}" \
  -n "${NPREDICT}" \
  --parallel "${NP}" \
  --cache-ram "${CACHE_RAM}" \
  --cache-type-k "${CACHE_K}"\
  --cache-type-v "${CACHE_V}" \
  --no-context-shift \
  --webui-config-file "${CONFIG}" \
  --no-mmproj \
  --log-colors off \
  $SAMPLING \
  --offline \


# --offline

# T/S Eval per flag
  # --jinja === 7.6 t/s
  # without === 8.1 t/s





# STATIC ANSWERS:
#  --temp 0.0 \
#  --top-k 1 \
#  --top-p 1.0 \
#  --min-p 0.0 \
#  --repeat-penalty 1.0 \
#  --repeat-last-n 0 \
#  --presence-penalty 0.0 \
#  --frequency-penalty 0.0 \

