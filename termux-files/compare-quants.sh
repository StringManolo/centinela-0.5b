#!/usr/bin/env bash
set -euo pipefail

QUANTS=("Q2_K" "Q4_K_M" "Q8_0")
MODEL_DIR="$HOME/OpenMobile/models"
OUTDIR="$HOME/OpenMobile/quant-results"
LOGDIR="$HOME/OpenMobile/logs"
mkdir -p "$OUTDIR" "$LOGDIR"

cd "$HOME/OpenMobile"

for Q in "${QUANTS[@]}"; do
  GGUF="$MODEL_DIR/Centinela0.1-0.5B-Instruct-cybersec-abliterated-${Q}.gguf"
  if [ ! -f "$GGUF" ]; then
    echo "⚠️  No existe $GGUF — saltando $Q"
    continue
  fi
  SIZE=$(du -h "$GGUF" | cut -f1)
  LOG="$LOGDIR/server-${Q}.log"

  echo
  echo "═══════════════════════════════════════════════"
  echo "  $Q  ($SIZE)"
  echo "═══════════════════════════════════════════════"

  # Matar cualquier server previo
  pkill -f llama-server 2>/dev/null || true
  sleep 2

  # Mapear quant → flag de runCentinela.sh
  case "$Q" in
    Q2_K)   FLAG="--Q2" ;;
    Q3_K_M) FLAG="--Q3" ;;
    Q4_K_M) FLAG="--Q4" ;;
    Q5_K_M) FLAG="--Q5" ;;
    Q6_K)   FLAG="--Q6" ;;
    Q8_0)   FLAG="--Q8" ;;
    F16)    FLAG="--F16" ;;
    *)      echo "quant desconocida: $Q"; continue ;;
  esac

  echo "Log: $LOG"

  # Arrancar server en background con log
  ./runCentinela.sh $FLAG --classify > "$LOG" 2>&1 &
  SERVER_PID=$!

  # Esperar a que esté listo (máx 120s)
  echo -n "Esperando server"
  READY=false
  for i in $(seq 1 60); do
    if curl -sf --max-time 2 http://127.0.0.1:8192/health > /dev/null 2>&1; then
      READY=true
      break
    fi
    # Si el proceso murió, no esperar más
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo
      echo "❌ El proceso llama-server murió. Log:"
      tail -40 "$LOG"
      break
    fi
    echo -n "."
    sleep 2
  done
  echo

  if [ "$READY" = false ]; then
    echo "❌ Server no arrancó. Últimas 40 líneas del log:"
    tail -40 "$LOG" || true
    pkill -f llama-server 2>/dev/null || true
    continue
  fi

  echo "✅ Server listo"

  # Pasar eval-samples si existe
  SAMPLES_ARG=""
  if [ -f "$HOME/OpenMobile/eval-samples.jsonl" ]; then
    SAMPLES_ARG="--samples $HOME/OpenMobile/eval-samples.jsonl"
    echo "Usando eval-samples.jsonl ($(wc -l < "$HOME/OpenMobile/eval-samples.jsonl") ejemplos)"
  fi

  # Evaluar
  python3 eval-quant.py --tag "$Q" --url http://127.0.0.1:8192 \
    --out "$OUTDIR/eval-${Q}.json" $SAMPLES_ARG || true

  # Matar server
  pkill -f llama-server 2>/dev/null || true
  sleep 3
done

# Matar cualquier server zombie al final
pkill -f llama-server 2>/dev/null || true

# Tabla comparativa
echo
echo "═══════════════════════════════════════════════"
echo "  COMPARACIÓN FINAL"
echo "═══════════════════════════════════════════════"
python3 - <<'PY'
import json, glob, os
files = sorted(glob.glob(os.path.expanduser("~/OpenMobile/quant-results/eval-*.json")))
if not files:
    print("No hay resultados.")
    raise SystemExit
print(f"{'Quant':10s} {'Hand-crafted':14s} {'Test samples':14s} {'Total time':12s}")
print("-" * 55)
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    s = d["summary"]
    hc = f"{s['hand_crafted_acc']*100:.1f}%" if s.get("hand_crafted_acc") is not None else "n/a"
    ts = f"{s['test_samples_acc']*100:.1f}%" if s.get("test_samples_acc") is not None else "n/a"
    tt = f"{s.get('total_seconds', 0):.0f}s"
    print(f"{d['tag']:10s} {hc:14s} {ts:14s} {tt:12s}")

print()
print("Detalle por clase:")
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    pca = d["summary"].get("per_class_acc", {})
    if pca:
        print(f"\n  {d['tag']}:")
        for cls in sorted(pca):
            bar = "█" * int(pca[cls] * 10)
            print(f"    {cls:22s} {pca[cls]*100:5.1f}% {bar}")
PY
