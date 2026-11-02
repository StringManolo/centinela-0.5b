#!/usr/bin/env bash
set -euo pipefail

WORKFLOW="centinela-train.yml"
LIMIT=50
OUT_DIR="ranking"

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

echo "Fetching last $LIMIT runs of $WORKFLOW..."
for run_id in $(gh run list --workflow="$WORKFLOW" --limit "$LIMIT" --json databaseId --jq '.[].databaseId'); do
    gh run download "$run_id" -n "centinela-run-$run_id" -D "run_$run_id" 2>/dev/null || true
done

echo ""
printf "%-50s %-12s %-12s %-12s %-12s %-12s\n" "MODEL" "ACC_BEFORE" "ACC_AFTER" "IMPROVEMENT" "TRAIN_S" "TOTAL_S"
echo "------------------------------------------------------------------------------------------------------------------------------------"

for f in $(find . -name metrics.json); do
    jq -r '[.base_model, .acc_before, .acc_after, .improvement, .timings.train_s, .timings.total_s] | @tsv' "$f"
done | sort -t$'\t' -k4 -nr | \
while IFS=$'\t' read -r model ab aa imp ts tt; do
    printf "%-50s %-12s %-12s %-12s %-12s %-12s\n" \
        "$(echo "$model" | cut -c1-50)" "$ab" "$aa" "$imp" "$ts" "$tt"
done
