#!/usr/bin/env bash
# Wait for the retry batch to finish, then run eval_mascarade_lora.py
# on all 10 published LoRA and push the real Qwen3-4B bench numbers
# to each HF model card.

set -uo pipefail

LOG_ROOT="/tmp/ship_qwen3_mascarade"
RETRY_LOG="$LOG_ROOT/_master_retry.log"
EVAL_LOG="/tmp/eval_mascarade_outer.log"

echo "=== chainer started $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$EVAL_LOG"

# Poll until retry batch wrote its "=== retry done" footer.
while ! grep -q "=== retry done" "$RETRY_LOG" 2>/dev/null; do
    sleep 30
    if ! pgrep -lf ship_qwen3_mascarade_retry > /dev/null; then
        # Master script gone but no done line? Maybe it crashed.
        if grep -q "=== retry done" "$RETRY_LOG" 2>/dev/null; then
            break
        fi
        echo "  retry script gone without done line — checking _master_retry.log tail:" | tee -a "$EVAL_LOG"
        tail -10 "$RETRY_LOG" >> "$EVAL_LOG"
        break
    fi
done

echo "=== retry batch finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$EVAL_LOG"
echo "--- master_retry tail ---" | tee -a "$EVAL_LOG"
tail -30 "$RETRY_LOG" | tee -a "$EVAL_LOG"

# Switch to venv and run eval.
cd "$HOME/ailiance-models-tuning" || exit 1
source .venv/bin/activate || exit 1

echo "" | tee -a "$EVAL_LOG"
echo "=== eval_mascarade_lora.py start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$EVAL_LOG"
python /tmp/eval_mascarade_lora.py --samples 10 --update-cards 2>&1 | tee -a "$EVAL_LOG"
RC=$?
echo "=== eval finished rc=$RC at $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$EVAL_LOG"
exit $RC
