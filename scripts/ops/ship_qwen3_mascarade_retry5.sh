#!/usr/bin/env bash
# Retry the 5 FAILED domains from the 12:45 UTC batch.
# Run AFTER `modprobe -r nvidia && modprobe nvidia` has restored CUDA.

set -uo pipefail

PROJECT_ROOT="$HOME/ailiance-models-tuning"
cd "$PROJECT_ROOT"
source .venv/bin/activate

LOG_ROOT="/tmp/ship_qwen3_mascarade"
mkdir -p "$LOG_ROOT"
MASTER_LOG="$LOG_ROOT/_master_retry.log"

DOMAINS=(platformio freecad dsp iot power)
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"

echo "=== RETRY START $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$MASTER_LOG"

# Sanity check CUDA before starting (avoid wasting 5x 4s if mismatch persists).
python -c 'import torch; assert torch.cuda.is_available(), "CUDA not available"; print(f"CUDA OK, {torch.cuda.device_count()} device(s)")' \
    >> "$MASTER_LOG" 2>&1 || {
    echo "  ABORT: CUDA still broken — reload nvidia module first" | tee -a "$MASTER_LOG"
    exit 1
}

for domain in "${DOMAINS[@]}"; do
    echo "" | tee -a "$MASTER_LOG"
    echo "================== $domain  ($(date -u +%H:%M:%SZ)) ==================" | tee -a "$MASTER_LOG"

    OUT_DIR="$PROJECT_ROOT/outputs/qwen3-4b-mascarade-$domain"
    DATASET_DIR="$PROJECT_ROOT/datasets/processed"
    DATASET_FILE="$DATASET_DIR/${domain}_chat.jsonl"
    mkdir -p "$DATASET_DIR" "$OUT_DIR"
    DOMAIN_LOG="$LOG_ROOT/${domain}_retry.log"

    echo "[1/3] Pulling Ailiance-fr/mascarade-${domain}-dataset" | tee -a "$MASTER_LOG"
    python -c "
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download(
    repo_id='Ailiance-fr/mascarade-${domain}-dataset',
    filename='${domain}_chat.jsonl',
    repo_type='dataset',
)
shutil.copy(p, '${DATASET_FILE}')
print(f'staged {p} -> ${DATASET_FILE}')
" >> "$DOMAIN_LOG" 2>&1 || {
        echo "  FAIL dataset pull" | tee -a "$MASTER_LOG"
        continue
    }

    echo "[2/3] train_sft Qwen3-4B + ${domain}" | tee -a "$MASTER_LOG"
    T0=$(date +%s)
    python scripts/train_sft.py \
        --base-model "$BASE_MODEL" \
        --dataset "$DATASET_FILE" \
        --output-dir "$OUT_DIR" \
        --epochs 1 \
        --lora-r 16 \
        --max-seq-length 2048 \
        --push-to-hub \
        --hub-model-id "Ailiance-fr/qwen3-4b-mascarade-${domain}-lora" \
        >> "$DOMAIN_LOG" 2>&1
    RC=$?
    DT=$(( $(date +%s) - T0 ))
    if [ $RC -eq 0 ]; then
        echo "  OK $domain  (${DT}s)" | tee -a "$MASTER_LOG"
    else
        echo "  FAIL $domain  rc=$RC  (${DT}s) — see $DOMAIN_LOG" | tee -a "$MASTER_LOG"
        continue
    fi

    echo "[3/3] eval_adapters" | tee -a "$MASTER_LOG"
    python scripts/eval_adapters.py \
        --adapter "$OUT_DIR" \
        --domain "$domain" \
        >> "$DOMAIN_LOG" 2>&1 || echo "  eval failed (non-fatal)" | tee -a "$MASTER_LOG"
done

echo "" | tee -a "$MASTER_LOG"
echo "=== retry done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$MASTER_LOG"
