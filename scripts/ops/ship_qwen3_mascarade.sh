#!/usr/bin/env bash
# Batch training for Qwen3-4B mascarade LoRA family on kxkm-ai.
# Designed to run AS kxkm user on kxkm-ai itself (no ssh wrapping).
# Granite-30B may still hold 19 GB VRAM — we rely on bitsandbytes
# llm_int8_enable_fp32_cpu_offload=True (default in train_sft.py per the
# repo CLAUDE.md), so the 4 GB free VRAM + 50 GB CPU RAM should be enough,
# at the cost of ~2-3x slower training per step.

set -uo pipefail

PROJECT_ROOT="$HOME/ailiance-models-tuning"
cd "$PROJECT_ROOT"
source .venv/bin/activate

LOG_ROOT="/tmp/ship_qwen3_mascarade"
mkdir -p "$LOG_ROOT"
MASTER_LOG="$LOG_ROOT/_master.log"

DOMAINS=(kicad spice stm32 emc embedded platformio freecad dsp iot power)
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"

echo "=== START $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$MASTER_LOG"
echo "Base: $BASE_MODEL" | tee -a "$MASTER_LOG"
echo "Domains: ${DOMAINS[*]}" | tee -a "$MASTER_LOG"

for domain in "${DOMAINS[@]}"; do
    echo "" | tee -a "$MASTER_LOG"
    echo "================== $domain  ($(date -u +%H:%M:%SZ)) ==================" | tee -a "$MASTER_LOG"

    OUT_DIR="$PROJECT_ROOT/outputs/qwen3-4b-mascarade-$domain"
    DATASET_DIR="$PROJECT_ROOT/datasets/processed"
    DATASET_FILE="$DATASET_DIR/${domain}_chat.jsonl"
    mkdir -p "$DATASET_DIR" "$OUT_DIR"
    DOMAIN_LOG="$LOG_ROOT/$domain.log"

    # 1. Pull dataset from HF Ailiance-fr.
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

    # 2. Train. epochs=1 first pass to stay reasonable.
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

    # 3. Eval token-overlap (optional).
    echo "[3/3] eval_adapters" | tee -a "$MASTER_LOG"
    python scripts/eval_adapters.py \
        --adapter "$OUT_DIR" \
        --domain "$domain" \
        >> "$DOMAIN_LOG" 2>&1 || echo "  eval failed (non-fatal)" | tee -a "$MASTER_LOG"
done

echo "" | tee -a "$MASTER_LOG"
echo "=== done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$MASTER_LOG"
