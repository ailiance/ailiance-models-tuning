#!/usr/bin/env bash
# Build all domain datasets and validate them.
# Run from the ailiance-models-tuning project root.
#
# Usage:
#   ./scripts/build_all_datasets.sh               # Seeds only (default)
#   ./scripts/build_all_datasets.sh --with-hf     # Seeds + HuggingFace data
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

HF_FLAG=""
if [[ "${1:-}" == "--with-hf" ]]; then
    HF_FLAG="--with-hf"
fi

echo "=== Building all datasets ==="
echo "Project root: $PROJECT_ROOT"
echo ""

FAILED=0

for builder in datasets/builders/build_*.py; do
    echo "--- Running $builder..."
    if python "$builder" $HF_FLAG; then
        echo "    OK"
    else
        echo "    FAILED: $builder"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

echo "=== Validating datasets ==="
shopt -s nullglob
JSONL_FILES=(datasets/processed/*.jsonl)
shopt -u nullglob

if [[ ${#JSONL_FILES[@]} -eq 0 ]]; then
    echo "  No JSONL files found in datasets/processed/ — nothing to validate."
    exit 1
fi

VALIDATION_FAILED=0
for dataset in "${JSONL_FILES[@]}"; do
    if ! python scripts/validate_dataset.py "$dataset"; then
        VALIDATION_FAILED=$((VALIDATION_FAILED + 1))
    fi
done

echo ""
echo "=== Summary ==="
echo "Datasets written to datasets/processed/:"
ls -lh datasets/processed/*.jsonl 2>/dev/null || echo "  (none)"

if [[ $FAILED -gt 0 ]] || [[ $VALIDATION_FAILED -gt 0 ]]; then
    echo ""
    echo "FAILED: $FAILED builder(s), $VALIDATION_FAILED validation failure(s)"
    exit 1
fi

echo ""
echo "All datasets built and validated successfully."
