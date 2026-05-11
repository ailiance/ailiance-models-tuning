# RECIPES — writing a YAML training config

Each fine-tune is driven by one YAML file in `configs/`. This document is the human-facing guide for writing one. For agent-facing rules (required fields, versioning discipline), see `configs/CLAUDE.md`.

## Minimal recipe

```yaml
# configs/sft_<domain>.yaml
base_model: Qwen/Qwen2.5-32B-Instruct
dataset:    datasets/processed/<domain>_train.jsonl
output_dir: outputs/sft-<domain>
epochs:     3
batch_size: 1                         # per-device; use 1 for 32B on RTX 4090
lr:         0.0002
lora_r:     16
max_seq_length: 2048
hub_model_id:   clemsail/kiki-<domain>-sft
```

Shape mirrors `TrainingConfig` in `src/ailiance_tuning/config.py`. Required fields are hard-fail on missing: `base_model`, `dataset`, `output_dir`. All others have sensible defaults.

## Field guide

| Field | What it controls | Typical value |
|---|---|---|
| `base_model` | HF model ID for the base (LoRA is base-specific) | `Qwen/Qwen2.5-32B-Instruct` or `Qwen/Qwen3-8B` |
| `dataset` | JSONL path (relative to repo root) | `datasets/processed/stm32_train.jsonl` |
| `output_dir` | Where adapters land | `outputs/sft-<domain>` |
| `epochs` | Full passes over dataset | 2-4; over-tune degrades generalization |
| `batch_size` | `per_device_train_batch_size` | 1 for 32B on 4090, up to 4 for 8B |
| `lr` | Learning rate | 2e-4 is the default; drop to 1e-4 for noisy seeds |
| `lora_r` | LoRA rank | 8 for small domains, 16 for niche (see `DOMAINS.md`) |
| `max_seq_length` | Max tokens per example | 2048 for chat, 4096 for code-heavy |
| `hub_model_id` | HF Hub slug | `clemsail/kiki-<domain>-sft` |

## Rank vs alpha

`train_sft.py` currently **hardcodes `alpha = 2 × r`**. If you change the ratio, edit the trainer too — otherwise the YAML value is silently overridden.

## Naming convention

- `sft_<domain>.yaml` — one-shot, default base
- `sft_<domain>_<basemodel-tag>.yaml` — when multiple bases coexist (e.g. `sft_stm32_qwen25_32b.yaml`)

No dataset hashes or timestamps in filenames — the YAML file's git history carries that.

## Examples by domain size

```yaml
# Large domain (>8k examples): embedded, kicad, spice, power
epochs: 2
lora_r: 16
max_seq_length: 2048
```

```yaml
# Medium domain (2-8k examples): stm32, dsp, emc, electronics
epochs: 3
lora_r: 16
max_seq_length: 2048
```

```yaml
# Small domain (<2k examples): freecad, platformio, electronics
epochs: 4
lora_r: 8
max_seq_length: 2048
```

## Running a recipe

```bash
# From repo root, on KXKM-AI (RTX 4090)
python scripts/train_sft.py \
  --config configs/sft_stm32.yaml

# Eval after training
python scripts/eval_adapters.py --samples 5

# Publish to HF Hub
python scripts/publish_adapters.py --org clemsail
```

Training writes to `outputs/sft-<domain>-qwen25-32b/` but eval expects `outputs/sft-<domain>/` — **rename or symlink** before eval. This is a known gotcha; see `CLAUDE.md` → Gotchas.

## Don'ts

- Do not embed secrets (HF token, wandb key) in YAML.
- Do not edit a YAML in place during a run — commit the YAML first, then launch. The model registry stores a copy, but the canonical source is the versioned YAML.
- Do not reference `~/` or `$HOME` — breaks portability between KXKM-AI, GrosMac, and CI.
- Do not mix train + eval configs in the same file — `EvalConfig` is a sibling dataclass; use `eval_<domain>.yaml`.

## Related

- `configs/CLAUDE.md` — agent rules (required fields, no-default policy).
- `DOMAINS.md` — per-domain rank / alpha / system-prompt catalog.
- `CLAUDE.md` (root) — hardware reality + pipeline overview.
