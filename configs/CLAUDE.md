# configs/ — YAML training recipes

One YAML per SFT run. Keeps reproducibility explicit: given a config + a commit hash + a dataset file, the run is deterministic modulo nondeterministic CUDA kernels.

## Shape

Mirrors `TrainingConfig` in `src/ailiance_tuning/config.py`. Example (`sft_stm32.yaml`):

```yaml
base_model: Qwen/Qwen3-8B            # or Qwen/Qwen2.5-32B-Instruct
dataset: datasets/processed/stm32_train.jsonl
output_dir: outputs/sft-stm32
epochs: 3
batch_size: 4
lr: 0.0002
lora_r: 16
max_seq_length: 2048
hub_model_id: clemsail/kiki-stm32-sft
```

## Naming

- `sft_<domain>.yaml` — one-shot domain recipe
- `sft_<domain>_<basemodel-tag>.yaml` when multiple base models coexist (e.g. `sft_stm32_qwen25_32b.yaml`)
- Do NOT include the dataset hash / timestamp in the filename — track that in git history of the YAML itself

## Versioning discipline

- **Change YAML, commit it, then run** — never edit a YAML in place during a run. The registry entry should be reproducible.
- **Keep deprecated YAMLs** until their adapters are unpublished. The registry's `training_config` field is a copy, but the canonical source is the YAML file.
- **Required fields** : `base_model`, `dataset`, `output_dir`. Missing = hard fail, not default.
- **Dataset paths are relative to repo root** (`datasets/processed/*.jsonl`). Never absolute — breaks kxkm-ai ↔ local parity.

## Gotchas

- YAML `batch_size` here is `per_device_train_batch_size`. On the 4090, **stick to 1** for 32B models; tune via `gradient_accumulation_steps` (not yet in YAML — CLI-only today).
- `lora_r` and `lora_alpha` : keep `alpha = 2 × r` for the current scripts (`train_sft.py` hardcodes this). If you break the ratio, touch `train_sft.py` too.
- `hub_model_id` is advisory — `publish_adapters.py` recomputes from `--org` flag. If they disagree, the flag wins.
- No validator yet for YAML schemas — a missing required field will only fail inside training, often after model load. When adding a new field, add it to `TrainingConfig` dataclass first.

## Anti-patterns

- Do NOT embed secrets (HF token, wandb key) in YAML.
- Do NOT mix train + eval config in the same file — `EvalConfig` is a sibling dataclass, create `eval_<domain>.yaml` if needed.
- Do NOT reference `~/` or `$HOME` paths — breaks portability between kxkm-ai, GrosMac, CI.
