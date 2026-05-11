# scripts/ — Ailiance training entry points

Real entry points for the pipeline. `src/ailiance_tuning/` only holds config + registry dataclasses — everything with side effects lives here.

## What's what

| File | Role | Runs where |
|------|------|-----------|
| `train_sft.py` | QLoRA 4-bit SFT, `trl.SFTTrainer`, `all-linear` LoRA | **kxkm-ai** only |
| `train_all_32b.sh` | Loops 7 domains through `train_sft.py` with Qwen2.5-32B | kxkm-ai |
| `train_missing_32b.sh` | Same, skip domains with existing adapter | kxkm-ai |
| `eval_adapters.py` | Loads base + each adapter, token-overlap metric, `outputs/eval_report.json` | GPU preferred (can CPU) |
| `publish_adapters.py` | Pushes to `clemsail/kiki-<domain>-sft`, builds model card from eval report | Any host with HF token |
| `validate_dataset.py` | Pure-python JSONL schema check, `messages/role/content` | Anywhere |
| `build_all_datasets.sh` | Runs every `datasets/builders/build_*.py` then validates outputs | Anywhere |

## Conventions

- **Entry scripts are `python scripts/foo.py` from repo root**, not `python -m ...`. CWD = repo root, not `scripts/`.
- **Lazy heavy imports** inside `main()` (torch/transformers/peft/trl). Keeps `--help` fast and lets validator scripts run without GPU deps.
- **Argparse everywhere** — no YAML parsing in scripts; YAML is pre-processed by the caller (shell wrapper) or the dataclass in `src/ailiance_tuning/config.py`.
- **Logging** : `logging.basicConfig(level=INFO)` + named logger (`ailiance_tuning.train_sft`, `ailiance_eval`, `ailiance_publish`). No `print()` except final summary.
- **Path convention** : adapters in `outputs/sft-<domain>/` or `outputs/sft-<domain>-<basemodel-tag>/`. Eval + publish scan these paths — if you rename, update both.
- **Graceful skip** : missing dataset / missing adapter → `logger.warning` + return status, never raise. Shell loops rely on `set -euo pipefail` but expect non-fatal skips.

## Anti-patterns (do not)

- Do NOT add a new training script — extend `train_sft.py` with flags instead. Keeping one path prevents config drift across 10 domains.
- Do NOT hardcode `outputs/sft-<domain>` in new code — accept it as argument. The 32B training runs write to `-qwen25-32b` suffix; keep the scanner flexible.
- Do NOT import `torch` at module scope — breaks `validate_dataset.py` on pure-CPU hosts.
- Do NOT parse YAML in the script — if a config needs parsing, build a loader in `src/ailiance_tuning/config.py` and return a `TrainingConfig`.
- Do NOT put secrets (HF token, wandb key) in the script — rely on env vars or `huggingface-cli login` beforehand.
