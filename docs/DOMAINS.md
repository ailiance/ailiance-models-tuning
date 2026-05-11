# DOMAINS — catalog of fine-tuned expert domains

ailiance-models-tuning trains 10 domain-expert LoRA adapters on top of a shared base (`Qwen/Qwen2.5-32B-Instruct`). Each domain has its own builder, system prompt, and recipe.

## Overview

| Domain | Category | Dataset size | LoRA rank | Notes |
|---|---|---|---|---|
| `stm32` | Firmware / MCU | ~2.7 k | 16 | HAL, LL, peripheral init, CubeMX-style |
| `embedded` | Systems | ~13.8 k | 16 | RTOS, bare-metal, HALs, drivers |
| `platformio` | Build | ~2.1 k | 8 | PlatformIO config, `platformio.ini`, lib management |
| `iot` | Networking | ~4 k | 16 | MQTT, LoRaWAN, ESP-NOW, BLE |
| `kicad` | EDA | ~8.5 k | 16 | KiCad S-expressions, schematic + PCB DSL |
| `spice` | Simulation | ~9.2 k | 16 | SPICE netlists, transient / AC / DC analysis |
| `emc` | Compliance | ~4.1 k | 16 | EMC shielding, filter design, harmonics |
| `power` | Power | ~3.5 k | 16 | Buck / boost, thermal, LDO, current sense |
| `dsp` | Signal processing | ~5.6 k | 16 | FFT, FIR / IIR, windowing, fixed-point |
| `freecad` | CAD | ~1.2 k | 8 | FreeCAD Python scripting, parametric design |

Additional domain via `datasets/builders/expand_espidf.py`:

- `espidf` — ESP-IDF framework (v5.4 / 5.5), component model, menuconfig

## Builder convention

Each domain has `datasets/builders/build_<domain>_dataset.py` which:

1. Loads hand-crafted seed examples (~50-200 Q/A per domain).
2. Optionally merges Hugging Face datasets via `--with-hf`.
3. Converts ShareGPT (`conversations/from/value`) → OpenAI (`messages/role/content`).
4. Writes `datasets/processed/<domain>_train.jsonl`.

Always run `scripts/validate_dataset.py` afterward to check role/content invariants.

## System prompt pattern

Each builder defines a domain-specific system prompt. Canonical shape:

```
You are an expert <domain> engineer. Answer questions with
precise, reproducible technical detail. When writing code,
match the idiomatic style of <domain>.
```

See `datasets/builders/build_stm32_dataset.py` for the reference prompt.

## Rank / alpha budget

- `alpha = 2 × rank` (hardcoded in `train_sft.py`; changing the ratio requires a code change).
- Rank 16 for niche, technical domains (STM32 HAL, SPICE netlists, KiCad DSL, EMC).
- Rank 8 for smaller or more generic domains (`freecad`, `platformio`).

Bigger ranks = more capacity but slower convergence and higher VRAM. Stay at 16 unless the base is clearly inadequate.

## Dataset size guidelines

| Size | Epochs | Rationale |
|---|---|---|
| < 2 k | 4 | Small dataset needs more passes for loss convergence |
| 2-8 k | 3 | Default regime; balances loss vs. over-tune |
| > 8 k | 2 | Large dataset; 2 epochs reaches asymptote |

Over-tuning shows up as hallucination on held-out domain probes, not just training loss.

## Adding a new domain

1. Write `datasets/builders/build_<domain>_dataset.py` using an existing builder as template.
2. Add a system prompt + 50-200 seed examples.
3. Run `./scripts/build_all_datasets.sh` (or target one) to generate JSONL.
4. Validate: `python scripts/validate_dataset.py datasets/processed/<domain>_train.jsonl`.
5. Write `configs/sft_<domain>.yaml` (see `RECIPES.md`).
6. Train on KXKM-AI: `python scripts/train_sft.py --config configs/sft_<domain>.yaml`.
7. Eval: `python scripts/eval_adapters.py --samples 5`.
8. Publish: `python scripts/publish_adapters.py --org clemsail`.

After publish, `src/ailiance_tuning/registry.py` records the new entry in `artifacts/model_registry.json`.

## Consumer: downstream repos

The trained adapters are loaded at inference time by:

| Repo | Role |
|---|---|
| [**mascarade**](https://github.com/electron-rare/mascarade) | Multi-provider LLM orchestration; dispatches queries to the right adapter via its own router |
| [**micro-kiki**](https://github.com/electron-rare/micro-kiki) | 35-domain MoE-LoRA runtime with cognitive layer; loads up to 4 adapters simultaneously |

When adding a new domain here, verify downstream consumers know about the new `hub_model_id`.

## Related

- `RECIPES.md` — YAML recipe shape and per-domain field guide.
- `CLAUDE.md` (root) — hardware reality, pipeline stages, gotchas.
- `configs/CLAUDE.md` — recipe versioning discipline.
