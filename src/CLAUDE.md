# src/ailiance_tuning/ — library core

Intentionally thin. Two modules, two dataclasses, no side effects. All the training/eval/publish logic lives in `scripts/`.

## Modules

| Module | Contents | Why it's here |
|--------|----------|---------------|
| `config.py` | `TrainingConfig`, `EvalConfig` (frozen-style `@dataclass`) | Single source of truth for defaults; mirrored by `configs/*.yaml` |
| `registry.py` | `ModelEntry`, `ModelRegistry` (JSON-backed at `artifacts/model_registry.json`) | Track trained adapters + eval scores without a DB |

## Conventions

- **Pure python only**. No torch / transformers / peft imports here — keeps the lib importable on CPU-only machines (tests, validator hosts, CI).
- **Dataclasses, not pydantic**. We don't need runtime validation and pydantic pulls a big tree. If you add a schema, use `dataclasses.dataclass` + type hints.
- **Registry is append-only semantics** — `register()` overwrites by name on purpose (re-training replaces the entry). If you want history, add a versioned field (`version: int = 1`), don't mutate the contract silently.
- **`from __future__ import annotations`** at the top. Python 3.12+, but keep string-form annotations for dataclass fields.
- **Logger name** : `ailiance_tuning.<module>`. Parent configures handlers — don't `basicConfig()` here.

## Anti-patterns

- Do NOT add a new module that imports torch — put that in `scripts/` or a new `scripts/lib/` if it grows.
- Do NOT import from `scripts/` into `src/` — the direction is one-way (scripts may import from src, never the reverse). Tests break this intentionally (`from scripts.validate_dataset import ...`) because the validator is small + pure.
- Do NOT read YAML here. YAML parsing belongs to whoever calls into `TrainingConfig(**yaml.safe_load(...))`, typically a script.
- Do NOT persist registry to anywhere other than `artifacts/` — parent `.gitignore` keeps that out of git, which is deliberate.

## Adding a new field to `TrainingConfig`

1. Add with default in `config.py`
2. Update `tests/test_config.py` assertion for the new default
3. Thread through `scripts/train_sft.py` argparse + usage
4. Document in `configs/CLAUDE.md` YAML shape table
5. Add to at least one example YAML in `configs/`
