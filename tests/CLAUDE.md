# tests/ — pytest suite

Unit tests only. No GPU, no HF network, no trainer instantiation. The real pipeline is validated by running it on kxkm-ai, not by these tests.

## Run

```bash
uv run python -m pytest tests/ -v               # all
uv run python -m pytest tests/test_config.py    # single file
uv run python -m pytest -k "persistence"        # single test
```

## What's covered

| File | Target | Notes |
|------|--------|-------|
| `test_config.py` | `src/ailiance_tuning/config.py` | Defaults + override semantics. If you change a default, update the assertion. |
| `test_registry.py` | `src/ailiance_tuning/registry.py` | Register / get / list / persistence round-trip via `tempfile`. |
| `test_validate_dataset.py` | `scripts/validate_dataset.py` | Role + content schema on synthetic JSONL. |

## Conventions

- **Import style** : `from src.ailiance_tuning.config import ...` and `from scripts.validate_dataset import ...`. Both work because pytest adds repo root to `sys.path` (no `pip install -e .` needed).
- **`tempfile.NamedTemporaryFile(delete=False)`** for registry tests — the registry reopens the file, so it cannot be held open by the context manager. Leaks files in `/tmp`; acceptable for a tiny suite.
- **No fixtures directory** yet. If one test file grows to need fixtures, add `conftest.py` next to it, not at root (keeps scope tight).
- **No mocks for HF / torch**. Tests should not touch those layers — if you're tempted to mock `AutoModelForCausalLM`, you're testing the wrong thing; rewrite the target to isolate pure logic first.

## What's NOT covered (by design)

- `scripts/train_sft.py`, `scripts/eval_adapters.py`, `scripts/publish_adapters.py` — integration-tested by actually running on kxkm-ai
- `datasets/builders/build_*.py` — covered indirectly by `scripts/validate_dataset.py` on their output (run `build_all_datasets.sh`, which also calls the validator)
- HuggingFace Hub interactions — requires network + token; tested manually

## Anti-patterns

- Do NOT import `torch`, `transformers`, `peft`, `trl` here. If a test needs them, it belongs in an `integration/` marker-gated suite (does not exist yet — create one if truly needed).
- Do NOT hit the real filesystem outside `tempfile.*` or the `tests/` tree.
- Do NOT test argparse parsing of the script layer — that's a thin wrapper; test the functions it calls instead.
- Do NOT assert on log output. Assert on return values, file contents, dataclass state.
