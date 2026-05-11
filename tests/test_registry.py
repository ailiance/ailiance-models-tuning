"""Tests for model registry."""

import json
import tempfile
from pathlib import Path

from src.ailiance_tuning.registry import ModelEntry, ModelRegistry


def test_register_and_get():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        registry = ModelRegistry(path=Path(f.name))

    entry = ModelEntry(
        name="test-model",
        base_model="Qwen/Qwen3-8B",
        adapter_path="outputs/test",
        domain="stm32",
    )
    registry.register(entry)
    assert registry.get("test-model") is not None
    assert registry.get("test-model").domain == "stm32"


def test_list_models():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        registry = ModelRegistry(path=Path(f.name))

    registry.register(ModelEntry(name="m1", base_model="b1", adapter_path="p1"))
    registry.register(ModelEntry(name="m2", base_model="b2", adapter_path="p2"))
    assert len(registry.list_models()) == 2


def test_persistence():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)

    registry1 = ModelRegistry(path=path)
    registry1.register(ModelEntry(name="persist", base_model="b", adapter_path="p"))

    registry2 = ModelRegistry(path=path)
    assert registry2.get("persist") is not None
