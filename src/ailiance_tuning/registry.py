"""Model registry — track trained adapters and their metadata."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("ailiance_tuning.registry")

REGISTRY_FILE = Path("artifacts/model_registry.json")


@dataclass
class ModelEntry:
    """A registered model adapter."""

    name: str
    base_model: str
    adapter_path: str
    hub_id: str | None = None
    domain: str = "general"
    eval_scores: dict[str, float] = field(default_factory=dict)
    training_config: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


class ModelRegistry:
    """Simple JSON-file model registry."""

    def __init__(self, path: Path = REGISTRY_FILE):
        self.path = path
        self.models: dict[str, ModelEntry] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            text = self.path.read_text().strip()
            if not text:
                return
            data = json.loads(text)
            for name, entry in data.items():
                self.models[name] = ModelEntry(**entry)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: asdict(entry) for name, entry in self.models.items()}
        self.path.write_text(json.dumps(data, indent=2))

    def register(self, entry: ModelEntry) -> None:
        self.models[entry.name] = entry
        self._save()
        logger.info(f"Registered model: {entry.name}")

    def get(self, name: str) -> ModelEntry | None:
        return self.models.get(name)

    def list_models(self) -> list[ModelEntry]:
        return list(self.models.values())
