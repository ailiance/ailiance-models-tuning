#!/usr/bin/env python3
"""Publish all SFT adapters to HuggingFace Hub.

Usage:
    huggingface-cli login  # first time
    python scripts/publish_adapters.py [--domains stm32,kicad] [--org clemsail]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from huggingface_hub import HfApi, create_repo

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ailiance_publish")

DOMAINS = [
    "stm32", "kicad", "embedded", "platformio", "iot",
    "freecad", "dsp", "emc", "power", "spice",
]

BASE_MODEL = "Qwen/Qwen3-8B"


def build_model_card(domain: str, adapter_dir: Path, eval_report: dict | None) -> str:
    """Generate a model card for the adapter."""
    # Count dataset examples
    dataset_path = Path(f"datasets/processed/{domain}_train.jsonl")
    n_examples = 0
    if dataset_path.exists():
        with open(dataset_path) as f:
            n_examples = sum(1 for _ in f)

    # Get training metrics from adapter_config
    config_path = adapter_dir / "adapter_config.json"
    lora_r = "16"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
            lora_r = str(cfg.get("r", 16))

    # Eval metrics
    eval_section = ""
    if eval_report:
        domain_eval = next((d for d in eval_report.get("domains", []) if d["domain"] == domain), None)
        if domain_eval and domain_eval.get("status") == "ok":
            overlap = domain_eval["avg_token_overlap"]
            eval_section = f"""
## Evaluation

| Metric | Value |
|--------|-------|
| Token Overlap | {overlap:.1%} |
| Samples | {domain_eval['n_samples']} |
"""

    return f"""---
library_name: peft
base_model: {BASE_MODEL}
license: apache-2.0
tags:
  - electronics
  - embedded-systems
  - {domain}
  - lora
  - sft
  - kiki-tuning
language:
  - en
  - fr
datasets:
  - custom
pipeline_tag: text-generation
---

# Ailiance {domain.upper()} SFT — LoRA Adapter

Fine-tuned LoRA adapter for **{domain}** domain expertise, based on `{BASE_MODEL}`.

Part of the [Ailiance Models Tuning](https://github.com/ailiance/ailiance-models-tuning) pipeline
for the [Ailiance](https://github.com/ailiance) platform.

## Training Details

| Parameter | Value |
|-----------|-------|
| Base Model | `{BASE_MODEL}` |
| Method | QLoRA (4-bit NF4) |
| LoRA Rank | {lora_r} |
| Epochs | 3 |
| Dataset | {n_examples} examples |
| Domain | {domain} |
{eval_section}
## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{BASE_MODEL}", device_map="auto")
model = PeftModel.from_pretrained(model, "clemsail/kiki-{domain}-sft")
tokenizer = AutoTokenizer.from_pretrained("{BASE_MODEL}")
```

## License

Apache 2.0
"""


def publish_domain(api: HfApi, domain: str, org: str, eval_report: dict | None) -> str:
    """Publish one adapter to HuggingFace Hub."""
    adapter_dir = Path(f"outputs/sft-{domain}")
    if not adapter_dir.exists():
        logger.warning(f"Adapter not found: {adapter_dir}")
        return "missing"

    repo_id = f"{org}/kiki-{domain}-sft"

    # Create repo if needed
    try:
        create_repo(repo_id, exist_ok=True, repo_type="model")
    except Exception as e:
        logger.error(f"Failed to create repo {repo_id}: {e}")
        return "error"

    # Write model card
    card = build_model_card(domain, adapter_dir, eval_report)
    card_path = adapter_dir / "README.md"
    card_path.write_text(card)

    # Upload all adapter files
    files_to_upload = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "README.md",
    ]

    for fname in files_to_upload:
        fpath = adapter_dir / fname
        if fpath.exists():
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fname,
                repo_id=repo_id,
                repo_type="model",
            )
            logger.info(f"  Uploaded {fname} → {repo_id}")

    logger.info(f"Published {repo_id}")
    return "ok"


def main():
    parser = argparse.ArgumentParser(description="Publish SFT adapters to HuggingFace")
    parser.add_argument("--domains", default=",".join(DOMAINS))
    parser.add_argument("--org", default="clemsail")
    parser.add_argument("--eval-report", default="outputs/eval_report.json", help="Path to eval report")
    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(",")]

    # Load eval report if available
    eval_report = None
    eval_path = Path(args.eval_report)
    if eval_path.exists():
        with open(eval_path) as f:
            eval_report = json.load(f)
        logger.info(f"Loaded eval report from {eval_path}")

    api = HfApi()

    print(f"\n=== Publishing {len(domains)} adapters to {args.org}/ ===\n")
    results = {}
    for domain in domains:
        logger.info(f"=== Publishing {domain} ===")
        status = publish_domain(api, domain, args.org, eval_report)
        results[domain] = status

    print("\n=== PUBLISH SUMMARY ===")
    for domain, status in results.items():
        print(f"  {domain:15s} → {status}")
    print(f"\nTotal: {sum(1 for s in results.values() if s == 'ok')}/{len(results)} published")


if __name__ == "__main__":
    main()
