#!/usr/bin/env python3
"""Evaluate SFT adapters by sampling from each domain's dataset.

Loads base model + LoRA adapter, generates responses to held-out prompts,
and reports token accuracy + sample outputs for manual review.

Usage:
    python scripts/eval_adapters.py [--domains stm32,kicad] [--samples 5]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ailiance_eval")

DOMAINS = [
    "stm32", "kicad", "embedded", "platformio", "iot",
    "freecad", "dsp", "emc", "power", "spice",
]

BASE_MODEL = "Qwen/Qwen3-8B"


def load_eval_samples(domain: str, n: int = 5) -> list[dict]:
    """Load n random samples from domain dataset."""
    path = Path(f"datasets/processed/{domain}_train.jsonl")
    if not path.exists():
        logger.warning(f"Dataset not found: {path}")
        return []
    with open(path) as f:
        lines = f.readlines()
    if not lines:
        return []
    random.seed(42)
    samples = random.sample(lines, min(n, len(lines)))
    return [json.loads(line) for line in samples]


def extract_prompt(sample: dict) -> str:
    """Extract the user prompt from a chat-format sample."""
    messages = sample.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            return msg["content"]
    return ""


def extract_reference(sample: dict) -> str:
    """Extract the expected assistant response."""
    messages = sample.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg["content"]
    return ""


def eval_domain(model, tokenizer, domain: str, n_samples: int = 5) -> dict:
    """Evaluate one domain adapter, return metrics."""
    import torch

    adapter_dir = Path(f"outputs/sft-{domain}")
    if not adapter_dir.exists():
        logger.warning(f"Adapter not found: {adapter_dir}")
        return {"domain": domain, "status": "missing"}

    # Load adapter
    logger.info(f"Loading adapter: {adapter_dir}")
    model.load_adapter(str(adapter_dir), adapter_name=domain)
    model.set_adapter(domain)

    samples = load_eval_samples(domain, n_samples)
    if not samples:
        return {"domain": domain, "status": "no_dataset"}

    results = []
    for i, sample in enumerate(samples):
        prompt = extract_prompt(sample)
        reference = extract_reference(sample)
        if not prompt:
            continue

        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Simple token overlap metric
        ref_tokens = set(reference.lower().split())
        gen_tokens = set(generated.lower().split())
        overlap = len(ref_tokens & gen_tokens) / max(len(ref_tokens), 1)

        results.append({
            "prompt": prompt[:100],
            "reference_excerpt": reference[:150],
            "generated_excerpt": generated[:150],
            "token_overlap": round(overlap, 3),
        })
        logger.info(f"  [{domain}] sample {i+1}: overlap={overlap:.3f}")

    # Unload adapter
    model.delete_adapter(domain)

    avg_overlap = sum(r["token_overlap"] for r in results) / max(len(results), 1)
    return {
        "domain": domain,
        "status": "ok",
        "n_samples": len(results),
        "avg_token_overlap": round(avg_overlap, 3),
        "samples": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT adapters")
    parser.add_argument("--domains", default=",".join(DOMAINS), help="Comma-separated domain list")
    parser.add_argument("--samples", type=int, default=5, help="Samples per domain")
    parser.add_argument("--output", default="outputs/eval_report.json", help="Output report path")
    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(",")]

    # Lazy imports
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info(f"Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    report = {"base_model": BASE_MODEL, "domains": []}
    for domain in domains:
        logger.info(f"=== Evaluating {domain} ===")
        result = eval_domain(model, tokenizer, domain, n_samples=args.samples)
        report["domains"].append(result)
        logger.info(f"  → {domain}: {result.get('avg_token_overlap', 'N/A')}")

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Report saved to {output_path}")

    # Summary
    print("\n=== EVAL SUMMARY ===")
    for d in report["domains"]:
        status = d.get("status", "?")
        overlap = d.get("avg_token_overlap", "N/A")
        print(f"  {d['domain']:15s} status={status:8s} overlap={overlap}")


if __name__ == "__main__":
    main()
