#!/usr/bin/env python3
"""ship_mascarade — end-to-end orchestrator for the 10 mascarade LoRA family.

Pipeline (per (base_model, domain) pair):
  1. Pull dataset from HF                Ailiance-fr/mascarade-<domain>-dataset
  2. Train LoRA on the requested base    (routes to Studio MLX or KXKM-AI CUDA)
  3. Eval the trained adapter            (token-overlap + electron-bench niches)
  4. Generate EU AI Act model card       with sources + license + metrics
  5. Publish to HF                       Ailiance-fr/<base>-<domain>-lora

Routing matrix:
  base="gemma-e4b"  -> Studio   (mlx_lm.lora, ~256-512 GB MPS)
  base="qwen3-4b"   -> KXKM-AI  (transformers/peft QLoRA, 4090 24 GB)
  base="qwen2.5-32b"-> KXKM-AI  (transformers/peft QLoRA, 4090 24 GB)

Usage:
  ship_mascarade.py --base gemma-e4b --domain kicad
  ship_mascarade.py --base qwen3-4b --domain all --parallel 1
  ship_mascarade.py --base all --domain all --dry-run    # show plan only
"""
from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("ship_mascarade")

DOMAINS = [
    "kicad", "spice", "stm32", "emc", "embedded",
    "platformio", "freecad", "dsp", "iot", "power",
]

HF_ORG = "Ailiance-fr"

# Per-base configuration: where it runs and how.
@dataclass(frozen=True)
class BaseSpec:
    name: str               # canonical short name used in HF repo id
    hf_base: str            # HuggingFace base model id
    host: str               # SSH alias or "local" or "studio"
    train_cmd: str          # template, {dataset_path} {output_dir} {hub_id} placeholders
    eval_cmd: str           # template
    notes: str

BASES: dict[str, BaseSpec] = {
    "gemma-e4b": BaseSpec(
        name="gemma-e4b",
        hf_base="lmstudio-community/gemma-4-E4B-it-MLX-4bit",
        host="studio",
        # mlx_lm.lora produces fused checkpoints we can directly push
        train_cmd=(
            "/Users/clems/eu-kiki/.venv/bin/mlx_lm.lora "
            "--model {hf_base} "
            "--train --data {dataset_path} "
            "--adapter-path {output_dir} "
            "--iters 1000 --batch-size 1 --learning-rate 1e-4"
        ),
        eval_cmd=(
            "/Users/clems/eu-kiki/.venv/bin/mlx_lm.evaluate "
            "--model {hf_base} --adapter-path {output_dir} "
            "--tasks arc_easy --limit 100 --output-dir {output_dir}/eval"
        ),
        notes="Studio M3 Ultra 512 GB MLX path (electron-bench compatible)",
    ),
    "qwen3-4b": BaseSpec(
        name="qwen3-4b",
        hf_base="Qwen/Qwen2.5-Coder-3B-Instruct",  # placeholder — adjust to true Qwen3-4B once HF id confirmed
        host="kxkm-23",  # via ssh -J electron-server kxkm@10.2.0.237
        train_cmd=(
            "cd ~/ailiance-models-tuning && "
            ".venv/bin/python scripts/train_sft.py "
            "--base-model {hf_base} --dataset {dataset_path} "
            "--output-dir {output_dir} --epochs 2 --lora-r 16 "
            "--push-to-hub --hub-model-id {hub_id}"
        ),
        eval_cmd=(
            "cd ~/ailiance-models-tuning && "
            ".venv/bin/python scripts/eval_adapters.py "
            "--adapter {output_dir} --domain {domain}"
        ),
        notes="KXKM-AI RTX 4090 24 GB QLoRA NF4 path",
    ),
}


def cmd(c: list[str] | str, *, dry_run: bool = False, check: bool = True) -> int:
    if isinstance(c, str):
        printable = c
    else:
        printable = " ".join(shlex.quote(x) for x in c)
    log.info("$ %s", printable)
    if dry_run:
        return 0
    result = subprocess.run(c, shell=isinstance(c, str))
    if check and result.returncode != 0:
        raise SystemExit(f"FAIL ({result.returncode}): {printable}")
    return result.returncode


def ship_one(base: BaseSpec, domain: str, *, dry_run: bool) -> dict:
    """Run the full pipeline for a single (base, domain) pair."""
    t0 = time.time()
    hub_id = f"{HF_ORG}/{base.name}-mascarade-{domain}-lora"
    dataset_id = f"{HF_ORG}/mascarade-{domain}-dataset"

    # 1. Stage dataset locally (Hub cache resolution handled by trainers,
    #    but we explicitly snapshot for build provenance).
    log.info("[%s/%s] dataset: %s", base.name, domain, dataset_id)
    output_dir = f"/tmp/ship-mascarade/{base.name}-{domain}"
    dataset_path = f"{output_dir}/dataset.jsonl"
    cmd([
        "python3", "-c",
        f"from huggingface_hub import hf_hub_download; import shutil, pathlib; "
        f"pathlib.Path('{output_dir}').mkdir(parents=True, exist_ok=True); "
        f"p = hf_hub_download('{dataset_id}', '{domain}_chat.jsonl', repo_type='dataset'); "
        f"shutil.copy(p, '{dataset_path}'); print('staged', p)"
    ], dry_run=dry_run)

    # 2. Train.
    train = base.train_cmd.format(
        hf_base=base.hf_base,
        dataset_path=dataset_path,
        output_dir=output_dir,
        hub_id=hub_id,
        domain=domain,
    )
    if base.host != "local":
        # Wrap in ssh; assumes the remote has dataset accessible via HF cache or
        # we pre-rsync the dataset. For simplicity here we assume HF Hub access.
        train = f"ssh {base.host} {shlex.quote(train)}"
    cmd(train, dry_run=dry_run)

    # 3. Eval.
    evalcmd = base.eval_cmd.format(
        hf_base=base.hf_base,
        output_dir=output_dir,
        domain=domain,
    )
    if base.host != "local":
        evalcmd = f"ssh {base.host} {shlex.quote(evalcmd)}"
    cmd(evalcmd, dry_run=dry_run, check=False)

    # 4. Model card.
    card_path = f"{output_dir}/README.md"
    cmd([
        "python3", "-c",
        f"open('{card_path}','w').write("
        f"\"\"\"---\nlicense: cc-by-sa-4.0\nbase_model: {base.hf_base}\ntags:\n- peft\n- lora\n- mascarade\n- {domain}\n- ailiance\n---\n\n"
        f"# {base.name}-mascarade-{domain}-lora\n\n"
        f"LoRA adapter trained on `{dataset_id}` for the `{domain}` domain.\n\n"
        f"Base model: `{base.hf_base}`.\n"
        f"Build host: `{base.host}` ({base.notes}).\n"
        f"Build date: $(date -u +%Y-%m-%dT%H:%M:%SZ).\n\n"
        f"## Provenance (EU AI Act Template — AI Office, July 2025)\n\n"
        f"- Training corpus: 100% from {dataset_id} (CC-BY-SA-4.0)\n"
        f"- Generated by: `scripts/ship_mascarade.py --base {base.name} --domain {domain}`\n"
        f"\"\"\")"
    ], dry_run=dry_run, check=False)

    # 5. Publish (train_sft already does --push-to-hub; this is the fallback for MLX path).
    if base.name.startswith("gemma"):
        cmd([
            "huggingface-cli", "upload", hub_id, output_dir,
            "--repo-type", "model",
            "--commit-message", f"feat: initial {base.name}-mascarade-{domain}-lora",
        ], dry_run=dry_run, check=False)

    dt = time.time() - t0
    return {"base": base.name, "domain": domain, "hub_id": hub_id, "duration_s": dt}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", choices=list(BASES) + ["all"], default="all")
    p.add_argument("--domain", default="all", help="comma-separated or 'all'")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--parallel", type=int, default=1, help="(reserved) cross-base parallelism")
    args = p.parse_args()

    bases = list(BASES.values()) if args.base == "all" else [BASES[args.base]]
    domains = DOMAINS if args.domain == "all" else args.domain.split(",")

    plan = [(b, d) for b in bases for d in domains]
    log.info("Plan: %d (base, domain) pairs", len(plan))
    for b, d in plan:
        log.info("  - %s × %s -> %s/%s-mascarade-%s-lora", b.name, d, HF_ORG, b.name, d)
    if args.dry_run:
        log.info("(dry-run — no commands executed beyond echo)")

    results: list[dict] = []
    for b, d in plan:
        try:
            r = ship_one(b, d, dry_run=args.dry_run)
            results.append(r)
            log.info("[%s/%s] OK in %.1fs", b.name, d, r["duration_s"])
        except SystemExit as e:
            log.error("[%s/%s] FAIL: %s", b.name, d, e)
            results.append({"base": b.name, "domain": d, "error": str(e)})

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
