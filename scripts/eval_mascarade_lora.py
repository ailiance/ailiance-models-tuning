#!/usr/bin/env python3
"""Evaluate published Qwen3-4B-mascarade-* LoRA adapters against held-out
dataset samples and emit per-domain bench snippets ready to inject into
the model card README.

Usage:
    python eval_mascarade_lora.py                       # all 10 domains
    python eval_mascarade_lora.py --domains kicad,spice # selected
    python eval_mascarade_lora.py --samples 20          # more samples (default 10)
    python eval_mascarade_lora.py --update-cards        # also push the snippet to HF

Pulls:
  - base model:  Qwen/Qwen3-4B-Instruct-2507
  - adapter:     Ailiance-fr/qwen3-4b-mascarade-<domain>-lora
  - eval data:   Ailiance-fr/mascarade-<domain>-dataset   (samples re-seeded
                 with seed=101 to differ from train seed=42)

Metrics (token-overlap & generation length):
  - Jaccard token overlap (lower-cased, word-split) vs reference
  - Average generation length
  - Latency per sample (cold model already loaded)

Designed to be re-runnable per-LoRA without unloading the base model.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_mascarade")

DOMAINS = [
    "kicad", "spice", "stm32", "emc", "embedded",
    "platformio", "freecad", "dsp", "iot", "power",
]
HF_ORG = "Ailiance-fr"
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
EVAL_SEED = 101  # differ from train seed=42


def load_eval_samples(domain: str, n: int = 10) -> list[dict]:
    """Pull dataset from HF and return n random samples (seed=101)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=f"{HF_ORG}/mascarade-{domain}-dataset",
        filename=f"{domain}_chat.jsonl",
        repo_type="dataset",
    )
    with open(path) as f:
        lines = [line for line in f if line.strip()]
    random.seed(EVAL_SEED)
    chosen = random.sample(lines, min(n, len(lines)))
    return [json.loads(l) for l in chosen]


def extract_prompt_ref(sample: dict) -> tuple[str, str]:
    msgs = sample.get("messages") or sample.get("conversations") or []
    prompt = ""
    ref = ""
    for m in msgs:
        role = m.get("role") or m.get("from")
        content = m.get("content") or m.get("value") or ""
        if role in ("user", "human"):
            prompt = content
        elif role in ("assistant", "gpt"):
            ref = content
    return prompt, ref


def jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def eval_domain(model, tokenizer, domain: str, n_samples: int) -> dict:
    """Load adapter for `domain`, eval on n_samples, unload."""
    import torch

    adapter_id = f"{HF_ORG}/qwen3-4b-mascarade-{domain}-lora"
    log.info("loading adapter %s", adapter_id)
    try:
        model.load_adapter(adapter_id, adapter_name=domain)
        model.set_adapter(domain)
    except Exception as e:
        log.error("adapter load failed %s: %r", adapter_id, e)
        return {"domain": domain, "status": "adapter_load_failed", "error": repr(e)}

    samples = load_eval_samples(domain, n_samples)
    if not samples:
        return {"domain": domain, "status": "no_samples"}

    rows = []
    t_total = 0.0
    for i, s in enumerate(samples):
        prompt, ref = extract_prompt_ref(s)
        if not prompt or not ref:
            continue

        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        dt = time.perf_counter() - t0
        t_total += dt

        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        ovl = jaccard(gen, ref)
        rows.append({
            "prompt_head": prompt[:100],
            "ref_head": ref[:120],
            "gen_head": gen[:120],
            "jaccard": round(ovl, 3),
            "gen_tokens": len(gen.split()),
            "latency_s": round(dt, 2),
        })
        log.info("  [%s] sample %d/%d jaccard=%.3f latency=%.2fs",
                 domain, i + 1, len(samples), ovl, dt)

    model.delete_adapter(domain)

    if not rows:
        return {"domain": domain, "status": "no_valid_rows"}

    avg_jac = sum(r["jaccard"] for r in rows) / len(rows)
    avg_len = sum(r["gen_tokens"] for r in rows) / len(rows)
    avg_lat = t_total / len(rows)
    return {
        "domain": domain,
        "status": "ok",
        "n_samples": len(rows),
        "avg_jaccard": round(avg_jac, 3),
        "avg_gen_tokens": round(avg_len, 1),
        "avg_latency_s": round(avg_lat, 2),
        "samples": rows[:3],  # truncate verbose rows for json size
    }


def card_snippet(report_domain: dict) -> str:
    """Render a Markdown bench section ready to splice into the model card."""
    d = report_domain
    if d["status"] != "ok":
        return f"\n## Bench results — held-out token-overlap (eval_mascarade_lora)\n\n_Eval skipped: {d['status']}_\n"
    return (
        "\n## Bench results — held-out token-overlap "
        f"(eval_mascarade_lora, n={d['n_samples']})\n\n"
        "Evaluated on 10 random held-out prompts from "
        f"`Ailiance-fr/mascarade-{d['domain']}-dataset` "
        f"(seed={EVAL_SEED} ≠ train seed 42).\n\n"
        "| Metric | Value |\n|---|---:|\n"
        f"| Avg Jaccard token-overlap | **{d['avg_jaccard']}** |\n"
        f"| Avg generation tokens | {d['avg_gen_tokens']} |\n"
        f"| Avg latency (per sample, RTX 4090) | {d['avg_latency_s']}s |\n\n"
        "_Token-overlap is a coarse quality proxy — high overlap (>0.4) "
        "suggests the LoRA reproduces domain vocabulary; low overlap "
        "indicates either domain-shift or stylistic divergence from the "
        "reference. See `ailiance/ailiance-bench` for richer functional "
        "evaluations (KiCad DRC, SPICE convergence, etc.) on the same "
        "family._\n"
    )


def update_card(domain: str, snippet: str) -> bool:
    """Replace the existing 'Bench results' section in the HF card."""
    import os, tempfile, re
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    repo = f"{HF_ORG}/qwen3-4b-mascarade-{domain}-lora"
    try:
        path = hf_hub_download(repo_id=repo, filename="README.md", repo_type="model")
        readme = open(path).read()
    except Exception as e:
        log.error("card fetch failed %s: %r", repo, e)
        return False

    # Replace any existing "## Bench results" block (up to next "## " or end)
    new_section = snippet.lstrip("\n")
    pattern = re.compile(r"## Bench results.*?(?=\n## |\Z)", re.S)
    if pattern.search(readme):
        new_readme = pattern.sub(new_section, readme, count=1)
    else:
        # Append before final "## Citations" if present, else at end
        if "## Citations" in readme:
            new_readme = readme.replace("## Citations", new_section + "\n## Citations", 1)
        else:
            new_readme = readme.rstrip() + "\n\n" + new_section

    if new_readme == readme:
        log.info("card %s: no change", repo)
        return True

    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as tf:
        tf.write(new_readme)
        tmp = tf.name
    try:
        api.upload_file(
            path_or_fileobj=tmp,
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="model",
            commit_message=f"docs: replace Phase 6 Gemma bench with real Qwen3-4B held-out eval ({domain})",
        )
        log.info("card %s: updated", repo)
        return True
    except Exception as e:
        log.error("card upload failed %s: %r", repo, e)
        return False
    finally:
        os.unlink(tmp)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--domains", default=",".join(DOMAINS))
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--output", default="outputs/eval_mascarade_report.json")
    p.add_argument("--update-cards", action="store_true",
                   help="Push the bench snippet back to each HF model card")
    args = p.parse_args()

    requested = [d.strip() for d in args.domains.split(",") if d.strip()]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    log.info("loading base %s", BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb,
        device_map="auto", trust_remote_code=True,
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    report = {
        "base_model": BASE_MODEL,
        "eval_seed": EVAL_SEED,
        "n_samples_target": args.samples,
        "domains": [],
    }
    for d in requested:
        log.info("=== %s ===", d)
        r = eval_domain(model, tok, d, args.samples)
        report["domains"].append(r)
        if args.update_cards:
            snip = card_snippet(r)
            ok = update_card(d, snip)
            r["card_updated"] = ok

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== EVAL SUMMARY ===")
    print(f"{'domain':<12} {'status':<22} {'jaccard':>8} {'gen_tok':>8} {'lat_s':>6}")
    for d in report["domains"]:
        s = d.get("status", "?")
        j = d.get("avg_jaccard", "—")
        g = d.get("avg_gen_tokens", "—")
        l = d.get("avg_latency_s", "—")
        print(f"  {d['domain']:<10} {s:<22} {j:>8} {g:>8} {l:>6}")
    print(f"\nReport saved to {out}")


if __name__ == "__main__":
    main()
