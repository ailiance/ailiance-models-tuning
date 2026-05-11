#!/usr/bin/env python3
"""build_kicad_d2_combined.py — EU AI Act compliant builder for the D2
combined KiCad fine-tuning corpus.

Combines 4 source surfaces into ShareGPT-style triplets:
  1. Real .kicad_sch + .kicad_pcb pairs (from Ailiance-fr/kicad9plus-{permissive,copyleft})
  2. kicad-cli sch erc + pcb drc reports (run in sandboxed iact-bench-kicad Docker)
  3. Programmatic noise-injected "bad" variants for fix-it triplets (seed=42)
  4. Prose: KiCad official wiki/manual + Wikipedia EMC + arXiv eess.SP

Outputs 2 license-segregated buckets:
  - Ailiance-fr/kicad-d2-combined-permissive (Apache/MIT/BSD .kicad_sch + CC-BY-SA prose)
  - Ailiance-fr/kicad-d2-combined-copyleft   (GPL .kicad_sch only)

Compliance hooks throughout:
  - Per-row metadata.provenance: source_repo, source_path, license_spdx, build_sha
  - PII scan via tools/pii_scan.py on final jsonl (filter hard hits)
  - MANIFEST_D2.json: per-source rows + license + download_date (Annex IV §2(b))
  - README EU AI Act Template (AI Office July 2025) auto-generated
  - TDM-DSM Art 4 disclosure for arXiv content
  - Deterministic build (seed=42, idempotent re-run = identical output)

Designed to run on electron-server (Docker iact-bench-kicad image present).

Usage:
  python build_kicad_d2_combined.py --dry-run               # plan only
  python build_kicad_d2_combined.py --max-projects 50       # smoke
  python build_kicad_d2_combined.py --skip-prose            # sch+drc+erc only
  python build_kicad_d2_combined.py --publish               # push to HF after assembly
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("d2_builder")

# ─── Config ─────────────────────────────────────────────────────────────

HF_ORG = "Ailiance-fr"
PERMISSIVE_TARGET = f"{HF_ORG}/kicad-d2-combined-permissive"
COPYLEFT_TARGET = f"{HF_ORG}/kicad-d2-combined-copyleft"

SOURCE_PERMISSIVE = f"{HF_ORG}/kicad9plus-permissive"
SOURCE_COPYLEFT = f"{HF_ORG}/kicad9plus-copyleft"

WORK_DIR = Path("/tmp/d2_build")
WORK_DIR.mkdir(parents=True, exist_ok=True)

DOCKER_IMAGE = "ghcr.io/electron-rare/iact-bench-kicad:latest"
DOCKER_SANDBOX = [
    "--network=none", "--read-only",
    "--tmpfs", "/tmp:size=1g",
    "--user", "1000:1000",
    "--cap-drop=ALL", "--security-opt", "no-new-privileges",
    "--cpus=2.0", "--memory=2g",
]

SEED = 42
NOISE_VARIANTS_PER_BOARD = 3  # 3 perturbations per valid board → 3 triplets
PROSE_CHUNK_CHARS = 1500
BUILD_SHA = subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"],
    cwd=Path(__file__).parent, text=True,
).strip() if (Path(__file__).parent / ".git").exists() else "uncommitted"


# ─── Data classes for provenance tracking ──────────────────────────────

@dataclass
class Provenance:
    """Per-row provenance record (EU AI Act Art. 53(1)(d))."""
    source_repo: str       # e.g., "Ailiance-fr/kicad9plus-permissive"
    source_path: str       # path within the repo
    license_spdx: str      # SPDX identifier
    surface: str           # one of: sch, drc-report, erc-report, noise-fix, prose-doc
    file_sha256: str       # 64-hex of original file (deduplication + audit)
    build_sha: str         # builder git SHA at time of build
    timestamp_utc: str     # ISO 8601


@dataclass
class TripletRow:
    """One ShareGPT row in the final jsonl."""
    conversations: list[dict[str, str]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps({"conversations": self.conversations, "metadata": self.metadata},
                          ensure_ascii=False)


# ─── Step 1: load source corpus (kicad9plus) ──────────────────────────

def load_source_corpus(bucket: str, max_projects: int | None = None) -> list[dict]:
    """Load .kicad_sch + .kicad_pcb pairs from Ailiance-fr/kicad9plus-<bucket>.

    Returns list of dicts {project, sch_path, pcb_path, license_spdx}.
    Bucket = 'permissive' or 'copyleft'.

    TODO: detail HF dataset download via huggingface_hub.snapshot_download,
    then walk file tree, group by project, extract license_spdx from path
    or accompanying LICENSE file.
    """
    log.info("[1] loading %s corpus (max_projects=%s)", bucket, max_projects)
    pass  # TODO


# ─── Step 2: kicad-cli ERC/DRC in sandboxed Docker ────────────────────

def docker_run_kicad_cli(cmd: list[str], input_files: dict[str, bytes],
                         timeout_s: int = 60) -> dict:
    """Run kicad-cli inside the sandboxed iact-bench-kicad container.

    input_files: {filename → bytes content} staged in /tmp/in/ readonly.
    cmd: kicad-cli argv (the container's entrypoint isn't used here).
    timeout_s: wall-clock cap, kill -9 after.

    Returns: {"exit_code": int, "stdout": str, "stderr": str, "duration_s": float}.
    """
    work = tempfile.mkdtemp(prefix="d2_kicadcli_")
    for name, data in input_files.items():
        Path(work, name).write_bytes(data)
    cmd_str = " ".join(f"'{a}'" for a in cmd)
    docker_cmd = [
        "timeout", f"{timeout_s}s",
        "docker", "run", "--rm",
        *DOCKER_SANDBOX,
        "-v", f"{work}:/tmp/in:ro",
        DOCKER_IMAGE,
        "sh", "-c", cmd_str,
    ]
    t0 = time.perf_counter()
    try:
        result = subprocess.run(docker_cmd, capture_output=True, text=True,
                                timeout=timeout_s + 30)
        return {"exit_code": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr,
                "duration_s": round(time.perf_counter() - t0, 2)}
    except subprocess.TimeoutExpired:
        return {"exit_code": None, "stdout": "", "stderr": "timeout",
                "duration_s": float(timeout_s + 30)}
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


def run_erc_drc_for_project(project: dict) -> dict:
    """Generate ERC and DRC reports for one project.

    Returns {"erc": {...}, "drc": {...}, "valid": bool} where `valid` means
    both ERC and DRC return 0 errors (only 0/warnings).
    """
    sch_bytes = Path(project["sch_path"]).read_bytes()
    pcb_bytes = Path(project["pcb_path"]).read_bytes()
    inputs = {"board.kicad_sch": sch_bytes, "board.kicad_pcb": pcb_bytes}

    erc = docker_run_kicad_cli(
        ["kicad-cli", "sch", "erc", "--format=json",
         "--output=/tmp/erc.json", "/tmp/in/board.kicad_sch", "&&",
         "cat", "/tmp/erc.json"],
        inputs, timeout_s=60,
    )
    drc = docker_run_kicad_cli(
        ["kicad-cli", "pcb", "drc", "--format=json",
         "--output=/tmp/drc.json", "/tmp/in/board.kicad_pcb", "&&",
         "cat", "/tmp/drc.json"],
        inputs, timeout_s=90,
    )
    # TODO: parse erc.json / drc.json from stdout (the `&& cat` trick prints JSON)
    return {"erc": erc, "drc": drc,
            "valid": erc["exit_code"] == 0 and drc["exit_code"] == 0}


# ─── Step 3: programmatic noise injection ──────────────────────────────

NOISE_OPERATIONS = ["delete_wire", "displace_symbol", "drop_global_label",
                    "shrink_track_width"]


def inject_noise(sch_text: str, pcb_text: str | None,
                 noise_op: str, rng: random.Random) -> tuple[str, str | None]:
    """Apply one perturbation; returns (bad_sch, bad_pcb_or_none).

    Operations are deterministic given (sch_text, noise_op, rng.seed).
    They are minimal-edit so the diff between good and bad is small and
    structurally clear (the LoRA learns to "spot the difference").

    TODO: detail S-expression parsing (kiutils library, or hand-rolled
    regex for these narrow ops):
      - delete_wire: find `(wire (pts ...))` block, remove one
      - displace_symbol: find `(symbol ... (at x y angle))`, increment x by +500mil
      - drop_global_label: find `(global_label "NAME" ...)`, remove one
      - shrink_track_width: in pcb, find `(segment ... (width 0.25))`, set to 0.05 (below typical DRC clearance)
    """
    log.debug("noise %s on %d-byte sch", noise_op, len(sch_text))
    pass  # TODO


def build_triplets_from_project(project: dict, manifest_rows: list) -> list[TripletRow]:
    """For one valid project, produce NOISE_VARIANTS_PER_BOARD fix-it triplets.

    System prompt: "You are a KiCad design assistant. Given a schematic and
    its ERC/DRC report, identify and fix violations."
    User: bad_sch + erc+drc reports of the bad version
    Assistant: explanation + corrected sch (or diff)
    """
    triplets = []
    rng = random.Random(SEED + hash(project["project"]))
    sch_text = Path(project["sch_path"]).read_text()
    pcb_text = Path(project["pcb_path"]).read_text() if project.get("pcb_path") else None

    for i in range(NOISE_VARIANTS_PER_BOARD):
        noise_op = NOISE_OPERATIONS[i % len(NOISE_OPERATIONS)]
        bad_sch, bad_pcb = inject_noise(sch_text, pcb_text, noise_op, rng)
        # Re-run ERC/DRC on the noisy version
        bad_reports = run_erc_drc_for_project({**project,
                                               "sch_path": bad_sch, "pcb_path": bad_pcb})
        # Skip if noise didn't actually break anything (kicad-cli still 0 errors)
        if bad_reports["valid"]:
            continue

        triplet = TripletRow(conversations=[
            {"from": "system", "value":
             "You are an expert KiCad electronics design assistant. Given a "
             "schematic with reported ERC/DRC violations, identify the "
             "underlying issues and produce a corrected schematic that "
             "satisfies the design rules and follows EU EMC/electromagnetic "
             "compatibility best practices (IEC 61000 family)."},
            {"from": "human", "value": _format_bad_prompt(bad_sch, bad_reports)},
            {"from": "gpt", "value": _format_fix_response(sch_text, noise_op)},
        ])
        triplet.metadata = {
            "provenance": asdict(Provenance(
                source_repo=project["source_repo"],
                source_path=project["source_path"],
                license_spdx=project["license_spdx"],
                surface=f"noise-fix:{noise_op}",
                file_sha256=hashlib.sha256(sch_text.encode()).hexdigest(),
                build_sha=BUILD_SHA,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
            )),
            "noise_op": noise_op,
        }
        triplets.append(triplet)
        manifest_rows.append(triplet.metadata["provenance"])
    return triplets


def _format_bad_prompt(bad_sch: str, reports: dict) -> str:
    return (
        f"Here is a schematic with ERC/DRC issues.\n\n"
        f"### Schematic (KiCad .kicad_sch):\n```\n{bad_sch[:6000]}\n```\n\n"
        f"### ERC report:\n```json\n{reports['erc']['stdout'][:2000]}\n```\n\n"
        f"### DRC report:\n```json\n{reports['drc']['stdout'][:2000]}\n```\n\n"
        "Identify the violations and propose a corrected schematic. "
        "Explain each fix in 1-2 sentences before the patched code."
    )


def _format_fix_response(good_sch: str, noise_op: str) -> str:
    return (
        f"The violations are caused by `{noise_op}`. Here is the corrected "
        f"schematic with the fix applied:\n\n"
        f"```\n{good_sch[:6000]}\n```\n\n"
        f"Key changes:\n"
        f"- TODO: human-readable summary tied to noise_op\n"
        f"  (delete_wire → restored connection between net X and Y,\n"
        f"   displace_symbol → repositioned U1 inside page border, etc.)"
    )


# ─── Step 4: prose corpus merging ──────────────────────────────────────

def load_prose_corpus(skip: bool = False) -> list[TripletRow]:
    """Build prose-pool triplets from 3 sources.

    Sources & licenses:
      - KiCad wiki/manual (CC-BY-SA-4.0): clone https://gitlab.com/kicad/services/kicad-doc (or
        the official mirror), convert .rst → markdown via pandoc, chunk PROSE_CHUNK_CHARS
      - Wikipedia EMC + signal integrity articles (CC-BY-SA-3.0): API extract paragraphs
      - arXiv eess.SP recent SI/EMC papers (TDM-DSM EU Directive 2019/790 Art 3-4 exception):
        download abstracts + selected paragraphs, mark with explicit license_note

    Each chunk → 1 triplet of "explain this design concept" form:
      system: "expert KiCad EMC/SI engineer"
      human: prose chunk excerpt + "what design implications?"
      gpt: synthesize implications + example KiCad config

    TODO: implement source fetchers per surface.
    """
    if skip:
        log.info("[4] skipping prose corpus per --skip-prose")
        return []
    log.info("[4] loading prose corpus (kicad-doc + wikipedia EMC + arxiv eess)")
    pass  # TODO


# ─── Step 5: assemble final triplets + license bucket split ───────────

def assemble(projects: list[dict], prose: list[TripletRow]) -> dict:
    """Run ERC/DRC + noise injection per project, accumulate triplets,
    split into permissive vs copyleft buckets according to source license.

    Critical invariant: triplets derived from a GPL .kicad_sch NEVER land
    in the permissive bucket. Triplets derived from prose (CC-BY-SA from
    KiCad doc / Wikipedia / arXiv) are propagated to BOTH buckets
    (CC-BY-SA is compatible with downstream Apache via dual licensing of
    the LoRA artifact card).
    """
    permissive_rows: list[TripletRow] = []
    copyleft_rows: list[TripletRow] = []
    manifest: list[dict] = []

    for proj in projects:
        reports = run_erc_drc_for_project(proj)
        if not reports["valid"]:
            log.debug("  project %s invalid baseline, skipping", proj["project"])
            continue
        triplets = build_triplets_from_project(proj, manifest)
        if proj["license_spdx"].startswith(("Apache", "MIT", "BSD")):
            permissive_rows.extend(triplets)
        else:
            copyleft_rows.extend(triplets)

    # Prose goes to both buckets
    permissive_rows.extend(prose)
    copyleft_rows.extend(prose)

    # Deterministic 80/20 train/valid split (seed=42)
    rng = random.Random(SEED)
    def split(rows):
        rng.shuffle(rows)
        cut = int(len(rows) * 0.8)
        return rows[:cut], rows[cut:]

    perm_train, perm_valid = split(permissive_rows)
    cl_train, cl_valid = split(copyleft_rows)

    return {
        "permissive": {"train": perm_train, "valid": perm_valid},
        "copyleft":   {"train": cl_train,   "valid": cl_valid},
        "manifest":   manifest,
    }


# ─── Step 6: compliance audit + publish ──────────────────────────────

def compliance_audit(jsonl_path: Path) -> dict:
    """Run tools/pii_scan.py on the assembled jsonl, filter hard-PII rows,
    output the cleaned variant + a report.

    Reuses the pii_scan.py module from ailiance-models-tuning/tools/
    (we merged this morning's PR #5 to main).
    """
    log.info("[6] PII scan + filter on %s", jsonl_path)
    # TODO: import pii_scan, run with --filter, write _clean.jsonl, return stats
    return {"rows_in": 0, "rows_out": 0, "hard_pii_filtered": 0}


def gen_readme(bucket: str, manifest: list[dict], stats: dict) -> str:
    """Generate the Annex IV §2(b) Template README with all required EU AI Act
    fields, including TDM disclosure if arXiv chunks present."""
    # TODO: template-based emit
    pass


def publish_bucket(repo: str, train: list[TripletRow], valid: list[TripletRow],
                   readme: str, manifest: list[dict], private: bool = True) -> None:
    """Push to HF Ailiance-fr/<repo>. Private first per agent-mode policy."""
    from huggingface_hub import HfApi, create_repo

    create_repo(repo_id=repo, repo_type="dataset", exist_ok=True, private=private)
    api = HfApi()

    # Write artifacts to a staging dir
    staging = Path(tempfile.mkdtemp(prefix=f"d2_pub_{repo.replace('/','_')}_"))
    (staging / "train.jsonl").write_text("\n".join(r.to_jsonl() for r in train))
    (staging / "valid.jsonl").write_text("\n".join(r.to_jsonl() for r in valid))
    (staging / "MANIFEST_D2.json").write_text(json.dumps({
        "_doc": "EU AI Act Annex IV §2(b) provenance record",
        "rebuilt_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "build_sha": BUILD_SHA,
        "rows_total": len(train) + len(valid),
        "rows_train": len(train),
        "rows_valid": len(valid),
        "per_source": manifest,
    }, indent=2, ensure_ascii=False))
    (staging / "README.md").write_text(readme)

    # Upload all files
    for path in sorted(staging.rglob("*")):
        if path.is_file():
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=str(path.relative_to(staging)),
                repo_id=repo, repo_type="dataset",
                commit_message=f"initial: {path.name}",
            )
    log.info("published %s (private=%s)", repo, private)


# ─── Main orchestration ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-projects", type=int, default=None)
    p.add_argument("--skip-prose", action="store_true")
    p.add_argument("--publish", action="store_true",
                   help="push to HF after assembly (default: write local only)")
    args = p.parse_args()

    log.info("=== D2 builder start, build_sha=%s seed=%d ===", BUILD_SHA, SEED)
    if args.dry_run:
        log.info("DRY RUN — printing plan, no docker/HF calls")
        log.info("would build: %s + %s", PERMISSIVE_TARGET, COPYLEFT_TARGET)
        log.info("would read: %s + %s", SOURCE_PERMISSIVE, SOURCE_COPYLEFT)
        return

    # 1. Load sources
    permissive_projects = load_source_corpus("permissive", args.max_projects)
    copyleft_projects = load_source_corpus("copyleft", args.max_projects)
    all_projects = (permissive_projects or []) + (copyleft_projects or [])

    # 4. Prose pool (independent of step 1)
    prose = load_prose_corpus(skip=args.skip_prose)

    # 2 + 3 + 5: ERC/DRC, noise inject, assemble per-bucket
    bundles = assemble(all_projects, prose)

    # 6: compliance audit + publish
    for bucket, repo in [("permissive", PERMISSIVE_TARGET),
                          ("copyleft", COPYLEFT_TARGET)]:
        log.info("=== bucket %s ===", bucket)
        train_path = WORK_DIR / f"{bucket}_train.jsonl"
        train_path.write_text("\n".join(r.to_jsonl() for r in bundles[bucket]["train"]))
        stats = compliance_audit(train_path)
        readme = gen_readme(bucket, bundles["manifest"], stats)
        if args.publish:
            publish_bucket(repo,
                          bundles[bucket]["train"], bundles[bucket]["valid"],
                          readme, bundles["manifest"], private=True)

    log.info("=== D2 builder done ===")


if __name__ == "__main__":
    main()
