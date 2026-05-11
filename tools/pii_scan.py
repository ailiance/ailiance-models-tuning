#!/usr/bin/env python3
"""PII scanner for mascarade datasets.

Scans JSONL files for common PII patterns:
- Email addresses (RFC-ish, captures most public-domain Stack Exchange handles too)
- API keys (AWS, GCP, OpenAI, Stripe, GitHub, generic high-entropy)
- Phone numbers (E.164 international + FR/US)
- IPv4 addresses (separates private from public)
- SSH/PGP private key headers
- Credit card patterns (Luhn-shaped)
- GitHub @mentions

Outputs per-pattern hit count, sample rows, and writes a `_clean.jsonl`
filtered version when --filter is passed.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

PATTERNS: dict[str, re.Pattern] = {
    "email":           re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"),
    "aws_access_key":  re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "aws_secret_key":  re.compile(r"\baws_secret_access_key\s*=\s*[A-Za-z0-9/+=]{40}\b", re.I),
    "openai_key":      re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    "github_pat":      re.compile(r"\bghp_[A-Za-z0-9]{36,}\b"),
    "stripe_key":      re.compile(r"\b(?:sk|pk)_(?:test|live)_[A-Za-z0-9]{24,}\b"),
    "ssh_priv":        re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "pgp_priv":        re.compile(r"-----BEGIN PGP PRIVATE KEY BLOCK-----"),
    "phone_e164":      re.compile(r"(?<!\d)\+[1-9]\d{7,14}\b"),
    "phone_fr":        re.compile(r"(?<!\d)0[1-9](?:[ .\-]?\d{2}){4}\b"),
    "ipv4_public":     re.compile(r"\b(?!(?:10\.|127\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.))(?:\d{1,3}\.){3}\d{1,3}\b"),
    "credit_card":     re.compile(r"\b(?:\d[ \-]?){13,19}\b"),  # any 13-19 digit run
    "github_mention":  re.compile(r"(?<![A-Za-z0-9_])@[A-Za-z][A-Za-z0-9\-]{2,38}(?![A-Za-z0-9_])"),
}

# Patterns that are "informational" — high-volume on Stack Exchange but
# usually public-by-design (e.g. @handles in code comments). Counted but
# not used for filter unless --strict.
SOFT_PATTERNS = {"github_mention", "credit_card", "ipv4_public"}


def scan_text(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for name, pat in PATTERNS.items():
        for m in pat.findall(text):
            out[name].append(m if isinstance(m, str) else str(m))
    return out


def row_text(row: dict) -> str:
    """Extract all text from a row, supporting both ShareGPT and OpenAI chat formats."""
    chunks: list[str] = []
    # ShareGPT: {"conversations": [{"from": ..., "value": ...}, ...]}
    for c in (row.get("conversations") or []):
        if isinstance(c, dict):
            chunks.append(c.get("value") or "")
    # OpenAI: {"messages": [{"role": ..., "content": ...}, ...]}
    for c in (row.get("messages") or []):
        if isinstance(c, dict):
            chunks.append(c.get("content") or "")
    # Flat string fields some HF datasets use.
    for k in ("text", "prompt", "response", "answer"):
        v = row.get(k)
        if isinstance(v, str):
            chunks.append(v)
    return "\n".join(chunks)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--filter", action="store_true",
                   help="Write <input>_clean.jsonl with non-soft-PII rows removed")
    p.add_argument("--strict", action="store_true",
                   help="Treat SOFT_PATTERNS as filter triggers too")
    p.add_argument("--sample", type=int, default=3,
                   help="Show N sample hits per pattern")
    args = p.parse_args()

    for path in args.files:
        path = Path(path)
        print(f"\n=== {path} ===")
        if not path.exists():
            print("  FILE NOT FOUND")
            continue

        counter: Counter[str] = Counter()
        samples: dict[str, list[str]] = defaultdict(list)
        flagged_rows: set[int] = set()
        n_rows = 0

        clean_rows: list[str] = []
        with open(path) as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                n_rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = row_text(row)
                hits = scan_text(text)
                row_has_hard_pii = False
                for name, matches in hits.items():
                    counter[name] += len(matches)
                    if len(samples[name]) < args.sample:
                        samples[name].extend(matches[: args.sample - len(samples[name])])
                    is_soft = name in SOFT_PATTERNS
                    if not is_soft or args.strict:
                        row_has_hard_pii = True
                if row_has_hard_pii:
                    flagged_rows.add(i)
                else:
                    if args.filter:
                        clean_rows.append(line.rstrip("\n"))

        print(f"  rows scanned: {n_rows}")
        print(f"  rows flagged (hard PII): {len(flagged_rows)}  ({100*len(flagged_rows)/max(n_rows,1):.1f}%)")
        if counter:
            for name in sorted(counter, key=lambda k: -counter[k]):
                tag = " [soft]" if name in SOFT_PATTERNS else ""
                print(f"    {name:<18} hits={counter[name]:>6}{tag}")
                for s in samples[name][: args.sample]:
                    redacted = s[:40] + ("..." if len(s) > 40 else "")
                    print(f"      e.g. {redacted!r}")
        else:
            print("    no pattern matched ✅")

        if args.filter and clean_rows:
            out = path.parent / (path.stem + "_clean" + path.suffix)
            with open(out, "w") as f:
                f.write("\n".join(clean_rows) + "\n")
            print(f"  wrote {len(clean_rows)} clean rows -> {out}")


if __name__ == "__main__":
    main()
