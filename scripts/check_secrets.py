#!/usr/bin/env python3
"""Scan the repository for credentials and machine-specific paths before publishing.

    python scripts/check_secrets.py            # scan the working tree
    python scripts/check_secrets.py --staged   # scan only files staged for commit

Exits non-zero if anything suspicious is found. Values are masked in the output.
Install as a pre-commit hook with:

    ln -s ../../scripts/check_secrets.py .git/hooks/pre-commit
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PATTERNS = {
    "private key block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GCP service account": re.compile(r'"type"\s*:\s*"service_account"'),
    "OpenAI-style key": re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_\-]{20,}"),
    "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
    "hardcoded key assignment": re.compile(
        r"""(?ix)(?:api[_-]?key|subscription[_-]?key|secret|token|password)\s*[:=]\s*["'][A-Za-z0-9+/_\-]{24,}["']"""
    ),
    "long hex / base64 literal": re.compile(r"""["'](?=[A-Za-z0-9+/]*\d)(?=[A-Za-z0-9+/]*[A-Za-z])[A-Za-z0-9+/]{64,}={0,2}["']"""),
    "cluster path": re.compile(r"/(?:gpfs|mmfs1|mmfs)/[\w./-]+"),
    "home path": re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/"),
}

SKIP_DIRS = {".git", "models", "outputs", "wandb", "logs", "__pycache__", ".venv", "node_modules"}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".safetensors", ".bin", ".pt", ".parquet", ".zip", ".gz"}
FORBIDDEN_NAMES = {".env", "cookies.pkl"}
ALLOW_FILE = {"scripts/check_secrets.py"}


def candidate_files(staged: bool):
    if staged:
        out = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.split()
        yield from (REPO_ROOT / p for p in out)
        return
    for path in REPO_ROOT.rglob("*"):
        if path.is_file() and not any(part in SKIP_DIRS for part in path.relative_to(REPO_ROOT).parts):
            yield path


def mask(value: str) -> str:
    return value if len(value) <= 12 else f"{value[:6]}…{value[-3:]} ({len(value)} chars)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--staged", action="store_true", help="scan only files staged in git")
    args = parser.parse_args()

    findings = []
    for path in candidate_files(args.staged):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in ALLOW_FILE or path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if path.name in FORBIDDEN_NAMES:
            findings.append((rel, 0, "forbidden file", path.name))
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for label, pattern in PATTERNS.items():
                for match in pattern.finditer(line):
                    findings.append((rel, lineno, label, match.group(0)))

    for rel, lineno, label, value in findings:
        print(f"{rel}:{lineno}: {label}: {mask(value)}")
    if findings:
        print(f"\n{len(findings)} potential issue(s). Move secrets to .env / environment variables "
              "and paths to CLI arguments.", file=sys.stderr)
        return 1
    print("No secrets or machine-specific paths found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
