#!/usr/bin/env python3
"""
Training-format parity check.

Verifies that sampled examples from each training JSONL match the production
system prompt format used at inference. Run this before any SFT job to catch
the class of bug fixed in c3acdc9 (training data missing production preamble).

Usage:
    python finetuning/check_format_parity.py                    # check all data/ files
    python finetuning/check_format_parity.py --files data/synthetic_train_targeted_fix.jsonl
    python finetuning/check_format_parity.py --sample 50 --strict

Exit code 0 = all checks pass. Exit code 1 = failures found (safe to block SFT submission).
"""

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Canonical first line of the production system prompt (anchorSystemPrompt.ts BASE_PROMPT)
PRODUCTION_PREAMBLE = "You are Anchor, a warm and caring AI companion"

# Expected memory header (contextBuilder.ts _MEMORY_HEADER)
MEMORY_HEADER_MARKER = "ABOUT THIS USER (you know this"

# Files that must have both preamble + memory header (they include memory context)
MEMORY_FILES = {
    "synthetic_train_targeted_fix.jsonl",
    "synthetic_train_biometric.jsonl",
    "synthetic_train_targeted_fixes.jsonl",
}

# Files that must have at minimum the base preamble
BASE_PREAMBLE_FILES = {
    "synthetic_train_friend_1.jsonl",
    "synthetic_train_therapist_.jsonl",
    "synthetic_train_transition.jsonl",
    "synthetic_train_casual.jsonl",
    "synthetic_train.jsonl",
    "synthetic_train_conv_memory.jsonl",
}

ALL_KNOWN_FILES = MEMORY_FILES | BASE_PREAMBLE_FILES

# Files to skip — not training data
SKIP_FILES = {"synthetic_train_debug.jsonl"}


def get_system_message(example: dict) -> str | None:
    """Extract system message text from a training example."""
    turns = example.get("conversations") or example.get("messages") or []
    for turn in turns:
        role = turn.get("role") or turn.get("from", "")
        if role in ("system", "human_system"):
            return turn.get("content") or turn.get("value") or ""
    return None


def check_file(path: Path, sample_n: int, strict: bool, require_memory: bool) -> tuple[int, int, list[str]]:
    """
    Returns (passed, total_sampled, list_of_failure_messages).
    """
    failures = []
    lines = path.read_text(encoding="utf-8").splitlines()
    lines = [l for l in lines if l.strip()]

    sample = random.sample(lines, min(sample_n, len(lines)))
    passed = 0

    for i, line in enumerate(sample):
        try:
            ex = json.loads(line)
        except json.JSONDecodeError as e:
            failures.append(f"  line parse error: {e}")
            continue

        sys_msg = get_system_message(ex)

        if sys_msg is None:
            failures.append(f"  example {i}: no system message found")
            continue

        if PRODUCTION_PREAMBLE not in sys_msg:
            snippet = sys_msg[:120].replace("\n", "\\n")
            failures.append(f"  example {i}: missing production preamble. System starts: {snippet!r}")
            continue

        if require_memory and MEMORY_HEADER_MARKER not in sys_msg:
            if strict:
                snippet = sys_msg[:120].replace("\n", "\\n")
                failures.append(f"  example {i}: missing memory header. System starts: {snippet!r}")
                continue

        passed += 1

    return passed, len(sample), failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files", nargs="+",
        help="Specific JSONL files to check (paths relative to project root or absolute). "
             "Defaults to all known training files in data/."
    )
    parser.add_argument("--sample", type=int, default=100,
                        help="Number of examples to sample per file (default: 100)")
    parser.add_argument("--strict", action="store_true",
                        help="Fail if memory-context files are missing the ABOUT THIS USER header")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if args.files:
        targets = [(Path(f) if Path(f).is_absolute() else PROJECT_ROOT / f) for f in args.files]
    else:
        targets = sorted((PROJECT_ROOT / "data").glob("synthetic_train*.jsonl"))

    if not targets:
        print("No training files found. Run from the project root or pass --files.")
        sys.exit(1)

    total_pass = total_sampled = 0
    any_fail = False

    for path in targets:
        if not path.exists():
            print(f"  SKIP  {path.name} — not found")
            continue
        if path.name in SKIP_FILES:
            print(f"  SKIP  {path.name} — excluded (not training data)")
            continue

        fname = path.name
        require_memory = fname in MEMORY_FILES

        passed, sampled, failures = check_file(path, args.sample, args.strict, require_memory)
        total_pass += passed
        total_sampled += sampled
        status = "✅ PASS" if not failures else "❌ FAIL"

        print(f"{status}  {fname}  ({passed}/{sampled} checked)")
        for msg in failures[:5]:  # cap output per file
            print(msg)
        if len(failures) > 5:
            print(f"  ... {len(failures) - 5} more failures")

        if failures:
            any_fail = True

    print()
    print(f"{'PASS' if not any_fail else 'FAIL'}  {total_pass}/{total_sampled} examples match production format")

    if any_fail:
        print()
        print("Fix: run finetuning/normalize_system_prompts.py to repair affected files.")
        print("Root cause reference: commit c3acdc9 (2026-05-09).")
        sys.exit(1)


if __name__ == "__main__":
    main()
