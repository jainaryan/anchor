#!/usr/bin/env python3
"""Normalize the system-prompt memory blocks in training JSONL to the exact
production format defined by benchmarks/scenarios.py::_sys().

c3acdc9 normalized the preamble across all files but left two residual drifts
that this script fixes:
  - targeted_fix / biometric: single '\n' between [User] and [Recent sessions];
    production uses '\n\n'.
  - targeted_fixes (gold): truncated 'ABOUT THIS USER' header instead of the full
    _MEMORY_HEADER (with the '====' bars and instruction lines).

Approach: for every record whose system message contains a real [User] or
[Recent sessions] block (a line equal to that marker), extract the profile and
memory text verbatim and rebuild the system prompt via _sys(). Records with no
memory block are left untouched. Idempotent — re-running is a no-op.

Usage:
    python scripts/normalize_memory_format.py            # dry run, report only
    python scripts/normalize_memory_format.py --write    # rewrite files in place
"""
import argparse
import glob
import json
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "benchmarks"))
from scenarios import _APP_BASE_PROMPT, _MEMORY_HEADER, _sys  # noqa: E402

DATA_GLOB = os.path.join(PROJECT_ROOT, "data", "synthetic_train*.jsonl")


def msgs_of(rec):
    return rec.get("conversations") or rec.get("messages")


# A line like "[Apr 9] ..." — recent-sessions memory content with the marker stripped.
_RECENT_LINE = re.compile(r"^\[[A-Z][a-z]{2} \d{1,2}\]")


def rebuild(system: str):
    """Canonical-path parse for records that already carry the preamble + header.

    Line-based on purpose: the canonical _MEMORY_HEADER references "[User]" and
    "[Recent sessions]" *inside* its instruction lines, so substring matching
    would corrupt correct records. Real block markers are always whole lines.
    Returns the canonical system prompt, or None if the record has no memory.
    """
    lines = system.split("\n")
    ui = lines.index("[User]") if "[User]" in lines else None
    ri = lines.index("[Recent sessions]") if "[Recent sessions]" in lines else None
    if ui is None and ri is None:
        return None  # no memory — out of scope
    profile = ""
    memory = ""
    if ui is not None:
        end = ri if (ri is not None and ri > ui) else len(lines)
        profile = "\n".join(lines[ui + 1:end]).strip()
    if ri is not None:
        memory = "\n".join(lines[ri + 1:]).strip()
    return _sys(profile, memory)


def rebuild_stripped(system: str):
    """Salvage parse for records whose preamble + header were stripped.

    Safe to substring-match here precisely because there is no canonical header
    to false-match against. Handles inline markers ("[Recent sessions] [Apr 3] …")
    and flattened prose profiles. Returns canonical prompt, or None if not memory.
    """
    s = system.strip()
    ui = s.find("[User]")
    ri = s.find("[Recent sessions]")
    if ui == -1 and ri == -1:
        # Bare recent-sessions lines like "[Apr 9] Mood 3/10 …" with no marker.
        return _sys("", s) if _RECENT_LINE.match(s) else None
    profile = ""
    memory = ""
    if ri != -1:
        memory = s[ri + len("[Recent sessions]"):].strip()
    if ui != -1:
        end = ri if (ri != -1 and ri > ui) else len(s)
        profile = s[ui + len("[User]"):end].strip()
    elif ri > 0:
        # Prose profile preceding an inline [Recent sessions] marker (no [User]).
        lead = s[:ri].strip()
        if lead:
            profile = lead
    return _sys(profile, memory)


def process(path, write, rebuild_missing):
    out = []
    n = mem = changed = rebuilt = skipped = 0
    for line in open(path):
        line = line.rstrip("\n")
        if not line:
            continue
        n += 1
        rec = json.loads(line)
        msgs = msgs_of(rec)
        sys_msg = next((m for m in msgs if m.get("role") == "system"), None)
        if sys_msg:
            content = sys_msg["content"]
            if content.startswith(_APP_BASE_PROMPT):
                # Has canonical preamble — only block formatting may differ.
                canon = rebuild(content)
                if canon is not None:
                    mem += 1
                    if content != canon:
                        sys_msg["content"] = canon
                        changed += 1
            elif rebuild_missing:
                # Preamble/header was stripped — reconstruct from the bare blocks.
                canon = rebuild_stripped(content)
                if canon is not None:
                    mem += 1
                    if content != canon:
                        sys_msg["content"] = canon
                        rebuilt += 1
            elif rebuild(content) is not None:
                # Non-canonical preamble and not asked to rebuild — leave + warn.
                mem += 1
                skipped += 1
        out.append(json.dumps(rec, ensure_ascii=False))
    status = (f"{os.path.basename(path):<44} n={n:<6} mem={mem:<6} "
              f"changed={changed:<6} rebuilt={rebuilt:<6} skipped={skipped}")
    print(status)
    if write and (changed or rebuilt):
        with open(path, "w") as f:
            f.write("\n".join(out) + "\n")
    return changed, rebuilt, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="rewrite files in place")
    ap.add_argument("--rebuild-missing-preamble", action="store_true",
                    help="also reconstruct records whose preamble/header was stripped "
                         "(default off — keeps the safety guard for normal runs)")
    ap.add_argument("files", nargs="*",
                    help="optional specific JSONL paths to process (default: all data/synthetic_train*.jsonl)")
    args = ap.parse_args()
    paths = args.files if args.files else sorted(glob.glob(DATA_GLOB))
    total_changed = total_rebuilt = total_skipped = 0
    for path in paths:
        c, r, s = process(path, args.write, args.rebuild_missing_preamble)
        total_changed += c
        total_rebuilt += r
        total_skipped += s
    print(f"\nTotal: changed={total_changed} | rebuilt={total_rebuilt} | "
          f"skipped (non-canonical preamble)={total_skipped}")
    if not args.write and (total_changed or total_rebuilt):
        print("Dry run — re-run with --write to apply.")


if __name__ == "__main__":
    main()
