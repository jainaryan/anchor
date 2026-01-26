#!/usr/bin/env python3
"""
process_mindmate_data.py

Lightweight post-processing for MindMate SFT JSONL produced by make_dataset_mindmate.py.
- Preserves roles and order exactly as-is (no re-labeling, no trimming, no windowing).
- Applies safe text cleanup:
    * HTML tag stripping (useful for CounselChat answers)
    * ESConv token fixes (_comma_, _period_, etc.)
    * Whitespace normalization & spacing before punctuation
- Optional utilities (off by default; enable by flags):
    * --ensure-assistant-last : drop trailing user lines so each sample ends on assistant
    * --drop-min-turns N      : drop examples with fewer than N tagged lines
    * --dedup                 : drop exact duplicate examples

Usage example:
    python finetuning/clean_dataset.py \
      --train-in ./data/new_raw_data/mindmate_train.jsonl \
      --val-in   ./data/new_raw_data/mindmate_val.jsonl \
      --train-out ./data/cleaned_data/mindmate_train.cleaned.jsonl \
      --val-out   ./data/cleaned_data/mindmate_val.cleaned.jsonl \
      --ensure-assistant-last \
      --drop-min-turns 2

If you want *zero* behavior changes beyond text cleanup, simply omit the last two flags.
"""

import argparse
import html
import json
import os
import re
from typing import List, Tuple, Iterable

# Role tags (must match your make_dataset_mindmate.py)
TAG_USER = "<|user|>"
TAG_ASSIST = "<|assistant|>"

# ESConv-style token artifacts to restore back to normal punctuation.
PUNCT_MAP = {
    "_comma_": ",",
    "_period_": ".",
    "_question_": "?",
    "_exclamation_": "!",
    "_quote_": '"',
    "_apos_": "'",
    "_dash_": "-",
    # Add more tokens here if you encounter them in samples
}


# ---------------------------
# Text cleaning primitives
# ---------------------------
def clean_text(s: str) -> str:
    """HTML-unescape, strip HTML tags, restore ESConv tokens, normalize spacing."""
    if not s:
        return s
    # Convert HTML entities (&amp;, &nbsp;, &quot;, etc.)
    s = html.unescape(s)

    # Remove any HTML tags like <p>, <a href=...>, <br>, etc.
    s = re.sub(r"<[^>]+>", "", s)

    # Restore ESConv token artifacts back to punctuation
    for k, v in PUNCT_MAP.items():
        s = s.replace(k, v)

    # Collapse runs of spaces/tabs and strip line ends
    s = re.sub(r"[ \t]+", " ", s).strip()

    # Tighten spaces before punctuation: "hello !" -> "hello!"
    s = re.sub(r"\s+([,.\?!:;])", r"\1", s)

    return s


# ---------------------------
# Parsing / Serialization
# ---------------------------
def parse_tagged_lines(block: str) -> List[Tuple[str, str]]:
    """
    Parse a single JSONL 'text' block into [(role, content), ...].

    IMPORTANT: We do NOT change roles here. We just read what’s present.
    """
    turns: List[Tuple[str, str]] = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line:
            continue

        if line.startswith(TAG_USER):
            turns.append(("user", line[len(TAG_USER):].strip()))
        elif line.startswith(TAG_ASSIST):
            turns.append(("assistant", line[len(TAG_ASSIST):].strip()))
        else:
            # A line without a tag is unexpected given your generator, but keep it as unknown.
            # We store it as role="" and will preserve content during serialization.
            turns.append(("", line))
    return turns


def serialize_tagged_lines(turns: List[Tuple[str, str]]) -> str:
    """
    Convert [(role,text),...] back to a multi-line tagged string.

    NOTE: We DO NOT change roles here. Whatever role is present is what we write.
    Unknown role "" (empty) lines are written back *as-is* (no tag added).
    """
    out_lines: List[str] = []
    for role, text in turns:
        if role == "user":
            out_lines.append(f"{TAG_USER} {text}")
        elif role == "assistant":
            out_lines.append(f"{TAG_ASSIST} {text}")
        else:
            # Unknown role: write raw line without adding a tag
            out_lines.append(text)
    return "\n".join(out_lines)


# ---------------------------
# Optional utilities
# ---------------------------
def ensure_assistant_last(turns: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    If the last tagged line is <|user|>, drop trailing user lines until the last
    tagged line is an <|assistant|>, if such a line exists. This does NOT alter
    any remaining content or roles; it only removes trailing user lines.
    """
    # Find last assistant index
    last_assist_idx = None
    for i in range(len(turns) - 1, -1, -1):
        if turns[i][0] == "assistant":
            last_assist_idx = i
            break

    if last_assist_idx is None:
        # No assistant at all -> return as-is (do not invent/normalize roles)
        return turns

    # If the final turn is already assistant, nothing to do
    if turns[-1][0] == "assistant":
        return turns

    # Otherwise, cut at the last assistant turn (drop trailing user lines)
    return turns[: last_assist_idx + 1]


def drop_min_turns(turns: List[Tuple[str, str]], min_turns: int) -> bool:
    """
    Returns True if the example should be DROPPED because it has fewer than
    min_turns tagged lines (user/assistant). Unknown-role lines don't count.
    """
    tagged = sum(1 for r, _ in turns if r in {"user", "assistant"})
    return tagged < min_turns


def token_count(s: str) -> int:
    """Very rough token count by whitespace; good enough for quick stats."""
    return len(s.split())


def merge_consecutive_user_turns(turns: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Merges adjacent user turns into a single user turn with newline-separated content.
    Sequence: (user, A), (user, B), (assistant, C) -> (user, A\nB), (assistant, C)
    """
    if not turns:
        return []
    
    merged = []
    for role, text in turns:
        if not merged:
            merged.append((role, text))
            continue
        
        last_role, last_text = merged[-1]
        
        if role == "user" and last_role == "user":
            # Merge with previous user turn
            new_text = f"{last_text}\n{text}"
            merged[-1] = (last_role, new_text)
        else:
            merged.append((role, text))
    return merged

# ---------------------------
# File processing
# ---------------------------
def process_file(
    inp_path: str,
    out_path: str,
    ensure_assistant_final: bool = False,
    min_turns_to_keep: int = 0,
    dedup: bool = False,
    merge_user: bool = False,
    sample_peek: int = 2,
):
    """
    Read JSONL at inp_path, clean text safely, optionally enforce assistant-last / min turns / dedup,
    and write cleaned JSONL to out_path. Also prints summary stats.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    total = 0
    kept = 0
    uniq_guard = set()  # for --dedup

    # Stats accumulators
    turn_counts: List[int] = []
    token_counts: List[int] = []
    samples_before: List[str] = []
    samples_after: List[str] = []

    with open(inp_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue

            raw = obj.get("text", "")
            if not raw:
                continue

            # Save a tiny preview of the raw record (for console peek)
            if len(samples_before) < sample_peek:
                samples_before.append(raw[:300])

            # Parse tags (no role normalization)
            turns = parse_tagged_lines(raw)

            # Clean ONLY the text content; leave roles untouched
            cleaned = []
            for r, t in turns:
                ct = clean_text(t)
                if ct:
                    cleaned.append((r, ct))
            turns = cleaned

            # Optional: merge consecutive user turns
            if merge_user:
                turns = merge_consecutive_user_turns(turns)

            # Optional: ensure we end on an assistant reply (just drops trailing user lines)
            if ensure_assistant_final:
                turns = ensure_assistant_last(turns)

            # Optional: drop examples with too few tagged turns (user/assistant lines)
            if min_turns_to_keep > 0 and drop_min_turns(turns, min_turns_to_keep):
                continue

            # Serialize back
            serialized = serialize_tagged_lines(turns)

            # Optional: deduplicate exact matches across this file
            if dedup:
                if serialized in uniq_guard:
                    continue
                uniq_guard.add(serialized)

            # Write out
            fout.write(json.dumps({"text": serialized}, ensure_ascii=False) + "\n")
            kept += 1

            # Stats
            tagged_only = [t for r, t in turns if r in {"user", "assistant"}]
            turn_counts.append(len(tagged_only))
            token_counts.append(token_count(serialized))

            # Save tiny preview of cleaned record
            if len(samples_after) < sample_peek:
                samples_after.append(serialized[:300])

    # Print summary
    print(f"[{os.path.basename(inp_path)}] kept {kept}/{total} → {os.path.basename(out_path)}")
    if samples_before:
        print("  sample BEFORE:\n ", samples_before[0])
    if samples_after:
        print("  sample AFTER:\n ", samples_after[0])

    if kept > 0:
        avg_turns = sum(turn_counts) / len(turn_counts)
        p90_turns = percentile(turn_counts, 90)
        avg_tokens = sum(token_counts) / len(token_counts)
        p90_tokens = percentile(token_counts, 90)
        print(f"  stats: avg_turns={avg_turns:.2f} p90_turns={p90_turns} | "
              f"avg_tokens={avg_tokens:.0f} p90_tokens={p90_tokens}")

def percentile(xs: List[int], p: float) -> int:
    """Simple percentile for ints (no numpy dependency)."""
    if not xs:
        return 0
    xs_sorted = sorted(xs)
    idx = int(round((p / 100.0) * (len(xs_sorted) - 1)))
    return xs_sorted[idx]


# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Light post-processing for MindMate JSONL.")
    ap.add_argument("--train-in", required=True, help="Path to the original train JSONL")
    ap.add_argument("--val-in",   required=True, help="Path to the original val JSONL")
    ap.add_argument("--train-out", required=True, help="Where to write cleaned train JSONL")
    ap.add_argument("--val-out",   required=True, help="Where to write cleaned val JSONL")

    # Optional toggles (all default OFF)
    ap.add_argument("--ensure-assistant-last", action="store_true",
                    help="If set, drop trailing user lines so each example ends with an assistant turn.")
    ap.add_argument("--drop-min-turns", type=int, default=0,
                    help="If > 0, drop examples with fewer than this many tagged (user/assistant) lines.")
    ap.add_argument("--dedup", action="store_true",
                    help="If set, drop exact duplicate examples within each file.")
    ap.add_argument("--merge-consecutive-user", action="store_true",
                    help="If set, merge consecutive user turns into a single turn separated by newline.")

    args = ap.parse_args()

    process_file(
        inp_path=args.train_in,
        out_path=args.train_out,
        ensure_assistant_final=args.ensure_assistant_last,
        min_turns_to_keep=args.drop_min_turns,
        dedup=args.dedup,
        merge_user=args.merge_consecutive_user,
    )
    process_file(
        inp_path=args.val_in,
        out_path=args.val_out,
        ensure_assistant_final=args.ensure_assistant_last,
        min_turns_to_keep=args.drop_min_turns,
        dedup=args.dedup,
        merge_user=args.merge_consecutive_user,
    )


if __name__ == "__main__":
    main()
