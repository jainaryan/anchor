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
import hashlib
import html
import json
import math
import os
import random
import re
from typing import List, Tuple, Iterable, Optional

# Role mapping (must match your build_dataset.py)
ROLE_USER = "user"
ROLE_ASSIST = "assistant"

# ESConv-style token artifacts to restore back to normal punctuation.
PUNCT_MAP = {
    "_comma_": ",",
    "_period_": ".",
    "_question_": "?",
    "_exclamation_": "!",
    "_quote_": '"',
    "_apos_": "'",
    "_dash_": "-",
}


# ---------------------------
# Text cleaning primitives
# ---------------------------
def clean_text(s: str) -> str:
    """HTML-unescape, strip HTML tags, restore ESConv tokens, normalize spacing."""
    if not s:
        return s
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", "", s)
    for k, v in PUNCT_MAP.items():
        s = s.replace(k, v)
    s = re.sub(r"[ \t]+", " ", s).strip()
    s = re.sub(r"\s+([,.\?!:;])", r"\1", s)
    return s


# ---------------------------
# Optional flows
# ---------------------------
def ensure_assistant_last(messages: List[dict]) -> List[dict]:
    """Drop trailing user lines so each sample ends on assistant."""
    last_assist_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == ROLE_ASSIST:
            last_assist_idx = i
            break

    if last_assist_idx is None:
        return messages

    return messages[: last_assist_idx + 1]


def drop_min_turns(messages: List[dict], min_turns: int) -> bool:
    """Returns True if the example should be DROPPED due to low turn count."""
    tagged = sum(1 for m in messages if m.get("role") in {ROLE_USER, ROLE_ASSIST})
    return tagged < min_turns


def merge_consecutive_user_turns(messages: List[dict]) -> List[dict]:
    """Merges adjacent user turns into a single user turn."""
    if not messages:
        return []
    
    merged = []
    for m in messages:
        role = m.get("role")
        text = m.get("content", "")
        if not merged:
            merged.append({"role": role, "content": text})
            continue
        
        last = merged[-1]
        if role == ROLE_USER and last["role"] == ROLE_USER:
            last["content"] = f"{last['content']}\n{text}"
        else:
            merged.append({"role": role, "content": text})
    return merged


# ---------------------------
# MinHash LSH near-duplicate detection (stdlib only, no new deps)
# ---------------------------
# Hashes assistant-turn text only — that's what gets trained on, and that's where
# near-dups from same-profile Qwen regenerations accumulate.

_NUM_PERM = 128
_SHINGLE_K = 4          # char 4-grams
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 32) - 1


def _make_hash_params(num_perm: int, seed: int = 42):
    rng = random.Random(seed)
    a = [rng.randint(1, _MERSENNE_PRIME - 1) for _ in range(num_perm)]
    b = [rng.randint(0, _MERSENNE_PRIME - 1) for _ in range(num_perm)]
    return a, b


_HASH_A, _HASH_B = _make_hash_params(_NUM_PERM)


def _shingle_hashes(text: str, k: int = _SHINGLE_K):
    text = text.lower()
    seen: set = set()
    for i in range(len(text) - k + 1):
        gram = text[i : i + k]
        if gram not in seen:
            seen.add(gram)
            raw = int.from_bytes(hashlib.md5(gram.encode()).digest()[:4], "little")
            yield raw


def minhash_signature(text: str) -> list:
    sig = [_MAX_HASH] * _NUM_PERM
    for raw in _shingle_hashes(text):
        for i in range(_NUM_PERM):
            h = ((_HASH_A[i] * raw + _HASH_B[i]) % _MERSENNE_PRIME) & _MAX_HASH
            if h < sig[i]:
                sig[i] = h
    return sig


def _best_br(num_perm: int, threshold: float):
    """Find (b, r) that minimises |LSH crossover - threshold|. Crossover ≈ (1/b)^(1/r)."""
    best_b, best_r, best_err = 2, num_perm // 2, float("inf")
    for b in range(2, num_perm + 1):
        r = num_perm // b
        if r < 1:
            break
        crossover = (1.0 / b) ** (1.0 / r)
        err = abs(crossover - threshold)
        if err < best_err:
            best_b, best_r, best_err = b, r, err
    return best_b, best_r


class LSHIndex:
    """Banded LSH index for MinHash near-duplicate detection."""

    def __init__(self, threshold: float = 0.8, num_perm: int = _NUM_PERM):
        self.b, self.r = _best_br(num_perm, threshold)
        self.num_perm = num_perm
        effective = (1.0 / self.b) ** (1.0 / self.r)
        print(f"[near-dedup] LSH params: b={self.b} r={self.r} "
              f"| target threshold={threshold:.2f} effective≈{effective:.2f}")
        self.tables: list = [{} for _ in range(self.b)]

    def _band_keys(self, sig):
        for bi in range(self.b):
            yield bi, tuple(sig[bi * self.r : (bi + 1) * self.r])

    def is_near_dup(self, sig: list) -> bool:
        return any(band in self.tables[bi] for bi, band in self._band_keys(sig))

    def add(self, sig: list):
        for bi, band in self._band_keys(sig):
            self.tables[bi][band] = True


def _assistant_text(msgs: list) -> str:
    """Concatenate all assistant turn content — the text being trained on."""
    return " ".join(
        m.get("content", "") for m in msgs if m.get("role") == ROLE_ASSIST
    )


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
    drop_no_assist: bool = False,
    sample_peek: int = 2,
    lsh_index: Optional[LSHIndex] = None,
):
    """
    Read JSONL at inp_path, clean text safely, optionally enforce assistant-last / min turns /
    exact dedup / near-dedup (MinHash LSH), and write cleaned JSONL to out_path.

    Pass a shared LSHIndex via lsh_index to deduplicate across train+val simultaneously.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    total = 0
    kept = 0
    near_dup_dropped = 0
    uniq_guard = set()  # for exact --dedup

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

            msgs = obj.get("conversations", [])
            if not msgs:
                continue
            # Preserve per-example loss weight if upstream tagged it.
            loss_weight = obj.get("loss_weight", None)

            # Clean ONLY the text content
            cleaned = []
            for m in msgs:
                m["content"] = clean_text(m.get("content", ""))
                if m["content"]:
                    cleaned.append(m)
            msgs = cleaned

            if merge_user:
                msgs = merge_consecutive_user_turns(msgs)

            if ensure_assistant_final:
                msgs = ensure_assistant_last(msgs)

            if min_turns_to_keep > 0 and drop_min_turns(msgs, min_turns_to_keep):
                continue

            # exact dedupe check
            serialized = json.dumps(msgs, ensure_ascii=False)
            if dedup:
                if serialized in uniq_guard:
                    continue
                uniq_guard.add(serialized)

            # near-duplicate check via MinHash LSH on assistant text
            if lsh_index is not None:
                asst_text = _assistant_text(msgs)
                if asst_text.strip():
                    sig = minhash_signature(asst_text)
                    if lsh_index.is_near_dup(sig):
                        near_dup_dropped += 1
                        continue
                    lsh_index.add(sig)

            out_obj = {"conversations": msgs}
            if loss_weight is not None:
                out_obj["loss_weight"] = loss_weight
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            kept += 1

            # Stats
            tagged_only = [m for m in msgs if m.get("role") in {ROLE_USER, ROLE_ASSIST}]
            turn_counts.append(len(tagged_only))
            token_counts.append(len(serialized.split()))

            # Save tiny preview of cleaned record
            if len(samples_after) < sample_peek:
                samples_after.append(serialized[:300])

    # Print summary
    near_dup_msg = f" (near-dup dropped: {near_dup_dropped})" if near_dup_dropped else ""
    print(f"[{os.path.basename(inp_path)}] kept {kept}/{total}{near_dup_msg} → {os.path.basename(out_path)}")
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
    ap.add_argument("--drop-no-assistant", action="store_true",
                    help="If set, drop examples that have ZERO assistant turns.")
    ap.add_argument("--near-dedup", action="store_true",
                    help="If set, drop near-duplicate examples using MinHash LSH on assistant text. "
                         "A single shared LSH index is built across both train and val files so "
                         "cross-file near-dups are also caught.")
    ap.add_argument("--near-dedup-threshold", type=float, default=0.8,
                    help="Approximate Jaccard similarity threshold for near-dup detection (default 0.8). "
                         "Actual LSH crossover is printed at runtime — it will be close but not exact.")

    args = ap.parse_args()

    # Build a shared LSH index if near-dedup is requested so train+val dedup together.
    lsh: Optional[LSHIndex] = LSHIndex(threshold=args.near_dedup_threshold) if args.near_dedup else None

    process_file(
        inp_path=args.train_in,
        out_path=args.train_out,
        ensure_assistant_final=args.ensure_assistant_last,
        min_turns_to_keep=args.drop_min_turns,
        dedup=args.dedup,
        merge_user=args.merge_consecutive_user,
        drop_no_assist=args.drop_no_assistant,
        lsh_index=lsh,
    )
    process_file(
        inp_path=args.val_in,
        out_path=args.val_out,
        ensure_assistant_final=args.ensure_assistant_last,
        min_turns_to_keep=args.drop_min_turns,
        dedup=args.dedup,
        merge_user=args.merge_consecutive_user,
        drop_no_assist=args.drop_no_assistant,
        lsh_index=lsh,
    )


if __name__ == "__main__":
    main()
