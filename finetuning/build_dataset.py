# make_dataset_mindmate.py
"""
Build a MindMate SFT dataset from:
  1) ESConv (multi-turn emotional support)
  2) EmpatheticDialogues (empathetic dialogs)
  3) CounselChat (Q&A counseling)

Outputs:
  ./data/mindmate_train.jsonl
  ./data/mindmate_val.jsonl

Format: each line is {"conversations": [{"role": "user", "content": "..."}, ...]}
"""

import os
import json
import random
from typing import List, Tuple
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from pathlib import Path
import re

# --------------------------
# Repro + paths
# --------------------------
SEED = 10
rng = random.Random(SEED)
random.seed(SEED)


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# NOTE: historically this was a hand-maintained list. It drifted out of sync with
# DATA_MIX_PRESETS — v3/v4 silently skipped targeted_fix.jsonl, biometric.jsonl, and
# the gold targeted_fixes.jsonl because they weren't in the list (see Bug Log 2026-05-19).
# Now derived from the active preset's keys so it cannot go stale again.
# Resolved below, after _preset_args is parsed.
extra_paths: list = []

# Data mix presets
DATA_MIX_PRESETS = {
    # v2 (2026-04-07): unified default for Llama v2 + Qwen3-1.7B v2
    # transition=30%, therapist=25%, casual=20%, grief=15%, friend=10%
    "v2": {
        "synthetic_train_transition.jsonl": 3000,
        "synthetic_train_therapist_.jsonl": 2500,
        "synthetic_train_casual.jsonl":     2000,
        "synthetic_train.jsonl":            1500,
        "synthetic_train_friend_1.jsonl":   1000,
    },
    # qwen25_3b (2026-04-08): heavier transition for 3B capacity
    # transition=35%, therapist=25%, casual=15%, grief=15%, friend=10%
    "qwen25_3b": {
        "synthetic_train_transition.jsonl": 3500,
        "synthetic_train_therapist_.jsonl": 2500,
        "synthetic_train_casual.jsonl":     1500,
        "synthetic_train.jsonl":            1500,
        "synthetic_train_friend_1.jsonl":   1000,
    },
    # gemma4_e2b (2026-04-14): same v2 mix — 2B model, good base; no extra transition weighting
    # transition=30%, therapist=25%, casual=20%, grief=15%, friend=10%
    "gemma4_e2b": {
        "synthetic_train_transition.jsonl": 3000,
        "synthetic_train_therapist_.jsonl": 2500,
        "synthetic_train_casual.jsonl":     2000,
        "synthetic_train.jsonl":            1500,
        "synthetic_train_friend_1.jsonl":   1000,
    },
    # gemma4_e4b (2026-04-17): 4B model — same mix as e2b for clean A/B comparison
    # transition=30%, therapist=25%, casual=20%, grief=15%, friend=10%
    "gemma4_e4b": {
        "synthetic_train_transition.jsonl": 3000,
        "synthetic_train_therapist_.jsonl": 2500,
        "synthetic_train_casual.jsonl":     2000,
        "synthetic_train.jsonl":            1500,
        "synthetic_train_friend_1.jsonl":   1000,
    },
    # v3 (2026-04-29): fresh training with targeted fix + biometric data
    # Fixes: MEMORY_USE (12%), HELP_MODE (50%), BIOMETRIC (20%) from benchmark baseline
    # transition=25%, targeted_fix=20%, therapist=15%, biometric=15%, friend=15%, casual=10%
    # +65 gold hand-crafted examples (always 100%)
    # Total: ~10,065
    "v3": {
        "synthetic_train_transition.jsonl":     2500,
        "synthetic_train_targeted_fix.jsonl":   2000,
        "synthetic_train_therapist_.jsonl":     1500,
        "synthetic_train_biometric.jsonl":      1500,
        "synthetic_train_friend_1.jsonl":       1500,
        "synthetic_train_casual.jsonl":         1000,
        "synthetic_train_targeted_fixes.jsonl":   65,
    },
    # v4 (2026-05-02): ~22k dataset, target ck400-600 Goldilocks zone
    # Fixes v3 weaknesses: MEMORY_USE+HELP_MODE via targeted_fix (28%), friend tone preserved (46% friend-adjacent)
    # Therapist kept at 2500 (matching v2) to protect CRISIS 5/5. hey-love not in data — DPO handles it.
    # targeted_fix=28%, transition=19%, friend=14%, casual=12%, biometric=11%, therapist=11%, grief=5%, +181 gold
    # Total: ~21,529
    "v4": {
        "synthetic_train_targeted_fix.jsonl":   6000,
        "synthetic_train_transition.jsonl":     4000,
        "synthetic_train_friend_1.jsonl":       3000,
        "synthetic_train_casual.jsonl":         2500,
        "synthetic_train_biometric.jsonl":      2348,
        "synthetic_train_therapist_.jsonl":     2500,
        "synthetic_train.jsonl":                1000,
        "synthetic_train_targeted_fixes.jsonl":  181,
    },
    # v5 (2026-05-19): first run with conv-memory + biometric Qwen data + help_mode data.
    # Key change vs v4: adds conv-memory (CROSS_SESSION_MEMORY fix), biometric Qwen shards
    # (BIOMETRIC fix), and help_mode Qwen data (HELP_MODE fix). Loss weighting handles
    # the safety-critical categories (crisis 4.0, help_mode 3.0, gold 3.0, targeted_fix 2.0,
    # conv_memory/biometric 1.5 via prefix match in _weight_for).
    # Core proportions kept from v4 to preserve COMPANION (92%) and FORMAT (56%).
    # Total: ~13,988. At batch=8: ~1,750 steps/epoch → run 2000 steps.
    # Use --model v5 arg. Output: adapters/genzv5, conversations_raw_v5, conversations_cleaned_v5.
    "v5": {
        # Core (same sources as v4, adjusted counts)
        "synthetic_train_targeted_fix.jsonl":        4000,
        "synthetic_train_friend_1.jsonl":            2000,
        "synthetic_train_transition.jsonl":          1500,
        "synthetic_train_casual.jsonl":              1000,
        "synthetic_train_therapist_.jsonl":           800,
        "synthetic_train.jsonl":                      300,
        "synthetic_train_targeted_fixes.jsonl":       181,
        # Conv-memory (all available shards — 1,617 total, loss_weight=1.5 via prefix)
        # s3-s8 added 2026-05-21 from new-pool PROFILE_SET (jobs 615485-615490, 176 ex total)
        "synthetic_train_conv_memory.jsonl":          167,
        "synthetic_train_conv_memory_qwen.jsonl":     299,
        "synthetic_train_conv_memory_qwen_s0.jsonl":  335,
        "synthetic_train_conv_memory_qwen_s1.jsonl":  316,
        "synthetic_train_conv_memory_qwen_s2.jsonl":  324,
        "synthetic_train_conv_memory_qwen_s3.jsonl":   44,
        "synthetic_train_conv_memory_qwen_s4.jsonl":   28,
        "synthetic_train_conv_memory_qwen_s5.jsonl":   21,
        "synthetic_train_conv_memory_qwen_s6.jsonl":   24,
        "synthetic_train_conv_memory_qwen_s7.jsonl":   33,
        "synthetic_train_conv_memory_qwen_s8.jsonl":   26,
        # Biometric (Qwen shards s0–s5 + old Gemma4 capped, loss_weight=1.5 via prefix)
        # s3-s5 added 2026-05-21 from jobs 615491-615493 (209 ex total)
        "synthetic_train_biometric_qwen_s0.jsonl":    993,
        "synthetic_train_biometric_qwen_s1.jsonl":     80,
        "synthetic_train_biometric_qwen_s2.jsonl":     78,
        "synthetic_train_biometric_qwen_s3.jsonl":     91,
        "synthetic_train_biometric_qwen_s4.jsonl":     57,
        "synthetic_train_biometric_qwen_s5.jsonl":     61,
        "synthetic_train_biometric.jsonl":            1500,
        # Help-mode (loss_weight=3.0 via exact key match)
        "synthetic_train_help_mode_qwen.jsonl":        115,
        # Crisis skipped — only 4 examples, insufficient signal
    },
    # v2_continued (2026-04-29): continued training from genzv2 ck1600
    # Base model already knows: casual tone, pivot, CRISIS, FORMAT, NO_HALLUCINATION
    # Only fixing: MEMORY_USE, BIOMETRIC, HELP_MODE
    # targeted_fix=30%, biometric=25%, transition=15%(replay), therapist=12%(replay),
    # casual=10%(replay), friend=5%(replay), +65 gold
    # Total: ~4,915
    "v2_continued": {
        "synthetic_train_targeted_fix.jsonl":   1500,
        "synthetic_train_biometric.jsonl":      1250,
        "synthetic_train_transition.jsonl":      750,
        "synthetic_train_therapist_.jsonl":      600,
        "synthetic_train_casual.jsonl":          500,
        "synthetic_train_friend_1.jsonl":        250,
        "synthetic_train_targeted_fixes.jsonl":   65,
    },
}

# --------------------------
# Category-conditional loss weights (filename → per-example loss multiplier)
# Applied in the trainer via WeightedLossTrainer.compute_loss.
# Rationale: CRISIS / HELP_MODE / NO_HALLUCINATION categories regress under
# undifferentiated SFT because friend-voice loss dominates. Upweighting their
# per-example loss steers gradients toward safety-critical behaviour without
# crowding out friend/casual data via raw sample-count rebalancing.
# Defaults: 1.0. Override below.
CATEGORY_WEIGHTS = {
    # gold hand-crafted examples — NO_HALLUCINATION + memory_recall fixes
    "synthetic_train_targeted_fixes.jsonl": 3.0,
    # help_mode + memory_recall targeted data (primary HELP_MODE fix)
    "synthetic_train_targeted_fix.jsonl": 2.0,
    # crisis data (Qwen-generated) — biggest regression vs base
    "synthetic_train_crisis_qwen.jsonl": 4.0,
    # help_mode data (Qwen-generated)
    "synthetic_train_help_mode_qwen.jsonl": 3.0,
    # conv-memory + biometric — currently weak vs base, modest upweight
    "synthetic_train_conv_memory.jsonl": 1.5,
    "synthetic_train_biometric.jsonl": 1.5,
    # everything else: 1.0 (default below)
}
DEFAULT_LOSS_WEIGHT = 1.0


def _weight_for(filename: str) -> float:
    """Resolve loss weight by exact filename, then by category prefix (handles _qwen_s0, _merged, etc.)."""
    if filename in CATEGORY_WEIGHTS:
        return CATEGORY_WEIGHTS[filename]
    for key, w in CATEGORY_WEIGHTS.items():
        stem = key.replace(".jsonl", "")
        if filename.startswith(stem):
            return w
    return DEFAULT_LOSS_WEIGHT


import argparse as _argparse
_preset_parser = _argparse.ArgumentParser(add_help=False)
_preset_parser.add_argument("--model", type=str, default="v2", choices=list(DATA_MIX_PRESETS.keys()))  # v5 is latest
_preset_args, _ = _preset_parser.parse_known_args()

SOURCE_CAPS = DATA_MIX_PRESETS[_preset_args.model]
OUT_DIR = PROJECT_ROOT / "data" / f"conversations_raw_{_preset_args.model}"
extra_paths = [PROJECT_ROOT / "data" / fname for fname in SOURCE_CAPS.keys()]
print(f"[build_dataset] Using data mix preset: {_preset_args.model}")
print(f"[build_dataset] Output dir: {OUT_DIR}")
print(f"[build_dataset] Loading {len(extra_paths)} source files: {[p.name for p in extra_paths]}")

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------
# Val split controls (per-source)
# --------------------------
VAL_RATIO = 0.15   # 15% per source
VAL_MIN   = 300    # at least this many per source (capped by available)

# Optional global prototype cap (None = use all)
MAX_TOTAL = None

def to_example(turns: List[Tuple[str, str]]) -> dict:
    """Pack [(role,text), ...] into one SFT example with structured messages."""
    conversations = []
    for role, text in turns:
        text = (text or "").strip()
        if not text:
            continue
        conversations.append({"role": role, "content": text})
    return {"conversations": conversations}

# =========================================================
# ESConv (robust loader; many snapshots present it as a 'text' blob)
# =========================================================
def load_esconv() -> List[dict]:
    """
    Handles:
      A) row['dialog'] : list of utterances (speaker/text variants)
      B) row['text']   : JSON/JSON-ish with a 'dialog' list or lines like 'usr: ...'
      C) fallback      : alternate roles per line
    Keeps dialogs with >=4 turns and at least one assistant turn.
    """
    ds = load_dataset("thu-coai/esconv", trust_remote_code=True, split="train")

    import json as _json, ast, re

    def map_role(raw):
        s = str(raw).lower().strip() if raw is not None else ""
        if s in {"sys","supporter","assistant","counselor","counsellor","therapist","helper"}:
            return "assistant"
        if s in {"usr","seeker","user","client","patient"}:
            return "user"
        return None

    def text_from(u):
        for k in ("text", "utterance", "content", "sentence"):
            v = u.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for v in u.values():  # last resort: first non-empty string
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    def packable(turns):
        return len(turns) >= 4 and any(r == "assistant" for r, _ in turns)

    out, total = [], 0

    for row in ds:
        total += 1
        turns: List[Tuple[str, str]] = []

        # Case A: structured dialog list
        dialog = (row.get("dialog") or row.get("dialogs")
                  or row.get("conversation") or row.get("conversations"))
        if isinstance(dialog, list) and dialog:
            for u in dialog:
                txt = text_from(u)
                if not txt:
                    continue
                role = map_role(u.get("speaker") or u.get("role") or u.get("agent_type") or u.get("who")) or "user"
                turns.append((role, txt))

        # Case B: everything inside row['text']
        elif isinstance(row.get("text"), str) and row["text"].strip():
            raw = row["text"].strip()

            parsed = None
            if raw[:1] in "{[":
                try:
                    parsed = _json.loads(raw)
                except Exception:
                    try:
                        parsed = ast.literal_eval(raw)
                    except Exception:
                        parsed = None

            # B1: dict with inner dialog
            if isinstance(parsed, dict):
                inner = (parsed.get("dialog") or parsed.get("dialogs")
                         or parsed.get("conversation") or parsed.get("conversations"))
                if isinstance(inner, list) and inner:
                    for u in inner:
                        txt = text_from(u)
                        if not txt:
                            continue
                        role = map_role(u.get("speaker") or u.get("role") or u.get("agent_type") or u.get("who")) or "user"
                        turns.append((role, txt))

            # B2: list [dicts or strings]
            if not turns and isinstance(parsed, list) and parsed:
                if isinstance(parsed[0], dict):
                    for u in parsed:
                        txt = text_from(u)
                        if not txt:
                            continue
                        role = map_role(u.get("speaker") or u.get("role") or u.get("agent_type") or u.get("who")) or "user"
                        turns.append((role, txt))
                elif isinstance(parsed[0], str):
                    for i, s in enumerate(parsed):
                        s = s.strip()
                        if not s:
                            continue
                        m = re.match(r'^\s*(usr|sys|seeker|supporter)\s*[:\-]\s*(.+)$', s, flags=re.I)
                        if m:
                            role = "assistant" if m.group(1).lower() in {"sys","supporter"} else "user"
                            turns.append((role, m.group(2).strip()))
                        else:
                            turns.append(("user" if i % 2 == 0 else "assistant", s))

            # B3: not JSON — parse line-by-line with usr/sys prefixes
            if not turns:
                lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
                for ln in lines:
                    m = re.match(r'^\s*(usr|sys|seeker|supporter)\s*[:\-]\s*(.+)$', ln, flags=re.I)
                    if m:
                        role = "assistant" if m.group(1).lower() in {"sys","supporter"} else "user"
                        turns.append((role, m.group(2).strip()))
                if not turns and lines:
                    for i, ln in enumerate(lines):
                        turns.append(("user" if i % 2 == 0 else "assistant", ln))

        if packable(turns):
            out.append(to_example(turns))

    print(f"[ESConv] kept {len(out)}/{total}")
    return out


# =========================================================
# EmpatheticDialogues (restitch convs; relax to >=4 turns)
# =========================================================
def load_empathetic_dialogues() -> List[dict]:
    """
    facebook/empathetic_dialogues:
    - We IGNORE speaker labels and just alternate roles by utterance order.
    - Always start with <|user|>, then <|assistant|>, etc.
    - Keep >=4 turns.
    """
    ds = load_dataset("facebook/empathetic_dialogues", "default", trust_remote_code=True)
    rows: List[dict] = []
    for split_name in ["train", "validation"]:
        df = pd.DataFrame(ds[split_name])
        # Group by conversation id then sort by utterance index
        for conv_id, grp in df.groupby("conv_id"):
            grp = grp.sort_values("utterance_idx")
            texts = [str(u) for u in grp["utterance"].tolist() if isinstance(u, str) and u.strip()]
            if len(texts) < 4:
                continue
            # Alternate roles starting with user
            turns = []
            role = "user"
            for t in texts:
                turns.append((role, t.strip()))
                role = "assistant" if role == "user" else "user"
            rows.append(to_example(turns))
    return rows



# =========================================================
# CounselChat (2-turn snippets: user question → assistant answer)
# =========================================================
def load_counselchat() -> List[dict]:
    ds = load_dataset("loaiabdalslam/counselchat", split="train")
    out: List[dict] = []
    for row in ds:
        qtitle = row.get("questionTitle") or ""
        qtext  = row.get("questionText") or ""
        atxt   = row.get("answerText") or ""
        q = (qtitle + "\n" + qtext).strip()
        a = (atxt or "").strip()
        if not q or not a or len(a) < 40:
            continue
        out.append(to_example([("user", q), ("assistant", a)]))
    return out


# --------------------------
# Helpers: tagging + stratified split
# --------------------------
def tag_rows(rows: List[dict], src: str) -> List[dict]:
    for r in rows:
        r["_src"] = src
    return rows

def stratified_split(rows: List[dict], ratio: float, val_min: int):
    """Split each source separately, then merge for balanced val."""
    by_src = {}
    for r in rows:
        by_src.setdefault(r["_src"], []).append(r)

    train, val = [], []
    for src, lst in by_src.items():
        rng.shuffle(lst)
        v = min(len(lst), max(val_min, int(len(lst) * ratio)))
        val.extend(lst[:v])
        train.extend(lst[v:])
    rng.shuffle(train)
    rng.shuffle(val)

    # drop tag
    for r in train: r.pop("_src", None)
    for r in val:   r.pop("_src", None)
    return train, val


def main():
    print("Skipping public datasets as requested. Only using synthetic data.")
    
    # adding extra synthetic data
    extra = []
    for ep in extra_paths:
        if not ep.exists():
            print(f"Warning: {ep} does not exist. Skipping.")
            continue
        
        count_before = len(extra)
        cap = SOURCE_CAPS.get(ep.name)
        weight = _weight_for(ep.name)
        rows_from_file = []
        with open(ep, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    # skip bad lines
                    continue
                if "conversations" in obj and isinstance(obj["conversations"], list):
                    rows_from_file.append({"conversations": obj["conversations"], "loss_weight": weight})
                elif "text" in obj and isinstance(obj["text"], str):
                    # Fallback for old format if mixed in
                    parts = re.split(r"(<\|user\|>|<\|assistant\|>)", obj["text"])
                    conv = []
                    role = "user"
                    for p in parts:
                        p = p.strip()
                        if p == "<|user|>": role = "user"
                        elif p == "<|assistant|>": role = "assistant"
                        elif p: conv.append({"role": role, "content": p})
                    rows_from_file.append({"conversations": conv, "loss_weight": weight})
        if weight != DEFAULT_LOSS_WEIGHT:
            print(f"  -> tagged {ep.name} with loss_weight={weight}")

        if cap is not None and len(rows_from_file) > cap:
            rng.shuffle(rows_from_file)
            rows_from_file = rows_from_file[:cap]
        extra.extend(rows_from_file)
        print(f"Loaded {len(extra) - count_before} extra training samples from {ep}")

    # Split synthetic data into train and eval
    rng.shuffle(extra)
    val_size = max(50, int(len(extra) * VAL_RATIO))
    if val_size > len(extra):
        val_size = len(extra) // 10
    
    val = extra[:val_size]
    train = extra[val_size:]
    
    print(f"Total synthetic data: {len(extra)}")


    # ----- Save -----
    train_path = os.path.join(OUT_DIR, "mindmate_train.jsonl")
    val_path   = os.path.join(OUT_DIR, "mindmate_val.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Saved: {len(train)} train ; {len(val)} val")
    print(f"Files: {train_path}  |  {val_path}")

if __name__ == "__main__":
    main()
