# make_dataset_mindmate.py
"""
Build a MindMate SFT dataset from:
  1) ESConv (multi-turn emotional support)
  2) EmpatheticDialogues (empathetic dialogs)
  3) CounselChat (Q&A counseling)

Outputs:
  ./data/mindmate_train.jsonl
  ./data/mindmate_val.jsonl

import os
import json
import random
from typing import List, Tuple
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# --------------------------
# Repro + paths
# --------------------------
SEED = 13
random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "new_raw_data"
extra_path = PROJECT_ROOT / "data" / "additional_training_samples.jsonl"

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------
# Val split controls (per-source)
# --------------------------
VAL_RATIO = 0.08   # 8% per source
VAL_MIN   = 300    # at least this many per source (capped by available)

# Optional global prototype cap (None = use all)
MAX_TOTAL = None

def to_example(turns: List[Tuple[str, str]]) -> dict:
    """Pack [(role,text), ...] into one SFT example with role tags."""
    lines = []
    for role, text in turns:
        text = (text or "").strip()
        if not text:
            continue
        lines.append(("<|user|> " if role == "user" else "<|assistant|> ") + text)
    return {"text": "\n".join(lines)}

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
        random.shuffle(lst)
        v = min(len(lst), max(val_min, int(len(lst) * ratio)))
        val.extend(lst[:v])
        train.extend(lst[v:])
    random.shuffle(train)
    random.shuffle(val)
    # drop tag
    for r in train: r.pop("_src", None)
    for r in val:   r.pop("_src", None)
    return train, val


def main():
    # ----- Load each dataset -----
    esconv_rows = tag_rows(load_esconv(), "ESConv")
    #ed_rows     = tag_rows(load_empathetic_dialogues(), "ED")
    cc_rows     = tag_rows(load_counselchat(), "CounselChat")

    # print(f"Loaded: ESConv={len(esconv_rows)}, ED={len(ed_rows)}, CounselChat={len(cc_rows)}")
    print(f"Loaded: ESConv={len(esconv_rows)}, CounselChat={len(cc_rows)}")
    # ----- Optional global cap (keeps relative proportions) -----
    # all_rows = esconv_rows + ed_rows + cc_rows
    all_rows = esconv_rows + cc_rows
    if MAX_TOTAL is not None and len(all_rows) > MAX_TOTAL:
        random.shuffle(all_rows)
        all_rows = all_rows[:MAX_TOTAL]
        # rebuild per-source lists after cap
        esconv_rows = [r for r in all_rows if r["_src"] == "ESConv"]
        # ed_rows     = [r for r in all_rows if r["_src"] == "ED"]
        cc_rows     = [r for r in all_rows if r["_src"] == "CounselChat"]

    # all_rows = esconv_rows + ed_rows + cc_rows
    all_rows = esconv_rows  + cc_rows

    # ----- Stratified split per source -----
    train, val = stratified_split(all_rows, VAL_RATIO, VAL_MIN)

    # adding extra synthetic data
    extra = []
    with open(extra_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # skip bad lines
                continue
            if "text" in obj and isinstance(obj["text"], str):
                extra.append({"text": obj["text"]})

    print(f"Loaded {len(extra)} extra training samples from {extra_path}")
    train.extend(extra)
    random.shuffle(train)

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
