---
tags: [anchor, data]
---

# Data

← [[Home]]

---

## SFT Training Files

| File | Examples | Content | Status |
|---|---|---|---|
| `synthetic_train_targeted_fix.jsonl` | **13,524** | help_mode + memory_recall | ✅ normalized (c3acdc9) |
| `synthetic_train_friend_1.jsonl` | 7,380 | Casual friend-style support | ✅ normalized |
| `synthetic_train_transition.jsonl` | 6,184 | Casual→emotional pivot (fix for joke-mode-lock) | ✅ normalized |
| `synthetic_train.jsonl` | 5,565 | Grief/loss | ✅ normalized |
| `synthetic_train_casual.jsonl` | 5,000 | Non-distress casual | ✅ normalized |
| `synthetic_train_therapist_.jsonl` | 2,637 | Therapeutic dialogue | ✅ normalized |
| `synthetic_train_biometric.jsonl` | **2,348** | Biometric health context | ✅ normalized (c3acdc9) |
| `synthetic_train_targeted_fixes.jsonl` | **181** | Hand-crafted gold examples | ✅ normalized; always 100% weight |
| `synthetic_train_conv_memory.jsonl` | **~7k–10k (generating)** | Multi-turn + memory, teacher-as-Anchor v2 | 🟢 jobs 609110–609111, 72h A100-80 |

**Total normalized: 42,038 examples** (excluding conv_memory which generates in correct format)

---

## Gold Examples (181 total)

Built into `synthetic_train_targeted_fixes.jsonl`. Always included at 100% weight in every data mix.

| Category | Count | What it teaches |
|---|---|---|
| name_resolution | 30 | "my friend/brother" → looks up profile name |
| crisis_safety | 20 | "better off without me" → includes safe/here/matter/alone/care + ends `?` |
| session_history_recall | 18 | Wedding/walking/mood-trend explicitly referenced |
| profile_coping | 15 | Uses marked-★-helpful strategy by name (not generic breathing) |
| biometric_profile | 12 | Avoids marked-✗-unhelpful strategy |
| help_cold_open | 12 | "help me calm down" → technique in FIRST sentence (not probe) |
| anti_hallucination | 8 | Cold "hey" → clean opener, no invented prior context |
| original (misc) | 65 | Mixed coverage from early sessions |

---

## DPO Files (abandoned)

DPO was abandoned — all 3 runs flat or worse than SFT. Files kept for reference.

| File | Pairs | Notes |
|---|---|---|
| `dpo_train.jsonl` | 5,750 | Active/last-used — biometric + help_mode + memory_recall |
| `dpo_val.jsonl` | 1,014 | Active/last-used |
| `dpo_train_v2.jsonl` | 6,120 | Superseded |
| `dpo_val_v2.jsonl` | 1,080 | Superseded |

---

## Excluded Data

By user decision (2026-04-14) — never used in any training run:
- `data/new_raw_data/mindmate_train.jsonl` — 20,662 examples
- `data/cleaned_data/mindmate_train_clean.jsonl` — 20,662 examples

---

## Data Schema

All active SFT data:
```json
{"conversations": [
  {"role": "system", "content": "<production anchor prompt + [User] + [Recent sessions]>"},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

Key: since c3acdc9, all 42,038 examples include the full production system prompt preamble. The model now sees the same format during training as it does at inference.

---

## Data Pipelines (`synthetic/`)

| Pipeline | Output | Status |
|---|---|---|
| `pipeline.py` | therapist/friend/grief | ✅ Done |
| `casual_pipeline.py` | casual non-distress | ✅ Done |
| `transition_pipeline.py` | casual→emotional pivots | ✅ Done |
| `targeted_fix_pipeline.py` | help_mode + memory_recall SFT | ✅ Done — 13,524 ex |
| `biometric_sft_pipeline.py` | Biometric SFT, 4 modes | ✅ Done — 2,348 ex |
| `biometric_dpo_pipeline.py` | Biometric DPO pairs | ✅ Done (DPO abandoned) |
| `dpo_targeted_fix_pipeline.py` | DPO pairs for help_mode | ✅ Done (DPO abandoned) |
| `conversation_memory_pipeline.py` | Multi-turn + memory (teacher-as-Anchor v2) | 🟢 RUNNING — jobs 609110–609111 |

### Teacher model
- `google/gemma-4-26B-A4B-it` bfloat16 (~52GB VRAM) — requires A100-80
- Gated model — `HF_TOKEN` must be set on cluster
- ~20-30 min first download; subsequent runs use HF cache

### conv_memory pipeline v2 design
Two-phase generation — training/inference distribution aligned by construction:
- **Phase 1:** Gemma4 as user simulator → generates all N user turns as JSON
- **Phase 2:** Gemma4 given **production anchor prompt** as its actual system message → generates assistant turns constrained exactly as at inference
- 6 conversation modes, 10 profile seeds, 20 new-fact seeds
- Output: 4–6 turn conversations, user introduces new fact mid-conversation

---

## See also

- [[Training]] — how data mixes are built and used
- [[Models]] — which checkpoints came from which data mix
- [[Benchmarks]] — per-category scores reveal data gaps
