---
tags: [anchor, data]
---

# Data

← [[Home]]

---

## File Index — What Each `.jsonl` Is For

**At-a-glance map of every `.jsonl` in `data/`.** Detailed tables further down.

### Active SFT training data (used by `build_dataset.py`)

| File | Role | Origin |
|---|---|---|
| `synthetic_train_targeted_fix.jsonl` | help_mode + memory_recall SFT (largest source) | `synthetic/targeted_fix_pipeline.py` |
| `synthetic_train_friend_1.jsonl` | Casual friend-style support | `synthetic/pipeline.py` |
| `synthetic_train_transition.jsonl` | Casual → emotional pivot (fixes joke-mode-lock) | `synthetic/transition_pipeline.py` |
| `synthetic_train.jsonl` | Grief / loss | `synthetic/pipeline.py` |
| `synthetic_train_casual.jsonl` | Non-distress casual chat | `synthetic/casual_pipeline.py` |
| `synthetic_train_therapist_.jsonl` | Therapeutic dialogue | `synthetic/pipeline.py` |
| `synthetic_train_biometric.jsonl` | Sleep / HRV / mood biometric context | `synthetic/biometric_sft_pipeline.py` |
| `synthetic_train_targeted_fixes.jsonl` | **Gold** — hand-crafted, always 100% weight | manual + targeted scripts |

### Conv-memory data (multi-turn + memory injection, in progress)

| File | Role | Origin |
|---|---|---|
| `synthetic_train_conv_memory.jsonl` | Gemma4 teacher, 65/35 mix (original — finishing) | jobs 609110, 609111 |
| `synthetic_train_conv_memory_qwen.jsonl` | Qwen3-30B teacher, 65/35 mix (pre-shard) | job 611377 |
| `synthetic_train_conv_memory_qwen_s0.jsonl` | Qwen shard 0 — 50/50 mix + overref routing | job 611379 |
| `synthetic_train_conv_memory_qwen_s1.jsonl` | Qwen shard 1 — same config, disjoint RNG | job 611380 |
| `synthetic_train_conv_memory_qwen_s2.jsonl` | Qwen shard 2 — same config, disjoint RNG | job 611381 |
| `synthetic_train_conv_memory_overref_qwen_s{0,1,2}.jsonl` | **Inspection only** — casual chats where Anchor shoehorned therapy/coping. NOT for training as-is; potential DPO negatives later | same jobs (filter side-channel) |
| `synthetic_train_conv_memory_qwen_s3.jsonl` | Qwen shard 3 — **new profiles only** (`PROFILE_SET=new`): 18 new clinical + 10 new companion (28 total), 47 facts, 2 new modes. **❌ FAILED** — job 612903 OOM'd during inference after generating only 10 conversations. Ran stuck in OOM loop for 2+ days before cancellation. | job 612903 (cancelled) |
| `synthetic_train_conv_memory_qwen_s4–s8.jsonl` | Shards 4–8 — same new-profile config. **❌ ALL FAILED** — jobs 612904, 612905, 613115–613117 all OOM-killed at weight loading (~55% through 531 shards). Zero data produced. Root cause: new-pool script loads Qwen3-30B differently and uses more RAM than the original shards 0–2. | jobs 612904, 612905, 613115–613117 (OOM-killed) |

### Biometric SFT data (biometric context handling, in progress)

Teaches Anchor when to reference vs. ignore health/biometric data in [Recent sessions].
Two-phase teacher-as-Anchor (v3): Phase 1 user simulator generates user turns per mode; Phase 2 Anchor responds under the production system prompt (same as conv-memory approach).
Four weighted modes: `irrelevant` (40% — default is to NOT inject), `adjacent` (25% — don't assume connection), `relevant` (25% — connect once naturally), `trend` (10% — name multi-session pattern).

| File | Role | Origin |
|---|---|---|
| `synthetic_train_biometric_qwen_s0.jsonl` | Biometric shard 0 — **v3 two-phase teacher-as-Anchor**, vLLM, weighted modes (irrel 40/adj 25/rel 25/trend 10) | job 613120 |
| `synthetic_train_biometric_qwen_s1.jsonl` | Biometric shard 1 — same config, disjoint RNG | job 613121 |
| `synthetic_train_biometric_qwen_s2.jsonl` | Biometric shard 2 — same config, disjoint RNG | job 613122 |
| `synthetic_train_biometric_qwen.jsonl` | **Merge target** — `cat s{0,1,2}` after jobs finish | (post-merge) |

Raw outputs with meta fields (profile name, mode, health_type) live in `synthetic/outputs/biometric_sft_qwen_raw_s{0,1,2}.jsonl` for debugging.

### DPO data (all abandoned — kept for reference, none in active training)

| File | Role | Status |
|---|---|---|
| `dpo_train.jsonl` | Last-used DPO train: biometric + help_mode + memory_recall, 5,750 pairs | ❌ Abandoned |
| `dpo_val.jsonl` | Last-used DPO val, 1,014 pairs | ❌ Abandoned |
| `dpo_train_v2.jsonl` | Earlier DPO train, 6,120 pairs | ❌ Superseded |
| `dpo_val_v2.jsonl` | Earlier DPO val, 1,080 pairs | ❌ Superseded |
| `dpo_biometric_partial.jsonl` | Biometric-only DPO pairs (subset of `dpo_train.jsonl`) | ❌ Partial, abandoned |
| `dpo_pairs_partial.jsonl` | Legacy partial DPO output | ❌ Legacy |
| `dpo_pairs_partial_v2.jsonl` (cluster only) | Legacy partial DPO output (v2 attempt) | ❌ Legacy |
| `dpo_targeted_fix_partial.jsonl` (cluster only) | help_mode + memory_recall DPO partial | ❌ Partial, abandoned |
| `dpo_onpolicy_train.jsonl` (cluster only) | On-policy DPO experiment train split | ❌ Abandoned |
| `dpo_onpolicy_val.jsonl` (cluster only) | On-policy DPO experiment val split | ❌ Abandoned |
| `dpo_onpolicy_raw.jsonl` (cluster only) | On-policy DPO raw generations | ❌ Abandoned |

### Misc / legacy / debug (do not include in training)

| File | Role |
|---|---|
| `additional_training_samples.jsonl` | Tiny hand-edited supplemental set (18KB). Not currently mixed in |
| `synthetic_train_debug.jsonl` (cluster only) | Pipeline debug dump (tiny) — discard |
| `compare_top3.jsonl` (cluster only) | One-off model output comparison artifact — not training data |

### Excluded by user decision (never trained on)

| File | Reason |
|---|---|
| `data/new_raw_data/mindmate_train.jsonl` | 20,662 public examples (ESConv, EmpatheticDialogues, CounselChat) — user opted out 2026-04-14 |
| `data/cleaned_data/mindmate_train_clean.jsonl` | Cleaned version of the above — same exclusion |

### Auto-generated intermediates (not edited directly)

These are *output* of `build_dataset.py` / `clean_dataset.py` and live in `data/conversations_raw_v4/` and `data/conversations_cleaned_v4/`. They are what `CUDA_train_qlora.py` actually reads. Regenerate from sources above whenever the mix changes.

| Dir | File | Role |
|---|---|---|
| `conversations_raw_v4/` | `mindmate_train.jsonl` | Sampled per `DATA_MIX_PRESETS["v4"]` |
| `conversations_raw_v4/` | `mindmate_val.jsonl` | 15% per-source val split (min 300) |
| `conversations_cleaned_v4/` | `mindmate_train.jsonl` | Deduped, min 4 turns, format-validated |
| `conversations_cleaned_v4/` | `mindmate_val.jsonl` | Cleaned val split |

---

## SFT Training Files

All files live in `~/projects/mindmate/data/` (local) and mirrored on cluster. All 42,038 examples were normalized to production system prompt format in commit c3acdc9 (2026-05-09).

| File | Examples | Content | Generation | Status |
|---|---|---|---|---|
| `synthetic_train_targeted_fix.jsonl` | **13,524** | help_mode + memory_recall; multi-turn | `synthetic/targeted_fix_pipeline.py`, job 599031 | ✅ normalized |
| `synthetic_train_friend_1.jsonl` | 7,380 | Casual friend-style support | `synthetic/pipeline.py` | ✅ normalized |
| `synthetic_train_transition.jsonl` | 6,184 | Casual→emotional pivot (fix for joke-mode-lock) | `synthetic/transition_pipeline.py` | ✅ normalized |
| `synthetic_train.jsonl` | 5,565 | Grief/loss | `synthetic/pipeline.py` | ✅ normalized |
| `synthetic_train_casual.jsonl` | 5,000 | Non-distress casual chat | `synthetic/casual_pipeline.py` | ✅ normalized |
| `synthetic_train_therapist_.jsonl` | 2,637 | Therapeutic dialogue | `synthetic/pipeline.py` | ✅ normalized |
| `synthetic_train_biometric.jsonl` | **2,348** | Biometric health context (sleep, HRV, mood) | `synthetic/biometric_sft_pipeline.py`, job 599030 | ✅ normalized |
| `synthetic_train_targeted_fixes.jsonl` | **181** | Hand-crafted gold examples | Manual + targeted scripts | ✅ normalized |
| `synthetic_train_conv_memory.jsonl` | 845KB | Multi-turn + memory context, teacher-as-Anchor v2, Gemma4, 65/35 mix | `synthetic/conversation_memory_pipeline.py`, jobs 609110–609111 | ✅ DONE |
| `synthetic_train_conv_memory_qwen.jsonl` | 1.6MB | Same pipeline, Qwen3-30B teacher, 65/35 casual/clinical | job 611377 | ✅ DONE |
| `synthetic_train_conv_memory_qwen_s{0,1,2}.jsonl` | ~1.6–1.7MB each | Qwen3-30B, **50/50 mix**, over-reference filter | jobs 611379–611381 | ✅ DONE |
| `synthetic_train_conv_memory_overref_qwen_s{0,1,2}.jsonl` | 42–72KB each | Inspection only — casual where Anchor over-therapized | jobs 611379–611381 | ✅ DONE (not for training) |
| `synthetic_train_conv_memory_qwen_s3.jsonl` | 48KB (10 convs) | New-profile pool attempt — **❌ OOM, barely started** | job 612903 | ❌ FAILED |
| `synthetic_train_conv_memory_qwen_s{4–8}.jsonl` | — (no files) | New-profile pool shards — **❌ all OOM-killed at weight load** | jobs 612904, 612905, 613115–613117 | ❌ FAILED |

**Total normalized: 42,038 examples** (not counting conv_memory, which generates in correct format already)

---

## Gold Examples — `synthetic_train_targeted_fixes.jsonl` (181 total)

Always included at **100% weight** in every data mix (never sampled down). Hand-crafted to target specific benchmark failures.

| Category | Count | What it teaches |
|---|---|---|
| `name_resolution` | 30 | "my friend/brother/partner" → looks up actual name from profile (Zoya, Kabir, etc.) |
| `crisis_safety` | 20 | "better off without me" type → response must include safe/here/matter/alone/care + end with `?` |
| `session_history_recall` | 18 | Wedding, walking, mood-trend specifically referenced from `[Recent sessions]` |
| `profile_coping` | 15 | Uses the ★-helpful strategy from profile by name (not generic breathing or box breathing) |
| `biometric_profile` | 12 | Avoids ✗-marked unhelpful strategy; connects sleep/HRV/mood trend naturally |
| `help_cold_open` | 12 | "help me calm down" → technique in FIRST sentence (not a probe question first) |
| `anti_hallucination` | 8 | Cold open ("hey") → clean response, no invented prior context ("you went quiet on me") |
| `original` | 65 | Earlier mixed examples from before the benchmark-targeted expansion |

Gold expanded from 65 → 181 on 2026-05-01 (commit on that date).

---

## DPO Files (abandoned — kept for reference)

DPO was tried 3 times, all flat or worse than SFT. Files kept but not used in any active training.

| File | Pairs | Date | Notes |
|---|---|---|---|
| `dpo_train.jsonl` | 5,750 | 2026-05-01 | Last-used; biometric + help_mode + memory_recall |
| `dpo_val.jsonl` | 1,014 | 2026-05-01 | Last-used |
| `dpo_train_v2.jsonl` | 6,120 | 2026-04-18 | Superseded |
| `dpo_val_v2.jsonl` | 1,080 | 2026-04-18 | Superseded |
| `dpo_biometric_partial.jsonl` | 1,971 | 2026-04-29 | Biometric-only pairs (subset of dpo_train.jsonl) |
| `dpo_pairs_partial.jsonl` | varies | earlier | Legacy partial output |

---

## Excluded Data (intentional — user decision 2026-04-14)

Never included in any training run:
- `data/new_raw_data/mindmate_train.jsonl` — 20,662 examples (public datasets: ESConv, EmpatheticDialogues, CounselChat)
- `data/cleaned_data/mindmate_train_clean.jsonl` — 20,662 cleaned version of above

---

## Data Schema

All active SFT data uses `conversations` format (normalized by c3acdc9):

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "You are Anchor, a warm and caring AI companion...\n\nABOUT THIS USER\n[User]\nName: ...\nAge: ...\n...\n[Recent sessions]\nSession 1: ..."
    },
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

**Before c3acdc9 (the format bug):**
- `targeted_fix` + `biometric`: had `[User]\n...\n[Recent sessions]\n...` only — missing "You are Anchor..." preamble and "ABOUT THIS USER" header
- All other files: had **no system message at all**
- This caused the model to never see the production format during training

**The normalization done in c3acdc9:**
- `targeted_fix` + `biometric`: prepended `_APP_BASE_PROMPT + "\n\nABOUT THIS USER\n"` before existing `[User]`/`[Recent sessions]` blocks
- `biometric`: 179 examples had raw session notes without `[Recent sessions]` label — wrapped with full format
- `friend_1`, `therapist_`, `transition`, `casual`, `grief`: injected `_APP_BASE_PROMPT` as system message
- `targeted_fixes` (gold): had abbreviated old format — prepended `_APP_BASE_PROMPT`
- All files normalized from `messages` key to `conversations` key where needed

---

## Build Pipeline — How Raw JSONL Becomes Training Data

```
data/synthetic_train_*.jsonl
        ↓  finetuning/build_dataset.py --model v4
data/conversations_raw_v4/mindmate_train.jsonl   (sampled per DATA_MIX_PRESETS["v4"])
data/conversations_raw_v4/mindmate_val.jsonl     (15% val split per source, min 300)
        ↓  finetuning/clean_dataset.py --drop-min-turns 4 --dedup
data/conversations_cleaned_v4/mindmate_train.jsonl   (deduped, min 4 turns, format-validated)
data/conversations_cleaned_v4/mindmate_val.jsonl
        ↓  finetuning/CUDA_train_qlora.py --data-dir data/conversations_cleaned_v4
(training)
```

Intermediate dirs for v4 already exist locally. For v5, new `conversations_raw_v5/` and `conversations_cleaned_v5/` dirs will be created by the scripts.

---

## Synthetic Data Pipelines (`synthetic/`)

| File | Purpose | Status | Output |
|---|---|---|---|
| `pipeline.py` | therapist/friend/grief SFT | ✅ Done | `synthetic_train_therapist_.jsonl`, `synthetic_train_friend_1.jsonl`, `synthetic_train.jsonl` |
| `casual_pipeline.py` | casual non-distress SFT | ✅ Done | `synthetic_train_casual.jsonl` |
| `transition_pipeline.py` | casual→emotional pivot SFT | ✅ Done | `synthetic_train_transition.jsonl` |
| `targeted_fix_pipeline.py` | help_mode + memory_recall SFT | ✅ Done (job 599031) | `synthetic_train_targeted_fix.jsonl` |
| `biometric_sft_pipeline.py` | Biometric SFT, 4 modes × 24 profiles | ✅ Done (job 599030) | `synthetic_train_biometric.jsonl` |
| `biometric_dpo_pipeline.py` | Biometric DPO pairs (4 pair types) | ✅ Done (DPO abandoned) | part of `dpo_train.jsonl` |
| `dpo_targeted_fix_pipeline.py` | help_mode + memory_recall DPO pairs | ✅ Done (DPO abandoned) | part of `dpo_train.jsonl` |
| `dpo_pipeline.py` | General DPO pairs v1 + v2 | ✅ Done (DPO abandoned) | `dpo_train_v2.jsonl` |
| `conversation_memory_pipeline.py` | Multi-turn + memory (teacher-as-Anchor v2). Supports `SHARD_IDX`/`OUT_LABEL` env vars; routes over-referencing casual examples to `_overref` file | 🟢 RUNNING — jobs 609110–609111 (gemma4), 611377 (qwen 65/35), 611379–611381 (qwen 50/50 sharded) | `synthetic_train_conv_memory{_qwen{_s0,_s1,_s2},}.jsonl` + `_overref_*` |
| `launch_conv_memory_shards.sh` | Launches N parallel shards of the conv-memory pipeline (each disjoint RNG seed, separate output file) | ✅ Active | `synthetic_train_conv_memory_qwen_s{0..N-1}.jsonl` |

### Teacher Model

- `google/gemma-4-26B-A4B-it` in bfloat16 (~52GB VRAM) — requires A100-80
- MoE: 26B total params, ~4B active per token — fast inference despite size
- Gated HuggingFace model — `HF_TOKEN` env var must be set before first use
- First download: ~52GB (~20–30 min); subsequent runs use `~/.cache/huggingface/`
- Set via env var: `export TEACHER_MODEL=gemma4` (now the default in `synthetic/utils.py`)

### conv_memory pipeline v2 — teacher-as-Anchor design

Two-phase generation (training/inference distribution aligned by construction):

**Phase 1 — User simulator:**
- Gemma4 given a "simulate a realistic user" system prompt
- Generates all N user turns upfront as a JSON array
- Encodes `mode` + `new_fact` from user's perspective (6 modes, 20 new-fact seeds, 10 profile seeds)

**Phase 2 — Anchor responder:**
- Gemma4 given the **production Anchor system prompt + `[User]` + `[Recent sessions]`** as its actual system message
- Generates one assistant turn at a time against growing history
- Teacher is constrained by the same format the student model sees at inference

This ensures: (1) multi-turn structure, (2) user introduces a new fact mid-conversation, (3) later turns require referencing both injected memory and within-conversation facts, (4) format is byte-for-byte production format.

**Session 7 enhancements** (for teaching *when to use memory*):
- **33 profiles** (25 clinical, 8 companion with no diagnoses, just interests)
- `pick_mode()` gives **~50/50 casual/clinical** overall — companion profiles always casual, clinical profiles 35% casual / 65% clinical
- **`MEMORY_REQUIRED_MODES = {"memory_callback", "asking_for_help"}`** — only these enforce a context-ref check in `heuristic_check`; all other modes pass without memory references
- **Casual `USER_MODE_INSTRUCTIONS` explicitly forbid** mentions of therapy/coping/diagnosis on the user side
- **Over-reference filter** routes (not drops) casual conversations where Anchor shoehorned therapy/coping talk to `synthetic_train_conv_memory_overref_*.jsonl` — for inspection or potential DPO negatives, not training as-is
- **Shard support** via `SHARD_IDX` env var — each shard gets a disjoint RNG seed and writes to `_s{N}.jsonl`. Launch parallel shards with `finetuning/launch_conv_memory_shards.sh N`

**Run commands:**
```bash
# Single job (gemma4, default)
sbatch synthetic/run_conv_memory.slurm

# Single job (qwen3-30B)
sbatch --gres=gpu:a100-80:1 finetuning/conv_memory_pipeline_qwen.slurm

# N parallel shards (qwen3-30B) — recommended for high throughput
./finetuning/launch_conv_memory_shards.sh 3   # spawns shard 0,1,2

# Manual
source mindmatenv/bin/activate
export TEACHER_MODEL=gemma4         # or 30b for qwen
export OUT_LABEL=qwen               # optional output suffix
export SHARD_IDX=0                  # optional shard index (seeds RNG)
export HF_TOKEN=<your_token>
python synthetic/conversation_memory_pipeline.py
```

---

## See also

- [[Training]] — how data mixes are configured and submitted
- [[Models]] — what each training run produced
- [[Benchmarks]] — per-category scores that reveal data gaps
