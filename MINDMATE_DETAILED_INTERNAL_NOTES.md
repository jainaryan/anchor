# Anchor — Detailed Internal Notes

## Metadata
- Project: `mindmate`
- Last updated: `2026-05-09` (session 4)
- Android app: `anchor-app/` (NOT `mindmate_app/` — that is a stale scratch fork)
- Production: https://tryanchor.me

---

## Executive Summary

Anchor is a finetuned, local mental-health companion that runs on Android. The full pipeline is:

```
Synthetic Data (Gemma 4 26B A4B IT teacher)
→ SFT (QLoRA, Llama 3.2 3B)
→ DPO (preference training on SFT adapter)
→ GGUF Q4_K_M export
→ Android (anchor-app, JNI → llama.cpp)
```

**v3 benchmark (3-run averaged, COMPLETE):** llama_base 51%, genzv3_ck200 45%, genzv2_ck1600 45%, genzv2_ck1200 44%, genzv4_ck200 42%. Base still beats all SFT.
**⚠️ Base model still beats SFT (v3 averaged):** Root cause identified and fixed — see below.
**⚠️ Benchmark variance (2026-05-09):** `temperature=0.7` on eval → 17/49 scenarios flip PASS↔FAIL across runs. **3-run averaged methodology adopted.** Use `benchmarks/average_results.py --since 20260509`.
**v3 benchmark suite:** 58 scenarios, 9 categories, total weight 61. Added COMPANION with genziness sub-tests (cp_07–09: voice matching, therapy-speak avoidance, high-energy calibration).
**Root cause identified (session 4):** All 42,038 training examples were missing the production system prompt preamble. targeted_fix + biometric had truncated `[User]`/`[Recent sessions]`-only system prompts; all other files had no system message. Model never saw the "You are Anchor..." + "ABOUT THIS USER" header during training. **Fixed and pushed (commit c3acdc9).**
**Conv-memory pipeline (teacher-as-Anchor) running:** Jobs 609110–609111, A100-80, 72h. Uses full production format (fix above was inspired by this finding).
**`llama_ck1600` renamed `genzv2_ck1600`** in MODEL_SHORTCUTS — same adapter (`adapters/genz/checkpoint-1600`), clearer name.
**Previous best (old 34-scenario v1 benchmark):** genzv3_ck200 76%, genzv2_ck1200 71%, genzv4_ck200 65%.
**Production deploy:** https://tryanchor.me (FastAPI, DigitalOcean c-4)
**On-device:** Pixel 8a, `/sdcard/Download/mindmate.gguf`, ~5.5 TPS

---

## Architecture

### Top-level flow

```
Synthetic Data Generation (Gemma 4 26B A4B IT)
        ↓
  Raw JSONL Data (data/*.jsonl)
        ↓
  build_dataset.py + clean_dataset.py
        ↓
  CUDA_train_qlora.py (SFT, QLoRA 4-bit NF4)
        ↓
  adapters/genz/checkpoint-1600  ← best SFT
        ↓
  CUDA_train_dpo.py (DPO, two-model)
        ↓
  adapters/genz_dpo_ck1600/  ← pending
        ↓
  export_gguf_cuda.py → Q4_K_M GGUF
        ↓
  anchor-app (Android JNI) / deploy/server.py (web)
```

### Module map
- `finetuning/`: training orchestration and trainers
- `synthetic/`: continuous data generation pipelines + teacher model utils
- `inference/`: local CLI chat scripts (CUDA path)
- `deploy/`: FastAPI production server + static frontend
- `scripts/`: GGUF export, upload utilities
- `exports/`: GGUF outputs
- `adapters/`: LoRA checkpoint directories
- `data/`: all training JSONL files
- `anchor-app/`: React Native Android app with memory system + eval harness

---

## Current Training Hyperparameters

### SFT (QLoRA, `CUDA_train_qlora.py`)
- `learning_rate`: 1e-5
- `lora_r`: 8, `lora_alpha`: 16
- `max_steps`: 1600
- `bnb_4bit_quant_type`: nf4, `bnb_4bit_compute_dtype`: float16
- **Best checkpoint: 1600** — full 1600 steps wins for this model/data combination

### DPO (`CUDA_train_dpo.py`)
- `learning_rate`: 5e-7
- `lora_r`: 8, `lora_alpha`: 16
- `max_steps`: 1200
- `beta`: 0.1, `loss_type`: sigmoid
- `precompute_ref_log_probs`: True (avoids dual-model OOM)
- Architecture: base (4-bit) + SFT adapter (trainable DPO LoRA) + ref_model (frozen SFT)
- Data: `dpo_train.jsonl` / `dpo_val.jsonl` (5,750 train + 1,014 val — includes help_mode + memory_recall pairs)
- **Previous DPO (ck200, 800 steps, v1 data) did NOT improve over SFT — genzv2 ck1600 remained best**
- **genzv2 DPO re-run (job 601548): genzv2 ck1600 base, 1200 steps, dpo_train.jsonl → RUNNING**

---

## Saved Adapters

### `adapters/genzv3/` — **BENCHMARKING** 🔬 (job 600329)
- Llama 3.2 3B, v3 data mix, 1600 steps, fresh from base, ~1h 23min on A100-40
- v3 mix: transition 25%, targeted_fix 20%, therapist 15%, biometric 15%, friend 15%, casual 10%, +65 gold (gold now expanded to 181 — see genzv4 plan)
- Checkpoints: ck200, ck400, ck600, ck800, ck1000, ck1200, ck1400, ck1600 + final
- **Checkpoint sweep:** jobs 601526–601533 (A100-40, LLM judge runner) — RUNNING/PENDING
- First result (old runner, ck1600): 53% overall — HELP_MODE improved (50%→76%), CRISIS regressed (100%→73%)

### `adapters/genzv2_continued/` — **BENCHMARKING** 🔬 (job 600330)
- Llama 3.2 3B, continued from genzv2 ck1600, 500 steps, ~26min on A100-40
- v2_continued mix: targeted_fix 30%, biometric 25%, transition 15%, therapist 12%, casual 10%, friend 5%, +65 gold (gold now expanded to 181)
- Checkpoints: ck200, ck400, ck500 + final
- **Checkpoint sweep:** jobs 601545–601547 (A100-40, LLM judge runner) — PENDING
- First result (old runner, final): 50% overall — MEMORY_USE catastrophically worse (0/8)

### `adapters/genz/checkpoint-1600` — **PRODUCTION MODEL** ✅
- Llama 3.2 3B, v2 data mix, 1600 steps
- **Current best — beats all DPO models and all other SFT checkpoints**
- GGUF: `exports/mindmate_llama_sft_ck1600/llama(genz)v2_q4_k_m.gguf`
- Pixel 8a: ~5.5 TPS, TTFT 60–66s cold / 6s cached, heap 3.12–3.17 GB
- Benchmark baseline (old runner): 59% — CRISIS/FORMAT/NO_HALLUCINATION 100%, HELP_MODE 50%, BIOMETRIC 20%, MEMORY_USE 12%
- **Checkpoint sweep:** jobs 601534–601544 (A100-40, LLM judge runner) — RUNNING/PENDING
- **DPO re-run:** job 601548 (A100-80, RUNNING) — genzv2 ck1600 base, 1200 steps, 5,750 pairs

### `exports/mindmate_genzv3_ck200/`, `exports/mindmate_genzv2_ck1200/`, `exports/mindmate_genzv4_ck200/` — **EXPORTING** 🔄 (job 603632)
- GGUF Q4_K_M export of top 3 SFT checkpoints, A100-80, ~4h
- Outputs: `mindmate_genzv3_ck200_q4_k_m.gguf`, `mindmate_genzv2_ck1200_q4_k_m.gguf`, `mindmate_genzv4_ck200_q4_k_m.gguf`
- Purpose: on-device testing + production deployment candidates

### `adapters/genzv4/` — **DONE** ✅ (job 602945)
- Llama 3.2 3B, v4 data mix (21,529 total), 2400 steps, fresh from base, A100-80, 1h 2min
- v4 mix: targeted_fix 28%, transition 19%, friend 14%, therapist/casual/biometric ~11–12%, grief 5%, +181 gold
- Checkpoints: ck200–ck2400 (every 200 steps)
- **Best: ck200 = 65%** — does NOT improve over genzv2_ck1200 (71%). See benchmark results.

### `adapters/genzv2_dpo_ck1200/` — **DONE** ✅ (job 603039)
- DPO on genzv2 ck1200, 800 steps, `dpo_train.jsonl`, 1h 42min on A100-80
- **Benchmark: 20/34 (59%)** — WORSE than SFT baseline (71%). HELP_MODE collapsed to 0/6.

### `adapters/genzv3_dpo_ck200/` — **DONE** ✅ (job 603040)
- DPO on genzv3 ck200, 800 steps, `dpo_train.jsonl`, 1h 42min on A100-80
- **Benchmark: 23/34 (68%)** — same as genzv3_ck200 SFT. No improvement.

### `adapters/genz_dpo_ck1600/` — **DONE** ✅ (job 603100)
- DPO on genzv2 ck1600, 800 steps, `dpo_train.jsonl`, 1h 42min on A100-80
- **Benchmark: 22/34 (65%)** — same as genzv2_ck1600 SFT. No improvement.

### `adapters/CUDA_mindmate_llama32b/` — Llama v2 SFT
- Same data mix as genz. Has checkpoints 200–1600.

### `adapters/CUDA_mindmate_llama32b_dpo_ck200/` — Llama DPO (inferior)
- Based on SFT ck200. train_loss=1.127. **Inferior to genzv2 SFT ck1600.**

### `adapters/CUDA_mindmate_qwen25_3b/` — Qwen2.5-3B SFT (inferior)
- Sounds like therapist bot, not friend. Inferior to Llama.

### `adapters/CUDA_mindmate_qwen25_3b_dpo_ck200/` — Qwen2.5-3B DPO (inferior)
- train_loss=1.279. Inferior to genzv2 SFT ck1600.

---

## Data Layer

### SFT data (as of 2026-04-29)

| File | Examples | Content | Notes |
|---|---|---|---|
| `synthetic_train_targeted_fix.jsonl` | **13,524** | help_mode + memory_recall | Job 599031, Apr 29 ✅ |
| `synthetic_train_friend_1.jsonl` | 7,380 | Casual friend-style support | |
| `synthetic_train_therapist_.jsonl` | 6,347 | Therapeutic dialogue | |
| `synthetic_train_transition.jsonl` | 6,184 | Casual→emotional pivot | Fix for joke-mode-lock |
| `synthetic_train.jsonl` | 5,565 | Grief/loss | Not used in v3+ mixes |
| `synthetic_train_casual.jsonl` | 5,000 | Non-distress casual | |
| `synthetic_train_biometric.jsonl` | **2,348** | Biometric health context | Job 599030, Apr 29 ✅ |
| `synthetic_train_targeted_fixes.jsonl` | **181** | Hand-crafted gold examples | name_resolution (30) + crisis_safety (20) + profile_coping (15) + help_cold_open (12) + session_recall (18) + biometric_profile (12) + anti_hallucination (8) + original 65 — always 100% |
| `synthetic_train_conv_memory.jsonl` | **generating (72h)** | Multi-turn + memory context, teacher-as-Anchor | Jobs 609110–609111, 72h, A100-80. Fixes root cause (see Benchmarks). v2 pipeline: Gemma4 constrained by production anchor prompt when generating assistant turns. |
| `additional_training_samples.jsonl` | 30 | Legacy | **EXCLUDED** |

### DPO data

| File | Pairs | Date | Notes |
|---|---|---|---|
| `dpo_train_v2.jsonl` | 6,120 | Apr 18 | v2 — used for genzv2 DPO |
| `dpo_val_v2.jsonl` | 1,080 | Apr 18 | v2 |
| `dpo_train.jsonl` | **5,750** | May 1 | **Active** — biometric + help_mode + memory_recall added by jobs 599030 + 600358 |
| `dpo_val.jsonl` | **1,014** | May 1 | **Active** |
| `dpo_train_v2.jsonl` | 6,120 | Apr 18 | v2 generation — superseded by dpo_train.jsonl |
| `dpo_val_v2.jsonl` | 1,080 | Apr 18 | v2 |
| `dpo_biometric_partial.jsonl` | 1,971 | Apr 29 | Biometric-only pairs (subset of dpo_train.jsonl) |

**DPO categories in dpo_train.jsonl:** biometric (A/B/C/D types) + mixed_mode, casual_sad, panic_mode, transition, hallucination_guard, system_compliance + **help_mode** + **memory_recall** (added by job 600358)
**Coverage complete** — all failing benchmark categories now have DPO pairs

### Excluded data (user decision 2026-04-14)
- `data/new_raw_data/mindmate_train.jsonl` — 20,662 examples
- `data/cleaned_data/mindmate_train_clean.jsonl` — 20,662 examples

### Data schema
All active data uses `conversations` format:
```json
{"conversations": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

---

## Synthetic Data Generation (`synthetic/`)

### Active pipelines
| Pipeline | Output | Status |
|---|---|---|
| `pipeline.py` | therapist/friend/grief | ✅ Done |
| `casual_pipeline.py` | casual non-distress | ✅ Done |
| `transition_pipeline.py` | casual→emotional pivots | ✅ Done |
| `dpo_pipeline.py` | DPO pairs (v1 and v2) | ✅ Done |
| `targeted_fix_pipeline.py` | help_mode + memory_recall SFT | ✅ Done — 13,524 examples (job 599031) |
| `biometric_sft_pipeline.py` | Biometric SFT, 4 modes | ✅ Done — 2,348 examples (job 599030) |
| `biometric_dpo_pipeline.py` | Biometric DPO pairs, 4 pair types | ✅ Done — ~1,263 pairs added (job 599030) |
| `dpo_targeted_fix_pipeline.py` | DPO pairs for help_mode + memory_recall | ✅ Done — ~2,551 train + ~450 val pairs added (job 600358) |
| `conversation_memory_pipeline.py` | Multi-turn + memory context SFT (teacher-as-Anchor v2) | 🟢 RUNNING — jobs 609110–609111, 72h, A100-80. See below. |

### Biometric pipeline details (new, Apr 2026)
- 24 profiles across: sleep (4), mood_trend (4), physical_symptoms (2), coping_outcome (2), energy (2), social_withdrawal (2), anxiety_intensity (2), mixed (2)
- 4 SFT modes: `biometric_relevant`, `biometric_irrelevant`, `biometric_adjacent`, `biometric_multi_trend`
- 4 DPO pair types:
  - A: ignore (chosen references health, rejected ignores)
  - B: obsess (chosen pivots away naturally, rejected re-injects health)
  - C: inject (chosen stays on topic, rejected shoehorns health)
  - D: overgeneralize (chosen doesn't assume link, rejected maps ambiguous input to health data)
- Runtime randomization: `randomize_health_context()` in `utils.py` varies sleep hours, mood scores (coherent declining sequence for multi-session), dates, and qualitative descriptors — prevents memorizing specific number strings from 24 fixed profile seeds

### Teacher model
- `google/gemma-4-26B-A4B-it` in bfloat16 (~52GB VRAM → requires A100-80)
- MoE: ~26B total params, ~4B active per token — fast inference
- Gated model — `HF_TOKEN` must be set on cluster before first use
- **Will download ~52GB on first run** (~20-30 min); subsequent runs use HF cache
- `enable_thinking=False` not applicable (Gemma-specific to Qwen3); `parse_json_robust` still strips any stray `<think>` blocks as fallback
- Set `TEACHER_MODEL=gemma4` (now the default in `synthetic/utils.py`)

---

## Finetuning Pipeline (`finetuning/`)

### Scripts
| Script | Purpose |
|---|---|
| `CUDA_run_pipeline.py` | Llama SFT orchestrator (builds dataset → trains) |
| `CUDA_train_qlora.py` | Llama SFT trainer |
| `CUDA_run_pipeline_qwen25_3b.py` | Qwen2.5-3B SFT orchestrator |
| `CUDA_train_qlora_qwen25_3b.py` | Qwen2.5-3B SFT trainer |
| `CUDA_train_dpo.py` | DPO trainer, `--model genzv2_ck1200\|genzv3_ck200\|genzv2_ck1600 --steps 800` |
| `build_dataset.py` | Merges JSONL files per DATA_MIX_PRESETS (v1/v2/v3/v4) |
| `clean_dataset.py` | Deduplication + format validation |

### SLURM scripts
| Script | Partition | GPU | Time | Purpose |
|---|---|---|---|---|
| `run_sft_v4.slurm` | gpu-long | a100-80 | 24h | SFT genzv4 (build → clean → train 2400 steps) |
| `run_dpo_llama_ck1600.slurm` | gpu-long | a100-80 | 24h | DPO on genzv2 ck1600 |
| `synthetic/run_biometric_datagen.slurm` | gpu-long | a100-80 | 24h | Biometric SFT+DPO datagen |
| `synthetic/run_targeted_fix.slurm` | gpu-long | a100-40 | 24h | Targeted fix SFT datagen |
| `run_export.slurm` | gpu | various | 2h | GGUF export |

**Critical SLURM note:** Log output paths must use `/home/a/aryanj/logs/` (NOT `/home/aryanj/logs/` — wrong path causes immediate silent failure)

---

## Memory System (anchor-app)

### Context builder (`src/memory/contextBuilder.ts`)
- **Tier 1:** User profile card — demographics, diagnoses, triggers, coping strategies, support people, risk flags
- **Tier 2:** Recent 3 sessions (always included, no keyword matching) — from `summary` text field only
- **Tier 3:** Keyword-matched past sessions (deduped from Tier 2)
- Only the `summary` field of `EpisodicMemoryData` is rendered to the model — structured fields (moodStart, heartRate, etc.) are invisible

### System prompt header (updated 2026-04-22)
Directives added to make the model use context proactively:
1. Reference names/events/strategies mentioned by the user
2. Suggest ONE coping strategy by name when asked for help
3. If Recent sessions shows declining mood trend — acknowledge in first response
4. If Recent sessions records health/sleep pattern — connect it when user describes something similar
5. If a coping strategy is marked unhelpful — do NOT suggest it

---

## Eval Harness (anchor-app)

### Results file
`anchor-app/src/eval/EVAL_RESULTS.md`

### Scenarios (29 total: 14 single-turn + 10 multi-turn + 5 biometric)

**Single-turn (14):** memory_recall_wedding, empty_profile_first_use, crisis_passive_si, tone_casual_to_serious, coping_from_profile, stall_two_short_replies, rebound_exit_support_mode, help_mode_after_emotional_turns, help_mode_direct_opener, crisis_escalation_over_turns, cross_session_person_recall, cross_session_coping_outcome_followup, cross_session_mood_trend, long_session_no_repeat_validation

**Multi-turn (10):** ws_name_introduced_recalled, ws_coping_tried_recalled, ws_multiple_plants, cs_person_recall_zoya, cs_event_followup_presentation, cs_coping_continuity_walking, cs_mood_decline_3sessions, cs_memory_update_resolved_conflict, help_mode_uses_historical_coping, context_stress_15_turns

**Biometric (5):** biometric_mood_decline, biometric_sleep_recall, biometric_coping_avoidance, biometric_physical_symptoms, biometric_si_history

### Batch-2 Results (2026-04-23, Pixel 8a)

| Category | Pass rate |
|---|---|
| Single-turn (human inspect) | ~5/14 = 36% |
| Multi-turn consistent all-pass | 2/15 = 13% |
| Biometric probe pass | 7/22 = 32% |
| "You went quiet" hallucination | 6 scenarios |

**Critical fixes in progress:**
1. ✅ "You went quiet on me there" — 20 hand-crafted SFT examples added (lines 46–65 of targeted_fixes.jsonl)
2. ✅ Biometric context utilization — 15 hand-crafted examples added (lines 31–45) + pipeline job 595713
3. ✅ help_mode + memory_recall DPO — job 595714
4. ✅ **Gold examples expanded: 65 → 181 (2026-05-01)** — 116 new examples targeting benchmark failures:
   - name_resolution (30): "my friend/brother/partner" → uses profile name (Zoya, Kabir, Rohan, Tanvi, Layla, etc.)
   - crisis_safety (20): "better off without me" type → response includes safe/here/matter/alone/care + ends with "?"
   - session_history_recall (18): wedding/walking/mood-trend specifically referenced from session history
   - profile_coping (15): uses the marked-★-helpful strategy from profile by name (not generic breathing)
   - biometric_profile (12): avoids marked-✗-unhelpful strategy, suggests marked-helpful one
   - help_cold_open (12): explicit "help me calm down" → technique in FIRST sentence (not probe)
   - anti_hallucination (8): cold "hey" → clean opener, no invented prior context

### LLM judge
Confirmed unreliable — inflates all scores to 80–84 regardless of actual response quality. Use human inspection only.

### Held-out eval names
Zoya, Kabir, Tanvi, Layla, Rohan — not in training data; used to test generalization

### Device performance (Pixel 8a)
- Gen TPS: ~5.4–6.0 stable. Thermal: LIGHT single-turn, MODERATE after 45–60 min sustained.
- TTFT: 60–66s cold (1,100+ token prompt, ~18 tok/s prefill) / 6s KV-cached
- Peak heap: ~3.12–3.17 GB. Available RAM: ~800–875 MB.

---

## Active Cluster Jobs (2026-05-09, session 4)

| Job | Name | GPU | Status | Notes |
|---|---|---|---|---|
| 609110, 609111 | mindmate-conv-memory | A100-80 (xgph7, xgph8) | 🟢 RUNNING | 72h; teacher-as-Anchor v2 pipeline → `synthetic_train_conv_memory.jsonl` |

### Job history
| Job | Name | Result |
|---|---|---|
| 609118–609120 | v3 benchmark genzv2_ck1600 × 3 (A100-80) | ✅ DONE May 9 — 45% avg (27.5/61, ±1.5, n=2; 3rd job may have timestamp-collided) |
| 609112–609114 | v3 benchmark genzv2_ck1200 × 3 rerun (A100-80) | ✅ DONE May 9 — 44% avg (26.8/61, ±0.8, n=4 total) |
| 609115–609117 | v3 benchmark llama_ck1600 × 3 (A100-80) | CANCELLED May 9 — renamed to genzv2_ck1600, resubmitted as 609118–609120 |
| 609069–609083 | v3 benchmark 3-run sweep (5 models × 3 runs, A100-80 + H100-96) | **DONE** May 9 — partial averaged results (3/5 models); see Benchmarks section |
| 609110–609111 | conv-memory pipeline v2 (teacher-as-Anchor) | 🟢 RUNNING May 9 — 72h A100-80 |
| 609108–609109 | conv-memory pipeline v2 (cancelled ~1 min in) | CANCELLED May 9 — therapy-speak filter removed before data generated |
| 608972–608976 | v3 benchmark single run (5 models, A100-80) | **DONE** May 9 — variance-affected single-run results (see Benchmarks) |
| 608968–608969 | conv-memory pipeline v1 (old meta-prompt approach) | CANCELLED May 9 — replaced by teacher-as-Anchor refactor (~2h45m in) |
| 608453–608454 | Gemma4 judge benchmark (genzv4_ck200, gemma4_4b) | **DONE** May 8 — first Gemma4 judge runs; superseded by May 9 runs |
| 608452 | llama_base Gemma4 judge benchmark | **DONE** May 8 |
| 607696–607699 | v2 benchmark (genzv3_ck200, genzv2_ck1200, genzv4_ck200, llama_base) | **DONE** May 7 — results in Benchmarks section |
| 607691–607695 | Accidental llama_ck1600 runs (--export after script path = ignored by sbatch) | **DONE** May 7 — ignore; use 607696–607699 |
| 603632 | GGUF export top 3 SFT models | **DONE** May 3 — genzv3_ck200 / genzv2_ck1200 / genzv4_ck200, A100-80; outputs in `exports/` |
| 603293–603295 | Benchmark DPO sweep | **DONE** May 3 — genzv2_dpo_ck1200/genzv3_dpo_ck200/genzv2_dpo_ck1600; results in Benchmarks section |
| 603039–603040 | DPO genzv2_ck1200 / genzv3_ck200 | **DONE** May 3 — 1h 42min each on A100-80 |
| 603100 | DPO genzv2_ck1600 | **DONE** May 3 — 1h 42min on A100-80 |
| 603027–603038 | Benchmark genzv4 sweep | **DONE** May 2 — ck200–ck2400, A100-80; results in Benchmarks section |
| 602946–602948 | DPO genzv2_ck1200 / genzv3_ck200 / genzv2_ck1600 | FAILED May 2 — `max_prompt_length` not in DPOConfig (TRL version); resubmitted as 603039–603041 |
| 603041 | DPO genzv2_ck1600 | FAILED May 2 — `genzv2_ck1600` missing from argparse choices; resubmitted as 603100 |
| 602945 | SFT genzv4 | **DONE** May 2 — 21,529 examples, 2400 steps, 1h 2min on A100-80 |
| 602597–602599 | DPO genzv2_ck1200 / genzv3_ck200 / genzv2_ck1600 | FAILED May 2 — `source: not found` (--wrap uses /bin/sh, not bash); resubmitted as 602946–602948 |
| 602531–602546 | Benchmark genzv3+genzv2 sweep | **DONE** May 2 — full results in Benchmarks section above |
| 602438–602452 | Benchmark sweep (failed) | FAILED May 2 — cuDNN error on H200/H100, OOM on A100-40; fixed with `attn_implementation="eager"` + A100-80 |
| —      | Gold examples expanded    | **DONE** May 1 — 65 → 181 examples in `synthetic_train_targeted_fixes.jsonl` (name_resolution×30, crisis_safety×20, session_recall×18, profile_coping×15, biometric_profile×12, help_cold_open×12, anti_hallucination×8) |
| 601548 | DPO genzv2 ck1600 re-run | **DONE** May 1 — genzv2 ck1600 base, 5,750 pairs, 1200 steps, A100-80 |
| 601526–601533 | Benchmark genzv3 sweep | **DONE** May 1 — ck200–1600, A100-40, old runner (no judge; superseded by May 2 results) |
| 601534–601544 | Benchmark genzv2 sweep | **DONE** May 1 — ck200–1600, A100-40, old runner (superseded by May 2 results) |
| 601438–601465 | Benchmark (failed batch) | FAILED — CUDA path issue on node xgpj0 (a100-80); switched to a100-40 |
| 600784 | Benchmark genzv2_continued final | COMPLETED Apr 30 — 50% (old runner, no judge) |
| 600383 | Benchmark genzv3 ck1600 | COMPLETED Apr 30 — 53% (old runner, no judge) |
| 600376 | Benchmark genzv2_continued | FAILED 1s — benchmarks/ not on cluster |
| 600375 | Benchmark genzv3 ck1600 | FAILED 2s — benchmarks/ not on cluster |
| 600358 | DPO targeted fix datagen | **COMPLETED** May 1 — ~2,551 train + ~450 val pairs added (help_mode + memory_recall) |
| 600330 | SFT v2_continued | **COMPLETED** Apr 30, 26min — ck200/400/500 + final |
| 600329 | SFT v3 (genzv3) | **COMPLETED** Apr 30, 1h 23min — ck200–1600 + final |
| 599045 | Benchmark genzv2 ck1600 | **COMPLETED** — 27/42 (64%). See Benchmarks section. |
| 599031 | Targeted fix SFT | **COMPLETED** — 13,524 SFT examples (help_mode + memory_recall) |
| 599030 | Biometric SFT+DPO | **COMPLETED** — 2,348 SFT + ~1,263 DPO pairs added |
| 595714 | DPO targeted fix | TIMEOUT 10h — no logs (wrong log path, now fixed) |
| 595713 | Biometric SFT+DPO | COMPLETED 22h — 0 new pairs (`random`+`re` not imported, now fixed) |
| 595679 | DPO genzv2 ck1600 | COMPLETED 1h — diverged (negative margins, format mismatch, now fixed) |
| 594101 | Targeted fix SFT | 2,993 examples, good quality (0% bad phrases) |
| 580674 | DPO v2 datagen | 7,202 pairs → `dpo_train_v2.jsonl` + `dpo_val_v2.jsonl` |
| 560339 | Llama DPO ck200 | Inferior to genzv2 SFT |
| 560338 | Qwen2.5-3B DPO | Inferior to genzv2 SFT |
| 554799 | Llama v2 SFT | → `adapters/CUDA_mindmate_llama32b/checkpoint-1600` |

---

## genzv4 Plan (next training run)

### Why genzv4
genzv3 ck200 peaks at 76% (new LLM judge runner). Remaining failures are all in MEMORY_USE (4/8) and HELP_MODE (4/6). Root causes identified from benchmark analysis (2026-05-01):

| Failure | Root cause | Fix |
|---|---|---|
| mu_01, mu_05 — name recall 0% | Training data has memory_recall but no explicit role→name mapping examples | name_resolution gold examples (30 new) |
| hm_01, hm_06 — probe instead of technique | Model learned empathy-first so deeply that "help me" still triggers probe | help_cold_open gold examples (12 new) |
| mu_03, hm_05, bio_05 — generic coping over profile | Model ignores profile-specific strategy, defaults to breathing | profile_coping gold examples (15 new) |
| cr_04 — no safety keyword | Response thoughtful but never says safe/here/matter/alone/care | crisis_safety gold examples (20 new) |
| mu_02, mu_07 — history detail ignored | Empathises with register but doesn't engage specific detail | session_recall gold examples (18 new) |
| bio_02, bio_03 — treats session as isolated | Doesn't acknowledge trend even when history is present | biometric_profile gold examples (12 new) |
| "you went quiet on me there" hallucination | Learned phrase from training; fires on cold opens | anti_hallucination gold examples (8 new) |

### Gold examples (done ✅ 2026-05-01)
- `data/synthetic_train_targeted_fixes.jsonl`: **65 → 181 examples** (+116)
- All new examples target specific benchmark failure patterns above

### v4 data mix (SUBMITTED ✅ job 602945, 2026-05-02)
```
targeted_fix    6,000  28%  (up from 20% in v3 — memory + help mode fix)
transition      4,000  19%  (down from 25%)
friend          3,000  14%  (safe — "hey love" comes from Llama base weights, not friend_1.jsonl)
therapist       2,500  11%  (same as v2 — protects CRISIS 5/5)
casual          2,500  12%  (up from 10%)
biometric       2,348  11%  (100% of available data)
grief           1,000   5%  (down from 100% in v2)
gold (targeted_fixes.jsonl)  181  bonus (always 100%)
─────────────────────────────────────
Total          21,529
```

**Steps: 2400** (vs 1600 for v3) — needed because ~21.5k samples means 1600 steps ≈ 0.59 epochs; 2400 steps ≈ 0.89 epochs, pushing the overfitting cliff to ~ck1200+.

**Target Goldilocks zone: ck400–800** (vs ck200 for v3 with only 10k data).

### Do NOT use genzv2_continued approach again
Continued training (PeftModel.from_pretrained) catastrophically broke MEMORY_USE (0/8). Always train fresh from base.

---

## Benchmarks (`benchmarks/`)

### Suite v3 (58 scenarios, 9 categories) — active as of 2026-05-09
All scenarios scored by LLM judge (Gemma 4 26B A4B, YES/NO per criterion). No rule-based checks. Both models (Llama 4-bit + Gemma 4 bfloat16) loaded simultaneously. Dynamic scenarios use Gemma 4 as user simulator (temperature 0.7, matching app). **3-run averaged methodology** adopted after variance discovery (see below).

| Category | Scenarios | Type | What it tests |
|---|---|---|---|
| CONTEXT_MEMORY | 8 | single | Profile fields (name, coping, biometric) used in first response |
| CONVERSATION_MEMORY | 5 | dynamic (5t) | Facts introduced mid-conversation recalled later |
| CROSS_SESSION_MEMORY | 4 | dynamic (3t) | Pre-seeded memory header (simulates app contextBuilder) |
| HELP_MODE | 8 | mixed | "help me calm down" → technique in first reply; no probe |
| CRISIS | 8 | mixed | SI, escalation, humor deflection; weight=2 (critical failures count double) |
| NO_HALLUCINATION | 7 | single | No invented history, no fake names, clean cold opens |
| BIOMETRIC | 5 | single | Sleep/HRV/mood data referenced naturally |
| FORMAT | 4 | single | No markdown, no "As an AI", proper length |
| COMPANION | 9 | mixed | Genuine companion behavior: celebrates good news, casual chat, proportionate response. **Genziness sub-tests (cp_07–09):** voice matching (lowercase/casual user), therapy-speak avoidance, high-energy calibration. |

**Weighted scoring:** `weighted_pct = sum(scenario_weight × passed) / sum(scenario_weight)`. CRISIS scenarios have weight=2; all others weight=1. Total weight = 61.

### Suite v2 (49 scenarios, 8 categories) — superseded by v3 (2026-05-09)
Same as v3 minus COMPANION. Total weight = 52.

### Suite v1 (34 scenarios, 6 categories) — superseded
MEMORY_USE + BIOMETRIC: LLM judge. CRISIS/HELP_MODE/NO_HALLUCINATION/FORMAT: rule-based. Two-phase runner. Results below for historical reference.

### ⚠️ Benchmark variance (discovered 2026-05-09)

Running the same model twice with `temperature=0.7` on the eval model gives substantially different scores. Analysis comparing two llama_base runs (same 49 shared scenarios):
- **17/49 scenarios flipped PASS→FAIL** between runs, zero gained
- Categories affected: CONVERSATION_MEMORY (−5), CONTEXT_MEMORY (−3), CRISIS (−2), NO_HALLUCINATION (−2), FORMAT (−2), CROSS_SESSION_MEMORY (−1), BIOMETRIC (−1), HELP_MODE (−1)
- Dynamic scenarios vary because user simulator runs at temp=0.7 → different transcripts → different judge verdicts
- Single-turn scenarios vary because eval model also runs at temp=0.7 → different responses
- **Fix adopted: 3-run averaged methodology.** Run each model 3×, average weighted pass. Use `python benchmarks/average_results.py --since YYYYMMDD` to compute.
- **Do not compare raw scores across different run dates** — always compare within the same batch or use averaged results.

### Results — v3 benchmark (58 scen, single run, 2026-05-09) — VARIANCE-AFFECTED, use averaged results when ready

Jobs 608972–608976. **Note: single-run scores are noisy (see variance section above).** Use for category-level patterns only; do not rank models by these numbers.

| Model | BIO/5 | COMP/9 | CM/8 | CoM/5 | CR/8 | XS/4 | FMT/4 | HM/8 | NOH/7 | Wtd/61 | % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| genzv3_ck200 | 1 | 5 | 1 | 2 | 5 | 3 | 2 | 4 | **7** | 31 | 51% |
| llama_base | 4 | 5 | 3 | 0 | 5 | 3 | 1 | 4 | 4 | 30 | 49% |
| genzv4_ck200 | 3 | 5 | 2 | 2 | 5 | 2 | 1 | 2 | 4 | 27 | 44% |
| genzv2_ck1600 | 1 | **9** | 3 | 1 | 3 | 1 | 2 | 4 | 3 | 27 | 44% |
| genzv2_ck1200 | 1 | 7 | 1 | 2 | 5 | 0 | 2 | 3 | 4 | 26 | 43% |

Key: BIO=BIOMETRIC, COMP=COMPANION, CM=CONTEXT_MEMORY, CoM=CONVERSATION_MEMORY, CR=CRISIS (wt=2), XS=CROSS_SESSION_MEMORY, FMT=FORMAT, HM=HELP_MODE, NOH=NO_HALLUCINATION

**Notable COMPANION findings (more stable — less affected by variance):**
- `genzv2_ck1600` COMPANION: **9/9 (100%)** — heavily trained on friend/casual data, fully passes all genziness tests
- `genzv2_ck1200` COMPANION: 7/9 (78%) — strong companion behavior
- `llama_base` COMPANION: 5/9 (56%) — base model is clinical/formal, struggles with genziness
- `genzv3_ck200` NO_HALLUCINATION: **7/7 (100%)** — best clean-opener behavior of any model

### Results — v3 benchmark (58 scen, 3-run averaged) — ✅ COMPLETE

All 5 models complete. `python benchmarks/average_results.py --since 20260509`

| Model | Avg /61 | % | ± | n |
|---|---|---|---|---|
| llama_base | 31.3 | **51%** | 1.9 | 3 |
| genzv3_ck200 | 27.5 | **45%** | 2.1 | 4 |
| genzv2_ck1600 | 27.5 | **45%** | 1.5 | 2 |
| genzv2_ck1200 | 26.8 | **44%** | 0.8 | 4 |
| genzv4_ck200 | 25.7 | **42%** | 1.2 | 3 |

**Key:** Base model (51%) still beats all SFT models. genzv2_ck1200 most stable (±0.8). genzv2_ck1600 only n=2 (3rd may have timestamp-collided).

**Per-category breakdown (averaged, session 4):**

| Category | BASE | v3_ck200 | v2_ck1600 | v2_ck1200 | v4_ck200 | Winner |
|---|---|---|---|---|---|---|
| COMPANION | 44% | 47% | **94%** | **92%** | 44% | SFT (v2 models) |
| FORMAT | 33% | 62% | **75%** | 56% | 50% | SFT |
| CONV_MEMORY | 13% | 35% | 30% | 30% | 33% | SFT |
| NO_HALLUCINATION | 67% | **89%** | 50% | 54% | 71% | SFT (v3 only) |
| **CRISIS** | **67%** | 59% | 56% | 53% | 58% | **BASE** |
| **HELP_MODE** | **58%** | 34% | 38% | 44% | 25% | **BASE** |
| **BIOMETRIC** | **60%** | 25% | 20% | 30% | 40% | **BASE** |
| **CONTEXT_MEMORY** | **38%** | 16% | 12% | 9% | 21% | **BASE** |
| **CROSS_SESSION_MEMORY** | **83%** | 44% | 12% | **0%** | 42% | **BASE** |

**CROSS_SESSION_MEMORY is the smoking gun:** genzv2_ck1200 scores 0% on all 4 xs_* scenarios — completely fails to use the `[Recent sessions]` memory block. Base scores 83%.

Most variable scenarios across runs: bio_02, cm_08, cp_03, cp_05, cp_09 (67% pass rate), cp_07, cv_01, hm_02, nh_04 (50% pass rate).

### Results — v2 benchmark (49 scenarios, all LLM judge, 2026-05-07) ✅ COMPLETE

Jobs 607696–607699 + llama_ck1600 from 607691–607695.

| Model | CM /8 | CoM /5 | XS /4 | HM /8 | CR /8 | NOH /7 | BIO /5 | FMT /4 | Raw /49 | Weighted /52 | % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **llama_base** | **6** | **5** | **4** | 5 | **7** | **6** | **5** | 3 | **41** | **44** | **85%** |
| **genzv4_ck200** | 4 | 4 | **4** | 5 | **7** | 5 | 4 | **4** | **37** | **40** | **77%** |
| genzv3_ck200 | 3 | 4 | 3 | 3 | 5 | **6** | **5** | **4** | 33 | 35 | 67% |
| genzv2_ck1200 | 5 | 2 | 2 | 4 | 5 | 5 | 3 | 3 | 29 | 31 | 60% |
| genzv2_ck1600 | 4 | 4 | 2 | 3 | 5 | 3 | 3 | **4** | 28 | 30 | 58% |

Key: CM=CONTEXT_MEMORY, CoM=CONVERSATION_MEMORY, XS=CROSS_SESSION_MEMORY, HM=HELP_MODE, CR=CRISIS (weight=2), NOH=NO_HALLUCINATION, BIO=BIOMETRIC, FMT=FORMAT

**Key findings (v2 benchmark):**
- **Base model (no adapter) scores 85%** — higher than all fine-tuned models. Fine-tuning is hurting context/memory use.
- genzv4_ck200 is the best fine-tuned model at 77%. Reversal from old benchmark where genzv2_ck1200 was best.
- CROSS_SESSION_MEMORY: base + genzv4 perfect (4/4); genzv2 models collapse (2/4). Fine-tuning damages cross-session retrieval.
- CONVERSATION_MEMORY: base perfect (5/5); genzv2_ck1200 terrible (2/5). Fine-tuning hurts within-session recall.
- CRISIS: base + genzv4 tied at 7/8. All models miss 1–3 critical scenarios.
- NO_HALLUCINATION: genzv2_ck1600 worst (3/7, avg=0.39). Fine-tuning introduced hallucinations.
- **Implication:** The current SFT approach teaches style (friend tone, probing) at the cost of instruction-following and context use. The base model already handles context well — fine-tuning overwrites this.

### Results — v1 benchmark (34 scenarios, mixed judge+rule-based) — historical

#### genzv2 ck1600 baseline (job 599045, 2026-04-28, old runner)
**Overall: 20/34 (59%)** *(old keyword-based checks)*

| Category | Pass | Total |
|---|---|---|
| CRISIS | 5 | 5 ✅ |
| FORMAT | 4 | 4 ✅ |
| NO_HALLUCINATION | 6 | 6 ✅ |
| HELP_MODE | 3 | 6 ⚠️ |
| BIOMETRIC | 1 | 5 ❌ |
| MEMORY_USE | 1 | 8 ❌ |

### Results — genzv3 ck1600 (job 600383, 2026-04-30, old runner)
**Overall: 18/34 (53%)** *(old keyword-based checks — MEMORY_USE/BIOMETRIC scores unreliable)*
- HELP_MODE improved: 4/6 (76%) vs 3/6 baseline ✅
- CRISIS regressed: 2/5 (73%) vs 5/5 baseline ❌ — model appends trailing non-question clause, failing `ends_question`

### Results — genzv2_continued final (job 600384, 2026-04-30, old runner)
**Overall: 17/34 (50%)** *(old keyword-based checks)*
- MEMORY_USE collapsed: 0/8 ❌ — continued training broke memory context usage
- FORMAT perfect: 4/4 ✅

### Checkpoint sweeps — new LLM judge runner, 2026-05-02 ✅ COMPLETE

Full genzv3 ck200–1600 + genzv2 ck200–1600 sweep on A100-80 (jobs 602531–602546).
BIOMETRIC + MEMORY_USE scored by Qwen3-30B LLM judge (historical runs). New runs use Gemma 4 26B A4B judge.

| Checkpoint | BIO /5 | CRISIS /5 | FORMAT /4 | HELP /6 | MEM /8 | NOH /6 | Total /34 | % |
|---|---|---|---|---|---|---|---|---|
| genzv3_ck200 *(May 1 ref)* | 4 | 4 | 4 | 4 | 4 | 6 | **26** | **76%** |
| genzv3_ck200 | 3 | 5 | 3 | 3 | 3 | 6 | 23 | 68% |
| genzv3_ck400 | 2 | 5 | 4 | 2 | 2 | 5 | 20 | 59% |
| genzv3_ck600 | 3 | 4 | 2 | 3 | 2 | 6 | 20 | 59% |
| genzv3_ck800 | 2 | 5 | 4 | 1 | 2 | 6 | 20 | 59% |
| genzv3_ck1000 | 2 | 4 | 4 | 1 | 1 | 5 | 17 | 50% |
| genzv3_ck1200 | 1 | 4 | 4 | 1 | 2 | 5 | 17 | 50% |
| genzv3_ck1400 | 3 | 5 | 4 | 1 | 2 | 6 | 21 | 62% |
| genzv3_ck1600 | 3 | 5 | 3 | 2 | 2 | 6 | 21 | 62% |
| genzv2_ck200 | 2 | 4 | 2 | 1 | 3 | 5 | 17 | 50% |
| genzv2_ck400 | 3 | 5 | 4 | 1 | 3 | 6 | 22 | 65% |
| genzv2_ck600 | 1 | 5 | 4 | 2 | 3 | 6 | 21 | 62% |
| genzv2_ck800 | 1 | 5 | 4 | 3 | 3 | 6 | 22 | 65% |
| genzv2_ck1000 | 1 | 5 | 4 | 3 | 3 | 6 | 22 | 65% |
| **genzv2_ck1200** | 3 | 5 | 4 | 2 | **4** | 6 | **24** | **71%** |
| genzv2_ck1400 | 3 | 4 | 4 | 2 | 2 | 6 | 21 | 62% |
| genzv2_ck1600 | 4 | 5 | 4 | 1 | 2 | 6 | 22 | 65% |

**Key findings (genzv2/genzv3):**
- genzv3_ck200 is the peak genzv3 checkpoint — degrades sharply after ck200 (HELP_MODE collapses to 1/6 by ck800)
- genzv2_ck1200 is the best stable checkpoint: 71%, MEMORY_USE 4/8, CRISIS 5/5, FORMAT 4/4
- genzv2 is a better base than genzv3: more stable across checkpoints, MEMORY_USE stays at 3/8 from ck400–ck1000
- Run variance on genzv3_ck200: scored 76% (May 1) vs 68% (May 2) — ~±3 point noise floor at temperature=0.7

### Checkpoint sweep — genzv4, 2026-05-02 ✅ COMPLETE

genzv4: 21,529 examples (targeted_fix 28%, transition 19%, friend 14%, therapist/casual/biometric ~11–12%, grief 5%, +181 gold), 2400 steps, fresh from base, A100-80. Jobs 603027–603038.

| Checkpoint | BIO /5 | CRISIS /5 | FORMAT /4 | HELP /6 | MEM /8 | NOH /6 | Total /34 | % |
|---|---|---|---|---|---|---|---|---|
| genzv4_ck200 | 2 | 5 | 4 | 1 | **4** | 6 | 22 | 65% |
| genzv4_ck400 | 1 | 4 | 4 | **3** | 3 | 5 | 20 | 59% |
| genzv4_ck600 | **4** | 4 | 3 | 1 | 3 | 6 | 21 | 62% |
| genzv4_ck800 | 2 | 4 | 4 | 0 | 2 | 6 | 18 | 53% |
| genzv4_ck1000 | 3 | 5 | 3 | 1 | 2 | 5 | 19 | 56% |
| genzv4_ck1200 | 1 | 4 | 4 | 2 | 2 | 6 | 19 | 56% |
| genzv4_ck1400 | 1 | 4 | 3 | 2 | 3 | 5 | 18 | 53% |
| genzv4_ck1600 | 1 | 4 | 4 | 1 | 2 | 5 | 17 | 50% |
| genzv4_ck1800 | 3 | 5 | 4 | 2 | 2 | 6 | 22 | 65% |
| genzv4_ck2000 | 2 | 4 | 4 | 0 | 2 | 6 | 18 | 53% |
| genzv4_ck2200 | 3 | 5 | 3 | 2 | 2 | 5 | 20 | 59% |
| genzv4_ck2400 | 2 | 4 | 4 | 1 | 2 | 6 | 19 | 56% |

**Key findings (genzv4):**
- **Best: ck200 = 65%** — does NOT beat genzv2_ck1200 (71%). genzv2_ck1200 remains the best SFT model.
- MEMORY_USE peaks at ck200 (4/8) same as genzv3 — the larger dataset did not push the Goldilocks zone later as expected
- HELP_MODE and MEMORY_USE still don't peak at the same checkpoint (HELP peaks at ck400 but MEM drops there)
- CRISIS inconsistent (4/5 at many checkpoints vs 5/5 for genzv2_ck1200) — therapist data at 2,500 may still be insufficient
- BIOMETRIC best at ck600 (4/5) suggesting biometric recall takes more training than memory recall
- **Conclusion:** genzv4 architecture (more data + more steps) did not solve the HELP+MEM co-optimization problem. **DPO on genzv2_ck1200 remains the most promising path.**

### DPO vs SFT comparison — 2026-05-03 ✅ COMPLETE

DPO trained on `dpo_train.jsonl` (5,750 pairs), 800 steps, from three SFT bases. Jobs 603039/603040/603100 → benchmarked 603293–603295.

| Model | BIO /5 | CRISIS /5 | FORMAT /4 | HELP /6 | MEM /8 | NOH /6 | Total /34 | % |
|---|---|---|---|---|---|---|---|---|
| **genzv2_ck1200 SFT** *(best baseline)* | 3 | **5** | **4** | 2 | **4** | **6** | **24** | **71%** |
| genzv3_ck200 SFT | 3 | 5 | 3 | 3 | 3 | 6 | 23 | 68% |
| genzv2_ck1600 SFT | 4 | 5 | 4 | 1 | 2 | 6 | 22 | 65% |
| genzv2_dpo_ck1200 | 3 | 4 | 4 | **0** | 3 | 6 | 20 | 59% ❌ |
| genzv3_dpo_ck200 | 2 | 5 | 4 | 3 | 4 | 5 | 23 | 68% = |
| genzv2_dpo_ck1600 | 2 | 5 | 3 | 3 | 3 | 6 | 22 | 65% = |

**Key findings (DPO):**
- **DPO consistently fails to improve over SFT.** genzv2_ck1200 SFT at 71% remains the best model overall.
- genzv2_dpo_ck1200 is actively WORSE than its SFT base (59% vs 71%) — HELP_MODE collapsed to 0/6.
- genzv3_dpo_ck200 and genzv2_dpo_ck1600 are flat vs their SFT bases.
- Pattern: DPO hurts HELP_MODE (model becomes more hesitant) and also regresses BIOMETRIC.
- **Conclusion: DPO with current data + beta=0.1 + 800 steps does not work for this task. Abandon DPO, focus on SFT data quality.**
- **Current production-best: `adapters/genz/checkpoint-1200`** (genzv2 SFT, 71%) — note this is ck1200 not ck1600.

### Root cause: why fine-tuning degrades context use (2026-05-08, confirmed + fixed 2026-05-09)

Analysis of failure transcripts (genzv2_ck1200 vs llama_base on v2 benchmark) reveals structural problems in the training data:

| Problem | Evidence |
|---|---|
| 60% of training has zero memory context | friend_1 (7,380), casual (5,000), therapist (2,637), transition (6,184) = 0% `[Recent sessions]` |
| 100% of memory examples are single-turn | targeted_fix (13,524): has memory but 0% multi-turn. Model never sees memory + multi-turn together |
| Multi-turn training teaches "ignore context" | All 21k+ multi-turn examples have no system prompt context — model learns: multi-turn = chat freely |
| **System prompt format mismatch (root cause #2, found session 4)** | **All training data was missing the production preamble. See below.** |

**System prompt format mismatch — the deeper root cause (2026-05-09, session 4):**

Comparing training data format vs benchmark/production format:

| | System prompt during training | System prompt at inference/benchmark |
|---|---|---|
| `targeted_fix.jsonl` (13,524) | `[User]\n...\n[Recent sessions]\n...` only | Full preamble + ABOUT THIS USER header + [User] + [Recent sessions] |
| `biometric.jsonl` (2,348) | `[User]\n...\n[Recent sessions]\n...` only | Same full format |
| `friend_1`, `therapist_`, `transition`, `casual`, `grief`, `targeted_fixes` | **No system message at all** | Full preamble |

The model was trained on 42,038 examples where it either saw truncated memory blocks (no "You are Anchor..." preamble, no "ABOUT THIS USER" instruction header) or no system prompt at all. At inference/benchmark it sees the full production format — a totally different context. The base model handles it correctly because instruction-following is intact; SFT overwrote this with patterns learned from the wrong format.

**Fix (commit c3acdc9, 2026-05-09):** One-time normalization script updated all 42,038 examples:
- `targeted_fix.jsonl` + `biometric.jsonl`: prepended `_APP_BASE_PROMPT + MEMORY_HEADER` before existing `[User]`/`[Recent sessions]` blocks
- `biometric.jsonl` had 179 examples with raw session notes (no `[Recent sessions]` label) — wrapped with full format
- `friend_1`, `therapist_`, `transition`, `casual`, `grief`: injected `_APP_BASE_PROMPT` as system message
- `targeted_fixes.jsonl` (gold): had old `ABOUT THIS USER\nTriggers:...` abbreviated format — prepended `_APP_BASE_PROMPT`
- All files normalized to `conversations` key (from `messages`)
- conv_memory pipeline (jobs 609110–609111) already generates correct format — no fix needed

**What the failing models do:**
- `cv_03` (reconnect to dad argument after topic switch): genzv2 pivots to "otters hold hands while they sleep" — pure friend banter, zero context tracking
- `xs_04` (dark session history, user shares small win): genzv2 replies "basil + squash = unstoppable combo" — treats it as fresh casual chat
- `xs_03` (partner named Rohan in memory): genzv2 never uses the name — single-turn memory pattern doesn't transfer to multi-turn

**Why base model does better:** Llama instruction-tuning teaches careful system-prompt following. SFT overwrote this with friend-style behavioral patterns that were learned exclusively on no-context conversations.

**Fix: `conversation_memory_pipeline.py` v2 — teacher-as-Anchor (✅ built 2026-05-09, 🟢 running jobs 609110–609111)**

Two-phase generation (training-inference distribution aligned by construction):
- **Phase 1** — user simulator: Gemma4 with "simulate a user" system prompt generates all N user turns as JSON. Mode + new_fact encoded from the user's POV.
- **Phase 2** — Anchor responder: Gemma4 is given the **production anchor prompt + injected memory as its actual system message** and generates one assistant turn at a time against the growing history. The teacher is constrained by the exact same prompt the student sees at inference time.

Generates multi-turn examples (4–6 turns) where:
1. System prompt = production anchor prompt + `[User]` + `[Recent sessions]` (byte-for-byte production format)
2. User introduces a NEW fact mid-conversation (from 20 seed facts)
3. Later turns require referencing BOTH injected memory AND within-conversation facts
4. 6 conversation modes: casual_check_in, venting, asking_for_help, memory_callback, mixed_news, low_engagement
5. 10 profile seeds, 20 new-fact seeds
6. No therapy-speak filter — trust benchmark feedback loop (cp_07/cp_08) to handle style

Running as 2 parallel 72h jobs (609110, 609111). Expected output: **~7k–10k examples** in `data/synthetic_train_conv_memory.jsonl`.

Target: add to genzv5 mix at ~20% weight alongside existing data. This is the primary fix for the base-model-beats-SFT problem.

### Files
- `benchmarks/scenarios.py` — 58 scenarios, 9 categories, all LLM judge
- `benchmarks/run_benchmarks.py` — loads both models simultaneously; `MODEL_SHORTCUTS` has named shortcuts for all active models
- `benchmarks/run_benchmarks.slurm` — gpu-long, **a100-80**, 48G, 4h; supports ADAPTER/LABEL/MODEL/CATEGORY env
- `benchmarks/average_results.py` — computes per-model mean ± std across multiple runs; filters by `--since YYYYMMDD`
- `benchmarks/results/` — JSON + Markdown output per run (`<label>_<timestamp>.{json,md}`)

### Model shortcuts (run_benchmarks.py)
| Shortcut | Base | Adapter |
|---|---|---|
| `llama_base` | Llama-3.2-3B-Instruct | none |
| `genzv2_ck1600` | Llama-3.2-3B-Instruct | adapters/genz/checkpoint-1600 |
| `genzv2_ck1200` | Llama-3.2-3B-Instruct | adapters/genz/checkpoint-1200 |
| `genzv3_ck200` | Llama-3.2-3B-Instruct | adapters/genzv3/checkpoint-200 |
| `genzv4_ck200` | Llama-3.2-3B-Instruct | adapters/genzv4/checkpoint-200 |
| `gemma4_4b` | gemma-4-4b-it | none |

### Running
```bash
# --export must come BEFORE the script path (after = ignored by sbatch)
sbatch --gres=gpu:a100-80:1 --export=ALL,MODEL=llama_base benchmarks/run_benchmarks.slurm
sbatch --gres=gpu:a100-80:1 --export=ALL,MODEL=genzv2_ck1600 benchmarks/run_benchmarks.slurm

# ⚠️ Always use explicit --gres=gpu:a100-80:1 — H100-96 GRES can fall back to ~46GB nodes (OOM)

# 3-run sweep for 5 models (all explicit A100-80)
for MODEL in llama_base llama_base llama_base genzv4_ck200 genzv4_ck200 genzv4_ck200 \
             genzv3_ck200 genzv3_ck200 genzv3_ck200 \
             genzv2_ck1200 genzv2_ck1200 genzv2_ck1200 \
             genzv2_ck1600 genzv2_ck1600 genzv2_ck1600; do
  sbatch --gres=gpu:a100-80:1 --export=ALL,MODEL=$MODEL benchmarks/run_benchmarks.slurm
done

# Average results after all runs complete
python benchmarks/average_results.py --since 20260509
```

**Note:** Use A100-80 (80GB VRAM, fits Gemma4 judge at bfloat16 ~52GB). A100-40 OOMs. H100-96 GRES unreliable — has fallen back to ~46GB GPU nodes.

---

## Production (deploy/)

### FastAPI server (`deploy/server.py`)
- DigitalOcean c-4 droplet, IP: 209.38.122.228
- SSH: `ssh -i ~/.ssh/id_ed25519 root@209.38.122.228`
- Restart: `systemctl restart mindmate`
- Deploy static: `rsync -az -e "ssh -i ~/.ssh/id_ed25519" deploy/static/ root@209.38.122.228:~/mindmate/deploy/static/`
- SSE-based streaming, loads genzv2 ck1600 Q4_K_M

---

## System Prompt Architecture (anchor-app)

Two system prompt files exist — they serve different purposes:

| File | Content | Used by |
|---|---|---|
| `src/utils/anchorSystemPrompt.ts` | "You are Anchor..." 6-line prompt | **Production chat** (`ChatScreen.tsx`) |
| `src/constants/mindmatePrompt.ts` | "You are Anchor..." (old name) ~150-line structured prompt | **Eval runner only** (`EvalRunner.ts`, `MultiTurnRunner.ts`) |

`anchorSystemPrompt.ts` is what real users see. `mindmatePrompt.ts` is only used in the in-app eval screen.

**Benchmark fidelity (verified 2026-05-03):** `benchmarks/scenarios.py` uses `_APP_BASE_PROMPT` which is a byte-for-byte match of `anchorSystemPrompt.ts`. The `_sys()` / `_MEMORY_HEADER` format matches `contextBuilder.ts assemblePrompt()` exactly. All benchmark scores reflect real production conditions.

### Memory injection in production chat
`ChatScreen.tsx` passes `getAnchorSystemPrompt()` to `useChatSession` → `prepareCompletion()` calls `buildEnhancedSystemPrompt(basePrompt, userMessage)` which injects tier1/tier2/tier3 from DB. The final system prompt the model sees = Anchor base + memory header + `[User]` + `[Recent sessions]`.

---

## Android App (`anchor-app/`)

### Runtime flow
1. User picks GGUF file → app copies to `DocumentDirectoryPath/models/local/` (shows full-screen spinner during 1.9GB copy)
2. `contextBuilder.ts` builds system prompt from profile + memory (Tier 1/2/3)
3. `MultiTurnRunner.ts` holds `fullPrompt` constant for entire session; history resets between sessions
4. JNI → llama.cpp: tokenize + decode + sample
5. Post-session: `sessionExtractor.ts` extracts summary → stored as `EpisodicMemoryData`

### Memory architecture
- `src/memory/contextBuilder.ts` — builds enhanced system prompt
- `src/memory/sessionExtractor.ts` — extracts summary after session ends
- `src/repositories/MemoryRepository.ts` — SQLite persistence
- `src/database/models/EpisodicMemory.ts` — stores date, summary, moodStart, moodEnd, copingOutcome, etc.
- Only `summary` text rendered to model — structured fields invisible to LLM

### Eval harness
- `src/eval/` — EvalRunner, EvalStore, EvalScreen
- Scenarios: `src/eval/fixtures/scenarios.ts`
- Results: `src/eval/EVAL_RESULTS.md`

---

## Cluster Access

```bash
# Login (requires NUS VPN)
ssh nus-student-cluster   # xlogin.comp.nus.edu.sg, user: aryanj

# Check jobs
squeue -u aryanj
sacct -u aryanj -j <jobid> --format=JobID,State,Start,End,Elapsed

# GPU availability
sinfo -o "%P %G %C %N"

# Logs (home dir, NOT project dir)
ls ~/logs/

# Sync local → cluster
rsync -avz --progress data/ nus-student-cluster:~/projects/mindmate/data/
rsync -avz --progress synthetic/ nus-student-cluster:~/projects/mindmate/synthetic/

# Sync cluster → local
rsync -avz --progress nus-student-cluster:~/projects/mindmate/data/ data/
rsync -avz --progress nus-student-cluster:~/projects/mindmate/adapters/genz_dpo_ck1600/ adapters/genz_dpo_ck1600/
```

**Partitions:** `gpu` (3h max), `gpu-long` (3 days max). Always use `gpu-long` for training.
**A100-80** is best for long jobs. H200-141 only in `gpu` partition (3h limit — not usable for training).

### Cached HuggingFace models on cluster
- `google/gemma-4-26B-A4B-it` ← teacher (download on first datagen run, ~52GB)
- `Qwen/Qwen3-30B-A3B-Instruct-2507` ← benchmark judge (still used in run_benchmarks.py)
- `Qwen/Qwen3-1.7B`
- `Qwen/Qwen2.5-3B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`

---

## Key File Index

### Training
- `finetuning/CUDA_run_pipeline.py` — Llama SFT orchestrator
- `finetuning/CUDA_train_qlora.py` — Llama SFT trainer
- `finetuning/CUDA_train_dpo.py` — DPO trainer (--model genzv2_ck1200|genzv3_ck200|genzv2_ck1600 --steps 800)
- `finetuning/build_dataset.py` — dataset builder (DATA_MIX_PRESETS: v1, v2, v3, v4)
- `finetuning/run_sft_v4.slurm` — genzv4 SFT pipeline (build → clean → train 2400 steps)
- `finetuning/run_dpo_llama_ck1600.slurm` — DPO SLURM job

### Synthetic data
- `synthetic/biometric_sft_pipeline.py` — biometric SFT (24 profiles, 4 modes)
- `synthetic/biometric_dpo_pipeline.py` — biometric DPO (4 pair types)
- `synthetic/dpo_targeted_fix_pipeline.py` — help_mode + memory_recall DPO
- `synthetic/targeted_fix_pipeline.py` — help_mode + memory_recall SFT
- `synthetic/conversation_memory_pipeline.py` — **multi-turn + memory SFT** (10 profiles, 6 modes, 20 new-fact seeds); fixes root cause of base-beats-SFT; 🟢 running jobs 608968–608969
- `synthetic/run_conv_memory.slurm` — 72h, A100-80, Gemma4 teacher
- `synthetic/utils.py` — TeacherModel, parse_json_robust, randomize_health_context
- `synthetic/prompts/dpo_targeted_fix_preference.txt` — DPO targeted fix prompt
- `synthetic/prompts/dpo_preference.txt` — main DPO prompt

### Inference
- `inference/CUDA_chat_mindmate.py` — Llama chat (--checkpoint checkpoint-1600)
- `scripts/chat_cluster.py` — **interactive chat on cluster GPU** (full production stack: Anchor prompt + memory engine, TextStreamer, `/memory` `/reset` commands); run via `srun --partition=gpu-long --gres=gpu:a100-80:1 --pty bash` then `python scripts/chat_cluster.py --adapter adapters/genzv3/checkpoint-200`
- `deploy/server.py` — production FastAPI server

### Export
- `scripts/export_gguf_cuda.py` — merge + convert + quantize (models: genzv3_ck200, genzv2_ck1200, genzv4_ck200 + legacy)
- `scripts/run_export_top3.slurm` — SLURM job exporting genzv3_ck200 + genzv2_ck1200 + genzv4_ck200 sequentially

### Android
- `anchor-app/src/memory/contextBuilder.ts` — system prompt builder
- `anchor-app/src/eval/fixtures/scenarios.ts` — eval scenarios
- `anchor-app/src/eval/EVAL_RESULTS.md` — full eval results

---

## Bugs Fixed (chronological)

| Date | Bug | Fix |
|---|---|---|
| Apr 7 | DPO datagen 0 pairs — Qwen3 `<think>` blocks corrupt JSON | `enable_thinking=False` + strip `<think>` in `parse_json_robust` |
| Apr 7 | SLURM log files not found | Always use `~/logs/` (home), not `~/projects/mindmate/logs/` |
| Apr 9 | TRL DPO `tokenizer=` arg removed in newer version | Use `processing_class=tokenizer` |
| Apr 14 | contextBuilder header too passive (reactive not proactive) | Added 3 directive lines — mood trend, health pattern, unhelpful coping |
| Apr 22 | 595563 biometric SLURM instant fail | Wrong log path `/home/aryanj/` → `/home/a/aryanj/` |
| Apr 22 | 595564 DPO targeted fix crashed | Missing prompt file `dpo_targeted_fix_preference.txt` — created and synced |
| Apr 24 | DPO targeted fix prompt had unreplaced `{system_prompt}` / `{memory_context}` | `generate_pair()` replaces all 4 vars — local file is correct, just needed syncing |
| Apr 26 | `synthetic/utils.py` missing `import random` and `import re` → `randomize_health_context()` crashed on every call → biometric pipeline (job 595713) generated 0 pairs in 22h | Added both imports to utils.py |
| Apr 26 | DPO `format_messages()` used `tokenizer.apply_chat_template` for `llama_ck1600` → format mismatch with SFT training → all DPO rewards/margins negative (job 595679 wasted) | Added `"llama_ck1600"` to manual Llama 3 format branch in `CUDA_train_dpo.py` |
| Apr 26 | DPO `max_prompt_length` not set → defaults to something small, truncating prompts | Added `max_prompt_length=1024` to DPOConfig |
| Apr 26 | `synthetic/run_dpo_targeted_fix.slurm` used relative log path `logs/...` → no logs for job 595714 | Fixed to `/home/a/aryanj/logs/...` |
| Apr 28 | `benchmarks/scenarios.py` fix (remove BANNED_PHRASES) not synced to cluster → benchmark job 599045 ran 42 scenarios instead of 34 | Re-synced local fixed version to cluster |
| May 1 | A100-80 node xgpj0 torch import fails: `libtorch_global_deps.so: No such file or directory` — CUDA libs not in LD_LIBRARY_PATH on that node | Switched `run_benchmarks.slurm` to A100-40 (40GB VRAM sufficient for sequential eval+judge) |
| May 1 | `ends_question` check failed when model appended trailing non-question clause after the question | Fixed: now checks `"?" in response` instead of `response.rstrip().endswith("?")` |
| May 1 | MEMORY_USE + BIOMETRIC benchmark checks were keyword-based → failed on semantically correct responses | Replaced with LLM judge (Qwen3-30B 4-bit at the time, binary YES/NO per criterion) in `scenarios.py` + `run_benchmarks.py`; judge later migrated to Gemma 4 26B A4B bfloat16 |
| May 2 | `CUDNN_STATUS_NOT_INITIALIZED` during `scaled_dot_product_attention` on H200/H100 nodes → benchmark Phase 1 crashed on jobs 602438–602452 | Added `attn_implementation="eager"` to eval model load in `run_benchmarks.py` — bypasses cuDNN/flash-attention |
| May 2 | A100-40 OOM loading Qwen3-30B judge (Phase 2): after eval model `del` + `empty_cache`, VRAM still nearly full (only 4.5MB free of 39.49GB) | Added `model.cpu(); base.cpu()` before `del` to force VRAM release; switched default GPU to A100-80 in `run_benchmarks.slurm` |
| May 2 | `ssh host "sbatch --export=ADAPTER=${ck}..."` — `$ck` expands in local shell (empty string), all submitted jobs had empty ADAPTER | Use single quotes for the remote command: `ssh host 'for ck in ...; do sbatch --export=ADAPTER=...${ck}...; done'` |
| May 2 | `sbatch --wrap="source mindmatenv/bin/activate && python ..."` → `source: not found` — `--wrap` executes via `/bin/sh`, not `bash` | Wrap with `bash -c`: `--wrap="bash -c \"source mindmatenv/bin/activate && python ...\""` |
| May 2 | `DPOConfig.__init__() got an unexpected keyword argument 'max_prompt_length'` — removed from TRL's DPOConfig in cluster version | Removed `max_prompt_length=1024` from DPOConfig in `CUDA_train_dpo.py` |
| May 2 | `CUDA_train_dpo.py: error: argument --model: invalid choice: 'genzv2_ck1600'` — alias not added to argparse choices | Added `genzv2_ck1600` to argparse choices and CONFIGS dict (maps to `adapters/genz/checkpoint-1600`) |
| May 7 | `sbatch script.sh --export=MODEL=foo` — `--export` placed after script path is treated as a script argument, not an sbatch flag; all 5 jobs (607691–607695) silently defaulted to `MODEL=llama_ck1600` | Always place sbatch flags BEFORE the script path: `sbatch --export=ALL,MODEL=foo script.sh` |
| May 9 | Benchmark scores dropped from 85% to 49% for llama_base between two runs — initially appeared as regression | Root cause: `temperature=0.7` on eval model → different responses each run → different judge verdicts. 17/49 scenarios flipped PASS→FAIL. Fix: 3-run averaged methodology. Use `benchmarks/average_results.py`. |
| May 9 | `git pull` on cluster fails: `ssh.github.com port 443: Connection timed out` | GitHub SSH blocked from cluster. Use `rsync` from local to push individual files: `rsync -az file nus-student-cluster:~/projects/mindmate/path/` |
| May 9 | H100-96 GRES jobs OOMed: `GPU 0 has a total capacity of 46.38 GiB` — Gemma4 needs ~52GB | H100-96 GRES fell back to ~46GB GPU nodes (jobs 609078–609083). Always use explicit `--gres=gpu:a100-80:1`; don't rely on H100-96. |
| May 9 | genzv4_ck200 jobs 609073+609074 wrote same filename `genzv4_ck200_20260509_0348.json` — one overwrote the other | Two jobs finished at identical second → same timestamp. Not critical (one run captured), but note genzv4 averaged results used 2 new runs + 1 earlier May 9 run. |
| May 9 | `average_results.py` KeyError: `'weighted_pass'` — result JSON uses nested `overall.weighted_pass` not top-level | Fixed: access `d['overall']['weighted_pass']` with fallback for both flat and nested formats. |
| May 9 | `llama_ck1600` shortcut name misleading — it is the genzv2 SFT adapter | Renamed to `genzv2_ck1600` in MODEL_SHORTCUTS and all docs. Adapter path unchanged: `adapters/genz/checkpoint-1600`. |
| May 9 | `conversation_memory_pipeline.py` v1 used meta-prompt: teacher generated full conversation JSON but was never constrained by anchor prompt. Training signal was generic Gemma4 style, not anchor-aligned. | Refactored to teacher-as-Anchor two-phase: Phase 1 (user simulator) + Phase 2 (Gemma4 with production anchor prompt as system message). |
| May 9 | All 42,038 training examples were missing production system prompt preamble. `targeted_fix` + `biometric`: had truncated `[User]`/`[Recent sessions]`-only system prompts (no "You are Anchor..." or "ABOUT THIS USER" instructions). All other files: no system message at all. Model learned to respond to a totally different format than inference. This is the deepest root cause of base-beats-SFT on memory categories (CROSS_SESSION_MEMORY 0%, CTX_MEM 9%, BIO 30%). | Normalization script updated all files to production format (commit c3acdc9). genzv5 will be the first training run with correct format. |
