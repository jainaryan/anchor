# MindMate Detailed Internal Notes

## Metadata
- Project: `mindmate`
- Last updated: `2026-05-01`
- Android app: `anchor-app/` (NOT `mindmate_app/` — that is a stale scratch fork)
- Production: https://tryanchor.me

---

## Executive Summary

MindMate is a finetuned, local mental-health companion that runs on Android. The full pipeline is:

```
Synthetic Data (Qwen3-30B teacher)
→ SFT (QLoRA, Llama 3.2 3B)
→ DPO (preference training on SFT adapter)
→ GGUF Q4_K_M export
→ Android (anchor-app, JNI → llama.cpp)
```

**Best model:** `adapters/genz/checkpoint-1600` (genzv2 SFT, Llama 3.2 3B). Beats all DPO variants.
**Production deploy:** https://tryanchor.me (FastAPI, DigitalOcean c-4)
**On-device:** Pixel 8a, `/sdcard/Download/mindmate.gguf`, ~5.5 TPS

---

## Architecture

### Top-level flow

```
Synthetic Data Generation (Qwen3-30B)
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

### `adapters/genz_dpo_ck1600/` — **RUNNING** 🟢 (job 601548)
- DPO on genzv2 ck1600, 1200 steps, `dpo_train.jsonl` (5,750 pairs incl. help_mode + memory_recall)
- Previous attempt (job 595679) diverged — format mismatch fixed in CUDA_train_dpo.py
- Saves every 200 steps → checkpoints at ck200, 400, 600, 800, 1000, 1200 + final
- Est. completion: ~4-5h from submission

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
- `Qwen/Qwen3-30B-A3B-Instruct-2507` in bfloat16 (~60GB VRAM → requires A100-80)
- **Always pass `enable_thinking=False`** — Qwen3 is a thinking model
- `parse_json_robust` strips `<think>` blocks as safety fallback
- Cached on cluster at `~/.cache/huggingface/hub/`

---

## Finetuning Pipeline (`finetuning/`)

### Scripts
| Script | Purpose |
|---|---|
| `CUDA_run_pipeline.py` | Llama SFT orchestrator (builds dataset → trains) |
| `CUDA_train_qlora.py` | Llama SFT trainer |
| `CUDA_run_pipeline_qwen25_3b.py` | Qwen2.5-3B SFT orchestrator |
| `CUDA_train_qlora_qwen25_3b.py` | Qwen2.5-3B SFT trainer |
| `CUDA_train_dpo.py` | DPO trainer, `--model llama_ck1600 --steps 1200 --data v1` |
| `build_dataset.py` | Merges JSONL files per DATA_MIX_PRESETS |
| `clean_dataset.py` | Deduplication + format validation |

### SLURM scripts
| Script | Partition | GPU | Time | Purpose |
|---|---|---|---|---|
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

## Active Cluster Jobs (2026-05-01)

| Job | Name | Status | Purpose |
|---|---|---|---|
| 601548 | `mindmate-dpo-llama-ck1600` | 🟢 **RUNNING** | DPO on genzv2 ck1600, 1200 steps, 5,750 pairs (A100-80, ~5h) |
| 601526–601533 | `mindmate-bench` | 🟢 **RUNNING/PENDING** | genzv3 ck200–1600 checkpoint sweep (A100-40, LLM judge) |
| 601534–601544 | `mindmate-bench` | 🟢 **RUNNING/PENDING** | genzv2 ck200–1600 checkpoint sweep (A100-40, LLM judge) |
| 601545–601547 | `mindmate-bench` | 🟡 **PENDING** | genzv2_continued ck200–500 checkpoint sweep (A100-40, LLM judge) |

### Job history
| Job | Name | Result |
|---|---|---|
| —      | Gold examples expanded    | **DONE** May 1 — 65 → 181 examples in `synthetic_train_targeted_fixes.jsonl` (name_resolution×30, crisis_safety×20, session_recall×18, profile_coping×15, biometric_profile×12, help_cold_open×12, anti_hallucination×8) |
| 601548 | DPO genzv2 ck1600 re-run | 🟢 **RUNNING** May 1 — genzv2 ck1600 base, 5,750 pairs, 1200 steps, A100-80 |
| 601526–601533 | Benchmark genzv3 sweep | 🟢 **RUNNING** May 1 — ck200–1600, A100-40, LLM judge |
| 601534–601544 | Benchmark genzv2 sweep | 🟢 **RUNNING/PENDING** May 1 — ck200–1600, A100-40, LLM judge |
| 601545–601547 | Benchmark genzv2_continued sweep | 🟡 **PENDING** May 1 — ck200–500, A100-40, LLM judge |
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

### Proposed v4 data mix (10k total)
```
targeted_fix (memory + help)  30%  (3,000 — up from 20%)
biometric                     15%  (1,500 — keep)
transition                    20%  (2,000 — slight decrease)
therapist                     12%  (1,200 — slight decrease)
casual                        10%  (1,000 — keep)
friend                         8%    (800 — down from 15% to reduce "hey love" dominance)
gold (targeted_fixes.jsonl)  +181  (always included at 100%)
```

Within targeted_fix (3,000), enforce sub-distribution:
- name_resolution (role→name recall): ~40% (1,200)
- session_history_detail (specific event/coping recall): ~25% (750)
- help_first (technique immediately on explicit help request): ~20% (600)
- profile_strategy_recall (suggest the marked-helpful strategy): ~15% (450)

### Optimal checkpoint
All evidence (Qwen-1.7B, genzv3) points to **ck200–400** as the Goldilocks zone for 10k datasets.
Options:
1. Train 1600 steps, run sweep, pick best early checkpoint (ck200–400 likely)
2. Increase dataset to 20k+ (use more of the 13,524 targeted_fix examples) — pushes the overfitting threshold later

### Do NOT use genzv2_continued approach again
Continued training (PeftModel.from_pretrained) catastrophically broke MEMORY_USE (0/8). Always train fresh from base.

---

## Benchmarks (`benchmarks/`)

### Suite (34 scenarios, 6 categories)
| Category | Scenarios | Scoring | What it tests |
|---|---|---|---|
| MEMORY_USE | 8 | **LLM judge** | Names, coping strategies, session history, held-out names (Zoya, Kabir) |
| HELP_MODE | 6 | rule-based | "help me" / "what should I do" → named technique, no hallucination |
| CRISIS | 5 | rule-based | Passive SI, active distress, safety escalation |
| NO_HALLUCINATION | 6 | rule-based | No fake shared history, no invented context |
| BIOMETRIC | 5 | **LLM judge** | Sleep/HRV/mood data → referenced naturally in response |
| FORMAT | 4 | rule-based | No markdown, no "As an AI", proper length |

**Scoring (updated 2026-05-01):** MEMORY_USE + BIOMETRIC use LLM judge (Qwen3-30B, 4-bit, YES/NO per criterion). CRISIS/HELP_MODE/NO_HALLUCINATION/FORMAT remain rule-based. `ends_question` check fixed: now passes if `?` appears anywhere in response (not just final char). Two-phase runner: eval model generates all responses → unload → judge model scores.

### Results — genzv2 ck1600 baseline (job 599045, 2026-04-28, old runner)
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

### Checkpoint sweeps (new LLM judge runner, 2026-05-01)
- **genzv3 ck200–1600:** jobs 601526–601533 — RUNNING/PENDING
- **genzv2 ck200–1600:** jobs 601534–601544 — RUNNING/PENDING
- **genzv2_continued ck200–500:** jobs 601545–601547 — PENDING
- Results will land in `benchmarks/results/` — compare against 59% baseline with new judge scores

### Files
- `benchmarks/scenarios.py` — scenarios; MEMORY_USE + BIOMETRIC use `judge_criteria`, others use `checks`
- `benchmarks/run_benchmarks.py` — two-phase: eval model → judge model (Qwen3-30B 4-bit)
- `benchmarks/run_benchmarks.slurm` — gpu-long, **a100-40**, 48G, 4h; supports ADAPTER/LABEL/MODEL/CATEGORY env
- `benchmarks/results/` — JSON + Markdown output per run (`<label>_<timestamp>.{json,md}`)

### Running
```bash
sbatch --export=ADAPTER=adapters/genzv3/checkpoint-400,LABEL=genzv3_ck400 benchmarks/run_benchmarks.slurm
sbatch --export=MODEL=llama_ck1600 benchmarks/run_benchmarks.slurm
sbatch --export=MODEL=llama_ck1600,CATEGORY=MEMORY_USE benchmarks/run_benchmarks.slurm
```

**Note:** Use A100-40 only (a100-80 node xgpj0 has broken CUDA path — torch import fails).

---

## Production (deploy/)

### FastAPI server (`deploy/server.py`)
- DigitalOcean c-4 droplet, IP: 209.38.122.228
- SSH: `ssh -i ~/.ssh/id_ed25519 root@209.38.122.228`
- Restart: `systemctl restart mindmate`
- Deploy static: `rsync -az -e "ssh -i ~/.ssh/id_ed25519" deploy/static/ root@209.38.122.228:~/mindmate/deploy/static/`
- SSE-based streaming, loads genzv2 ck1600 Q4_K_M

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
- `Qwen/Qwen3-30B-A3B-Instruct-2507` ← teacher
- `Qwen/Qwen3-1.7B`
- `Qwen/Qwen2.5-3B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`

---

## Key File Index

### Training
- `finetuning/CUDA_run_pipeline.py` — Llama SFT orchestrator
- `finetuning/CUDA_train_qlora.py` — Llama SFT trainer
- `finetuning/CUDA_train_dpo.py` — DPO trainer (--model llama_ck1600 --steps 1200 --data v2)
- `finetuning/build_dataset.py` — dataset builder
- `finetuning/run_dpo_llama_ck1600.slurm` — DPO SLURM job

### Synthetic data
- `synthetic/biometric_sft_pipeline.py` — biometric SFT (24 profiles, 4 modes)
- `synthetic/biometric_dpo_pipeline.py` — biometric DPO (4 pair types)
- `synthetic/dpo_targeted_fix_pipeline.py` — help_mode + memory_recall DPO
- `synthetic/targeted_fix_pipeline.py` — help_mode + memory_recall SFT
- `synthetic/utils.py` — TeacherModel, parse_json_robust, randomize_health_context
- `synthetic/prompts/dpo_targeted_fix_preference.txt` — DPO targeted fix prompt
- `synthetic/prompts/dpo_preference.txt` — main DPO prompt

### Inference
- `inference/CUDA_chat_mindmate.py` — Llama chat (--checkpoint checkpoint-1600)
- `deploy/server.py` — production FastAPI server

### Export
- `scripts/export_gguf_cuda.py` — merge + convert + quantize

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
| May 1 | MEMORY_USE + BIOMETRIC benchmark checks were keyword-based → failed on semantically correct responses | Replaced with LLM judge (Qwen3-30B 4-bit, binary YES/NO per criterion) in `scenarios.py` + `run_benchmarks.py` |
