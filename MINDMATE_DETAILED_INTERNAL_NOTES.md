# MindMate Detailed Internal Notes

## Metadata
- Project: `mindmate`
- Last updated: `2026-04-28`
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
- `max_steps`: 1200 (increased from 800 — v2 data is 3× larger)
- `beta`: 0.1, `loss_type`: sigmoid
- `precompute_ref_log_probs`: True (avoids dual-model OOM)
- Architecture: base (4-bit) + SFT adapter (trainable DPO LoRA) + ref_model (frozen SFT)
- Data: `dpo_train_v2.jsonl` / `dpo_val_v2.jsonl` (7,200 pairs)
- **Previous DPO (ck200, 800 steps, v1 data) did NOT improve over SFT — genzv2 ck1600 remained best**
- **New DPO run (job 595679): genzv2 ck1600 base, 1200 steps, v2 data → pending**

---

## Saved Adapters

### `adapters/genz/checkpoint-1600` — **PRODUCTION MODEL** ✅
- Llama 3.2 3B, v2 data mix, 1600 steps
- **Beats all DPO models and all other SFT checkpoints**
- GGUF: `exports/mindmate_llama_sft_ck1600/llama(genz)v2_q4_k_m.gguf`
- Pixel 8a: ~5.5 TPS, TTFT 60–66s cold / 6s cached, heap 3.12–3.17 GB

### `adapters/genz_dpo_ck1600/` — **COMPLETED but FAILED** ❌ (job 595679)
- DPO on top of genzv2 ck1600, 1200 steps on dpo_v2 data (~1h on A100-80)
- Saved to cluster, but training diverged: all `rewards/margins` negative, `rewards/accuracies` 0.11–0.31
- Root cause: `format_messages()` used `tokenizer.apply_chat_template` for `llama_ck1600`, but genzv2 SFT was trained with manual Llama 3 format → format mismatch → DPO had no useful gradient signal
- **Fixed in CUDA_train_dpo.py**: added `"llama_ck1600"` to manual-format branch; added `max_prompt_length=1024`
- **Needs re-run** after pushing fix to cluster

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
| `synthetic_train_targeted_fixes.jsonl` | 65 | Hand-crafted gold examples | help_mode (30) + biometric (15) + banned opener (20) — always 100% |
| `additional_training_samples.jsonl` | 30 | Legacy | **EXCLUDED** |

### DPO data

| File | Pairs | Date | Notes |
|---|---|---|---|
| `dpo_train_v2.jsonl` | 6,120 | Apr 18 | v2 — used for genzv2 DPO |
| `dpo_val_v2.jsonl` | 1,080 | Apr 18 | v2 |
| `dpo_train.jsonl` | **3,199** | Apr 29 | **Active** — v2 + biometric pairs added by job 599030 |
| `dpo_val.jsonl` | **564** | Apr 29 | **Active** — v2 + biometric |
| `dpo_biometric_partial.jsonl` | 1,971 | Apr 29 | Biometric-only pairs (subset of dpo_train.jsonl) |
| `dpo_pairs_partial_v2.jsonl` | 7,202 | Apr 18 | Raw partial — redundant |

**DPO categories in dpo_train.jsonl:** biometric (A/B/C/D types) + mixed_mode, casual_sad, panic_mode, transition, hallucination_guard, system_compliance
**Still missing:** help_mode, memory_recall DPO pairs — not yet generated

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
| `pipeline.py` | therapist/friend/grief | Done |
| `casual_pipeline.py` | casual non-distress | Done |
| `transition_pipeline.py` | casual→emotional pivots | Done |
| `dpo_pipeline.py` | DPO pairs (v1 and v2) | Done |
| `targeted_fix_pipeline.py` | help_mode + memory_recall SFT | Done (job 594101) |
| `dpo_targeted_fix_pipeline.py` | DPO pairs for help_mode + memory_recall | **Needs re-run** (job 595714 timed out at 10h, no logs — SLURM log path bug fixed) |
| `biometric_sft_pipeline.py` | Biometric health context SFT, 4 modes | **Needs re-run** (job 595713 generated 0 — `random`+`re` not imported in utils.py; fixed) |
| `biometric_dpo_pipeline.py` | Biometric DPO pairs, 4 pair types | **Needs re-run** (same bug) |

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
| `CUDA_train_dpo.py` | DPO trainer, `--model llama_ck1600 --steps 1200 --data v2` |
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
2. ⏳ Biometric context utilization — 15 hand-crafted examples added (lines 31–45) + pipeline job 595713
3. ⏳ help_mode + memory_recall DPO — job 595714

### LLM judge
Confirmed unreliable — inflates all scores to 80–84 regardless of actual response quality. Use human inspection only.

### Held-out eval names
Zoya, Kabir, Tanvi, Layla, Rohan — not in training data; used to test generalization

### Device performance (Pixel 8a)
- Gen TPS: ~5.4–6.0 stable. Thermal: LIGHT single-turn, MODERATE after 45–60 min sustained.
- TTFT: 60–66s cold (1,100+ token prompt, ~18 tok/s prefill) / 6s KV-cached
- Peak heap: ~3.12–3.17 GB. Available RAM: ~800–875 MB.

---

## Active Cluster Jobs (2026-04-29)

| Job | Name | Status | Purpose |
|---|---|---|---|
| 600329 | `mindmate-sft-v3` | **RUNNING/PENDING** | SFT v3 fresh training, 1600 steps |
| 600330 | `mindmate-sft-v2c` | **RUNNING/PENDING** | SFT v2_continued from genzv2 ck1600, 500 steps |

**Still needs submission:**
- `finetuning/run_dpo_llama_ck1600.slurm` — DPO on genzv2 ck1600 (format + max_prompt_length fixed, not yet submitted)

### Job history
| Job | Name | Result |
|---|---|---|
| 600329 | SFT v3 (genzv3) | QUEUED Apr 29 |
| 600330 | SFT v2_continued (genzv2_continued) | QUEUED Apr 29 |
| 599045 | Benchmark genzv2 ck1600 | **COMPLETED** — 27/42 (64%). See Benchmarks section. |
| 599031 | Targeted fix SFT | **COMPLETED** — 13,524 SFT examples (help_mode + memory_recall) |
| 599030 | Biometric SFT+DPO | **COMPLETED** — 2,348 SFT + ~1,263 DPO pairs added |
| 595714 | DPO targeted fix | TIMEOUT 10h — no logs (wrong log path, now fixed) |
| 595713 | Biometric SFT+DPO | COMPLETED 22h — 0 new pairs (`random`+`re` not imported, now fixed) |
| 595679 | DPO genzv2 ck1600 | COMPLETED 1h — diverged (negative margins, format mismatch, now fixed) |
| 594101 | Targeted fix SFT | 2,993 examples, good quality (0% bad phrases) |
| 594102 | DPO targeted fix | Failed — no output (missing prompt file) |
| 595563 | Biometric SFT+DPO | Failed — wrong SLURM log path |
| 595564 | DPO targeted fix retry | Crashed — missing `dpo_targeted_fix_preference.txt` prompt |
| 580674 | DPO v2 datagen | 7,202 pairs → `dpo_train_v2.jsonl` + `dpo_val_v2.jsonl` |
| 560339 | Llama DPO ck200 | Inferior to genzv2 SFT |
| 560338 | Qwen2.5-3B DPO | Inferior to genzv2 SFT |
| 554799 | Llama v2 SFT | → `adapters/CUDA_mindmate_llama32b/checkpoint-1600` |

---

## Benchmarks (`benchmarks/`)

### Suite (34 scenarios, 6 categories — after removing BANNED_PHRASES)
| Category | Scenarios | What it tests |
|---|---|---|
| MEMORY_USE | 8 | Names, coping strategies, session history, held-out names (Zoya, Kabir) |
| HELP_MODE | 6 | "help me" / "what should I do" → named technique, no hallucination |
| CRISIS | 5 | Passive SI, active distress, safety escalation |
| NO_HALLUCINATION | 6 | No fake shared history, no invented context |
| BIOMETRIC | 5 | Sleep/HRV/mood data → referenced naturally in response |
| FORMAT | 4 | No markdown, no "As an AI", proper length |

### Results — genzv2 ck1600 baseline (job 599045, 2026-04-28)
**Overall: 27/42 (64%)** — *note: ran against old scenarios.py that still had 8 BANNED_PHRASES scenarios; true score on 34-scenario suite estimated ~20/34 (59%)*

| Category | Pass | Total | Avg score |
|---|---|---|---|
| CRISIS | 5 | 5 | 1.00 ✅ |
| FORMAT | 4 | 4 | 1.00 ✅ |
| NO_HALLUCINATION | 6 | 6 | 1.00 ✅ |
| HELP_MODE | 3 | 6 | 0.75 ⚠️ |
| BIOMETRIC | 1 | 5 | 0.54 ❌ |
| MEMORY_USE | 1 | 8 | 0.43 ❌ |

**Key failures:**
- Memory: ignores names (Zoya, Kabir), coping strategies, session history — responds generically
- Biometric: ignores sleep hours/HRV/mood scores in context entirely
- Help mode: asks a question instead of naming a technique when user says "help me calm down"

**Fix in progress:** Targeted fix datagen (job 599031) + biometric datagen (job 599030) — will add training data for these exact failure modes, then re-SFT and re-benchmark.

### Files
- `benchmarks/scenarios.py` — all scenarios + `ALL_SCENARIOS`, `CATEGORIES` dicts
- `benchmarks/run_benchmarks.py` — model loader (4-bit NF4 + PeftModel) + scorer + output
- `benchmarks/run_benchmarks.slurm` — gpu-long, a100-40, 2h, supports MODEL/CATEGORY env
- `benchmarks/results/` — JSON + Markdown output per run (`<label>_<timestamp>.{json,md}`)

### Running
```bash
sbatch benchmarks/run_benchmarks.slurm                          # default: llama_ck1600
sbatch --export=MODEL=llama_dpo_ck1600 benchmarks/run_benchmarks.slurm
sbatch --export=MODEL=llama_ck1600,CATEGORY=MEMORY_USE benchmarks/run_benchmarks.slurm
```

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
