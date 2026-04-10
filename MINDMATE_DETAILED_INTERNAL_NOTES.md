# MindMate Detailed Internal Notes (Local)

## Metadata
- Project: `mindmate`
- Last deep audit: `2026-04-02`
- Last updated: `2026-04-09`
- Branch at audit time: `main`
- Scope covered: first-party code, training/inference pipelines, web app, Android app, model artifacts, data assets, deployment scripts.
- Scope intentionally not line-by-line reviewed: full upstream `llama.cpp` source trees (treated as external vendored dependency).

## Executive Summary
MindMate is a multi-surface mental-health assistant project with three parallel execution paths:
1. Training/finetuning pipelines (MLX + CUDA/Hugging Face).
2. Local inference surfaces (CLI + Flask web app with memory/profile layer).
3. On-device Android app using a JNI bridge to `llama.cpp` for GGUF inference.

The repo already contains substantial assets (datasets, adapters, exported GGUFs, Android installer/model), but it also has drift between scripts, docs, and runtime assumptions. The biggest blockers for a stable "finetuned local Android mental wellness assistant" are:
- fragmented data schemas and pipeline drift,
- some broken or stale scripts,
- safety/privacy/security hygiene gaps,
- duplicated model/runtime paths.

## Repo Footprint Snapshot

### Directory sizes
- `models/`: ~6.0G
- `exports/`: ~7.9G
- `android/`: ~3.2G
- `adapters/`: ~284M
- `data/`: ~106M
- `synthetic/`: ~18M
- `web/`: ~6.5M

### Notable large tracked artifacts
- `exports/mindmate_llama32_3b_f16.gguf` (~6.4G)
- `exports/mindmate_llama32_3b_q4_k_m.gguf` (~2.0G)
- `models/mlx_base_llama32_3b/model-00001-of-00002.safetensors` (~5.0G)
- `models/mlx_base_llama32_3b/model-00002-of-00002.safetensors` (~1.0G)

### Current git state at audit time
- Tracked modifications made in this audit: `.gitignore`
- Pre-existing untracked file: `android/temp_lib.so`

## Intended Product Direction (from user context)
You are building a **finetuned, local, mental-health wellness assistant that runs on Android**.

This repo already supports that direction conceptually:
- finetuning path exists,
- GGUF export path exists,
- Android app + native inference bridge exists,
- mobile download packaging exists (`android/serve_apk`).

But there are integration-quality gaps before this can be considered robust for production-ish personal use.

---

## Top-Level Architecture

```
Synthetic Data Generation
        ↓
  Raw JSONL Data
        ↓
  Dataset Build/Clean
        ↓
  SFT Finetuning (QLoRA)
        ↓
  SFT Adapter Checkpoints
        ↓
  DPO Preference Data Generation
        ↓
  DPO Training (on top of SFT)
        ↓
  DPO Adapter Checkpoints
        ↓
  Model Merge + GGUF Export (Q4_K_M)
        ↓
  ┌─────────────────────┐
  │  Desktop/Web Inf.   │   Android Local Inference
  └─────────────────────┘          (JNI → llama.cpp)
```

## High-Level Module Map
- `finetuning/`: dataset prep + train orchestration scripts (MLX + CUDA variants).
- `synthetic/`: continuous synthetic scenario/dialogue generation with heuristic filtering.
- `inference/`: local CLI chat scripts (CUDA and MLX variants).
- `web/`: Flask app, model service queue, memory/profile engine, browser UI.
- `android/app`: Jetpack Compose app + state/viewmodel + prompt handling.
- `android/llama`: JNI + C++ bridge to `llama.cpp` static libs.
- `scripts/`: export/util/debug helpers.
- `exports/`: fused GGUF outputs.
- `models/`: base model/tokenizer assets.
- `adapters/`: LoRA adapter checkpoints.

---

## Current Active Pipeline (as of 2026-04-09)

```
CUDA_run_pipeline.py → CUDA_train_qlora.py → CUDA_train_dpo.py → export_gguf_cuda.py → Android
```

### Canonical Models (current best)
1. **Llama 3.2 3B Instruct** — primary; `checkpoint-200` + DPO → `CUDA_mindmate_llama32b_dpo_ck200/` ✅ COMPLETE
2. **Qwen2.5-3B Instruct** — secondary; `checkpoint-200` + DPO → `CUDA_mindmate_qwen25_3b_dpo_ck200/` ✅ COMPLETE
3. **Qwen3-1.7B** — deprecated for now; 1.7B capacity causes hallucinations and gender confusion

### Why checkpoint-200?
Evaluated all checkpoints (200/800/1600) on both models. checkpoint-200 consistently outperforms later ones — later checkpoints overfit to therapy-speak and hallucinate fake shared history ("since day 1", "when we met"). See `memory/INFERENCE_LOGS.md`.

### System prompt
Simplified from 150-line ruleset to 13-line natural-language direction (`system_prompt.txt`). Temperature raised 0.4→0.75 for more natural responses.

---

## Data Layer Deep Dive

### Synthetic Data Sources (current)
| File | Examples | Content |
|---|---|---|
| `data/synthetic_train_therapist_.jsonl` | 6,347 | Supportive therapeutic dialogue |
| `data/synthetic_train_friend_1.jsonl` | 7,380 | Casual friend-style support |
| `data/synthetic_train.jsonl` | 5,565 | Grief/loss focused |
| `data/synthetic_train_casual.jsonl` | 5,000 | Non-distress casual conversation |
| `data/synthetic_train_transition.jsonl` | **6,170** | Casual→emotional pivot dialogues |
| `data/dpo_pairs_partial.jsonl` | growing | DPO pairs (incremental, survives kill) |
| `data/dpo_train.jsonl` | TBD | DPO training split (85%) |
| `data/dpo_val.jsonl` | TBD | DPO validation split (15%) |

### All data generated by teacher model
- `Qwen/Qwen3-30B-A3B-Instruct-2507` in bfloat16 on A100-80
- This is a MoE model: 30B total params, only 3B active per token — fast + high quality

### Transition Data (critical fix)
- `synthetic/transition_pipeline.py` — new pipeline for casual→emotional shift dialogues
- Phase 1: starts casual (2-4 turns), Phase 2: user drops real emotion, Phase 3: AI pivots
- 25 situations: grief, loneliness, anxiety, family issues, etc.
- Heuristic check: no therapy-speak in opener, last turn shouldn't end with lol/haha
- **Purpose:** fix "joke-mode-lock" — model stuck in casual mode when user gets serious
- Job 544107 completed (Apr 7) → 6,170 examples in 20h on A100-80

### SFT Data Mixes

**Llama (genz adapter, job 543708) — 50/50 casual/distress:**
| Source | Samples | % |
|---|---|---|
| casual | 5,000 | 50% |
| friend | 1,912 | 19.1% |
| therapist | 1,645 | 16.5% |
| grief/loss | 1,443 | 14.4% |
| **Total** | **10,000** | |

**Qwen3-1.7B (job 544016) — Option B mix:**
| Source | Samples | % |
|---|---|---|
| casual | 4,000 | 40% |
| therapist | 2,500 | 25% |
| friend | 2,200 | 22% |
| grief/loss | 1,300 | 13% |
| **Total** | **10,000** | |

### Schema
All data uses unified `conversations` schema:
```json
{"conversations": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```
Both Llama and Qwen use this — `apply_chat_template` handles model-specific formatting.

### Schema drift (legacy)
Two old shapes still exist in `data/conversations_raw/` and `data/new_raw_data/`:
1. `{"conversations": [{"role": ..., "content": ...}]}`
2. `{"text": "<|user|> ... <|assistant|> ..."}`
These are no longer consumed by the active pipeline — synthetic data only.

---

## Synthetic Generation Pipeline (`synthetic/`)

### Core behavior
- `synthetic/pipeline.py` — original therapist/friend/grief pipeline
- `synthetic/casual_pipeline.py` — casual non-distress pipeline (added Apr 3)
- `synthetic/transition_pipeline.py` — casual→emotional pivot pipeline (added Apr 5)
- `synthetic/dpo_pipeline.py` — DPO preference pair generation (added Apr 7)
- Uses `TeacherModel` from `synthetic/utils.py`
- Default teacher: `Qwen/Qwen3-30B-A3B-Instruct-2507` in bfloat16
- Writes incrementally; survives SLURM kills

### DPO Pipeline (`synthetic/dpo_pipeline.py`)
- Target: 2,500 preference pairs
- 50% mixed-mode (casual→emotional→panic arc)
- 50% single-mode (one failure type: casual_sad / transition / panic_mode / system_compliance / hallucination_guard)
- Heuristics: similarity < 0.70, length parity < 2.5x, no grounding in chosen for casual_sad/transition, mixed_mode needs ≥7 prompt turns
- Saves pair immediately to `data/dpo_pairs_partial.jsonl` on accept (survives SIGKILL)
- Rewrites train/val split every 25 pairs
- SIGTERM handler saves immediately before exit
- Resume: on restart, loads from partial file automatically

### Teacher model notes
- `enable_thinking=False` MUST be passed to `apply_chat_template` — Qwen3 is a thinking model and outputs `<think>...</think>` blocks otherwise
- `parse_json_robust` strips `<think>` blocks before JSON extraction as a safety fallback
- Do NOT use `Qwen/Qwen3-72B-Instruct` — this model ID does not exist on HuggingFace
- Only cached on cluster: `Qwen3-1.7B`, `Qwen3-30B-A3B-Instruct-2507`, `Llama-3.2-3B-Instruct`

### Known issues
- `synthetic/generate_scenarios.py` has a syntax error: `NUM_SCENARIOS =  # 2 mock scenarios`
- `USE_VLLM` path is stubbed and not implemented

---

## Finetuning Pipeline (`finetuning/`)

### Current active CUDA path
- **Llama:** `finetuning/CUDA_run_pipeline.py` → `build_dataset.py` → `clean_dataset.py` → `CUDA_train_qlora.py`
- **Qwen:** `finetuning/CUDA_run_pipeline_qwen.py` → `build_dataset.py` → `clean_dataset.py` → `CUDA_train_qlora_qwen.py`
- **DPO:** `finetuning/CUDA_train_dpo.py` (runs after SFT, loads SFT adapter as frozen reference)

### SFT Hyperparameters (as of 2026-04-04)
- `learning_rate`: 1e-5 (reduced from 3e-5)
- `lora_r`: 8, `lora_alpha`: 16 (reduced from r=16, alpha=32)
- `max_steps`: 1600
- **Why:** Model was catastrophically forgetting base conversational capabilities. Smaller LR and rank reduce overwriting.

### DPO Hyperparameters (as of 2026-04-07)
- `learning_rate`: 5e-7
- `lora_r`: 8, `lora_alpha`: 16
- `max_steps`: 800
- `beta`: 0.1, `loss_type`: sigmoid
- `precompute_ref_log_probs`: True (avoids dual-model OOM on single GPU)
- Architecture: base (frozen 4-bit) → SFT adapter (frozen, `"reference"`) → DPO LoRA (trainable, `"policy"`)

### Loss masking boundaries
- **Llama:** `<|start_header_id|>assistant<|end_header_id|>\n\n` ... `<|eot_id|>`
- **Qwen:** `<|im_start|>assistant\n` ... `<|im_end|>`

### Qwen3-specific notes
- `trust_remote_code=True` required for tokenizer and model
- Always `enable_thinking=False` in `apply_chat_template`
- Base model is a thinking model — without this flag it outputs reasoning tokens in responses

### MLX path (legacy/stale)
- `finetuning/run_pipeline.py` references undefined `MLX_LORA_BIN` — not used
- `finetuning/chunk.py` is broken (undeclared variables) — not used

---

## Saved Adapters

### `adapters/genz/` — GenZ model (Llama 3.2 3B, job 543708)
- **Personality:** Very casual, joke-heavy. Doesn't switch to serious mode when user gets emotional.
- **Problem:** Joke-mode-lock — the 50% casual data mix taught it casual IS the default personality.
- **Status:** Archived. Will be superseded by retrained model with transition data.

### `adapters/CUDA_mindmate_llama32b/` — Llama SFT adapter (job 543708)
- Same weights as genz (copy). `checkpoint-200` is the best checkpoint.

### `adapters/CUDA_mindmate_llama32b_dpo_ck200/` — Llama DPO adapter ✅ TRAINED (job 560339)
- DPO LoRA trained on top of SFT ck200
- Final loss: 1.127 after 800 steps on A100-40
- Load: base → merge SFT ck200 → apply DPO LoRA

### `adapters/CUDA_mindmate_qwen25_3b/` — Qwen2.5-3B SFT adapter (job TBD)
- `checkpoint-200` is the best checkpoint (same finding as Llama)
- **Inference:** `inference/CUDA_chat_qwen25_3b.py`

### `adapters/CUDA_mindmate_qwen25_3b_dpo_ck200/` — Qwen2.5-3B DPO adapter ✅ TRAINED (job 560338)
- DPO LoRA trained on top of Qwen2.5-3B SFT ck200
- Final loss: 1.279 after 800 steps on A100-40
- Load: base → merge SFT ck200 → apply DPO LoRA

### `adapters/CUDA_mindmate_qwen3_1p7b/` — Qwen3-1.7B SFT adapter (job 544016) — ARCHIVED
- 1.7B capacity causes hallucinations and gender confusion — not pursuing further

---

## Job History (as of 2026-04-09)

| Job | Name | Status | Node | Notes |
|---|---|---|---|---|
| 543708 | mindmate-finetune | COMPLETED | — | Llama SFT (genz adapter) |
| 544016 | mindmate-qwen-finetune | COMPLETED | — | Qwen3-1.7B SFT |
| 544107 | mindmate-transition | COMPLETED | xgph3 | 6,170 transition dialogues |
| 552529 | mindmate_dpo_datagen | COMPLETED | A100-80 (gpu-long) | 2,500 DPO pairs generated |
| 554799 | mindmate-llama-v2 | COMPLETED | A100-40 (gpu-long) | Llama SFT v2 (new data mix) |
| 554800 | mindmate-qwen-v2 | COMPLETED | A100-40 (gpu-long) | Qwen v2 SFT (new data mix) |
| 560338 | mindmate-dpo-qwen | COMPLETED | A100-40 (gpu-long) | Qwen2.5-3B DPO, loss=1.279 |
| 560339 | mindmate-dpo-llama | COMPLETED | A100-40 (gpu-long) | Llama DPO, loss=1.127 |
| 562150 | mindmate-export | FAILED | H100-96 | llama_sft_ck200 — wrong script version |
| 562151 | mindmate-export | FAILED | H100-96 | llama_dpo_ck200 — wrong script version |
| 562152 | mindmate-export | FAILED | H100-96 | qwen25_dpo_ck200 — wrong script version |

---

## SLURM Scripts

| Script | Partition | GPU | Time | Purpose |
|---|---|---|---|---|
| `run_finetune.slurm` | gpu-long | H100-96 | 24h | Llama SFT |
| `run_finetune_qwen.slurm` | gpu | H100-96 | 24h | Qwen SFT |
| `run_dpo_datagen.slurm` | gpu-long | A100-80 | 48h | DPO pair generation |
| `run_dpo.slurm` | gpu | H200-141 | 12h | DPO training (llama or qwen) |
| `run_export.slurm` | gpu | H200-141 | 2h | GGUF export |

**Critical SLURM notes:**
- Logs go to `~/logs/` (home dir), NOT `~/projects/mindmate/logs/` — SLURM resolves `--output` before `cd` runs
- `gpu` partition: max 3h
- `gpu-long` partition: max 3 days
- H200 not suitable for long jobs (3h limit on gpu partition)
- A100-80 on gpu-long is the best option for training/datagen jobs

---

## Inference Layer (`inference/`)

### CUDA finetuned chat
- `inference/CUDA_chat_mindmate.py` — Llama with SFT adapter
- `inference/CUDA_chat_qwen.py` — Qwen3-1.7B with SFT adapter (must pass `enable_thinking=False`)
- `inference/CUDA_chat_base.py` — baseline without adapter (comparison)

### MLX chat (legacy)
- `inference/chat_mindmate.py` — loads MLX model + adapter

### System prompt
- Root `system_prompt.txt` — canonical prompt used for training and inference
- Contains section 8B "CASUAL → SERIOUS TRANSITION (CRITICAL)" — drop humor instantly on distress signals
- `inference/system_prompt.txt` — old Anchor-persona prompt, inconsistent, do not use

---

## Web App Architecture (`web/`)

### Backend
- Framework: Flask (`web/app.py`)
- Model runtime abstraction: `web/model_service.py` with background worker + request queue
- Memory/profile layer: `web/memory_engine.py`
- `USE_ADAPTER = False` by default — loads base model unless toggled

### Key behaviors
- Onboarding flow collects name/role/style if profile missing
- Streaming chat via NDJSON chunks
- Session logs persisted to `web/chat_logs/*.json`
- Post-session analysis updates `web/user_profile.json` (mood/topics/risk)
- TTS endpoint integrated with Hume API

---

## Android App Deep Dive (`android/`)

### Runtime flow
1. `MainActivity` requests storage permissions
2. `ChatViewModel` searches expected GGUF paths (Downloads or app files dir)
3. `LlamaInference` loads model through `com.mindmate.llama.Llm` native wrapper
4. Prompt built in Llama 3-style special-token format with system prompt from `res/raw/system_prompt.txt`
5. JNI calls `llama.cpp` APIs to tokenize/decode/sample

### Native/JNI details (`android/llama/src/main/cpp/llama-android.cpp`)
- CPU-only (`n_gpu_layers = 0`)
- Context length hardcoded to `n_ctx = 512` ← needs increasing
- Batch size `n_batch = 8`
- Sampler chain: temp/top-p/dist
- Stop checks for eot markers and "User:"

### Android packaging
- `android/serve_apk/` hosts APK + GGUF + download page

---

## Export/Deployment Path

### GGUF export
- Main script: `scripts/export_gguf_cuda.py`
- Models: `"genz"` and `"qwen"` configs
- Process:
  1. Load base + SFT adapter → `merge_and_unload()` → save merged HF model
  2. `convert_hf_to_gguf.py --outtype f16`
  3. `llama-quantize Q4_K_M`
- `--skip-merge` flag to reuse existing merged model

### GGUFs uploaded to HuggingFace
- Repo: `jainaryan/mindmate-gguf` (private)
- Upload script: `scripts/upload_gguf_to_hf.py`

---

## Security / Privacy / Safety Findings

### High-priority security hygiene issues
1. Hardcoded HF token appears in `run_finetune.slurm` — rotate after use
2. `.env` contains live API credentials (not tracked, but present locally)
3. DPO SLURM script has HF_TOKEN inline — remove after job completes: `sed -i '/HF_TOKEN/d' run_dpo_datagen.slurm`

### Privacy concerns
- Web session logs and profile files store personal mental-health data as plain JSON
- Android chat logs appended to public Downloads path in plain text

### Safety alignment gaps
- Older raw corpora contain low-quality/unsafe counseling styles
- Crisis instructions differ across prompt files and surfaces
- Web quick crisis panel has specific hotline numbers; one prompt version instructs not to provide them (policy mismatch)

---

## Code Health Findings

### Confirmed breakages or drifts
1. `synthetic/generate_scenarios.py` syntax error (fails compilation)
2. `finetuning/chunk.py` currently inconsistent/broken
3. `finetuning/run_pipeline.py` references undefined `MLX_LORA_BIN`, disabled dataset build step
4. README command path drift (`scripts/chat_mindmate.py` vs actual `inference/chat_mindmate.py`)

### Resolved
- Training distribution imbalance → `casual_pipeline.py` + `transition_pipeline.py`
- Hyperparameters causing catastrophic forgetting → reduced LR and LoRA rank
- Qwen3 `<think>` tags in output → `enable_thinking=False` in inference + data gen
- DPO datagen 0/14k saves → `<think>` stripping in `parse_json_robust` + `enable_thinking=False` in teacher

---

## Bugs Fixed This Session (2026-04-07)

| Bug | Root Cause | Fix |
|---|---|---|
| DPO datagen 0 pairs after 14k+ attempts | Qwen3 teacher outputting `<think>` blocks, corrupting JSON extraction | `enable_thinking=False` in `apply_chat_template` + strip `<think>` in `parse_json_robust` |
| DPO log files not found | SLURM resolves `--output` from home dir before `cd` | Always check `~/logs/` not `~/projects/mindmate/logs/` |
| `Qwen/Qwen3-72B-Instruct` not found | Model doesn't exist on HuggingFace | Use `Qwen3-30B-A3B-Instruct-2507` (cached on cluster) |
| DPO data lost on timeout | `save_train_val_split` only called at end | Append each pair to `dpo_pairs_partial.jsonl` immediately; checkpoint every 25 pairs; SIGTERM handler saves on timeout |
| rsync files landed in wrong dir | Rsync flattened directory structure | Move files to correct subdirs on cluster after transfer |

---

## Practical Roadmap To Reach Goal (Local Finetuned Android Wellness Assistant)

### Immediate (pending)
1. ⏳ GGUF export — resubmit after syncing updated `export_gguf_cuda.py` to cluster
   - `sbatch run_export.slurm llama_sft_ck200`
   - `sbatch run_export.slurm llama_dpo_ck200`
   - `sbatch run_export.slurm qwen25_dpo_ck200`
2. ⏳ Inference testing — compare DPO vs SFT ck200 on same prompts

### Phase 1: Quality hardening
1. Evaluate DPO vs SFT models on casual/emotional/panic scenarios
2. Run automatic dataset linting for toxic/religious/moralizing patterns
3. Add eval suite: empathy quality, panic-mode behavior, unsafe request handling

### Phase 2: Android-first runtime hardening
1. Increase `n_ctx` from 512 (very limiting for multi-turn chat)
2. Add deterministic stop handling and output sanitization in JNI
3. Move logs to app-private encrypted storage

### Phase 3: Secure ops
1. Rotate any exposed HF tokens
2. Remove credentials from scripts — use env vars only
3. Add pre-commit secret scanning

### Phase 4: Product packaging
1. Standardize model filename/version metadata consumed by app
2. Implement in-app model validation (size/hash/format)

---

## Key File Index (Quick Navigation)

### Training/Data
- `finetuning/CUDA_run_pipeline.py` — Llama SFT orchestrator
- `finetuning/CUDA_run_pipeline_qwen.py` — Qwen SFT orchestrator
- `finetuning/CUDA_train_qlora.py` — Llama SFT trainer
- `finetuning/CUDA_train_qlora_qwen.py` — Qwen SFT trainer
- `finetuning/CUDA_train_dpo.py` — DPO trainer (both models via --model flag)
- `finetuning/build_dataset.py` — dataset builder
- `finetuning/clean_dataset.py` — dataset cleaner
- `synthetic/pipeline.py` — therapist/friend/grief datagen
- `synthetic/casual_pipeline.py` — casual datagen
- `synthetic/transition_pipeline.py` — casual→emotional pivot datagen
- `synthetic/dpo_pipeline.py` — DPO preference pair generation
- `synthetic/utils.py` — TeacherModel, parse_json_robust, load_prompt
- `synthetic/prompts/dpo_preference.txt` — DPO teacher prompt template

### Inference
- `inference/CUDA_chat_mindmate.py` — Llama chat
- `inference/CUDA_chat_qwen.py` — Qwen chat
- `system_prompt.txt` — canonical system prompt (source of truth)

### Export
- `scripts/export_gguf_cuda.py` — merge + convert + quantize
- `scripts/upload_gguf_to_hf.py` — upload to HuggingFace

### SLURM
- `run_finetune.slurm` — Llama SFT job
- `run_finetune_qwen.slurm` — Qwen SFT job
- `run_dpo_datagen.slurm` — DPO data generation job
- `run_dpo.slurm` — DPO training job
- `run_export.slurm` — GGUF export job

### Web
- `web/app.py`, `web/model_service.py`, `web/memory_engine.py`

### Android
- `android/app/src/main/java/com/mindmate/app/viewmodel/ChatViewModel.kt`
- `android/llama/src/main/cpp/llama-android.cpp`
