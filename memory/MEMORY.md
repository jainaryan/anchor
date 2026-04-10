# MindMate — Project Memory

## Quick Context
- Last session: 2026-04-10
- What was done:
  - Added Qwen2.5-3B finetune pipeline (scripts, SLURM, inference)
  - Evaluated all checkpoints across Qwen3-1.7B, Qwen2.5-3B, Llama v2 — checkpoint-200 is best on all models
  - Simplified system prompt from 150 lines → 13 lines of natural direction
  - Raised inference temperature 0.4 → 0.75
  - Fixed DPO pipeline through 10+ errors: `ref_adapter_name` removed (TRL version), `max_prompt_length` removed, `tokenizer` → `processing_class`, HF/Arrow column-major storage, Jinja2 dot-notation failure, torch version pinned to 2.10.0+cu128
  - DPO training COMPLETE: Llama ck200 (loss=1.127, job 560339) and Qwen2.5-3B ck200 (loss=1.279, job 560338)
  - GGUF export jobs resubmitted: 566379 (llama_sft), 566380 (llama_dpo), 566381 (qwen25_dpo)
  - Updated all docs and pushed to GitHub
- Next steps:
  1. Test DPO adapters at inference vs SFT ck200 baseline
  2. Once GGUF jobs complete — pull exports and deploy to Android app

---

## Goal
Build a finetuned, local mental-health wellness assistant that runs on Android.

**Why:** Personal use / product; user wants a private, on-device AI therapist companion.

**How to apply:** Frame all suggestions toward Android-first local inference. Prefer consolidation over new features.

---

## Stack
- Models: Llama 3.2 3B + Qwen2.5-3B, both finetuned with QLoRA (4-bit NF4 + PEFT) + DPO
  - Best SFT: `adapters/CUDA_mindmate_llama32b/checkpoint-200` (Llama v2)
  - Best DPO: `adapters/CUDA_mindmate_llama32b_dpo_ck200/` (Llama) and `adapters/CUDA_mindmate_qwen25_3b_dpo_ck200/` (Qwen2.5)
  - Qwen3-1.7B deprioritised — 1.7B capacity causes hallucinations and gender confusion
- Training: CUDA path on A100-40/80 cluster via SLURM (gpu-long partition)
  - Llama SFT: `finetuning/CUDA_run_pipeline.py` → `CUDA_train_qlora.py`
  - Qwen2.5-3B SFT: `finetuning/CUDA_run_pipeline_qwen25_3b.py` → `CUDA_train_qlora_qwen25_3b.py`
  - DPO: `finetuning/CUDA_train_dpo.py --model llama_ck200|qwen25_3b --steps 800`
- Data: Synthetic dialogues generated via Qwen3-30B-A3B-Instruct-2507 teacher model (bfloat16)
- Export: GGUF Q4_K_M via llama.cpp (`scripts/export_gguf_cuda.py`)
- Android runtime: JNI bridge to llama.cpp (`android/llama/src/main/cpp/llama-android.cpp`)
- Also: Flask web app (`web/app.py`) with memory/profile engine, macOS MLX CLI (legacy)

---

## Cluster Access (NUS SoC)
- Login node: `ssh nus-student-cluster` (xlogin.comp.nus.edu.sg, user: aryanj)
- Interactive GPU: `srun --partition=gpu --gres=gpu:h200-141:1 --mem=24G --cpus-per-task=4 --time=01:00:00 --pty bash`
- Project path on cluster: `~/projects/mindmate`
- Venv: `source ~/projects/mindmate/mindmatenv/bin/activate`
- Sync adapters locally: `rsync -avz --progress nus-student-cluster:~/projects/mindmate/adapters/CUDA_mindmate_llama32b/ adapters/CUDA_mindmate_llama32b/`
- **SLURM partitions:**
  - `gpu` — max 3h, has H100-96, H200-141, A100-40/80
  - `gpu-long` — max 3 days, same nodes, use for all training/datagen jobs
- **GPU availability notes:**
  - H200-141: strong but 3h max on `gpu` partition — not suitable for long jobs
  - A100-80: best for long jobs (gpu-long), 80GB VRAM fits 30B bfloat16 or 72B 4-bit
  - A100-40: 40GB VRAM; good for SFT/DPO on 3B models
  - H100-96: often at high utilization (90%+)
- **SLURM job logs:** go to `~/logs/` (home dir), NOT `~/projects/mindmate/logs/`
- **Cached HuggingFace models on cluster:**
  - `Qwen/Qwen3-1.7B`
  - `Qwen/Qwen2.5-3B-Instruct`
  - `Qwen/Qwen3-30B-A3B-Instruct-2507` ← teacher model for all data gen
  - `meta-llama/Llama-3.2-3B-Instruct`

---

## Active Canonical Pipeline
`CUDA_run_pipeline.py` → `CUDA_train_qlora.py` → `CUDA_train_dpo.py` → GGUF export → Android JNI app

## Current Training Hyperparameters (as of 2026-04-09)
### SFT (QLoRA)
- `learning_rate`: 1e-5, `lora_r`: 8, `lora_alpha`: 16, `max_steps`: 1600
- **Best checkpoint:** 200 (not final) — later steps overfit to therapy-speak and hallucinate shared history

### DPO
- `learning_rate`: 5e-7, `lora_r`: 8, `lora_alpha`: 16, `max_steps`: 800
- `beta`: 0.1, `loss_type`: sigmoid, `precompute_ref_log_probs`: True
- Architecture: two-model (policy + frozen ref_model), NOT multi-adapter (TRL version incompatibility)
- `processing_class=tokenizer` (NOT `tokenizer=` — that arg was removed in newer TRL)

---

## Saved Adapters

### `adapters/CUDA_mindmate_llama32b/` — Llama 3.2 3B SFT (job 554799, Apr 8 2026)
- v2 data mix: transition=30%, therapist=25%, casual=20%, grief=15%, friend=10%
- **Best checkpoint: checkpoint-200** — natural, pivots on distress, asks real questions
- Inference: `inference/CUDA_chat_mindmate.py --checkpoint checkpoint-200`

### `adapters/CUDA_mindmate_llama32b_dpo_ck200/` — Llama DPO (job 560339, Apr 9 2026) ✅
- Based on Llama SFT checkpoint-200. train_loss=1.127, 800 steps, ~3.3h on A100-80

### `adapters/CUDA_mindmate_qwen25_3b/` — Qwen2.5-3B SFT (job 555117, Apr 9 2026)
- Data mix: transition=35%, therapist=25%, casual=15%, grief=15%, friend=10%
- **Best checkpoint: checkpoint-200** — coherent but passive; DPO needed
- Inference: `inference/CUDA_chat_qwen25_3b.py --checkpoint checkpoint-200`

### `adapters/CUDA_mindmate_qwen25_3b_dpo_ck200/` — Qwen2.5-3B DPO (job 560338, Apr 9 2026) ✅
- Based on Qwen2.5-3B SFT checkpoint-200. train_loss=1.279, 800 steps, ~3.5h on A100-80

### `adapters/genz/` — ARCHIVED
- Llama 3.2 3B joke-mode-lock model. Superseded.

---

## Data Sources
- `data/synthetic_train_therapist_.jsonl` — 6,347 examples
- `data/synthetic_train_friend_1.jsonl` — 7,380 examples
- `data/synthetic_train.jsonl` — 5,565 examples (grief/loss heavy)
- `data/synthetic_train_casual.jsonl` — 5,000 examples
- `data/synthetic_train_transition.jsonl` — 6,170 examples (casual→emotional pivot)
- `data/dpo_train.jsonl` / `data/dpo_val.jsonl` — 2,125 / 375 pairs (85/15 split) ✅ COMPLETE

---

## DPO Pipeline

### Data Generation (`synthetic/dpo_pipeline.py`)
- 2,500 pairs: 50% mixed-mode (casual→emotional→panic arc) / 50% single-mode
- Single-mode categories: `casual_sad`, `transition`, `panic_mode`, `system_compliance`, `hallucination_guard`
- Teacher: `Qwen/Qwen3-30B-A3B-Instruct-2507` bfloat16, `enable_thinking=False`
- Incremental saves to `dpo_pairs_partial.jsonl`; train/val split every 25 pairs

### DPO Training (`finetuning/CUDA_train_dpo.py`)
- `--model llama_ck200` or `--model qwen25_3b`
- Two-model architecture: policy (base + SFT LoRA + DPO LoRA trainable) + ref_model (base + SFT frozen)
- `processing_class=tokenizer` in DPOTrainer (NOT `tokenizer=`)
- No `max_prompt_length` (removed from TRL DPOConfig)
- torch pinned to `2.10.0+cu128` in SLURM scripts to prevent CUDA breakage

---

## Qwen2.5-Specific Notes
- `Qwen/Qwen2.5-3B-Instruct` — no thinking mode, no `enable_thinking` needed
- `trust_remote_code=True` required
- Loss masking: `<|im_start|>assistant\n` / `<|im_end|>`

## Qwen3-Specific Notes
- Always `enable_thinking=False` in `apply_chat_template` at inference AND data gen
- `trust_remote_code=True` required

## Known TRL Compatibility Issues (cluster)
- `ref_adapter_name` not in DPOConfig → use two-model approach instead
- `max_prompt_length` not in DPOConfig → remove it
- `tokenizer=` not in DPOTrainer → use `processing_class=`
- `uv pip install --upgrade trl` pulls torch 2.11+cu130 which breaks CUDA → pin torch to 2.10.0+cu128
