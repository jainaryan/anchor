# MindMate — Project Memory

## Quick Context
- Last session: 2026-04-10
- What was done:
  - Added Qwen2.5-3B finetune pipeline (scripts, SLURM, inference)
  - Evaluated all checkpoints across Qwen3-1.7B, Qwen2.5-3B, Llama v2 — checkpoint-200 is best on all models
  - Simplified system prompt from 150 lines → 13 lines → 11 lines (current)
  - Raised inference temperature 0.4 → 0.75
  - Fixed DPO pipeline through 10+ errors (TRL compat, CUDA, Jinja2, Arrow storage)
  - DPO training COMPLETE: Llama ck200 (loss=1.127) and Qwen2.5-3B ck200 (loss=1.279)
  - All GGUF exports complete: llama_sft_ck200, llama_dpo_ck200, qwen25_dpo_ck200, llama_sft_ck1600
  - llama_sft_ck1600 uploaded to HF as llama(genz)v2_q4_k_m.gguf
  - Queued DPO for: Llama ck1600 (569555), Qwen2.5-3B ck1600 (569556), genz (569447)
  - Fixed CUDNN_STATUS_NOT_INITIALIZED on H100 — use attn_implementation="eager" everywhere
  - Updated inference scripts: eager attn, checkpoint-200 default, DPO stacked-adapter scripts
- Next steps:
  1. Wait for DPO jobs to complete (569555, 569556, 569447)
  2. Export GGUFs for ck1600 DPO models once done
  3. Compare all models at inference (SFT vs DPO, ck200 vs ck1600)

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
  - Also training: ck1600 DPO variants + genz DPO
  - Qwen3-1.7B deprioritised — 1.7B capacity causes hallucinations and gender confusion
- Training: CUDA path on A100/H100 cluster via SLURM (gpu-long partition)
  - Llama SFT: `finetuning/CUDA_run_pipeline.py` → `CUDA_train_qlora.py`
  - Qwen2.5-3B SFT: `finetuning/CUDA_run_pipeline_qwen25_3b.py` → `CUDA_train_qlora_qwen25_3b.py`
  - DPO: `finetuning/CUDA_train_dpo.py --model llama_ck200|qwen25_3b|llama_ck1600|qwen25_3b_ck1600|genz`
- Data: Synthetic dialogues generated via Qwen3-30B-A3B-Instruct-2507 teacher model (bfloat16)
- Export: GGUF Q4_K_M via llama.cpp (`scripts/export_gguf_cuda.py`)
- HF repo: `jainaryan/mindmate-gguf` (private)
- Android runtime: JNI bridge to llama.cpp (`android/llama/src/main/cpp/llama-android.cpp`)
- Also: Flask web app (`web/app.py`) with memory/profile engine, macOS MLX CLI (legacy)

---

## Cluster Access (NUS SoC)
- Login node: `ssh nus-student-cluster` (xlogin.comp.nus.edu.sg, user: aryanj)
- Interactive GPU: `srun --partition=gpu --gres=gpu:h200-141:1 --mem=24G --cpus-per-task=4 --time=02:00:00 --pty bash`
- Project path on cluster: `~/projects/mindmate`
- Venv: `source ~/projects/mindmate/mindmatenv/bin/activate`
- **SLURM partitions:**
  - `gpu` — max 3h, has H100-96, H200-141, A100-40/80
  - `gpu-long` — max 3 days, same nodes, use for all training/datagen jobs
- **GPU notes:**
  - H200-141: free most of the time, best for interactive sessions (use 2h limit)
  - A100-40: often at 100% — avoid
  - H100-47: good for DPO (20-40% utilisation)
  - **IMPORTANT:** All H100/H200 nodes need `attn_implementation="eager"` — SDPA causes CUDNN_STATUS_NOT_INITIALIZED
- **SLURM job logs:** go to `~/logs/` (home dir), NOT `~/projects/mindmate/logs/`
- **HF_TOKEN:** already set in `~/.bashrc` and `~/.bash_profile`
- **Cached HuggingFace models on cluster:**
  - `Qwen/Qwen3-1.7B`
  - `Qwen/Qwen2.5-3B-Instruct`
  - `Qwen/Qwen3-30B-A3B-Instruct-2507` ← teacher model for all data gen
  - `meta-llama/Llama-3.2-3B-Instruct`

---

## Active Canonical Pipeline
`CUDA_run_pipeline.py` → `CUDA_train_qlora.py` → `CUDA_train_dpo.py` → GGUF export → HF upload → Android

## Current Training Hyperparameters (as of 2026-04-10)
### SFT (QLoRA)
- `learning_rate`: 1e-5, `lora_r`: 8, `lora_alpha`: 16, `max_steps`: 1600
- **Best checkpoint:** 200 (not final) — later steps overfit to therapy-speak and hallucinate shared history

### DPO
- `learning_rate`: 5e-7, `lora_r`: 8, `lora_alpha`: 16, `max_steps`: 800
- `beta`: 0.1, `loss_type`: sigmoid, `precompute_ref_log_probs`: True
- Architecture: two-model (policy + frozen ref_model), NOT multi-adapter (TRL version incompatibility)
- `processing_class=tokenizer` (NOT `tokenizer=`)
- `attn_implementation="eager"` required on H100/H200 nodes

---

## Saved Adapters

### `adapters/CUDA_mindmate_llama32b/` — Llama 3.2 3B SFT (job 554799)
- v2 data mix: transition=30%, therapist=25%, casual=20%, grief=15%, friend=10%
- **Best checkpoint: checkpoint-200**
- Inference: `inference/CUDA_chat_mindmate.py` (defaults to ck200)

### `adapters/CUDA_mindmate_llama32b_dpo_ck200/` — Llama DPO ck200 ✅
- loss=1.127, 800 steps
- Inference: `inference/CUDA_chat_mindmate_dpo.py`

### `adapters/CUDA_mindmate_llama32b_dpo_ck1600/` — Llama DPO ck1600 ⏳ TRAINING (job 569555)

### `adapters/CUDA_mindmate_qwen25_3b/` — Qwen2.5-3B SFT
- **Best checkpoint: checkpoint-200**
- Inference: `inference/CUDA_chat_qwen25_3b.py` (defaults to ck200)

### `adapters/CUDA_mindmate_qwen25_3b_dpo_ck200/` — Qwen2.5-3B DPO ck200 ✅
- loss=1.279, 800 steps
- Inference: `inference/CUDA_chat_qwen25_3b_dpo.py`

### `adapters/CUDA_mindmate_qwen25_3b_dpo_ck1600/` — Qwen2.5-3B DPO ck1600 ⏳ TRAINING (job 569556)

### `adapters/genz/` — GenZ model (Llama 3.2 3B) — ARCHIVED SFT
- Joke-mode-lock issue. DPO running (job 569447) to see if it can be fixed.

### `adapters/genz_dpo/` — genz DPO ⏳ TRAINING (job 569447)

---

## GGUFs on HuggingFace (`jainaryan/mindmate-gguf`)
| Filename | What |
|---|---|
| `llama(genz)v2_q4_k_m.gguf` | Llama SFT ck1600 — uploaded Apr 10 |

## GGUFs on Cluster (`exports/`)
| Folder | What |
|---|---|
| `mindmate_llama_sft_ck200/` | Llama SFT ck200 |
| `mindmate_llama_dpo_ck200/` | Llama DPO ck200 |
| `mindmate_llama_sft_ck1600/` | Llama SFT ck1600 ← uploaded to HF |
| `mindmate_qwen25_dpo_ck200/` | Qwen2.5-3B DPO ck200 |

---

## Inference Scripts
| Script | Model |
|---|---|
| `inference/CUDA_chat_mindmate.py` | Llama SFT (default ck200, pass `--checkpoint` to change) |
| `inference/CUDA_chat_mindmate_dpo.py` | Llama DPO (SFT ck200 + DPO, stackable) |
| `inference/CUDA_chat_qwen25_3b.py` | Qwen2.5-3B SFT (default ck200) |
| `inference/CUDA_chat_qwen25_3b_dpo.py` | Qwen2.5-3B DPO (SFT ck200 + DPO, stackable) |

---

## Data Sources
- `data/synthetic_train_therapist_.jsonl` — 6,347 examples
- `data/synthetic_train_friend_1.jsonl` — 7,380 examples
- `data/synthetic_train.jsonl` — 5,565 examples (grief/loss heavy)
- `data/synthetic_train_casual.jsonl` — 5,000 examples
- `data/synthetic_train_transition.jsonl` — 6,170 examples (casual→emotional pivot)
- `data/dpo_train.jsonl` / `data/dpo_val.jsonl` — 2,125 / 375 pairs ✅

---

## Known Issues / Gotchas
- `attn_implementation="eager"` required on H100/H200 — SDPA causes CUDNN_STATUS_NOT_INITIALIZED
- `torch_dtype` deprecated — use `dtype` instead
- `processing_class=tokenizer` in DPOTrainer (NOT `tokenizer=`)
- `max_prompt_length` removed from TRL DPOConfig
- `ref_adapter_name` not in DPOConfig → use two-model approach
- torch pinned to `2.10.0+cu128` in SLURM — upgrading pulls incompatible CUDA
- Interactive sessions: use 2h limit, H200-141 is usually free
- SLURM logs go to `~/logs/` not project logs dir
