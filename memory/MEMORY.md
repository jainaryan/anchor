# MindMate — Project Memory

## Quick Context
- Last session: 2026-04-18
- What was done:
  - Full model evaluation: **genzv2 SFT checkpoint-1600 is the best model overall** — beats all DPO variants and all other SFT checkpoints
  - DPO did NOT improve over SFT (Llama DPO ck200 and Qwen2.5 DPO ck200 both inferior)
  - App deployed live at https://tryanchor.me (DigitalOcean c-4 droplet, IP: 209.38.122.228)
  - Running: `adapters/genz/checkpoint-1600` → GGUF Q4_K_M → llama-cpp-python CPU inference
  - UI built: FastAPI + SSE streaming, sidebar chat history, mobile-responsive, welcome screen
  - Feedback button: filled accent pill, strobe animation at 30s, toast after 3rd message
  - Welcome screen: anchor emoji + description of on-device privacy-focused vision
  - 20 real users have chatted as of 2026-04-18
  - HF repo `jainaryan/mindmate-gguf` has 4 models (llama genzv2, llama genz, qwen3 1.7b, gemma4 e2b)
  - Gemma 4 E2B finetune completed and uploaded to HF (not yet deployed on server)
  - Added Gemma 4 E2B IT pipeline scripts and `gemma4_e2b` data mix preset
- Next steps:
  1. Deploy Gemma 4 E2B to server and compare with genzv2 in production
  2. Export `adapters/genz/checkpoint-1600` to GGUF Q4_K_M for Android (if not done)
  3. Deploy GGUF to Android app
  4. Set up periodic email of chat logs (Gmail app password needed)

---

## Goal
Build a finetuned, local mental-health wellness assistant that runs on Android.

**Why:** Personal use / product; user wants a private, on-device AI therapist companion.

**How to apply:** Frame all suggestions toward Android-first local inference. Prefer consolidation over new features.

---

## Stack
- Models: Llama 3.2 3B finetuned with QLoRA (4-bit NF4 + PEFT)
  - **Best model overall: `adapters/genz/checkpoint-1600`** (genzv2 SFT) — beats all DPO variants and all other SFT checkpoints
  - Other trained: Qwen2.5-3B SFT/DPO, Llama DPO — all inferior to genzv2 ck1600
  - Qwen3-1.7B deprioritised — 1.7B capacity causes hallucinations and gender confusion
- Training: CUDA path on A100-40/80 cluster via SLURM (gpu-long partition)
  - Llama SFT: `finetuning/CUDA_run_pipeline.py` → `CUDA_train_qlora.py`
  - Qwen2.5-3B SFT: `finetuning/CUDA_run_pipeline_qwen25_3b.py` → `CUDA_train_qlora_qwen25_3b.py`
  - Gemma 4 E2B IT SFT: `finetuning/CUDA_run_pipeline_gemma4_e2b.py` → `CUDA_train_qlora_gemma4_e2b.py`
  - DPO: `finetuning/CUDA_train_dpo.py --model llama_ck200|qwen25_3b --steps 800`
- Data: Synthetic dialogues generated via Gemma 4 26B A4B IT teacher model (bfloat16, `google/gemma-4-26B-A4B-it`)
- Export: GGUF Q4_K_M via llama.cpp (`scripts/export_gguf_cuda.py`)
- **Production webapp:** FastAPI + SSE (`deploy/server.py`), nginx reverse proxy, systemd service
  - URL: https://tryanchor.me
  - Server: DigitalOcean c-4 dedicated CPU droplet, IP 209.38.122.228
  - SSH: `ssh -i ~/.ssh/id_ed25519 root@209.38.122.228`
  - Deploy static: `rsync -az -e "ssh -i ~/.ssh/id_ed25519" deploy/static/ root@209.38.122.228:~/mindmate/deploy/static/`
  - Restart service: `systemctl restart mindmate`
  - Logs: `journalctl -u mindmate -f` or nginx logs at `/var/log/nginx/access.log`
- Android runtime: JNI bridge to llama.cpp (`android/llama/src/main/cpp/llama-android.cpp`)
- HF repo: `jainaryan/mindmate-gguf` (public, gated)

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
  - `google/gemma-4-26B-A4B-it` ← teacher model for all data gen (not yet cached — downloads on first run)
  - `meta-llama/Llama-3.2-3B-Instruct`

---

## Active Canonical Pipeline
`CUDA_run_pipeline.py` → `CUDA_train_qlora.py` → GGUF export → tryanchor.me / Android JNI app

## Current Training Hyperparameters (as of 2026-04-18)
### SFT (QLoRA)
- `learning_rate`: 1e-5, `lora_r`: 8, `lora_alpha`: 16, `max_steps`: 1600
- **Best checkpoint: 1600** (genzv2) — full 1600 steps wins; DPO did not improve it

### DPO
- `learning_rate`: 5e-7, `lora_r`: 8, `lora_alpha`: 16, `max_steps`: 800
- `beta`: 0.1, `loss_type`: sigmoid, `precompute_ref_log_probs`: True
- Architecture: two-model (policy + frozen ref_model), NOT multi-adapter (TRL version incompatibility)
- `processing_class=tokenizer` (NOT `tokenizer=` — that arg was removed in newer TRL)
- **DPO did NOT improve over SFT** — genzv2 SFT ck1600 beats both DPO models

---

## Saved Adapters

### `adapters/genz/checkpoint-1600` — **BEST MODEL** ✅ (genzv2 SFT)
- Llama 3.2 3B, v2 data mix (base: `meta-llama/Llama-3.2-3B-Instruct`)
- **In production at tryanchor.me** as `llama(genz)v2_q4_k_m.gguf`
- Beats all DPO models and all other SFT checkpoints

### `adapters/CUDA_mindmate_llama32b/` — Llama 3.2 3B SFT (job 554799)
- v2 data mix: transition=30%, therapist=25%, casual=20%, grief=15%, friend=10%
- Checkpoints 200–1600 available; best is ck200 for this adapter (not genz)

### `adapters/CUDA_mindmate_llama32b_dpo_ck200/` — Llama DPO (job 560339)
- Based on Llama SFT checkpoint-200. train_loss=1.127, 800 steps
- **Inferior to genzv2 SFT ck1600**

### `adapters/CUDA_mindmate_qwen25_3b/` — Qwen2.5-3B SFT (job 555117)
- Data mix: transition=35%, therapist=25%, casual=15%, grief=15%, friend=10%
- **Inferior to genzv2 SFT ck1600**

### `adapters/CUDA_mindmate_qwen25_3b_dpo_ck200/` — Qwen2.5-3B DPO (job 560338)
- train_loss=1.279, 800 steps
- **Inferior to genzv2 SFT ck1600**

---

## GGUFs on HuggingFace (`jainaryan/mindmate-gguf`)
| Filename | What | Size |
|---|---|---|
| `llama(genz)v2_q4_k_m.gguf` | Llama genzv2 SFT ck1600 — **production model** | 2.02 GB |
| `mindmate_genz_llama32_3b_q4_k_m.gguf` | Same as above (duplicate upload) | 2.02 GB |
| `mindmate_gemma4_e2b_q4_k_m.gguf` | Gemma 4 E2B IT finetune | 3.43 GB |
| `mindmate_qwen3_1p7b_q4_k_m.gguf` | Qwen3 1.7B finetune | 1.28 GB |

---

## Production Webapp (`deploy/`)
- `deploy/server.py` — FastAPI + SSE streaming
- `deploy/config.py` — N_CTX=2048, MAX_TOKENS=150, INFERENCE_TIMEOUT=120, N_GPU_LAYERS=0 (CPU)
- `deploy/static/index.html` — UI: sidebar, chat, welcome screen
- `deploy/static/script.js` — chat logic, SSE reader, sidebar, feedback pulse/toast
- `deploy/static/style.css` — styling
- `inference/system_prompt.txt` — 6-line Anchor system prompt (warm friend, no lecturing)
- nginx config: `/etc/nginx/sites-available/mindmate` — proxy_http_version 1.1, proxy_buffering off
- systemd: `/etc/systemd/system/mindmate.service` — Restart=always
- Billing guard: `/root/billing_guard.sh` — hourly cron, destroys droplet at $175/month spend

---

## Data Sources

### Active synthetic data (usable by build_dataset.py)
- `data/synthetic_train_friend_1.jsonl` — 7,380 examples
- `data/synthetic_train_therapist_.jsonl` — 6,347 examples
- `data/synthetic_train_transition.jsonl` — 6,184 examples
- `data/synthetic_train.jsonl` — 5,565 examples (grief/loss heavy)
- `data/synthetic_train_casual.jsonl` — 5,000 examples
- **Total available: 30,476** — pipeline caps at 10,000 via `DATA_MIX_PRESETS`

### DPO data
- `data/dpo_train.jsonl` / `data/dpo_val.jsonl` — 2,125 / 375 pairs ✅

### Legacy data — **DO NOT USE**
- `data/new_raw_data/mindmate_train.jsonl` — 20,662 examples — excluded
- Old `{"text": "<|user|>..."}` format — not worth converting

---

## Known Issues / Gotchas
- `attn_implementation="eager"` required on H100/H200 — SDPA causes CUDNN_STATUS_NOT_INITIALIZED
- Same rule applies to Gemma 4 E2B under 4-bit
- `processing_class=tokenizer` in DPOTrainer (NOT `tokenizer=`)
- `max_prompt_length` removed from TRL DPOConfig
- `ref_adapter_name` not in DPOConfig → use two-model approach
- torch pinned to `2.10.0+cu128` in SLURM — upgrading pulls incompatible CUDA
- SLURM logs go to `~/logs/` not project logs dir

## Gemma 4 E2B IT Notes
- Use `attn_implementation="eager"` — SDPA causes NaN gradients under 4-bit
- `bfloat16` compute dtype (not float16)
- Loss masking: `<start_of_turn>model\n` / `<end_of_turn>`
- Gated model — `HF_TOKEN` required on cluster
