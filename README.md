# MindMate

A finetuned, local mental-health wellness assistant that runs on Android. Models are trained with QLoRA on a CUDA cluster, exported to GGUF, and deployed in a React Native Android app via a llama.cpp JNI bridge.

**Production webapp:** https://tryanchor.me (FastAPI + SSE, DigitalOcean)
**Android app:** `anchor-app/` (React Native, package `com.pocketpal`)

---

## Best Model

**`adapters/genz/checkpoint-1600`** — Llama 3.2 3B SFT (genzv2), Q4_K_M
- Beats all DPO variants and all other SFT checkpoints
- Deployed as `exports/mindmate_llama_sft_ck1600/llama(genz)v2_q4_k_m.gguf`
- On device at `/sdcard/Download/mindmate.gguf` (Pixel 8a)
- Gen: ~5.4–6.0 TPS, TTFT: 6s (cached) / 60–66s (cold 1100-tok prompt)

---

## Stack

| Layer | Tool |
|---|---|
| Base model | Llama 3.2 3B Instruct |
| SFT | QLoRA (4-bit NF4, PEFT), `CUDA_train_qlora.py` |
| DPO | TRL DPOTrainer, two-model architecture, `CUDA_train_dpo.py` |
| Teacher (datagen) | Gemma 4 26B A4B IT (bfloat16, ~52GB, A100-80) |
| Export | GGUF Q4_K_M via llama.cpp, `scripts/export_gguf_cuda.py` |
| Android runtime | JNI → llama.cpp (`android/llama/src/main/cpp/llama-android.cpp`) |
| Webapp | FastAPI + SSE (`deploy/server.py`) |
| Cluster | NUS SoC SLURM, `gpu-long` partition, A100-80 |

---

## Training Pipeline

```
Synthetic Data (Gemma 4 26B A4B IT teacher)
        ↓
  SFT: CUDA_run_pipeline.py → CUDA_train_qlora.py
        ↓
  adapters/genz/checkpoint-1600  ← best SFT checkpoint
        ↓
  DPO: CUDA_train_dpo.py --model llama_ck1600
        ↓
  GGUF export: scripts/export_gguf_cuda.py
        ↓
  Android app (anchor-app/) or webapp (deploy/)
```

### Running SFT on cluster
```bash
cd ~/projects/mindmate
sbatch finetuning/run_finetune_llama_v2.slurm
```

### Running DPO on cluster
```bash
sbatch finetuning/run_dpo_llama_ck1600.slurm
# Trains on dpo_train_v2.jsonl (7,200 pairs), 1200 steps, A100-80
# Output: adapters/genz_dpo_ck1600/
```

### Exporting to GGUF
```bash
sbatch run_export.slurm
# or directly:
python scripts/export_gguf_cuda.py --model llama_ck1600
```

---

## Data

### SFT training data (33,534 usable examples)

| File | Examples | Content |
|---|---|---|
| `data/synthetic_train_friend_1.jsonl` | 7,380 | Casual friend-style support |
| `data/synthetic_train_therapist_.jsonl` | 6,347 | Therapeutic dialogue |
| `data/synthetic_train_transition.jsonl` | 6,184 | Casual→emotional pivot |
| `data/synthetic_train.jsonl` | 5,565 | Grief/loss focused |
| `data/synthetic_train_casual.jsonl` | 5,000 | Non-distress casual |
| `data/synthetic_train_targeted_fix.jsonl` | 2,993 | Targeted: help_mode + memory_recall |
| `data/synthetic_train_targeted_fixes.jsonl` | 65 | Hand-crafted gold examples |
| `data/synthetic_train_biometric.jsonl` | pending | Biometric health context (job 595713) |

### DPO data

| File | Pairs | Notes |
|---|---|---|
| `data/dpo_train_v2.jsonl` | 6,120 | **Active** — use this |
| `data/dpo_val_v2.jsonl` | 1,080 | **Active** — use this |
| `data/dpo_train.jsonl` | 2,125 | v1, superseded |
| `data/dpo_val.jsonl` | 375 | v1, superseded |

Categories in v2: mixed_mode, casual_sad, panic_mode, transition, hallucination_guard, system_compliance. **help_mode and memory_recall pairs are pending** (job 595714).

---

## Active Cluster Jobs (as of 2026-04-24)

| Job | What | GPU | Status |
|---|---|---|---|
| 595679 | DPO genzv2 ck1600, 1200 steps | A100-80 | PENDING |
| 595713 | Biometric SFT+DPO datagen (24h) | A100-80 | PENDING |
| 595714 | DPO targeted fix (help_mode+memory_recall) | A100-80 | PENDING |

---

## Eval

14-scenario eval harness in `anchor-app/src/eval/`. Run from the EvalScreen in the Android app.

**Latest results (Batch-2, 2026-04-23):**
- Single-turn human inspect: ~5/14 = 36% pass
- Multi-turn consistent all-pass: 2/15 = 13%
- Biometric probe pass: 7/22 = 32%
- Critical issue: "you went quiet on me there" hallucination in 6 scenarios

Full results: `anchor-app/src/eval/EVAL_RESULTS.md`

---

## Saved Adapters

| Adapter | Base | Notes |
|---|---|---|
| `adapters/genz/checkpoint-1600` | Llama 3.2 3B | **BEST — in production** |
| `adapters/CUDA_mindmate_llama32b/` | Llama 3.2 3B | v2 SFT, same data mix |
| `adapters/CUDA_mindmate_llama32b_dpo_ck200/` | Llama 3.2 3B | DPO ck200 — inferior to genzv2 SFT |
| `adapters/CUDA_mindmate_qwen25_3b/` | Qwen2.5-3B | Inferior to Llama |
| `adapters/CUDA_mindmate_qwen25_3b_dpo_ck200/` | Qwen2.5-3B | DPO — inferior to Llama |
| `adapters/genz_dpo_ck1600/` | Llama 3.2 3B | **Pending** — job 595679 |

---

## Cluster Access

```bash
ssh nus-student-cluster   # xlogin.comp.nus.edu.sg, user: aryanj
# Requires NUS VPN
squeue -u aryanj          # check jobs
sinfo                     # check node availability

# Sync local → cluster
rsync -avz --progress data/ nus-student-cluster:~/projects/mindmate/data/

# Sync cluster → local (adapters)
rsync -avz --progress nus-student-cluster:~/projects/mindmate/adapters/ adapters/
```

**Partitions:** `gpu` (max 3h), `gpu-long` (max 3 days). All training on `gpu-long` with `a100-80`.
**Logs:** `~/logs/` on cluster (NOT `~/projects/mindmate/logs/`).

---

## Android App

App: `anchor-app/` (React Native, NOT `mindmate_app/` which is a stale scratch fork)

```bash
cd anchor-app
yarn typecheck               # typecheck
cd android && ./gradlew assembleDebug   # build APK
```

Model file expected at `/sdcard/Download/mindmate.gguf` on device.

---

## Project Structure

```
mindmate/
├── adapters/               # LoRA checkpoints
│   ├── genz/checkpoint-1600/   ← BEST MODEL
│   └── genz_dpo_ck1600/        ← pending job 595679
├── data/                   # Training data (JSONL)
├── exports/                # GGUF exports
├── finetuning/             # Training scripts
│   ├── CUDA_run_pipeline.py
│   ├── CUDA_train_qlora.py
│   ├── CUDA_train_dpo.py
│   └── run_dpo_llama_ck1600.slurm
├── synthetic/              # Data generation pipelines
│   ├── biometric_sft_pipeline.py
│   ├── biometric_dpo_pipeline.py
│   ├── dpo_targeted_fix_pipeline.py
│   ├── utils.py
│   └── prompts/
├── inference/              # CUDA chat scripts
├── deploy/                 # Production FastAPI server
│   └── server.py
├── scripts/                # Export + utility scripts
└── memory/                 # Project memory files
```
