---
tags: [anchor, training]
---

# Training Pipeline

← [[Home]]

---

## Pipeline Overview

```
Gemma 4 26B A4B IT (teacher, bfloat16, ~52GB)
        ↓ teacher-as-Anchor mode
  data/synthetic_train_*.jsonl  ← all normalized to production format (c3acdc9)
        ↓
  finetuning/build_dataset.py + clean_dataset.py
        ↓
  finetuning/CUDA_train_qlora.py  (SFT, QLoRA 4-bit NF4, Llama 3.2 3B)
        ↓
  adapters/genz/checkpoint-1200  ← BEST
        ↓
  scripts/export_gguf_cuda.py  →  Q4_K_M GGUF
        ↓
  Android app (anchor-app) / webapp (deploy/)
```

**DPO is abandoned** — all 3 runs flat or worse than SFT.

---

## SFT Hyperparameters

| Param | Value |
|---|---|
| `learning_rate` | 1e-5 |
| `lora_r` | 8 |
| `lora_alpha` | 16 |
| `max_steps` | 1600 (v2/v3), 2400 (v4) |
| `bnb_4bit_quant_type` | nf4 |
| `bnb_4bit_compute_dtype` | float16 |
| Base model | `meta-llama/Llama-3.2-3B-Instruct` |

**Always train fresh from base** — continued training (`PeftModel.from_pretrained`) catastrophically breaks MEMORY_USE (0/8, confirmed on genzv2_continued).

---

## Data Mixes

### v2 (genzv2 — produced best model)
| Source | Samples | % |
|---|---|---|
| synthetic_train_targeted_fix | 13,524 | dominant |
| synthetic_train_friend_1 | 7,380 | |
| synthetic_train_therapist_ | 2,637 | |
| synthetic_train_transition | 6,184 | |
| synthetic_train.jsonl (grief) | 5,565 | |
| synthetic_train_casual | 5,000 | |
| synthetic_train_biometric | 2,348 | |
| synthetic_train_targeted_fixes (gold) | 181 | always 100% |

### v4 (genzv4 — 21,529 total, 2400 steps)
targeted_fix 28%, transition 19%, friend 14%, therapist/casual/biometric ~11–12%, grief 5%, +181 gold

### v5 (planned — first run with correct format)
All v2 files (normalized) + `synthetic_train_conv_memory.jsonl` at ~20% weight

---

## Running SFT on Cluster

```bash
# Build dataset + train (v4 template — edit for v5)
sbatch finetuning/run_sft_v4.slurm

# Manual steps
cd ~/projects/mindmate
source mindmatenv/bin/activate
python finetuning/build_dataset.py --version v5
python finetuning/clean_dataset.py
sbatch --gres=gpu:a100-80:1 finetuning/run_sft_v4.slurm
```

### SLURM checklist
- [ ] Partition: `gpu-long` (3-day max)
- [ ] GPU: `--gres=gpu:a100-80:1` explicitly — H100-96 can fall back to 46GB nodes
- [ ] Log path: `~/logs/` (NOT `~/projects/mindmate/logs/`)
- [ ] `--export` flags BEFORE script path

---

## GGUF Export

```bash
# On cluster
python scripts/export_gguf_cuda.py --model genzv2_ck1200

# Via SLURM
sbatch scripts/run_export_top3.slurm
```

---

## Key Scripts

| Script | Purpose |
|---|---|
| `finetuning/CUDA_run_pipeline.py` | Orchestrator (build → clean → train) |
| `finetuning/CUDA_train_qlora.py` | SFT trainer |
| `finetuning/build_dataset.py` | Dataset builder (DATA_MIX_PRESETS) |
| `finetuning/clean_dataset.py` | Dedup + format validation |
| `scripts/export_gguf_cuda.py` | Merge adapter + quantize to GGUF |
| `synthetic/conversation_memory_pipeline.py` | Multi-turn memory data gen (active) |

---

## See also

- [[Data]] — training data files
- [[Cluster]] — SLURM, SSH, GPU notes
- [[Models]] — checkpoint results
