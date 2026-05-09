---
tags: [anchor, training]
---

# Training

← [[Home]]

---

## Pipeline Overview

```
data/synthetic_train_*.jsonl        (all normalized to production format — c3acdc9)
        ↓
finetuning/build_dataset.py --model v4
        → data/conversations_raw_v4/mindmate_train.jsonl
        → data/conversations_raw_v4/mindmate_val.jsonl
        ↓
finetuning/clean_dataset.py --train-in ... --val-in ... --train-out ... --val-out ...
        → data/conversations_cleaned_v4/mindmate_train.jsonl
        → data/conversations_cleaned_v4/mindmate_val.jsonl
        ↓
finetuning/CUDA_train_qlora.py --iters 2400 --data-dir ... --out-dir adapters/genzv4
        → adapters/genzv4/checkpoint-{200,400,...,2400}/
        ↓
scripts/export_gguf_cuda.py --model genzv4_ck200
        → exports/mindmate_genzv4_ck200_q4_k_m.gguf
        ↓
Android app (anchor-app JNI → llama.cpp) / webapp (deploy/server.py)
```

**DPO is abandoned** — all 3 DPO runs (genzv2_dpo, genzv3_dpo, genzv2_ck1600_dpo) flat or worse than SFT. Focus: SFT data quality and format correctness.

---

## SFT Hyperparameters

All defined in `finetuning/CUDA_train_qlora.py`:

| Parameter | Value | Notes |
|---|---|---|
| `learning_rate` | `1e-5` | |
| `per_device_train_batch_size` | `4` | |
| `gradient_accumulation_steps` | `2` | effective batch = 8 |
| `max_steps` | `1600` (v2/v3), `2400` (v4) | via `--iters` arg |
| `lr_scheduler_type` | `cosine` | |
| `warmup_ratio` | `0.03` | |
| `optim` | `paged_adamw_32bit` | |
| `bf16` | `True` | |
| `gradient_checkpointing` | `True` | |
| `lora_r` | `8` | |
| `lora_alpha` | `16` | |
| `lora_dropout` | `0.05` | |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` | all attention + MLP |
| `bnb_4bit_quant_type` | `nf4` | |
| `bnb_4bit_compute_dtype` | `float16` | |
| `bnb_4bit_use_double_quant` | `True` | |
| `max_length` | `2048` tokens | tokenizer truncation |
| `save_steps` / `eval_steps` | `200` | checkpoints every 200 steps |
| Base model | `meta-llama/Llama-3.2-3B-Instruct` | HF model ID |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | env var, prevents OOM fragmentation |

**Loss masking:** Assistant-only loss masking via `tokenizer.apply_chat_template`. System and user tokens get label=-100; only assistant response tokens are trained on.

**⚠️ NEVER use `--adapter-path` (continued training).** Tested in `genzv2_continued`: MEMORY_USE collapsed from 4/8 to 0/8 on all checkpoints. The approach is broken. Always train fresh from base model.

---

## Data Mix Presets

Defined in `finetuning/build_dataset.py` → `DATA_MIX_PRESETS`. Each key maps to `{filename: sample_count}`.

### v4 — current (job 602945, genzv4)

```python
"v4": {
    "synthetic_train_targeted_fix.jsonl":   6000,   # 28%
    "synthetic_train_transition.jsonl":     4000,   # 19%
    "synthetic_train_friend_1.jsonl":       3000,   # 14%
    "synthetic_train_casual.jsonl":         2500,   # 12%
    "synthetic_train_therapist_.jsonl":     2500,   # 11%
    "synthetic_train_biometric.jsonl":      2348,   # 11% (100% of available)
    "synthetic_train.jsonl":                1000,   # 5%  (grief/loss)
    "synthetic_train_targeted_fixes.jsonl":  181,   # gold — always 100%
}
# Total: ~21,529 examples
# Steps: 2400  (≈ 0.89 epochs; Goldilocks zone target: ck400–800)
```

### v3 — used for genzv3 (job 600329)

```python
"v3": {
    "synthetic_train_transition.jsonl":     2500,   # 25%
    "synthetic_train_targeted_fix.jsonl":   2000,   # 20%
    "synthetic_train_therapist_.jsonl":     1500,   # 15%
    "synthetic_train_biometric.jsonl":      1500,   # 15%
    "synthetic_train_friend_1.jsonl":       1500,   # 15%
    "synthetic_train_casual.jsonl":         1000,   # 10%
    "synthetic_train_targeted_fixes.jsonl":   65,   # gold (old count, before expansion)
}
# Total: ~10,065 | Steps: 1600 | Goldilocks zone: ck200
```

### v5 — planned (not yet in DATA_MIX_PRESETS)

See [[Next Steps]] for exact proposed mix. Key additions: `synthetic_train_conv_memory.jsonl` at ~20%. Must add v5 preset to `build_dataset.py` before running.

### Older presets (kept for reference, do not reuse)

- `v2` — 10k, grief-heavy, no targeted_fix, no biometric — used for genzv2 (best checkpoint ck1200)
- `v2_continued` — ~4,915, attempted continued training from ck1600 — **approach broken, never use**
- `qwen25_3b`, `gemma4_e2b`, `gemma4_e4b` — early experiments on non-Llama bases

---

## Running SFT on Cluster

### Submit via SLURM (recommended)

```bash
# From local machine — sync and submit v4 SLURM script
rsync -az finetuning/run_sft_v4.slurm nus-student-cluster:~/projects/mindmate/finetuning/
ssh nus-student-cluster "sbatch ~/projects/mindmate/finetuning/run_sft_v4.slurm"
```

**What `finetuning/run_sft_v4.slurm` does (in order):**

```bash
# 1. Build raw dataset
python -u finetuning/build_dataset.py --model v4
# → data/conversations_raw_v4/mindmate_{train,val}.jsonl

# 2. Clean dataset
python -u finetuning/clean_dataset.py \
    --train-in  data/conversations_raw_v4/mindmate_train.jsonl \
    --val-in    data/conversations_raw_v4/mindmate_val.jsonl \
    --train-out data/conversations_cleaned_v4/mindmate_train.jsonl \
    --val-out   data/conversations_cleaned_v4/mindmate_val.jsonl \
    --ensure-assistant-last \
    --drop-min-turns 4 \
    --dedup
# → data/conversations_cleaned_v4/mindmate_{train,val}.jsonl

# 3. Train
python -u finetuning/CUDA_train_qlora.py \
    --iters    2400 \
    --data-dir data/conversations_cleaned_v4 \
    --out-dir  adapters/genzv4
# → adapters/genzv4/checkpoint-{200,400,...,2400}/
```

**SLURM header for all SFT jobs:**
```bash
#SBATCH --partition=gpu-long       # 3-day max — needed for training
#SBATCH --gres=gpu:a100-80:1       # ALWAYS explicit — H100-96 falls back to 46GB
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/home/a/aryanj/logs/sft_v4_%j.out   # /home/a/ not /home/
#SBATCH --error=/home/a/aryanj/logs/sft_v4_%j.err
```

### Creating the v5 SLURM script

```bash
cp finetuning/run_sft_v4.slurm finetuning/run_sft_v5.slurm
# Then edit:
# - build_dataset.py --model v4  →  --model v5
# - conversations_raw_v4  →  conversations_raw_v5
# - conversations_cleaned_v4  →  conversations_cleaned_v5
# - adapters/genzv4  →  adapters/genzv5
# - --iters 2400  →  appropriate value
# Also add v5 to DATA_MIX_PRESETS in finetuning/build_dataset.py first
```

### SLURM submission checklist

- [ ] `v5` preset added to `finetuning/build_dataset.py DATA_MIX_PRESETS`
- [ ] Conv-memory data pulled from cluster (`data/synthetic_train_conv_memory.jsonl` exists locally)
- [ ] Conv-memory data synced to cluster
- [ ] SLURM script updated (version, paths, `--iters`)
- [ ] Partition: `gpu-long`
- [ ] GPU: `--gres=gpu:a100-80:1` in `#SBATCH` header (not just `--export`)
- [ ] Log path: `/home/a/aryanj/logs/` (note the `/a/` subdirectory)
- [ ] Submit with `sbatch finetuning/run_sft_v5.slurm` (no `--export` needed for SFT)

---

## GGUF Export

Run after training. Merges LoRA adapter into base, converts to F16 GGUF, then quantizes to Q4_K_M.

```bash
# On cluster interactively (after srun)
python scripts/export_gguf_cuda.py --model genzv2_ck1200

# Via SLURM
sbatch --gres=gpu:a100-80:1 scripts/run_export_top3.slurm   # exports genzv3_ck200, genzv2_ck1200, genzv4_ck200

# For a new checkpoint, add entry to scripts/export_gguf_cuda.py MODELS dict first:
# "genzv5_ck800": {
#     "base": "meta-llama/Llama-3.2-3B-Instruct",
#     "adapter": PROJECT_ROOT / "adapters" / "genzv5" / "checkpoint-800",
#     "out_name": "mindmate_genzv5_ck800",
# },
```

Export output: `exports/mindmate_genzv5_ck800_q4_k_m.gguf`

Export pipeline inside `export_gguf_cuda.py`:
1. Load base model (float16, not quantized)
2. Load + merge LoRA adapter → full merged model
3. Save merged model to temp dir
4. Run `llama.cpp/convert_hf_to_gguf.py` → F16 GGUF
5. Run `llama.cpp/build/bin/llama-quantize` with Q4_K_M → final GGUF
6. Clean up temp files

---

## HuggingFace Upload

```bash
python scripts/upload_gguf_to_hf.py
# Uploads exports/*.gguf to jainaryan/mindmate-gguf on HuggingFace
```

---

## Key Script Reference

| Script | Purpose | Args |
|---|---|---|
| `finetuning/build_dataset.py` | Samples from JSONL files per `DATA_MIX_PRESETS` | `--model v4` |
| `finetuning/clean_dataset.py` | Dedup + format validation | `--train-in`, `--val-in`, `--train-out`, `--val-out`, `--drop-min-turns 4`, `--dedup` |
| `finetuning/CUDA_train_qlora.py` | QLoRA SFT trainer | `--iters 2400`, `--data-dir data/conversations_cleaned_v4`, `--out-dir adapters/genzv4` |
| `finetuning/CUDA_train_dpo.py` | DPO trainer (**abandoned**) | `--model genzv2_ck1200`, `--steps 800` |
| `finetuning/run_sft_v4.slurm` | Current SFT SLURM job (build+clean+train) | No args — edit script for v5 |
| `scripts/export_gguf_cuda.py` | Merge adapter + quantize to GGUF | `--model genzv2_ck1200` |
| `scripts/run_export_top3.slurm` | Export top 3 checkpoints sequentially | No args |
| `scripts/upload_gguf_to_hf.py` | Upload GGUFs to HuggingFace | No args |
| `scripts/chat_cluster.py` | Interactive inference on cluster GPU | `--adapter adapters/genz/checkpoint-1200` |

**⚠️ Root-level scripts are legacy.** `CUDA_train_qlora.py`, `CUDA_train_dpo.py`, `build_dataset.py` at project root are old copies. Always use `finetuning/` versions.

---

## See also

- [[Data]] — all training JSONL files and how they were generated
- [[Models]] — what each training run produced
- [[Cluster]] — SLURM, SSH, GPU notes
- [[Next Steps]] — genzv5 plan
