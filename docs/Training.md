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

**Near-duplicate deduplication (added 2026-05-19):** `clean_dataset.py --near-dedup` runs MinHash LSH over assistant-turn text (the part being trained on). Char 4-gram shingles, 128 permutations, banded LSH auto-tuned to `--near-dedup-threshold` (default 0.8 ≈ 80% Jaccard similarity). A single shared LSH index spans both train and val files, so cross-file near-dups (e.g., same-profile Qwen shard vs Gemma4 file) are also caught. No new dependencies — stdlib only. Effective threshold and drop count are printed at runtime. Both `run_sft_v4.slurm` and the planned `run_sft_v5.slurm` include this flag.

**Loss masking:** Assistant-only loss masking via `tokenizer.apply_chat_template`. System and user tokens get label=-100; only assistant response tokens are trained on.

**Category-conditional loss weights (added 2026-05-19):** Each training example carries a `loss_weight` field, set by source file in `finetuning/build_dataset.py::CATEGORY_WEIGHTS` and propagated through `clean_dataset.py`. `WeightedLossTrainer` in `CUDA_train_qlora.py` computes per-example assistant-token mean cross-entropy and reduces with `sum(w_i * L_i) / sum(w_i)` (weighted-mean — keeps gradient scale comparable to unweighted runs, so existing LR / cosine schedule still applies). Current weights:

| File (or prefix) | Weight | Why |
|---|---|---|
| `synthetic_train_targeted_fixes.jsonl` | 3.0 | Gold NO_HALLUCINATION + memory_recall fixes |
| `synthetic_train_targeted_fix.jsonl` | 2.0 | Primary HELP_MODE fix data |
| `synthetic_train_crisis_qwen*` | 4.0 | CRISIS is the largest SFT regression (67% → 48%) |
| `synthetic_train_help_mode_qwen*` | 3.0 | HELP_MODE 58% → 25–44% under prior SFT |
| `synthetic_train_conv_memory*` | 1.5 | Memory categories weak vs base |
| `synthetic_train_biometric*` | 1.5 | BIOMETRIC 60% → 30% under prior SFT |
| (default) | 1.0 | friend / casual / transition / therapist / grief |

Prefix matching means sharded files (`*_qwen_s0.jsonl`, `*_merged.jsonl`) inherit the same weight. Examples missing the field default to 1.0 — old datasets continue to work unchanged.

**⚠️ NEVER use `--adapter-path` (continued training).** Tested in `genzv2_continued`: MEMORY_USE collapsed from 4/8 to 0/8 on all checkpoints. The approach is broken. Always train fresh from base model.

---

## Data Mix Presets

Defined in `finetuning/build_dataset.py` → `DATA_MIX_PRESETS`. Each key maps to `{filename: sample_count}`.

**As of 2026-05-19, the iterator `extra_paths` is derived from `SOURCE_CAPS.keys()`** — adding a file to a preset automatically picks it up at load time. Previously `extra_paths` was a separate hand-maintained list and silently dropped files that weren't in it. v3 and v4 build_dataset runs against the git version of this file under-loaded by 3 files (`targeted_fix`, `biometric`, `targeted_fixes`); the documented `~21k` / `~10k` totals below reflect *intent*, not necessarily what was loaded historically. See Bug Log 2026-05-19.

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

### v5 — early run (job 616892, PENDING on A100-80, 2026-05-19)

Early genzv5 training with currently available data. Full v5 (with finished conv-memory new-pool shards and biometric shards 3–5) will be a follow-up run.

```python
"v5": {
    "synthetic_train_targeted_fix.jsonl":        4000,   # 29%  weight=2.0
    "synthetic_train_friend_1.jsonl":            2000,   # 14%
    "synthetic_train_transition.jsonl":          1500,   # 11%
    "synthetic_train_biometric.jsonl":           1500,   # 11%  weight=1.5  (Gemma4, keep until Qwen shards done)
    "synthetic_train_conv_memory.jsonl":          167,   #  1%  weight=1.5
    "synthetic_train_conv_memory_qwen.jsonl":     299,   #  2%  weight=1.5
    "synthetic_train_conv_memory_qwen_s0.jsonl":  335,   #  2%  weight=1.5
    "synthetic_train_conv_memory_qwen_s1.jsonl":  316,   #  2%  weight=1.5
    "synthetic_train_conv_memory_qwen_s2.jsonl":  324,   #  2%  weight=1.5
    "synthetic_train_biometric_qwen_s0.jsonl":    993,   #  7%  weight=1.5
    "synthetic_train_biometric_qwen_s1.jsonl":     80,   #  1%  weight=1.5
    "synthetic_train_biometric_qwen_s2.jsonl":     78,   #  1%  weight=1.5
    "synthetic_train_casual.jsonl":              1000,   #  7%
    "synthetic_train_therapist_.jsonl":           800,   #  6%
    "synthetic_train.jsonl":                      300,   #  2%  (grief)
    "synthetic_train_targeted_fixes.jsonl":       181,   #  1%  weight=3.0  (gold)
    "synthetic_train_help_mode_qwen.jsonl":        115,  #  1%  weight=3.0
}
# Total: ~13,988 examples
# Steps: 2000 (~1 epoch; checkpoints every 200)
```

Note: SLURM script uses `data/conversations_raw_v5` and `data/conversations_cleaned_v5`. Near-dedup flag NOT included (not available on cluster's `clean_dataset.py`). Full CUDA shim applied in `CUDA_train_qlora.py` (all cluster nodes have driver 12090 / PyTorch cu130 mismatch — see Bug Log 2026-05-20):
```python
torch.cuda.is_available = lambda: True          # guard + TrainingArguments check
torch.cuda.is_bf16_supported = lambda *a, **kw: True  # bf16 validation
torch.cuda._initialized = True                  # _lazy_init short-circuits hereafter
torch.cuda._queued_calls.clear()                # drop _check_capability from deferred queue
```
bitsandbytes uses its own compiled CUDA extension and is unaffected by these patches. Commit `69dd330`.

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

### SLURM submission checklist (genzv5 early run — 2026-05-19)

- [x] `v5` preset added to `finetuning/build_dataset.py DATA_MIX_PRESETS`
- [x] SLURM script `finetuning/run_sft_v5.slurm` created
- [x] CUDA guard in trainer replaced with warning (commit `16ad500`)
- [x] Scripts rsync'd to cluster
- [x] Job 616892 submitted → PENDING on A100-80
- [ ] Wait for jobs 615485–615490 (conv-memory new-pool) + 615491–615493 (biometric shards 3–5) — both finish ~72h from 2026-05-17 (~31h remaining)
- [ ] Merge shards + rsync to cluster → full v5 data (re-run build+clean+train with complete files)

**Checklist for any future v5+ job:**
- Partition: `gpu-long`
- GPU: `--gres=gpu:a100-80:1` in `#SBATCH` header (not `--export`)
- Log path: `/home/a/aryanj/logs/` (note the `/a/` subdirectory)
- `PYTHONUNBUFFERED=1` in SLURM script for visible print output

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
