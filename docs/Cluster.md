---
tags: [anchor, cluster]
---

# Cluster

← [[Home]]

---

## Access

```bash
# Requires NUS VPN to be active first (FortiClient on Mac)
ssh nus-student-cluster
# → xlogin.comp.nus.edu.sg, user: aryanj

# Activate environment
cd ~/projects/mindmate
source mindmatenv/bin/activate

# Python path sanity check
which python   # should be ~/projects/mindmate/mindmatenv/bin/python
```

---

## SLURM Quick Reference

```bash
# Check your jobs
squeue -u aryanj

# Detailed job info (start time, end time, elapsed, state)
sacct -u aryanj -j 609110 --format=JobID,State,Start,End,Elapsed,NodeList

# Watch job queue (refresh every 5s)
watch -n5 squeue -u aryanj

# GPU availability across nodes
sinfo -o "%P %G %C %N"

# Cancel a job
scancel 609110

# Check what GPU a running job got
squeue -u aryanj -o "%.18i %.8P %.20j %.8u %.8T %.10M %.9l %R %b"
```

---

## Partitions

| Partition | Max time | GPU options | Use for |
|---|---|---|---|
| `gpu` | 3h | H200-141, A100-80, A100-40, H100-96 | Quick benchmark runs, interactive |
| `gpu-long` | 3 days | A100-80, A100-40 | **All training and long datagen** |

**Always use `gpu-long` for training and the conv-memory pipeline (72h jobs).**

---

## GPU Notes

| GPU | VRAM | Notes |
|---|---|---|
| A100-80 | 80GB | **Best. Use `--gres=gpu:a100-80:1` for all training and benchmarking.** |
| A100-40 | 40GB | OOMs for Gemma4 bfloat16 judge (~52GB). OK for SFT training only. |
| H100-96 GRES | varies | ⚠️ Has fallen back to ~46GB nodes (jobs 609078–609083 OOMed). Unreliable. Do not use. |
| H200-141 | 141GB | `gpu` partition only (3h max) — not usable for training |

**⚠️ Always specify `--gres=gpu:a100-80:1` explicitly in the `#SBATCH` header.** Relying on H100-96 GRES is unreliable — it silently falls back to smaller nodes.

---

## Critical SLURM Gotchas

These have each cost at least one job:

### 1. `--export` must be BEFORE the script path

```bash
# ✅ Correct — MODEL env var is passed to job
sbatch --gres=gpu:a100-80:1 --export=ALL,MODEL=genzv2_ck1200 benchmarks/run_benchmarks.slurm

# ❌ Wrong — MODEL silently treated as script argument, job uses default
sbatch --gres=gpu:a100-80:1 benchmarks/run_benchmarks.slurm --export=ALL,MODEL=genzv2_ck1200
```

Lost jobs 607691–607695 to this. All silently ran with `MODEL=llama_ck1600`.

### 2. Log path must include `/a/` subdirectory

```bash
# ✅ Correct
#SBATCH --output=/home/a/aryanj/logs/%j.out
#SBATCH --error=/home/a/aryanj/logs/%j.err

# ❌ Wrong — silent failure, no logs written
#SBATCH --output=/home/aryanj/logs/%j.out
```

### 3. `--wrap` uses `/bin/sh`, not bash

```bash
# ✅ Correct
sbatch --wrap="bash -c \"source mindmatenv/bin/activate && python train.py\""

# ❌ Wrong — "source: not found" error
sbatch --wrap="source mindmatenv/bin/activate && python train.py"
```

### 4. Shell variable expansion in remote sbatch

```bash
# ✅ Correct — single quotes, $ck expands on cluster
ssh nus-student-cluster 'for ck in 200 400 600; do sbatch --export=ALL,ADAPTER=${ck} script.slurm; done'

# ❌ Wrong — double quotes, $ck expands locally (empty string)
ssh nus-student-cluster "for ck in 200 400 600; do sbatch --export=ALL,ADAPTER=${ck} script.slurm; done"
```

### 5. GitHub SSH is blocked from cluster

```bash
# ❌ This fails — ssh.github.com port 443: Connection timed out
git pull

# ✅ Use rsync from local machine instead (see File Sync section)
```

### 6. `set -euo pipefail` in SLURM scripts

All current SLURM scripts use `set -euo pipefail`. If any command fails, the job stops immediately. Good for catching issues early. Don't remove this.

---

## File Sync (local ↔ cluster)

Since GitHub SSH is blocked, rsync is the only way to push files to the cluster.

```bash
# Sync entire directory
rsync -avz --progress data/ nus-student-cluster:~/projects/mindmate/data/
rsync -avz --progress synthetic/ nus-student-cluster:~/projects/mindmate/synthetic/
rsync -avz --progress finetuning/ nus-student-cluster:~/projects/mindmate/finetuning/
rsync -avz --progress benchmarks/ nus-student-cluster:~/projects/mindmate/benchmarks/
rsync -avz --progress scripts/ nus-student-cluster:~/projects/mindmate/scripts/

# Single file (quiet)
rsync -az -e "ssh -o LogLevel=QUIET" path/to/file.py nus-student-cluster:~/projects/mindmate/path/to/file.py

# Pull results back from cluster
rsync -avz --progress nus-student-cluster:~/projects/mindmate/data/ data/
rsync -avz --progress nus-student-cluster:~/projects/mindmate/exports/ exports/
rsync -avz --progress nus-student-cluster:~/projects/mindmate/benchmarks/results/ benchmarks/results/

# Pull an adapter (large — checkpoint dirs are ~50MB each)
rsync -avz --progress nus-student-cluster:~/projects/mindmate/adapters/genzv5/ adapters/genzv5/
```

**Note:** `adapters/` and `exports/` are in `.gitignore` — large binary files. They live on the cluster and are synced to local as needed.

---

## Cached HuggingFace Models on Cluster

| Model | Size | Used for |
|---|---|---|
| `meta-llama/Llama-3.2-3B-Instruct` | ~6GB | Training base + inference |
| `google/gemma-4-26B-A4B-it` | ~52GB | Teacher (datagen) + benchmark judge |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | ~60GB | **Active datagen teacher** for conv-memory + biometric pipelines |
| `Qwen/Qwen3-1.7B` | ~3GB | Historical |
| `Qwen/Qwen2.5-3B-Instruct` | ~6GB | Historical (inferior base, not used) |

`HF_TOKEN` must be set for gated models (Gemma4, Llama). The `run_conv_memory.slurm` script checks for `HF_TOKEN` at startup and fails early if missing.

---

## Interactive Sessions

```bash
# Get a GPU node for interactive work
srun --partition=gpu-long --gres=gpu:a100-80:1 --mem=48G --cpus-per-task=4 --pty bash

# On the node:
cd ~/projects/mindmate
source mindmatenv/bin/activate

# Interactive chat with a model
python scripts/chat_cluster.py
python scripts/chat_cluster.py --adapter adapters/genz/checkpoint-1200
python scripts/chat_cluster.py --adapter adapters/genz/checkpoint-1200 --label genzv2_ck1200
python scripts/chat_cluster.py --no-memory   # skip memory setup, bare system prompt

# In-chat commands
/memory    # show current memory context (profile + sessions)
/reset     # reset conversation history (keep memory context)
```

`scripts/chat_cluster.py` implements the full production stack: Anchor system prompt, memory injection tiers (Tier 1/2/3), Llama 3 chat format, TextStreamer for real-time output.

---

## Monitoring Running Jobs

```bash
# See all your jobs
squeue -u aryanj

# Watch logs in real time
tail -f ~/logs/<jobid>.out

# Check job history with times
sacct -u aryanj --format=JobID,JobName,State,Start,End,Elapsed -S 2026-05-01

# Which node is a job running on?
squeue -u aryanj -o "%.7i %.8j %.8u %.2t %R %N"
```

---

## Active Jobs (2026-05-14, session 8)

| Job | Name | GPU | Status | Output |
|---|---|---|---|---|
| 611377 | mindmate-conv-memory-qwen | A100-80 (xgph2) | 🟢 RUNNING (~45h in) | `synthetic_train_conv_memory_qwen.jsonl` |
| 611379 | mindmate-conv-memory-qwen | A100-80 (xgph4) | 🟢 RUNNING (shard 0) | `synthetic_train_conv_memory_qwen_s0.jsonl` |
| 611380 | mindmate-conv-memory-qwen | A100-80 (xgph8) | 🟢 RUNNING (shard 1) | `synthetic_train_conv_memory_qwen_s1.jsonl` |
| 611381 | mindmate-conv-memory-qwen | A100-80 (xgph7) | 🟢 RUNNING (shard 2) | `synthetic_train_conv_memory_qwen_s2.jsonl` |
| 612894 | mindmate-biometric-qwen | A100-80 | 🟢 RUNNING (shard 0) | `synthetic_train_biometric_qwen_s0.jsonl` |
| 612895 | mindmate-biometric-qwen | A100-80 | 🟢 RUNNING (shard 1) | `synthetic_train_biometric_qwen_s1.jsonl` |
| 612896 | mindmate-biometric-qwen | A100-80 | 🟢 RUNNING (shard 2) | `synthetic_train_biometric_qwen_s2.jsonl` |
| 612897 | mindmate-conv-memory-qwen | A100-80 | 🟢 RUNNING (shard 3, new profiles) | `synthetic_train_conv_memory_qwen_s3.jsonl` |
| 612898 | mindmate-conv-memory-qwen | A100-80 | 🟢 RUNNING (shard 4, new profiles) | `synthetic_train_conv_memory_qwen_s4.jsonl` |
| 612899 | mindmate-conv-memory-qwen | A100-80 | 🟢 RUNNING (shard 5, new profiles) | `synthetic_train_conv_memory_qwen_s5.jsonl` |

All 72h jobs on `gpu-long`. Conv-memory shards 0–2 finishing ~2026-05-15. Biometric finishing ~2026-05-17. Conv-memory shards 3–5 finishing ~2026-05-17.

### Launching sharded pipelines

```bash
# Conv-memory (already running, for reference):
bash finetuning/launch_conv_memory_shards.sh 3

# Biometric SFT (already running, for reference):
bash finetuning/launch_biometric_shards.sh 3

# To add more shards (indices must not overlap existing):
sbatch --gres=gpu:a100-80:1 --export=ALL,SHARD_IDX=3 finetuning/biometric_pipeline_qwen.slurm
```

### Merge commands (run after jobs finish)

```bash
# Conv-memory: merge all qwen shards
cat data/synthetic_train_conv_memory_qwen_s*.jsonl > data/synthetic_train_conv_memory_qwen_merged.jsonl

# Biometric: merge all qwen shards
cat data/synthetic_train_biometric_qwen_s*.jsonl > data/synthetic_train_biometric_qwen.jsonl
```

---

## See also

- [[Training]] — SLURM scripts for SFT
- [[Benchmarks]] — SLURM commands for benchmark runs
- [[Data]] — rsync commands for data files
