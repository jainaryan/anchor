---
tags: [anchor, cluster]
---

# Cluster

← [[Home]]

---

## Access

```bash
# Requires NUS VPN to be active first
ssh nus-student-cluster   # → xlogin.comp.nus.edu.sg, user: aryanj

# Environment
cd ~/projects/mindmate
source mindmatenv/bin/activate
```

---

## SLURM Quick Reference

```bash
# Check your jobs
squeue -u aryanj

# Job details (start, end, elapsed, state)
sacct -u aryanj -j <jobid> --format=JobID,State,Start,End,Elapsed

# GPU availability
sinfo -o "%P %G %C %N"

# Submit a job
sbatch --gres=gpu:a100-80:1 --export=ALL,MODEL=genzv2_ck1200 benchmarks/run_benchmarks.slurm

# Interactive session (for chat_cluster.py)
srun --partition=gpu-long --gres=gpu:a100-80:1 --pty bash
```

---

## Partitions

| Partition | Max time | GPU options | Use for |
|---|---|---|---|
| `gpu` | 3h | H200-141, A100-80, A100-40 | Quick benchmark runs |
| `gpu-long` | 3 days | A100-80, A100-40 | Training, long datagen |

**Always use `gpu-long` for training and the conv-memory pipeline.**

---

## GPU Notes

| GPU | VRAM | Notes |
|---|---|---|
| A100-80 | 80GB | **Best for all jobs.** Use `--gres=gpu:a100-80:1` explicitly. |
| A100-40 | 40GB | OOMs for Gemma4 bfloat16 judge (~52GB). OK for SFT only. |
| H100-96 GRES | varies | ⚠️ Can fall back to ~46GB nodes — OOMs Gemma4. Unreliable. |
| H200-141 | 141GB | `gpu` partition only (3h max) — not usable for training |

**Always specify `--gres=gpu:a100-80:1` explicitly.** H100-96 GRES silently falls back to smaller nodes.

---

## Critical SLURM Gotchas

1. **`--export` BEFORE script path** — if placed after, sbatch treats it as a script argument and silently ignores it
   ```bash
   # ✅ Correct
   sbatch --gres=gpu:a100-80:1 --export=ALL,MODEL=genzv2_ck1200 benchmarks/run_benchmarks.slurm
   
   # ❌ Wrong — MODEL env var silently ignored
   sbatch --gres=gpu:a100-80:1 benchmarks/run_benchmarks.slurm --export=ALL,MODEL=genzv2_ck1200
   ```

2. **Log path** — use `~/logs/` (home dir), NOT `~/projects/mindmate/logs/`
   ```bash
   # ✅ Correct
   #SBATCH --output=/home/a/aryanj/logs/%j.out
   
   # ❌ Wrong — no logs generated, job silently fails
   #SBATCH --output=/home/aryanj/logs/%j.out
   ```

3. **`--wrap` uses `/bin/sh`** — `source` command not available; use `bash -c`:
   ```bash
   # ✅ Correct
   sbatch --wrap="bash -c \"source mindmatenv/bin/activate && python ...\""
   
   # ❌ Wrong — "source: not found"
   sbatch --wrap="source mindmatenv/bin/activate && python ..."
   ```

4. **Shell variable expansion in remote sbatch** — use single quotes:
   ```bash
   # ✅ Correct — $ck expands on cluster
   ssh nus-student-cluster 'for ck in 200 400 600; do sbatch --export=ALL,ADAPTER=${ck} script.slurm; done'
   
   # ❌ Wrong — $ck expands locally (empty string)
   ssh nus-student-cluster "for ck in 200 400 600; do sbatch --export=ALL,ADAPTER=${ck} script.slurm; done"
   ```

---

## File Sync

GitHub SSH is blocked from the cluster — use rsync:

```bash
# Local → cluster
rsync -avz --progress data/ nus-student-cluster:~/projects/mindmate/data/
rsync -avz --progress synthetic/ nus-student-cluster:~/projects/mindmate/synthetic/
rsync -avz --progress finetuning/ nus-student-cluster:~/projects/mindmate/finetuning/
rsync -avz --progress benchmarks/ nus-student-cluster:~/projects/mindmate/benchmarks/

# Single file
rsync -az -e "ssh -o LogLevel=QUIET" path/to/file.py nus-student-cluster:~/projects/mindmate/path/to/file.py

# Cluster → local (pull results)
rsync -avz --progress nus-student-cluster:~/projects/mindmate/data/ data/
rsync -avz --progress nus-student-cluster:~/projects/mindmate/adapters/genzv2/ adapters/genzv2/
rsync -avz --progress nus-student-cluster:~/projects/mindmate/exports/ exports/
rsync -avz --progress nus-student-cluster:~/projects/mindmate/benchmarks/results/ benchmarks/results/
```

---

## Cached HuggingFace Models on Cluster

| Model | Size | Used for |
|---|---|---|
| `meta-llama/Llama-3.2-3B-Instruct` | ~6GB | Training base + inference |
| `google/gemma-4-26B-A4B-it` | ~52GB | Teacher (datagen) + benchmark judge |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | ~60GB | Old benchmark judge (historical) |
| `Qwen/Qwen3-1.7B` | ~3GB | Historical |
| `Qwen/Qwen2.5-3B-Instruct` | ~6GB | Historical (inferior to Llama) |

`HF_TOKEN` must be set for gated models (Gemma4, Llama). Set once; persists in `~/.cache/huggingface/`.

---

## Interactive Inference

```bash
# Get a GPU node
srun --partition=gpu-long --gres=gpu:a100-80:1 --pty bash

# On the node
cd ~/projects/mindmate
source mindmatenv/bin/activate
python scripts/chat_cluster.py --adapter adapters/genz/checkpoint-1200

# In-chat commands
/memory   # show current memory context
/reset    # reset conversation history
```

---

## Active Jobs (2026-05-09)

| Job | Name | ETA | Notes |
|---|---|---|---|
| 609110, 609111 | conv-memory v2 (teacher-as-Anchor) | ~64h remaining | → `data/synthetic_train_conv_memory.jsonl` |

---

## See also

- [[Training]] — SLURM scripts for SFT and datagen
- [[Benchmarks]] — SLURM commands for benchmark runs
