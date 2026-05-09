---
tags: [anchor, next-steps]
---

# Next Steps

← [[Home]]

---

## Immediate — Waiting on Cluster

### 1. Wait for conv-memory pipeline to finish

Jobs 609110–609111 are running A100-80, 72h from 2026-05-09. Expected completion: ~2026-05-12 ~12:00.

```bash
# Check status
ssh nus-student-cluster "squeue -u aryanj"
ssh nus-student-cluster "sacct -u aryanj -j 609110,609111 --format=JobID,State,Start,End,Elapsed"

# Watch logs
ssh nus-student-cluster "tail -f ~/logs/mindmate-conv-memory_609110.out"

# Pull results once done
rsync -avz --progress nus-student-cluster:~/projects/mindmate/data/synthetic_train_conv_memory.jsonl data/
```

Expected output: `data/synthetic_train_conv_memory.jsonl`, ~7k–10k examples.

---

## genzv5 — The Priority Run

genzv5 will be the **first training run with correct system prompt format** (c3acdc9 applied). This is the real test of whether SFT can beat base model on memory categories.

### Step 1: Add v5 preset to `finetuning/build_dataset.py`

```python
# Add to DATA_MIX_PRESETS dict:
"v5": {
    # conv_memory first — this is the primary fix
    "synthetic_train_conv_memory.jsonl":    2000,   # ~20% — exact count depends on file size
    "synthetic_train_targeted_fix.jsonl":   5000,   # ~25%
    "synthetic_train_friend_1.jsonl":       3000,   # ~15%
    "synthetic_train_transition.jsonl":     2500,   # ~12%
    "synthetic_train_casual.jsonl":         2000,   # ~10%
    "synthetic_train_therapist_.jsonl":     1600,   # ~8%
    "synthetic_train_biometric.jsonl":      1600,   # ~8%
    "synthetic_train.jsonl":                500,    # ~2%  (grief)
    "synthetic_train_targeted_fixes.jsonl":  181,   # gold — always 100%
}
# Total: ~18,381 (adjust conv_memory count when actual file size is known)
# Steps: TBD — aim for ~1 epoch. At batch=8: 18k/8 ≈ 2250 steps → use 2400
```

Adjust `conv_memory` count to be ~20% of total once the file is downloaded and size is known.

### Step 2: Create v5 SLURM script

```bash
cp finetuning/run_sft_v4.slurm finetuning/run_sft_v5.slurm
```

Edit `run_sft_v5.slurm`:
- `--model v4` → `--model v5`
- `conversations_raw_v4` → `conversations_raw_v5`
- `conversations_cleaned_v4` → `conversations_cleaned_v5`
- `adapters/genzv4` → `adapters/genzv5`
- `--iters 2400` → decide based on dataset size

### Step 3: Sync to cluster and submit

```bash
# Sync data (conv_memory file) and scripts to cluster
rsync -avz data/synthetic_train_conv_memory.jsonl nus-student-cluster:~/projects/mindmate/data/
rsync -avz finetuning/ nus-student-cluster:~/projects/mindmate/finetuning/

# Submit
ssh nus-student-cluster "sbatch ~/projects/mindmate/finetuning/run_sft_v5.slurm"
```

### Step 4: Benchmark (3-run averaged)

```bash
# Add genzv5_ckXXX to MODEL_SHORTCUTS in benchmarks/run_benchmarks.py first
# Then submit 3 runs per checkpoint
for i in 1 2 3; do
  sbatch --gres=gpu:a100-80:1 --export=ALL,MODEL=genzv5_ck200 benchmarks/run_benchmarks.slurm
done

# Average when all 3 complete
python benchmarks/average_results.py --since <YYYYMMDD>
```

Run a checkpoint sweep: genzv5_ck200, ck400, ck600, ck800, ck1000, ck1200 — find the Goldilocks zone.

### Expected outcomes

- **CROSS_SESSION_MEMORY** should recover from 0% (genzv2_ck1200) toward base's 83%
- **CONTEXT_MEMORY** should recover from 9% toward base's 38%
- **BIOMETRIC** should recover from 30% toward base's 60%
- **COMPANION** should stay high — friend data still in mix
- **CRISIS + HELP_MODE** — unclear; base still beats SFT here, may need more therapist data or longer training

---

## Production Upgrade

Still serving genzv2_ck1600 at tryanchor.me. Should upgrade to genzv2_ck1200 (currently best).

See [[Production]] → "Upgrade Model to genzv2_ck1200" for exact commands.

---

## Open Questions

### 1. Will the format fix alone close the gap?

The base model passes CROSS_SESSION at 83% with zero training on it — instruction-following is intact. genzv5 will have conv_memory data that trains on cross-session context use, but it's unclear if 20% of conv_memory is enough. If memory categories don't recover, consider:
- Increasing conv_memory weight to 30-40%
- Generating more conv_memory data (rerun the pipeline for another 72h)

### 2. Optimal checkpoint for genzv5

v3 peaked at ck200 (10k examples). v4 peaked at ck200 (21k examples). More data did not push the Goldilocks zone later. genzv5 may also peak early. Run the full sweep and don't assume a higher checkpoint is better.

### 3. DPO worth reconsidering?

DPO was tried with corrected-format SFT data (pre-c3acdc9). If genzv5 SFT recovers base categories, DPO might be more useful. Would need: format-corrected DPO data, longer training (1200+ steps), possibly higher beta. Not a priority until genzv5 results are in.

### 4. Per-category targeted data

Currently the model learns crisis/help/biometric from style examples. Could generate:
- More multi-turn conv_memory examples with `[Recent sessions]` explicitly referenced
- More biometric SFT examples that combine health context with emotional support
- HELP_MODE examples where the user explicitly says "help me calm down" in turn 2+ (not just turn 1)

### 5. Server upgrade + performance

DigitalOcean c-4 does CPU inference only. If tryanchor.me traffic grows, consider upgrading to a GPU droplet or switching to HuggingFace Spaces for inference.

---

## See also

- [[Training]] — how to run SFT
- [[Benchmarks]] — how to run the benchmark suite
- [[Data]] — data files and conv_memory pipeline status
- [[Cluster]] — SLURM and sync commands
