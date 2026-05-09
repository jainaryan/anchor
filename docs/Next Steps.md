---
tags: [anchor, next-steps]
---

# Next Steps

← [[Home]]

---

## Immediate (waiting on cluster)

### 1. Wait for conv-memory jobs to finish

Jobs 609110–609111 are running on A100-80, 72h from 2026-05-09.

```bash
# Check status
ssh nus-student-cluster "squeue -u aryanj"

# Pull results when done
rsync -avz --progress nus-student-cluster:~/projects/mindmate/data/synthetic_train_conv_memory.jsonl data/
```

Expected output: **~7k–10k examples** in `data/synthetic_train_conv_memory.jsonl`.

---

## genzv5 — First Correct-Format Run

This is the primary next milestone. genzv5 will be the first training run where:
1. All training data uses the production system prompt format (fixed c3acdc9)
2. Multi-turn memory examples are included (`synthetic_train_conv_memory.jsonl`)

### Planned data mix

| Source | Target % | Rationale |
|---|---|---|
| `synthetic_train_conv_memory.jsonl` | ~20% | Fix for base-beats-SFT on memory categories |
| `synthetic_train_targeted_fix.jsonl` | ~25% | Help mode + memory recall |
| `synthetic_train_friend_1.jsonl` | ~15% | Companion behavior |
| `synthetic_train_transition.jsonl` | ~12% | Casual→emotional pivot |
| `synthetic_train_casual.jsonl` | ~10% | Non-distress casual |
| `synthetic_train_therapist_.jsonl` | ~8% | Crisis/therapeutic |
| `synthetic_train_biometric.jsonl` | ~8% | Biometric context |
| `synthetic_train.jsonl` (grief) | ~2% | Grief/loss coverage |
| `synthetic_train_targeted_fixes.jsonl` | 181 gold | Always 100% |

### Steps

```bash
# 1. Pull conv_memory data from cluster
rsync -avz --progress nus-student-cluster:~/projects/mindmate/data/synthetic_train_conv_memory.jsonl data/

# 2. Add v5 mix preset to finetuning/build_dataset.py (DATA_MIX_PRESETS)

# 3. Build + train
python finetuning/build_dataset.py --version v5
python finetuning/clean_dataset.py
sbatch --gres=gpu:a100-80:1 finetuning/run_sft_v4.slurm  # edit for v5 steps

# 4. Benchmark (3-run averaged)
for i in 1 2 3; do
  sbatch --gres=gpu:a100-80:1 --export=ALL,MODEL=genzv5_ck<N> benchmarks/run_benchmarks.slurm
done
python benchmarks/average_results.py --since <YYYYMMDD>
```

### Expected outcomes

- CROSS_SESSION_MEMORY should recover from 0% (genzv2_ck1200) toward base's 83%
- CONTEXT_MEMORY should recover from 9% toward base's 38%
- BIOMETRIC should recover from 30% toward base's 60%
- COMPANION should stay high (SFT advantage) — friend data still in mix
- genzv5 is the real test of the format fix

---

## Production Upgrade

- Upgrade tryanchor.me from `ck1600` to `mindmate_genzv2_ck1200_q4_k_m.gguf`
- See [[Production]] for rsync + restart commands

---

## Open Questions

1. **Will format fix alone close the gap?** Base still passes CROSS_SESSION 83% with zero training on it. genzv5 may need even more conv_memory data.

2. **Optimal conv_memory weight?** ~20% is a guess. If COMPANION or FORMAT degresses, reduce. If memory categories don't recover, increase.

3. **Steps for genzv5?** With ~conv_memory mixed in, total dataset ~50k examples. 1600 steps ≈ 0.49 epochs. Probably need 2400–3200 steps and watch checkpoint sweep.

4. **DPO reconsidered?** DPO was tried with beta=0.1, 800 steps, current data. Could retry with corrected-format data + longer training + higher beta. Not a priority until SFT v5 results are in.

---

## See also

- [[Training]] — how to run SFT
- [[Benchmarks]] — how to run the benchmark suite
- [[Data]] — data files and conv_memory pipeline status
