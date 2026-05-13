---
tags: [anchor, next-steps]
---

# Next Steps

← [[Home]]

---

## Immediate — Waiting on Cluster

### 1. Wait for conv-memory + biometric pipelines to finish

| Jobs | Expected finish | Output |
|---|---|---|
| 611377, 611379–611381 | ~2026-05-15 | `synthetic_train_conv_memory_qwen*.jsonl` (4 files, ~1,400 examples total) |
| 612894–612896 | ~2026-05-17 | `synthetic_train_biometric_qwen_s{0,1,2}.jsonl` |

```bash
# Check status
ssh nus-student-cluster "squeue -u aryanj"

# Pull conv-memory results once done (all shards + pre-shard run)
rsync -az -e "ssh -o LogLevel=QUIET" \
  "nus-student-cluster:projects/mindmate/data/synthetic_train_conv_memory_qwen.jsonl" \
  "nus-student-cluster:projects/mindmate/data/synthetic_train_conv_memory_qwen_s0.jsonl" \
  "nus-student-cluster:projects/mindmate/data/synthetic_train_conv_memory_qwen_s1.jsonl" \
  "nus-student-cluster:projects/mindmate/data/synthetic_train_conv_memory_qwen_s2.jsonl" \
  data/

# Merge shards locally (or on cluster)
cat data/synthetic_train_conv_memory_qwen_s*.jsonl > data/synthetic_train_conv_memory_qwen_merged.jsonl

# Pull biometric results once done
rsync -az -e "ssh -o LogLevel=QUIET" \
  "nus-student-cluster:projects/mindmate/data/synthetic_train_biometric_qwen_s0.jsonl" \
  "nus-student-cluster:projects/mindmate/data/synthetic_train_biometric_qwen_s1.jsonl" \
  "nus-student-cluster:projects/mindmate/data/synthetic_train_biometric_qwen_s2.jsonl" \
  data/
cat data/synthetic_train_biometric_qwen_s*.jsonl > data/synthetic_train_biometric_qwen.jsonl

# Also pull overref files for inspection (potential DPO negatives)
rsync -az -e "ssh -o LogLevel=QUIET" \
  "nus-student-cluster:projects/mindmate/data/synthetic_train_conv_memory_overref_qwen_s0.jsonl" \
  "nus-student-cluster:projects/mindmate/data/synthetic_train_conv_memory_overref_qwen_s1.jsonl" \
  "nus-student-cluster:projects/mindmate/data/synthetic_train_conv_memory_overref_qwen_s2.jsonl" \
  data/
```

---

## genzv5 — The Priority Run

genzv5 will be the **first training run with correct system prompt format** (c3acdc9 applied). This is the real test of whether SFT can beat base model on memory categories.

### Step 1: Add v5 preset to `finetuning/build_dataset.py`

Wait for both pipelines to finish + merge, then set counts based on actual file sizes. Rough plan:

```python
# Add to DATA_MIX_PRESETS dict:
"v5": {
    # New Qwen-generated data — primary fix targets
    "synthetic_train_conv_memory_qwen_merged.jsonl": 2000,  # ~15% — cross-session memory
    "synthetic_train_biometric_qwen.jsonl":          2000,  # ~15% — biometric context handling

    # Existing sources
    "synthetic_train_targeted_fix.jsonl":   4000,   # ~30%
    "synthetic_train_friend_1.jsonl":       2000,   # ~15%
    "synthetic_train_transition.jsonl":     1500,   # ~11%
    "synthetic_train_casual.jsonl":         1000,   # ~7%
    "synthetic_train_therapist_.jsonl":      800,   # ~6%
    "synthetic_train.jsonl":                 300,   # ~2%  (grief)
    "synthetic_train_targeted_fixes.jsonl":  181,   # gold — always 100%
}
# Total: ~13,781 (adjust conv_memory + biometric counts once file sizes known)
# Steps: TBD — aim for ~1 epoch. At batch=8: 14k/8 ≈ 1750 steps → use 2000
```

Key decisions before finalising:
- **conv_memory weight**: target ~15% of mix. If merged file has >2k examples, cap at 2k. If <2k, use 100%.
- **biometric_qwen weight**: same — use 100% if <2k final examples, cap at 2k if more.
- **Drop old `synthetic_train_biometric.jsonl`** from v5 — replaced by `biometric_qwen.jsonl` (Qwen teacher, correct format, richer mode coverage). The old file used Gemma4 with an older pipeline version.

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

---

## anchor-app Code Quality (session 5 — partially done)

### Done (2026-05-10)

- ✅ **Panic detection tiered** — `'urgent'` blocks, `'watch'` notifies only
- ✅ **Session-extraction race condition** — snapshot at schedule time
- ✅ **Person-name casing bug** — intro regex now matches regardless of case
- ✅ **Memory retrieval** — summary text included in search, recency bonus
- ✅ **Prompt token budgeting** — per-tier char caps in contextBuilder
- ✅ **Eval prompt drift** — EvalRunner defaults to production prompt

### Still to do

- ⬜ **Unify profile storage** — `profileStorage.ts` (AsyncStorage) and `MemoryRepository.ts` (WatermelonDB) are separate stores. Profile UI writes to AsyncStorage; inference memory reads from WatermelonDB. These can diverge. Migrate to one canonical store.
- ⬜ **Unit tests for critical memory path** — `contextBuilder`, `sessionExtractor`, `panicDetection`, `profileExtractor` have no focused tests. Add unit tests + regression fixtures. These directly affect user safety.
- ⬜ **Privacy hardening** — Profile and diary stored in plaintext local storage. Consider encrypted at-rest option or keychain-backed "privacy lock" mode.

### UX revamps to implement

- ⬜ **Calmer chat home** — move secondary controls (mode pills, thinking toggle, profile suggestion actions) behind a collapsed `Session tools` drawer. First-use/default view should show only mood check-in, input, and safe empty-state.
- ⬜ **Passive diary auto-capture** — replace persistent "Save to diary" button flow with session-end auto-generated reflection draft + lightweight `Review & Save` entry point into `DiaryEditor`.
- ⬜ **Progressive profile onboarding** — split `ProfileSetupScreen` into `2-minute essentials` and `optional deep profile` so first-run completion is faster and lower friction.

---

## See also

- [[Training]] — how to run SFT
- [[Benchmarks]] — how to run the benchmark suite
- [[Data]] — data files and conv_memory pipeline status
- [[Cluster]] — SLURM and sync commands
