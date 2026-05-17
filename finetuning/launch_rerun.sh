#!/bin/bash
# Relaunch conv-memory new-pool shards (3-8) and 3 more biometric shards (3-5).
#
# Conv-memory new-pool (shards 3-8):
#   - Fixes the OOM: conv_memory_pipeline_qwen.slurm now has --mem=64G
#   - Uses PROFILE_SET=new → draws from NEW_CLINICAL_PROFILES + NEW_COMPANION_PROFILES
#   - Appends to existing _s3.jsonl (48K partial), creates _s4.._s8 fresh
#
# Biometric (shards 3-5):
#   - s0 produced 993 examples in 72h; s1/s2 only 80 each (queued late, less wall time)
#   - 3 more shards → ~3 × 72h ≈ ~3000 more examples at similar throughput
#   - Uses existing biometric_pipeline_qwen.slurm (--mem=64G, gpu:a100-40:2, works)

set -euo pipefail

echo "=== Conv-memory new-pool shards 3-8 ==="
for i in 3 4 5 6 7 8; do
  JOB=$(sbatch \
    --export=ALL,SHARD_IDX=$i,PROFILE_SET=new \
    finetuning/conv_memory_pipeline_qwen.slurm | awk '{print $4}')
  echo "  conv-memory shard $i → job $JOB"
done

echo ""
echo "=== Biometric shards 3-5 ==="
for i in 3 4 5; do
  JOB=$(sbatch \
    --export=ALL,SHARD_IDX=$i \
    finetuning/biometric_pipeline_qwen.slurm | awk '{print $4}')
  echo "  biometric shard $i → job $JOB"
done

echo ""
echo "9 jobs submitted. Monitor with: squeue -u aryanj"
echo ""
echo "Output files (after jobs finish):"
echo "  Conv-memory: data/synthetic_train_conv_memory_qwen_s{3..8}.jsonl"
echo "  Biometric:   data/synthetic_train_biometric_qwen_s{3,4,5}.jsonl"
