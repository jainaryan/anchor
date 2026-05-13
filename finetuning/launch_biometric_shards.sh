#!/bin/bash
# Launch N parallel shards of the biometric SFT pipeline.
# Each shard writes to its own file:
#   data/synthetic_train_biometric_qwen_s{N}.jsonl   ← training data
#   synthetic/outputs/biometric_sft_qwen_raw_s{N}.jsonl  ← raw with meta fields
# Different RNG seeds per shard → no overlap, safe to merge after.
#
# Usage:
#   ./finetuning/launch_biometric_shards.sh          # default: 3 shards
#   ./finetuning/launch_biometric_shards.sh 4        # 4 shards

set -e

NUM_SHARDS="${1:-3}"

echo "Launching $NUM_SHARDS parallel shards of biometric SFT pipeline (Qwen3-30B)..."

for ((i=0; i<NUM_SHARDS; i++)); do
  JOB=$(sbatch --gres=gpu:a100-80:1 \
    --export=ALL,SHARD_IDX=$i \
    finetuning/biometric_pipeline_qwen.slurm | awk '{print $4}')
  echo "  shard $i → job $JOB"
done

echo ""
echo "Done. Monitor with: squeue -u \$USER"
echo "Output files will be:"
echo "  data/synthetic_train_biometric_qwen_s{0..$((NUM_SHARDS-1))}.jsonl  (training)"
echo "  synthetic/outputs/biometric_sft_qwen_raw_s{0..$((NUM_SHARDS-1))}.jsonl  (raw+meta)"
echo ""
echo "After jobs finish, merge training files with:"
echo "  cat data/synthetic_train_biometric_qwen_s*.jsonl > data/synthetic_train_biometric_qwen.jsonl"
