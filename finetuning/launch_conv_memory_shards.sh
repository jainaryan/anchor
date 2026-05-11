#!/bin/bash
# Launch N parallel shards of the conv-memory pipeline.
# Each shard writes to its own file (data/synthetic_train_conv_memory_qwen_s{N}.jsonl)
# and uses a different RNG seed → no overlap, safe to merge after.
#
# Usage:
#   ./finetuning/launch_conv_memory_shards.sh                # default: 3 shards
#   ./finetuning/launch_conv_memory_shards.sh 4              # 4 shards

set -e

NUM_SHARDS="${1:-3}"

echo "Launching $NUM_SHARDS parallel shards of conv-memory pipeline (Qwen3-30B)..."

for ((i=0; i<NUM_SHARDS; i++)); do
  JOB=$(sbatch --gres=gpu:a100-80:1 \
    --export=ALL,SHARD_IDX=$i \
    finetuning/conv_memory_pipeline_qwen.slurm | awk '{print $4}')
  echo "  shard $i → job $JOB"
done

echo ""
echo "Done. Monitor with: squeue -u \$USER"
echo "Output files will be: data/synthetic_train_conv_memory_qwen_s{0..$((NUM_SHARDS-1))}.jsonl"
echo "After they finish, merge with:"
echo "  cat data/synthetic_train_conv_memory_qwen_s*.jsonl > data/synthetic_train_conv_memory_qwen.jsonl"
