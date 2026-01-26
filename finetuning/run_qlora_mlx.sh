#!/usr/bin/env bash
set -e

# Go to project root
cd /Users/aryanjain/projects/mindmate

# QLoRA fine-tuning with MLX for Llama 3.2 3B
/Users/aryanjain/miniconda3/envs/mindmatenv/bin/mlx_lm.lora \
  --model ./mlx_llama32_3b \
  --train \
  --data ./data_mindmate \
  --batch-size 1 \
  --iters 2000 \
  --save-every 200 \
  --max-seq-length 3072 \
  --num-layers 10 \
  --learning-rate 3e-5 \
  --grad-checkpoint \
  --steps-per-eval 200 \
  --steps-per-report 50 \
  --adapter-path adapters/mindmate_llama32_3b_qlora_nl10_3072_lr3e5 \
  2>&1 | tee run_qlora_nl10_3072_lr3e5.log
