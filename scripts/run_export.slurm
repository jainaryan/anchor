#!/bin/bash
#SBATCH --job-name=mindmate-export
#SBATCH --output=logs/export_%j.log
#SBATCH --error=logs/export_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=02:00:00

set -euo pipefail

unset VIRTUAL_ENV
source mindmatenv/bin/activate

MODEL=${1:-genz}   # pass model as arg: sbatch run_export.slurm genz

echo "Exporting model: $MODEL"

# Build llama-quantize if not already compiled
if [ ! -f "llama.cpp/build/bin/llama-quantize" ]; then
    echo "Building llama-quantize..."
    cd llama.cpp
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release --target llama-quantize -j$(nproc)
    cd ..
fi

mindmatenv/bin/python scripts/export_gguf_cuda.py --model $MODEL

echo "Export complete!"
