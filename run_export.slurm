#!/bin/bash
#SBATCH --job-name=mindmate-export
#SBATCH --output=/home/a/aryanj/logs/export_%j.log
#SBATCH --error=/home/a/aryanj/logs/export_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=02:00:00

set -euo pipefail

cd ~/projects/mindmate

unset VIRTUAL_ENV
source mindmatenv/bin/activate

MODEL=${1:-genz}   # pass model as arg: sbatch run_export.slurm genz

echo "Exporting model: $MODEL"

# Install convert script dependencies
uv pip install gguf sentencepiece transformers torch accelerate peft

# Clone llama.cpp if not present
if [ ! -d "llama.cpp" ]; then
    echo "Cloning llama.cpp..."
    git clone --depth=1 https://github.com/ggerganov/llama.cpp.git
fi

# Install llama.cpp's gguf python package
uv pip install llama.cpp/gguf-py

# Build llama-quantize if not already compiled (use flock to prevent concurrent builds)
if [ ! -f "llama.cpp/build/bin/llama-quantize" ]; then
    echo "Building llama-quantize..."
    flock /tmp/llama_build.lock bash -c "
        if [ ! -f llama.cpp/build/bin/llama-quantize ]; then
            cmake -S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release
            cmake --build llama.cpp/build --config Release --target llama-quantize -j\$(nproc)
        fi
    "
fi

mindmatenv/bin/python scripts/export_gguf_cuda.py --model $MODEL

echo "Export complete!"
