#!/bin/bash
#SBATCH --job-name=mindmate-export
#SBATCH --output=/home/a/aryanj/logs/%x_%j.out
#SBATCH --error=/home/a/aryanj/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=02:00:00

set -euo pipefail

cd ~/projects/mindmate
unset VIRTUAL_ENV
source mindmatenv/bin/activate
export PYTHONUNBUFFERED=1

# MODEL passed via --export=ALL,MODEL=genzv6_ck1600; default genz
MODEL=${MODEL:-genz}
echo "Exporting model: $MODEL"

echo "[setup] Ensuring pip is in venv..."
if python -c "import pip" 2>/dev/null; then
    echo "[setup] pip already present in venv"
else
    python -m ensurepip --upgrade
fi
# Export is 100% CPU (device_map="cpu" throughout) — install CPU-only torch to
# avoid CUDA driver / libcufile issues on nodes with mixed driver versions.
echo "[setup] Uninstalling existing torch from venv..."
python -m pip uninstall -y torch torchvision 2>&1 | tail -5 || true
VENV_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
rm -rf "$VENV_SITE/torch" "$VENV_SITE"/torch-*.dist-info 2>/dev/null || true
rm -rf "$VENV_SITE/torchvision" "$VENV_SITE"/torchvision-*.dist-info 2>/dev/null || true
echo "[setup] Installing CPU-only PyTorch..."
python -m pip install -q --index-url https://download.pytorch.org/whl/cpu torch torchvision
echo "[setup] torch version:"
python -c "import torch; print('  torch:', torch.__version__); print('  [OK]')" \
  || { echo "[setup] FATAL: torch install failed"; exit 1; }

# Build llama-quantize if not already compiled
if [ ! -f "llama.cpp/build/bin/llama-quantize" ]; then
    echo "Building llama-quantize..."
    cd llama.cpp
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release --target llama-quantize -j$(nproc)
    cd ..
fi

python scripts/export_gguf_cuda.py --model $MODEL

echo "Export complete!"
