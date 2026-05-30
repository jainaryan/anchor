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

# Export is 100% CPU (device_map="cpu" for merge, llama.cpp binaries are C++).
# NEVER touch the venv's torch — it's GPU torch shared with all datagen/training jobs.
# Install CPU-only torch into /tmp (local disk, per-node) and prepend to PYTHONPATH.
# This avoids: (1) destroying GPU torch for other jobs, (2) libcufile.so.0 missing,
# (3) NFS stale-file-handle failures on pip install to home dir.
CPU_TMP=/tmp/cpu_torch_export_$$
echo "[setup] Installing CPU-only torch to $CPU_TMP (venv untouched)..."
mkdir -p "$CPU_TMP"
python -m pip install --target="$CPU_TMP" -q \
  --index-url https://download.pytorch.org/whl/cpu torch torchvision \
  || { echo "[setup] FATAL: CPU torch install to /tmp failed"; exit 1; }
export PYTHONPATH="$CPU_TMP:${PYTHONPATH:-}"
echo "[setup] torch version:"
python -c "import torch; print('  torch:', torch.__version__); print('  [OK]')" \
  || { echo "[setup] FATAL: torch import failed after /tmp install"; exit 1; }

# Compiled C extensions in the venv can have stale NFS file handles (Errno 116).
# Scrub and reinstall any known-fragile packages before running the export script.
VENV_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
for pkg in regex numpy; do
  if ! python -c "import $pkg" 2>/dev/null; then
    echo "[setup] $pkg import failed — reinstalling..."
    rm -rf "$VENV_SITE/$pkg" "$VENV_SITE"/$pkg-*.dist-info 2>/dev/null || true
    python -m pip install -q "$pkg" || echo "[setup] WARNING: $pkg reinstall failed, continuing"
  fi
done

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
