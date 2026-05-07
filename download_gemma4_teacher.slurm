#!/bin/bash
#SBATCH --job-name=download-gemma4-teacher
#SBATCH --output=/home/a/aryanj/logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

# Downloads google/gemma-4-26B-A4B-it (~52GB) to HF cache.
# Run this once before the first datagen job to avoid the 20-30min
# download happening inside a gpu-long slot.
# No GPU needed — this is CPU-only download.

source ~/projects/mindmate/mindmatenv/bin/activate

# Gemma 4 is gated — HF_TOKEN required
if [ -z "${HF_TOKEN:-}" ] && [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "ERROR: HF_TOKEN not set — cannot download gated model" && exit 1
fi

echo "Downloading google/gemma-4-26B-A4B-it (~52GB)..."
HF_HUB_DISABLE_CONCURRENT_DOWNLOADS=1 huggingface-cli download google/gemma-4-26B-A4B-it
echo "Download complete!"
