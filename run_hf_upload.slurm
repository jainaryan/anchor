#!/bin/bash
#SBATCH --job-name=mindmate-hf-upload
#SBATCH --output=/home/a/aryanj/logs/hf_upload_%j.log
#SBATCH --error=/home/a/aryanj/logs/hf_upload_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --partition=long
#SBATCH --time=02:00:00

set -euo pipefail
cd ~/projects/mindmate
unset VIRTUAL_ENV
source mindmatenv/bin/activate

MODEL=${1:-gemma4_e2b}
echo "Uploading model: $MODEL"
mindmatenv/bin/python scripts/upload_gguf_to_hf.py --model $MODEL
echo "Upload complete!"
