#!/bin/bash
#SBATCH --job-name=mindmate-export-top3
#SBATCH --output=/home/a/aryanj/logs/export_top3_%j.out
#SBATCH --error=/home/a/aryanj/logs/export_top3_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=04:00:00

set -euo pipefail

cd ~/projects/mindmate
source mindmatenv/bin/activate

echo "=========================================="
echo "  MindMate GGUF Export — Top 3 SFT"
echo "  Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "  Started: $(date)"
echo "=========================================="
echo ""
echo "  genzv3_ck200  — 26/34 (76%)"
echo "  genzv2_ck1200 — 24/34 (71%)"
echo "  genzv4_ck200  — 22/34 (65%)"
echo ""

# Build llama-quantize if needed
if [ ! -f "llama.cpp/build/bin/llama-quantize" ]; then
    echo "[Build] Compiling llama-quantize..."
    cd llama.cpp
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release --target llama-quantize -j$(nproc)
    cd ..
    echo "[Build] Done."
fi

echo "[1/3] Exporting genzv3_ck200..."
python -u scripts/export_gguf_cuda.py --model genzv3_ck200
echo ""

echo "[2/3] Exporting genzv2_ck1200..."
python -u scripts/export_gguf_cuda.py --model genzv2_ck1200
echo ""

echo "[3/3] Exporting genzv4_ck200..."
python -u scripts/export_gguf_cuda.py --model genzv4_ck200
echo ""

echo "=========================================="
echo "  All exports complete: $(date)"
echo "  Outputs:"
echo "    exports/mindmate_genzv3_ck200/mindmate_genzv3_ck200_q4_k_m.gguf"
echo "    exports/mindmate_genzv2_ck1200/mindmate_genzv2_ck1200_q4_k_m.gguf"
echo "    exports/mindmate_genzv4_ck200/mindmate_genzv4_ck200_q4_k_m.gguf"
echo "=========================================="
