# MindMate

MindMate is a finetuned local mental-health wellness assistant. Multiple model variants are trained via QLoRA + DPO and exported to GGUF for on-device Android inference.

## Models

| Model | Size | SFT Adapter | DPO Adapter | Notes |
|---|---|---|---|---|
| Llama 3.2 3B Instruct | 3B | `adapters/CUDA_mindmate_llama32b/checkpoint-200` | `adapters/CUDA_mindmate_llama32b_dpo_ck200/` | Best overall — good crisis pivot |
| Qwen2.5-3B Instruct | 3B | `adapters/CUDA_mindmate_qwen25_3b/checkpoint-200` | `adapters/CUDA_mindmate_qwen25_3b_dpo_ck200/` | No thinking mode; cleaner instruct |
| Qwen3-1.7B | 1.7B | `adapters/CUDA_mindmate_qwen3_1p7b/` | — | Smallest; 1.7B capacity limits quality |

## Features
- **Fine-tuning**: QLoRA (4-bit NF4) on CUDA via HuggingFace PEFT + TRL
- **DPO**: Preference training on top of SFT adapters using 30B teacher-generated pairs (2,500 pairs)
- **System prompt**: Simplified 13-line natural-language prompt (temperature 0.75)
- **Platform Support**:
  - **CUDA (cluster)**: Full training + inference pipeline
  - **Android**: On-device GGUF inference via llama.cpp JNI bridge
  - **macOS** (Apple Silicon): Legacy MLX inference path

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- macOS with Apple Silicon (for training/inference using MLX)
- (Optional) `llama.cpp` tools for export.

### Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/jainaryan/mindmate.git
cd mindmate

# Create a conda env (recommended)
conda create -n mindmate python=3.11
conda activate mindmate

# Install dependencies
pip install mlx-lm huggingface-hub gguf protobuf python-dotenv
```

---

## Training Pipeline

```
Synthetic Data Generation (Qwen3-30B teacher)
        ↓
  SFT via QLoRA (CUDA_train_qlora.py)
        ↓
  SFT Adapter Checkpoints
        ↓
  DPO Pair Generation (dpo_pipeline.py)
        ↓
  DPO Training (CUDA_train_dpo.py)
        ↓
  GGUF Export (Q4_K_M) → Android
```

### CUDA pipeline (active)
```bash
# SFT — Llama
python finetuning/CUDA_run_pipeline.py

# SFT — Qwen2.5-3B
python finetuning/CUDA_run_pipeline_qwen25_3b.py

# DPO (after SFT)
python finetuning/CUDA_train_dpo.py --model llama_ck200 --steps 800
python finetuning/CUDA_train_dpo.py --model qwen25_3b --steps 800
```

### SLURM jobs
| Script | GPU | Time | Purpose |
|---|---|---|---|
| `run_finetune.slurm` | A100-40, gpu-long | 48h | Llama SFT |
| `run_finetune_qwen25_3b.slurm` | A100-40, gpu-long | 48h | Qwen2.5-3B SFT |
| `run_dpo_llama_ck200.slurm` | A100-40, gpu-long | 24h | Llama DPO |
| `run_dpo_qwen25_3b.slurm` | A100-40, gpu-long | 24h | Qwen2.5-3B DPO |
| `run_export.slurm` | H100-96, gpu | 2h | GGUF export |

---

## Inference (Chat)

### CUDA cluster
```bash
# Llama (defaults to checkpoint-200)
python inference/CUDA_chat_mindmate.py
python inference/CUDA_chat_mindmate.py --checkpoint checkpoint-400

# Qwen2.5-3B (defaults to checkpoint-200)
python inference/CUDA_chat_qwen25_3b.py

# Qwen3-1.7B
python inference/CUDA_chat_qwen.py
```

### macOS (Legacy MLX)
```bash
python inference/chat_mindmate.py
```

---

## Export (Android)

To use the model on Android, export to GGUF format:

```bash
# SFT baseline
python scripts/export_gguf_cuda.py --model llama_sft_ck200

# DPO models
python scripts/export_gguf_cuda.py --model llama_dpo_ck200
python scripts/export_gguf_cuda.py --model qwen25_dpo_ck200
```

Or via SLURM:
```bash
sbatch run_export.slurm llama_sft_ck200
sbatch run_export.slurm llama_dpo_ck200
sbatch run_export.slurm qwen25_dpo_ck200
```

Outputs land in `exports/<model_name>/` as `*_f16.gguf` and `*_q4_k_m.gguf`.

See `exports/README.md` for Android deployment instructions.

---

## Project Structure

```
mindmate/
├── adapters/             # Trained LoRA adapters (SFT + DPO)
├── data/                 # Synthetic training data + DPO pairs
├── exports/              # Exported GGUF models
├── finetuning/           # QLoRA + DPO training scripts
├── inference/            # CUDA + MLX chat scripts
├── scripts/              # Export, upload, utility scripts
├── synthetic/            # Data generation pipelines
├── web/                  # Flask web app with memory engine
├── android/              # Android app + llama.cpp JNI bridge
└── system_prompt.txt     # Canonical system prompt (13 lines)
```
