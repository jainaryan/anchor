# MindMate

MindMate is a fine-tuned **Llama 3.2 3B** model optimized for **mental health support** interactions. It is trained using **QLoRA** on a custom dataset to provide empathetic, detailed, and supportive responses.

## 🚀 Features
- **Base Model**: Llama 3.2 3B Instruct
- **Fine-tuning**: QLoRA (4-bit quantization) via [MLX](https://github.com/ml-explore/mlx)
- **Platform Support**:
  - **macOS** (Apple Silicon): Native inference via MLX.
  - **Windows**: Via LM Studio or Ollama (GGUF).
  - **Android**: Via Layla or UserLAnd (GGUF).

---

## 🛠️ Setup & Installation

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

### Download Base Model
You need the base Llama 3.2 model weights for training or export:
```bash
huggingface-cli download mlx-community/Llama-3.2-3B-Instruct-4bit --local-dir mlx_llama32_3b
```

---

## 🧠 Training Pipeline

The project includes an end-to-end pipeline in `finetuning/run_pipeline.py` that handles everything from data processing to QLoRA training.

### Pipeline Steps
1.  **Build Dataset**: Compiles raw data from sources.
2.  **Clean Dataset**: Dedupes and formats data for chat (User/Assistant).
3.  **Chunk Data**: Splits conversations into 3072-token windows.
4.  **Train (QLoRA)**: Fine-tunes the model using MLX.

### Running the Pipeline
To run the full training process:
```bash
python finetuning/run_pipeline.py
```
*   **Input**: Raw data in `data/new_raw_data`.
*   **Output**: Adapters saved in `adapters/mindmate_llama32_3b_qlora_...`

---

## 💬 Inference (Chat)

### macOS (Native MLX)
To chat with the model on your Mac:
```bash
python scripts/chat_mindmate.py
```
This loads the base model + your trained adapters and launches an interactive CLI chat.

---

## 📦 Export (Windows / Android)

To usage the model on other platforms, we export it to **GGUF format**.

### 1. Run the Export Script
This script validates the environment, fuses the adapters, and creates the GGUF file.
```bash
python scripts/export_to_gguf.py
```
*   **Output**: `exports/mindmate_llama32_3b_f16.gguf`

### 2. Run on Windows
1.  Download **[LM Studio](https://lmstudio.ai/)**.
2.  Load the generated `.gguf` file.
3.  Paste the contents of `system_prompt.txt` into the system prompt configuration.

### 3. Run on Android
*   **Layla (App)**: Transfer the `.gguf` to your phone and select "Load Local Model".
*   **UserLAnd**: Install Ubuntu -> `llama.cpp` and run from command line.

See `exports/README.md` for more detailed export instructions.

---

## 📂 Project Structure

mindmate/
├── adapters/             # Trained LoRA adapters
├── data/                 # Raw/Cleaned training data
├── exports/              # Exported GGUF models & instructions
├── finetuning/           # Fine-tuning pipeline & scripts
│   ├── run_pipeline.py   # Training pipeline entry point
│   ├── build_dataset.py
│   ├── clean_dataset.py
│   └── chunk.py
├── mlx_export/           # Intermediate fused models
├── mlx_llama32_3b/       # Base Llama 3.2 model (downloaded)
├── scripts/              # Misc utility scripts
│   ├── chat_mindmate.py  # Inference script
│   ├── export_to_gguf.py # Export tool
│   └── ...
└── system_prompt.txt     # Core system instructions for the bot

## ☁️ Uploading to Hugging Face
To publish your fused model and ExecuTorch artifacts (if generated):
```bash
python upload_to_hf.py
```
*Requires `HF_TOKEN` in your environment variables.*
