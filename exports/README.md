# MindMate GGUF Exports

This folder contains MindMate models exported to **GGUF format** for on-device inference (Android, Windows, macOS).

## Available Models

| File | Base | Adapter | Size (Q4_K_M) | Notes |
|---|---|---|---|---|
| `mindmate_llama_sft_ck200/` | Llama 3.2 3B | SFT ck200 | ~2.0 GB | SFT baseline |
| `mindmate_llama_dpo_ck200/` | Llama 3.2 3B | SFT ck200 + DPO | ~2.0 GB | Best quality — DPO trained |
| `mindmate_qwen25_dpo_ck200/` | Qwen2.5-3B | SFT ck200 + DPO | ~2.0 GB | Alternative model |

Each folder contains:
- `*_f16.gguf` — full precision (larger, higher quality)
- `*_q4_k_m.gguf` — 4-bit quantized (recommended for Android)

---

## Android Usage

The MindMate Android app (`android/`) loads a GGUF from the device's Downloads folder.

1. Transfer the `*_q4_k_m.gguf` file to your phone's Downloads folder
2. Install the MindMate APK
3. The app auto-detects the GGUF on launch

**Required RAM:** ~2.5 GB for Q4_K_M on a 3B model

---

## Other Platforms

### macOS / Windows — LM Studio
1. Download [LM Studio](https://lmstudio.ai/)
2. Load the `.gguf` file
3. Paste `system_prompt.txt` as the system prompt

### macOS / Linux — Ollama
```bash
# Create a Modelfile
echo 'FROM ./mindmate_llama_dpo_ck200_q4_k_m.gguf' > Modelfile
echo 'SYSTEM """' >> Modelfile
cat ../system_prompt.txt >> Modelfile
echo '"""' >> Modelfile
ollama create mindmate -f Modelfile
ollama run mindmate
```

---

## Reproducing the Export

Run on the CUDA cluster via SLURM:

```bash
# From ~/projects/mindmate on the cluster
sbatch run_export.slurm llama_sft_ck200
sbatch run_export.slurm llama_dpo_ck200
sbatch run_export.slurm qwen25_dpo_ck200
```

Or directly:
```bash
python scripts/export_gguf_cuda.py --model llama_dpo_ck200
```

Export logs go to `~/logs/export_<jobid>.log`.
