# MindMate GGUF Exports

GGUF models for on-device inference (Android, macOS, Windows).

## Available Models

| Folder | Base | Adapter | Quant | Notes |
|---|---|---|---|---|
| `mindmate_llama_sft_ck1600/` | Llama 3.2 3B | genzv2 SFT ck1600 | Q4_K_M | **PRODUCTION — use this** |
| `mindmate_llama_sft_ck200/` | Llama 3.2 3B | SFT ck200 | Q4_K_M | Older baseline |
| `mindmate_llama_dpo_ck200/` | Llama 3.2 3B | SFT ck200 + DPO | Q4_K_M | Inferior to genzv2 SFT |
| `mindmate_qwen25_dpo_ck200/` | Qwen2.5-3B | SFT ck200 + DPO | Q4_K_M | Inferior to Llama |

**Current production model:** `mindmate_llama_sft_ck1600/llama(genz)v2_q4_k_m.gguf`
- On Pixel 8a: ~5.5 TPS, ~2.0 GB on disk, ~3.12 GB heap at runtime
- Running at https://tryanchor.me

**Pending:** `mindmate_genz_dpo_ck1600/` — DPO on top of genzv2, job 595679 running

---

## Android Usage

The anchor-app loads GGUF from device storage.

1. Transfer `llama(genz)v2_q4_k_m.gguf` to `/sdcard/Download/mindmate.gguf`
2. In the app: Add Model → pick from storage → app copies to internal storage with progress overlay
3. Load model → chat

**Required:** ~2.5 GB free RAM for Q4_K_M on Llama 3.2 3B

---

## Other Platforms

### macOS / Windows — LM Studio
1. Download [LM Studio](https://lmstudio.ai/)
2. Load the `.gguf` file
3. System prompt: use the Anchor `BASE_PROMPT` from `anchor-app/src/utils/anchorSystemPrompt.ts`

### macOS / Linux — Ollama
```bash
echo 'FROM ./llama_genzv2_q4_k_m.gguf' > Modelfile
ollama create mindmate -f Modelfile
ollama run mindmate
```

---

## Reproducing the Export

```bash
# On the cluster
cd ~/projects/mindmate
python scripts/export_gguf_cuda.py --model llama_ck1600

# Or via SLURM
sbatch run_export.slurm
```

Export logs: `~/logs/export_<jobid>.log`
