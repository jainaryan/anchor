# Anchor GGUF Exports

GGUF models for on-device inference (Android, webapp, macOS, Windows).

## Available GGUFs

| File | Base | Adapter | Quant | Notes |
|---|---|---|---|---|
| `mindmate_genzv2_ck1200_q4_k_m.gguf` | Llama 3.2 3B | genzv2 SFT ck1200 | Q4_K_M | **BEST MODEL — use this** |
| `mindmate_genzv3_ck200_q4_k_m.gguf` | Llama 3.2 3B | genzv3 SFT ck200 | Q4_K_M | 76% v1 / 45% v3 avg |
| `mindmate_genzv4_ck200_q4_k_m.gguf` | Llama 3.2 3B | genzv4 SFT ck200 | Q4_K_M | 65% v1 / 42% v3 avg |
| `mindmate_genz_llama32_3b_q4_k_m.gguf` | Llama 3.2 3B | genzv2 SFT ck1600 | Q4_K_M | Old production (ck1600) |
| `mindmate_llama32_3b_q4_k_m.gguf` | Llama 3.2 3B | none | Q4_K_M | Base model (no adapter) |
| `mindmate_llama32_3b_f16.gguf` | Llama 3.2 3B | none | F16 | Base model full precision |
| `mindmate_qwen3_1p7b_q4_k_m.gguf` | Qwen3 1.7B | — | Q4_K_M | Inferior — do not use |

**Benchmark summary (v1 /34, v3 /61 weighted):**
- genzv2_ck1200: 71% v1 / 44% v3 (most stable: ±0.8)
- genzv3_ck200: 76% v1 / 45% v3
- genzv4_ck200: 65% v1 / 42% v3
- llama_base (no adapter): 51% v3 — still beats all SFT on v3 (root cause fixed in c3acdc9, genzv5 pending)

**Current webapp server** (`tryanchor.me`) still serves the old ck1600 GGUF — should upgrade to `mindmate_genzv2_ck1200_q4_k_m.gguf`.

**DPO models: ABANDONED** — all DPO runs were flat or worse than SFT base. No DPO GGUFs exported.

---

## Android Usage

The anchor-app loads GGUF from device storage.

1. Transfer `mindmate_genzv2_ck1200_q4_k_m.gguf` to `/sdcard/Download/mindmate.gguf`
2. In the app: Add Model → pick from storage → app copies to internal storage with progress overlay
3. Load model → chat

**Performance (Pixel 8a, Q4_K_M):** ~5.4–6.0 TPS, TTFT 6s cached / 60–66s cold, heap ~3.12–3.17 GB.

---

## Other Platforms

### macOS / Windows — LM Studio
1. Download [LM Studio](https://lmstudio.ai/)
2. Load the `.gguf` file
3. System prompt: use the Anchor `BASE_PROMPT` from `anchor-app/src/utils/anchorSystemPrompt.ts`

### macOS / Linux — Ollama
```bash
echo 'FROM ./mindmate_genzv2_ck1200_q4_k_m.gguf' > Modelfile
ollama create anchor -f Modelfile
ollama run anchor
```

---

## Reproducing the Export

```bash
# On the cluster
cd ~/projects/mindmate
python scripts/export_gguf_cuda.py --model genzv2_ck1200

# Or via SLURM
sbatch scripts/run_export_top3.slurm
```

Export logs: `~/logs/export_<jobid>.log`
