# Anchor GGUF Exports

GGUF builds of the fine-tuned models for on-device inference. Scores and recommendations: [`docs/Models.md`](../docs/Models.md). The weight files themselves are gitignored.

## On the cluster (`~/projects/mindmate/exports/`, 2026-10-20)

| Dir | Model | Adapter | Exported | v4 score | Notes |
|---|---|---|---|---|---|
| `mindmate_genzv3_ck200/` | genzv3_ck200 | `adapters/genzv3/checkpoint-200` | 2026-05-03 | **61.5%** | Best on v4; CRISIS 64%. Not yet on HF |
| `mindmate_genzv6_ck1600/` | genzv6_ck1600 | `adapters/genzv6/checkpoint-1600` | 2026-05-30 (job 624121) | 56.7% | Latest SFT; on HF |
| `mindmate_genzv2_ck1200/` | genzv2_ck1200 | `adapters/genz/checkpoint-1200` | 2026-05-03 | 51.3% | Android app default (last known); on HF. CRISIS 28% |
| `mindmate_llama_sft_ck1600/` | genzv2_ck1600 | `adapters/genz/checkpoint-1600` | 2026-04-10 | 51.5% | tryanchor.me model; on HF as `llama(genz)v2_q4_k_m.gguf`. CRISIS 21% |

Each dir holds:
- `merged_hf/` — merged fp16 HF weights
- `*_f16.gguf` — 6.4 GB
- `*_q4_k_m.gguf` — ~1.9 GB, **the one to ship**

A whole dir is ~14 GB, so pull just the q4 file. Other exports (genzv4_ck200, DPO, Qwen, Gemma4, old-name ck1600) were deleted in the 2026-05-21 cleanup; regenerate with `scripts/export_gguf_cuda.py` if needed. The Mac's `exports/` may hold a different subset.

**No GGUF has been benchmarked yet** — all v4 scores are for the NF4 adapters. Before shipping, run `benchmarks/run_benchmarks_v4.slurm` with `GGUF_PATH=exports/mindmate_<name>/mindmate_<name>_q4_k_m.gguf`.

## HuggingFace

`jainaryan/mindmate-gguf` (private). Known uploads: `llama(genz)v2_q4_k_m.gguf`, `mindmate_genzv2_ck1200_q4_k_m.gguf`, `mindmate_genzv6_ck1600_q4_k_m.gguf`. To publish genzv3_ck200:

```bash
python scripts/upload_gguf_to_hf.py --model genzv3_ck200   # needs HF_TOKEN
```

## Using a GGUF

- **Anchor app:** download from HF in the Models screen, or Add Model → pick from storage (the app copies it to internal storage). The app builds the system prompt and memory itself.
  - Pixel 8a, Q4_K_M: ~5.4–6.0 tok/s, TTFT ~6s cached / 60–66s cold, heap ~3.1 GB.
- **LM Studio / Ollama / llama.cpp:** use chat format `llama-3`. As the system prompt, use the 6-line Anchor base prompt from [`docs/System Prompt.md`](../docs/System%20Prompt.md) (= `anchor-app/src/utils/anchorSystemPrompt.ts`). `system_prompt.txt` and `inference/system_prompt.txt` are legacy prompts, not what the app uses.

```bash
echo 'FROM ./mindmate_genzv3_ck200_q4_k_m.gguf' > Modelfile
ollama create anchor -f Modelfile && ollama run anchor
```

## Reproducing an export (cluster)

```bash
cd ~/projects/mindmate
# new checkpoint? add it to MODELS in scripts/export_gguf_cuda.py first
sbatch --export=ALL,MODEL=genzv6_ck1600 scripts/run_export.slurm
python scripts/upload_gguf_to_hf.py --model genzv6_ck1600
```

`run_export.slurm` is **CPU-only**: it installs CPU torch into `/tmp/cpu_torch_export_$$` and never touches the venv's GPU torch. Logs: `~/logs/mindmate-export_<jobid>.out`. Details: [`docs/Training.md`](../docs/Training.md) → GGUF Export.
