# Anchor (training repo, formerly MindMate)

Training repo for **Anchor**, a privacy-first mental-health companion that runs on-device on Android:
1. Llama 3.2 3B Instruct is fine-tuned with QLoRA on the NUS SoC SLURM cluster.
2. The adapter is quantized to a ~2 GB Q4_K_M GGUF.
3. llama.cpp serves it in two places:
   - the React Native app ([tanmaykay/Anchor](https://github.com/tanmaykay/Anchor), branch `aryan_branch`, Mac: `~/projects/anchor-app/`)
   - a demo webapp at https://tryanchor.me (`deploy/`)

**All project knowledge lives in the Obsidian vault in [`docs/`](docs/)** — start at [`docs/Home.md`](docs/Home.md). Agents: read [`CLAUDE.md`](CLAUDE.md) / [`AGENTS.md`](AGENTS.md) first.

## Pipeline

```
synthetic/*_pipeline.py        teacher-as-Anchor data generation (Qwen3-30B-A3B / Gemma4 26B, A100-80)
        ↓  data/synthetic_train_*.jsonl
finetuning/build_dataset.py --model vN → finetuning/clean_dataset.py → finetuning/CUDA_train_qlora.py
        ↓  adapters/genzvN/checkpoint-*
benchmarks/run_benchmarks_v4.py        83-scenario suite, Gemma4 judge, production sampling
        ↓
scripts/export_gguf_cuda.py → exports/mindmate_<name>/mindmate_<name>_q4_k_m.gguf → scripts/upload_gguf_to_hf.py
        ↓  jainaryan/mindmate-gguf (HF, private)
Android app (llama.rn)  /  deploy/server.py (FastAPI + SSE, CPU)
```

## Status (2026-10-20)

| | |
|---|---|
| **Best model (v4)** | `genzv3_ck200` — 61.5%. Base Llama 3.2 3B 57.4%; Qwen3-30B teacher 78.0% |
| **Latest SFT** | `genzv6_ck1600` — 56.7%; GGUF exported + on HF |
| **Next run** | `v7` preset + `finetuning/run_sft_v7.slurm` prepared, **not trained** — blocked on the P0 fixes in [`docs/Next Steps.md`](docs/Next%20Steps.md) |
| **Shipped** | App default: genzv2_ck1200 (last known). Webapp, per `deploy/config.py`: genzv2_ck1600 with the legacy 34-line `inference/system_prompt.txt` prompt, not the app's 6-line Anchor prompt. Both have the weakest CRISIS scores on v4 (21–28%) |
| **Open bugs** | (see [`docs/Bug Log.md`](docs/Bug%20Log.md) 2026-10-20) |

Open bugs in detail:
- `--drop-min-turns 4` drops all single-turn training data (all of targeted_fix, most gold).
- The cluster's old `clean_dataset.py` drops `loss_weight`.
- v4 repeat runs aren't independent (global reseed in the simulator).
- The holdout set is broken and its names leak into the training data.
- The help-mode heuristic rejects mid-session conversations.
- An HF token is committed in `misc` — revoke it.

## Layout

| Path | What |
|---|---|
| `synthetic/` | Data-generation pipelines (`crisis_help_pipeline.py`, `conversation_memory_pipeline.py`, `biometric_sft_pipeline.py`, …) + `utils.py` (`TeacherModel`, HF or vLLM backend) |
| `finetuning/` | Dataset build/clean, QLoRA trainer (`CUDA_train_qlora.py`, weighted loss), SFT + datagen SLURM scripts |
| `benchmarks/` | v4 suite (`scenarios.json`), runner, judge, crisis calibration, replay / diff / leaderboard; results in `benchmarks/results/` |
| `scripts/` | GGUF export (`export_gguf_cuda.py`, `run_export.slurm`), HF upload, `chat_cluster.py`, `normalize_memory_format.py`, `sync_mobile_eval.sh` |
| `deploy/` | tryanchor.me FastAPI server (`server.py`, `config.py`, `DEPLOY.md`) |
| `data/` | Training JSONL — see [`docs/Data.md`](docs/Data.md) |
| `adapters/`, `exports/` | LoRA checkpoints and GGUFs (weights gitignored) — see [`exports/README.md`](exports/README.md) |
| `docs/` | Obsidian vault — the documentation |
| `inference/`, `web/`, `mlx_export/`, root `*.py` / `*.slurm` | **Legacy**: old CLI chat scripts, Flask demo, MLX path, early duplicates. Use the `finetuning/`, `synthetic/`, `benchmarks/` and `scripts/` versions |

## Quick commands (cluster)

```bash
cd ~/projects/mindmate && source mindmatenv/bin/activate
sbatch finetuning/run_sft_v7.slurm                                         # build + clean + train (fix P0 items first)
sbatch --export=ALL,MODEL=genzv3_ck200 benchmarks/run_benchmarks_v4.slurm  # benchmark (A100-80, ~30 min)
sbatch --export=ALL,MODEL=genzv6_ck1600 scripts/run_export.slurm           # GGUF export (CPU-only)
python scripts/upload_gguf_to_hf.py --model genzv6_ck1600                  # needs HF_TOKEN
```

Cluster rules that cost jobs before ([`docs/Cluster.md`](docs/Cluster.md)):
- always `--gres=gpu:a100-80:1`
- `--export` goes before the script path
- logs go to `/home/a/aryanj/logs/`
- use `python -m pip`, never bare `pip`
- run Python from `/tmp`
- don't let parallel jobs modify the shared venv
- GitHub SSH is blocked from the cluster — rsync from the Mac, or use HTTPS
