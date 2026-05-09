---
tags: [anchor, index]
---

# Anchor — Project Hub

> Local, finetuned mental-health companion. Llama 3.2 3B SFT → GGUF → Android + webapp.
> **Production:** https://tryanchor.me | **Last updated:** 2026-05-09 (session 4)

---

## What This Project Is

Anchor is a privacy-first mental-health companion that runs entirely on-device (Android). It is a finetuned version of `meta-llama/Llama-3.2-3B-Instruct` using QLoRA SFT. The model is served as a GGUF Q4_K_M file loaded by llama.cpp on the phone. There is also a FastAPI webapp at tryanchor.me for demos.

The goal: a model that behaves like a close friend who listens, remembers context across sessions, recognizes crisis situations, and responds with appropriate tone (not therapist-speak, not generic chatbot). The app stores episodic memory in SQLite and injects it into the system prompt each session.

---

## Quick Status (2026-05-09)

| | |
|---|---|
| **Best model** | `genzv2_ck1200` — `adapters/genz/checkpoint-1200` |
| **Best GGUF** | `exports/mindmate_genzv2_ck1200_q4_k_m.gguf` |
| **v3 benchmark** | llama_base **51%** > genzv3_ck200 45% = genzv2_ck1600 45% > genzv2_ck1200 44% > genzv4_ck200 42% |
| **Production serving** | `exports/mindmate_llama_sft_ck1600/` at tryanchor.me — ⚠️ outdated, should be genzv2_ck1200 |
| **Root cause fixed** | ✅ commit c3acdc9 — all 42,038 training examples now use production system prompt format |
| **Active cluster jobs** | 609110–609111 — conv-memory pipeline v2, 72h A100-80, ~64h remaining as of session 4 |
| **Next milestone** | genzv5 SFT once conv-memory data lands (jobs above finish) |
| **DPO** | ❌ Abandoned — all 3 runs flat or worse than SFT |

---

## Critical Things to Know Before Touching Anything

These are the non-obvious invariants that burn time if unknown:

1. **Base model still beats all SFT on v3 benchmark.** Root cause: training data used wrong system prompt format. Fixed in c3acdc9. genzv5 will be the first real test of whether SFT can actually beat base.

2. **Never continued-train.** `PeftModel.from_pretrained` with `is_trainable=True` catastrophically collapses MEMORY_USE to 0/8. Confirmed on `genzv2_continued`. Always start fresh from base.

3. **Always use `--gres=gpu:a100-80:1` explicitly on the cluster.** H100-96 GRES can fall back to ~46GB nodes (seen on jobs 609078–609083), which OOMs the Gemma4 judge at bfloat16 (~52GB).

4. **`--export` flag must be BEFORE the script path in sbatch.** `sbatch script.slurm --export=MODEL=foo` silently treats it as a script argument. Burned 5 jobs (607691–607695) this way.

5. **GitHub SSH is blocked from the cluster.** `git pull` fails with `Connection timed out`. Use `rsync` to push individual files from local to cluster.

6. **Log path on cluster is `/home/a/aryanj/logs/`** — not `/home/aryanj/logs/` (the latter is a different, wrong path that causes silent SLURM failure).

7. **`finetuning/` has the current scripts. Root-level scripts are legacy.** `CUDA_train_qlora.py`, `build_dataset.py`, etc. at the project root are old duplicates from early development. Always use `finetuning/CUDA_train_qlora.py`, `finetuning/build_dataset.py`, etc.

8. **Benchmark scores have ±~2pt variance** at temperature=0.7. A single run can swing ±17 scenarios. Always use 3-run averaged results (`benchmarks/average_results.py --since YYYYMMDD`). Never rank models on single-run scores.

9. **`--wrap` in sbatch uses `/bin/sh`, not bash.** `source` command is not available. Use `bash -c "source ... && python ..."` instead.

10. **`deploy/config.py` still points to ck1600 GGUF.** It needs to be updated to `mindmate_genzv2_ck1200_q4_k_m.gguf` when upgrading production.

---

## Repos

| Repo | URL | Branch |
|---|---|---|
| mindmate (training) | github.com/jainaryan/mindmate | `main` |
| anchor-app (Android) | github.com/tanmaykay/Anchor | `aryan_branch` |

---

## Key Local Paths

| Resource | Path |
|---|---|
| mindmate repo | `~/projects/mindmate/` |
| anchor-app repo | `~/projects/anchor-app/` |
| Obsidian vault | `~/projects/obisidian/anchor/` → symlink → `~/projects/mindmate/docs/` |
| Best adapter (local) | `~/projects/mindmate/adapters/genz/checkpoint-1200` |
| Best GGUF (local) | `~/projects/mindmate/exports/mindmate_genzv2_ck1200_q4_k_m.gguf` |
| All adapters (cluster) | `~/projects/mindmate/adapters/` |
| Cluster project | `nus-student-cluster:~/projects/mindmate/` |
| Cluster logs | `nus-student-cluster:~/logs/` |

---

## Notes

- [[Models]] — leaderboard, per-category breakdown, adapter + GGUF inventory, all checkpoint results
- [[Benchmarks]] — full methodology, all results tables, variance analysis, running commands
- [[Training]] — SFT pipeline, exact hyperparams, data mixes, SLURM scripts with exact args
- [[Data]] — all training JSONL files with counts, status, normalization, pipelines
- [[Cluster]] — SSH, SLURM gotchas, GPU notes, file sync commands
- [[Production]] — webapp (DigitalOcean) + Android app architecture, model paths
- [[System Prompt]] — production vs eval prompt, memory injection tiers, training format
- [[Next Steps]] — genzv5 plan, open questions
- [[Bug Log]] — chronological bug history (Apr 7 → May 9)
