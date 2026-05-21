---
tags: [anchor, index]
---

# Anchor — Project Hub

> Local, finetuned mental-health companion. Llama 3.2 3B SFT → GGUF → Android + webapp.
> **Production:** https://tryanchor.me | **Last updated:** 2026-05-21 (🟢 **genzv5 SFT RUNNING** as job 618377 — loss 5.7→3.5 at step 190. Full bug trail: bare pip→miniconda, ensurepip, logging.py CWD shadow, torchvision/torch version mismatch. All fixed. See Bug Log 2026-05-21.
>
> **Cleanup (2026-05-21):** freed ~106 GB on cluster, ~40 GB locally. Cluster: deleted exports `mindmate_qwen3_1p7b`, `mindmate_qwen25_dpo_ck200`, `mindmate_llama_dpo_ck200`, `mindmate_gemma4_e2b` (teacher), `mindmate_genz_llama32_3b` (superseded), `mindmate_llama_sft_ck200`, `mindmate_genzv4_ck200`; deleted all DPO adapters (`CUDA_mindmate_llama32b_dpo_ck{200,1600}`, `genz_dpo`, `genz_dpo_ck1600`, `genzv2_dpo_ck1200`, `genzv3_dpo_ck200`, `CUDA_mindmate_qwen25_3b_dpo_ck{200,1600}`) — DPO abandoned. `exports/` 141G→42G, `adapters/` 19G→12G. Local: deleted `mindmate_app/` (superseded by `anchor-app/`, GitHub-backed), `exports.zip` (Jan-24 archive), `models/mlx_base_llama32_3b` (MLX abandoned), `.claude/worktrees`; pruned `adapters/` to keep only `genz/checkpoint-1200` (3.4G→219M); `git gc --aggressive --prune=now` (.git 11G→606M). Disk 5.3 GiB→45 GiB free. All deletions regeneratable via `scripts/export_gguf_cuda.py`.)

---

## What This Project Is

Anchor is a privacy-first mental-health companion that runs entirely on-device (Android). It is a finetuned version of `meta-llama/Llama-3.2-3B-Instruct` using QLoRA SFT. The model is served as a GGUF Q4_K_M file loaded by llama.cpp on the phone. There is also a FastAPI webapp at tryanchor.me for demos.

The goal: a model that behaves like a close friend who listens, remembers context across sessions, recognizes crisis situations, and responds with appropriate tone (not therapist-speak, not generic chatbot). The app stores episodic memory in SQLite and injects it into the system prompt each session.

---

## Quick Status (2026-05-19)

| | |
|---|---|
| **Best SFT checkpoint** | `genzv2_ck1200` — `adapters/genz/checkpoint-1200`, **44%** on v3 benchmark (still best SFT pending genzv5) |
| **Best on v3 overall** | `llama_base` — base model, no fine-tuning, **51%** on v3 benchmark |
| **Best SFT GGUF** | `exports/mindmate_genzv2_ck1200_q4_k_m.gguf` |
| **Production serving** | `exports/mindmate_llama_sft_ck1600/` at tryanchor.me — ⚠️ outdated, should upgrade to genzv2_ck1200 |
| **Benchmark suite** | **v4** (released 2026-05-15) — 83 scenarios (45 dynamic), unified cluster + mobile, Gemma4 judge, **production sampling params** (temp=0.7, top_p=0.95, top_k=40, min_p=0.05 — matches `defaultCompletionParams` in the app). v3 stays available for legacy comparisons. See [[Benchmarks]]. |
| **Root cause fixed** | ✅ commit c3acdc9 — all 42,038 training examples now use production system prompt format |
| **Active cluster jobs** | ✅ 615485–615493 conv-memory + biometric shards complete (385 ex added to v5) · **616643** — crisis (running, ⚠️ only 5/511 pass heuristic) · **618377** — genzv5 SFT 🟢 **TRAINING** (A100-80, cu126+torchvision fix, loss 5.7→3.5 at step 190, ~1.8s/step, ETA ~22h) |
| **Completed jobs** | ✅ **615518** help_mode — 200 examples total |
| **Next milestone** | genzv5 SFT 617977 start on A100-80 → checkpoints every 200 steps → v4 benchmark |
| **DPO** | ❌ Abandoned — all 3 runs flat or worse than SFT |

---

## Critical Things to Know Before Touching Anything

These are the non-obvious invariants that burn time if unknown:

1. **Base model still beats all SFT on v3 benchmark.** Root cause: training data used wrong system prompt format. Fixed in c3acdc9. genzv5 will be the first real test of whether SFT can actually beat base.

2. **Never continued-train.** `PeftModel.from_pretrained` with `is_trainable=True` catastrophically collapses MEMORY_USE to 0/8. Confirmed on `genzv2_continued`. Always start fresh from base.

3. **Always use `--gres=gpu:a100-80:1` explicitly on the cluster.** H100-96 GRES can fall back to ~46GB nodes (seen on jobs 609078–609083), which OOMs the Gemma4 judge at bfloat16 (~52GB). **A100-80 nodes have mixed drivers** — xgph6 has driver 575 (CUDA 12.9), others have driver 580 (CUDA 13.0). PyTorch cu130 only works on driver 580 nodes. The SLURM training script reinstalls torch+cu126 at job start (works on both driver versions) AND `CUDA_train_qlora.py` has a comprehensive Python-level CUDA shim (`is_available`, `is_bf16_supported`, `_initialized`, `set_device`, `manual_seed*`, `get_rng_state*`, `default_generators` etc.) as defense-in-depth. H200 (xgpk0, `gpu` partition) also has driver 575 and is not usable without the cu126 install step.

3b. **CRITICAL: use `python -m pip`, NEVER bare `pip`, in SLURM scripts.** On the cluster, `source mindmatenv/bin/activate` does NOT change which `pip` is resolved — bare `pip` keeps pointing to `/home/a/aryanj/miniconda3/bin/pip`. Every install goes to **miniconda3's site-packages**, NOT the venv. Burned 4 SFT jobs (617977, 618080, 618086, 618200) this way — `pip show torch` reported cu126 (miniconda3's view) while `python -c 'import torch'` reported cu130 (venv's actual install, untouched since April). Also `rm -rf $VENV_SITE/torch $VENV_SITE/torch-*.dist-info` before reinstall — pip's `--force-reinstall` leaves stale package files. See Bug Log 2026-05-21.

4. **`--export` flag must be BEFORE the script path in sbatch.** `sbatch script.slurm --export=MODEL=foo` silently treats it as a script argument. Burned 5 jobs (607691–607695) this way.

5. **GitHub SSH is blocked from the cluster.** `git pull` fails with `Connection timed out`. Use `rsync` to push individual files from local to cluster.

6. **Log path on cluster is `/home/a/aryanj/logs/`** — not `/home/aryanj/logs/` (the latter is a different, wrong path that causes silent SLURM failure).

7. **`finetuning/` has the current scripts. Root-level scripts are legacy.** `CUDA_train_qlora.py`, `build_dataset.py`, etc. at the project root are old duplicates from early development. Always use `finetuning/CUDA_train_qlora.py`, `finetuning/build_dataset.py`, etc.

7b. **Documented v3/v4 sample counts may be wrong.** `extra_paths` drift (fixed 2026-05-19) caused v3/v4 build_dataset.py runs to silently skip 3 of 8 files in SOURCE_CAPS. v4 documented as 21,529 examples may have actually been ~13,000 in git's version; v3 documented as 10,065 may have been ~8,500. Cluster copies may differ. Genzv5+ are unaffected (fix is in). See Bug Log 2026-05-19.

8. **Benchmarks use production sampling — variance is real.** v4 uses the same sampling params as the production chat (temperature=0.7, top_p=0.95, top_k=40, min_p=0.05). This means scores are non-deterministic on every scenario type — 3-run averaging is required for any leaderboard ranking. The rationale: a benchmark with temp=0 measures a model nobody ships. v3 had the same problem; v4 inherits the same fix (`--runs=3` + `benchmarks/diff_results.py`).

9. **`--wrap` in sbatch uses `/bin/sh`, not bash.** `source` command is not available. Use `bash -c "source ... && python ..."` instead.

10. **`deploy/config.py` still points to ck1600 GGUF.** It needs to be updated to `mindmate_genzv2_ck1200_q4_k_m.gguf` when upgrading production.

11. **Update these docs every time you change code.** See the "Doc update rule" section at the bottom of this file. This is not optional — stale docs are the primary source of wasted context across sessions.

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

---

## Doc Update Rule — Mandatory for All Agents

**Every time you make a code change in this project, update the docs before ending your turn.** This rule exists because stale docs are the #1 source of wasted context across sessions.

| What changed | Which docs to update |
|---|---|
| Training scripts, data pipeline, dataset | `Training.md`, `Data.md` |
| Benchmark logic or results | `Benchmarks.md`, `Models.md` |
| Cluster scripts, SLURM, ssh | `Cluster.md` |
| anchor-app memory system, prompts, chat lifecycle | `Anchor App.md`, `System Prompt.md` |
| anchor-app eval harness | `Anchor App.md` (eval section) |
| Any bug found + fixed | `Bug Log.md` — new entry under today's date |
| New work items or removed blockers | `Next Steps.md` |
| Anything at all | Bump `Home.md` last-updated date |

When in doubt: add a dated note to `Bug Log.md` and update this file's date.

**Where this rule is enforced** (one file per agent entry point):

| Agent | File |
|---|---|
| Claude Code | `CLAUDE.md` |
| Codex / any OpenAI agent | `AGENTS.md` |
| Cursor | `.cursorrules` |
| GitHub Copilot | `.github/copilot-instructions.md` |
| Any agent that reads the docs | This section |

---

## Notes
- [[Benchmarks]] — full methodology, all results tables, variance analysis, running commands
- [[Training]] — SFT pipeline, exact hyperparams, data mixes, SLURM scripts with exact args
- [[Data]] — all training JSONL files with counts, status, normalization, pipelines
- [[Cluster]] — SSH, SLURM gotchas, GPU notes, file sync commands
- [[Production]] — webapp (DigitalOcean) + Android app architecture, model paths
- [[System Prompt]] — production vs eval prompt, memory injection tiers, training format
- [[Next Steps]] — genzv5 plan, open questions
- [[Bug Log]] — chronological bug history (Apr 7 → May 9)
- [[Anchor App]] — Android app architecture, memory system, screens, chat lifecycle, eval harness
