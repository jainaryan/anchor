---
tags: [anchor, index]
---

# Anchor — Project Hub

> Local, finetuned mental-health companion. Llama 3.2 3B SFT → GGUF → Android + webapp.
> **Production:** https://tryanchor.me | **Last updated:** 2026-05-16 (session 11 wrap — v4 benchmark infrastructure complete. Sampling parameters now match the production app (`defaultCompletionParams`): temp=0.7, top_p=0.95, top_k=40, min_p=0.05, penalty_repeat=1.0, n_predict=1024. The cluster runner reads from a `PRODUCTION_SAMPLING` constant; mobile EvalRunnerV4 imports `defaultCompletionParams` directly so it can never drift. Both runtimes emit a `sampling.matches_production_app: bool` flag in each result doc — flips false the moment anyone overrides a param.)

---

## What This Project Is

Anchor is a privacy-first mental-health companion that runs entirely on-device (Android). It is a finetuned version of `meta-llama/Llama-3.2-3B-Instruct` using QLoRA SFT. The model is served as a GGUF Q4_K_M file loaded by llama.cpp on the phone. There is also a FastAPI webapp at tryanchor.me for demos.

The goal: a model that behaves like a close friend who listens, remembers context across sessions, recognizes crisis situations, and responds with appropriate tone (not therapist-speak, not generic chatbot). The app stores episodic memory in SQLite and injects it into the system prompt each session.

---

## Quick Status (2026-05-16)

| | |
|---|---|
| **Best SFT checkpoint** | `genzv2_ck1200` — `adapters/genz/checkpoint-1200`, **44%** on v3 benchmark (still best SFT pending genzv5) |
| **Best on v3 overall** | `llama_base` — base model, no fine-tuning, **51%** on v3 benchmark |
| **Best SFT GGUF** | `exports/mindmate_genzv2_ck1200_q4_k_m.gguf` |
| **Production serving** | `exports/mindmate_llama_sft_ck1600/` at tryanchor.me — ⚠️ outdated, should upgrade to genzv2_ck1200 |
| **Benchmark suite** | **v4** (released 2026-05-15) — 83 scenarios (45 dynamic), unified cluster + mobile, Gemma4 judge, **production sampling params** (temp=0.7, top_p=0.95, top_k=40, min_p=0.05 — matches `defaultCompletionParams` in the app). v3 stays available for legacy comparisons. See [[Benchmarks]]. |
| **Root cause fixed** | ✅ commit c3acdc9 — all 42,038 training examples now use production system prompt format |
| **Active cluster jobs** | 611377/611379–611381 — conv-memory shards 0–2 (finishing) · 613218–613220 — biometric PCIe HF (~0.7–0.9/hr) · 613666–613668 — biometric SXM vLLM (queued, ~5/hr when running) · 612903–612905, 613115–613117 — conv-memory new pool shards 3–8 |
| **Next milestone** | genzv5 SFT — first model on the v4 benchmark — once conv-memory + biometric data merge |
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
