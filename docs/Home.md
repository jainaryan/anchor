---
tags: [anchor, index]
---

# Anchor — Project Hub

> Local, finetuned mental-health companion. Llama 3.2 3B SFT → GGUF → Android + webapp.
> **Production:** https://tryanchor.me | **Last updated:** 2026-05-09 (session 4)

---

## Quick Status

| | |
|---|---|
| **Best model** | [[Models#genzv2 ck1200 — Best SFT\|genzv2_ck1200]] (71% v1 / 44% v3) |
| **Active jobs** | 609110–609111 — conv-memory pipeline v2 (72h from 2026-05-09) |
| **Next milestone** | [[Next Steps]] — genzv5 SFT once conv-memory data lands |
| **Root cause fixed?** | ✅ commit c3acdc9 — training format mismatch patched |
| **DPO** | ❌ Abandoned — all 3 runs flat or worse than SFT |

---

## Notes

- [[Models]] — leaderboard, adapter paths, GGUF inventory
- [[Benchmarks]] — methodology, v1/v2/v3 results, variance notes
- [[Training]] — SFT pipeline, hyperparams, SLURM commands
- [[Data]] — all training JSONL files, counts, status
- [[Cluster]] — SSH, SLURM, GPU notes, common commands
- [[Production]] — webapp (DigitalOcean) + Android app
- [[System Prompt]] — production vs eval prompt architecture
- [[Next Steps]] — genzv5 plan, open questions
- [[Bug Log]] — historical bugs fixed

---

## Repos

| Repo | URL | Branch |
|---|---|---|
| mindmate (training) | github.com/jainaryan/mindmate | `main` |
| anchor-app (Android) | github.com/tanmaykay/Anchor | `aryan_branch` |

## Key Local Paths

| | Path |
|---|---|
| Training | `~/projects/mindmate/` |
| Android app | `~/projects/anchor-app/` |
| Best adapter | `~/projects/mindmate/adapters/genz/checkpoint-1200` |
| Best GGUF | `~/projects/mindmate/exports/mindmate_genzv2_ck1200_q4_k_m.gguf` |
