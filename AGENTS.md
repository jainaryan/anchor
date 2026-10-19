# Anchor — mindmate repo (agent instructions)

All project knowledge lives in `docs/`. Read `docs/Home.md` first, then the note for your task.

## Doc update rule — enforced for every agent

**Whenever you change code in this repo or in `~/projects/anchor-app/`, update the relevant docs before ending your turn.** This applies to Claude, Codex, Cursor, Copilot, and any other agent.

| What you changed | Update these docs |
|---|---|
| Training scripts, data pipeline, dataset files | `docs/Training.md`, `docs/Data.md` |
| Benchmark logic, scenarios, or results | `docs/Benchmarks.md`, `docs/Models.md` |
| Cluster scripts, SLURM, SSH setup | `docs/Cluster.md` |
| anchor-app: memory system, prompts, chat lifecycle | `docs/Anchor App.md`, `docs/System Prompt.md` |
| anchor-app: eval harness | `docs/Anchor App.md` (eval section) |
| Any bug found and fixed | `docs/Bug Log.md` — new dated entry |
| New tasks unblocked or old ones completed | `docs/Next Steps.md` |
| Anything at all | Bump the "Last updated" date in `docs/Home.md` |

When in doubt: add a dated entry to `docs/Bug Log.md` and update the date in `docs/Home.md`.

---

## Must-know invariants

1. **`finetuning/` is current. Root-level scripts are legacy.** Use `finetuning/CUDA_train_qlora.py`, not `./CUDA_train_qlora.py`.
2. **Never continue-train** (`--adapter-path`). MEMORY_USE collapses to 0/8. Always train fresh from base.
3. **`--gres=gpu:a100-80:1` must be explicit in every sbatch.** H100-96 GRES can fall back to ~46 GB nodes (OOMs the Gemma4 judge at bfloat16 ~52 GB).
4. **`--export` flag must go BEFORE the script path in sbatch.** After the path = silently ignored.
5. **GitHub SSH is blocked from the cluster.** Use rsync. See `docs/Cluster.md`.
6. **3-run averaged benchmark scores only.** Single runs swing ±17 scenarios at temp=0.7. ⚠️ But v4 repeats are currently ~identical (global `torch.manual_seed` in the simulator — 75/83 transcripts repeat). Fix seeding before trusting averages. See `docs/Bug Log.md` 2026-10-20.
7. **Best v4 model is genzv3_ck200 (61.5%)**, base 57.4%, Qwen3-30B teacher 78.0%.
8. **Check the cleaned dataset, not the preset.** `clean_dataset.py --drop-min-turns 4` drops all single-turn rows (all of targeted_fix, 175/181 gold). The cluster's old `clean_dataset.py` also drops `loss_weight`. Count rows per source and `loss_weight` in `conversations_cleaned_vN` before training.
