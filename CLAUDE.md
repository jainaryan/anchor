# Anchor — mindmate repo

**Start here:** Read `docs/Home.md`. All project knowledge lives in `docs/`.

## What this is
Finetuned local mental-health companion. Llama 3.2 3B + QLoRA SFT → GGUF → Android app + webapp.
- Training, data, benchmarks, cluster jobs: **this repo**
- Android app: `~/projects/anchor-app/` (separate repo)
- Obsidian vault: `~/projects/obisidian/anchor/` → symlink → `docs/`

## Docs to read (in order)
1. `docs/Home.md` — quick status, all critical invariants, links to everything
2. Pick the relevant note for your task:
   - Training a new model → `docs/Training.md` + `docs/Data.md`
   - Running benchmarks → `docs/Benchmarks.md` + `docs/Cluster.md`
   - Understanding results → `docs/Models.md`
   - Production deploy → `docs/Production.md`
   - Android app → `docs/Anchor App.md`
   - System prompt / memory → `docs/System Prompt.md`
   - What to do next → `docs/Next Steps.md`
   - Debugging a past issue → `docs/Bug Log.md`

## Must-know before touching anything

1. **`finetuning/` scripts are current. Root-level scripts are legacy.** Use `finetuning/CUDA_train_qlora.py`, not `./CUDA_train_qlora.py`.
2. **Never continue-train** (`--adapter-path`). MEMORY_USE collapses to 0/8. Always fresh from base.
3. **`--gres=gpu:a100-80:1` must be explicit** in every sbatch. H100-96 falls back to 46GB nodes (OOMs Gemma4).
4. **`--export` flag goes BEFORE the script path** in sbatch. After = silently ignored.
5. **GitHub SSH blocked from cluster.** Use rsync. See `docs/Cluster.md`.
6. **3-run averaged benchmark scores only.** Single runs swing ±17 scenarios at temp=0.7.
7. **Base model beats all SFT on v3 benchmark (51% vs 44%).** Root cause fixed (c3acdc9). genzv5 is next.

## Doc update rule — mandatory

See `AGENTS.md` for the full routing table. Short version: update the relevant doc in `docs/` before ending your turn, and always bump the "Last updated" date in `docs/Home.md`. When in doubt, add a dated entry to `docs/Bug Log.md`.
