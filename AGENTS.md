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
6. **3-run averaged benchmark scores only.** Single runs swing ±17 scenarios at temp=0.7.
7. **Base model beats all SFT on v3 benchmark (51% vs 44%).** Root cause fixed in commit c3acdc9. genzv5 is the next real test.
