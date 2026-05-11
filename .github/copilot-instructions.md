# Anchor — mindmate repo — Copilot instructions

Read `docs/Home.md` first. All project knowledge lives in `docs/`.

## Required after every code change: update the docs

| Change | Doc to update |
|---|---|
| Training scripts, data pipeline | `docs/Training.md`, `docs/Data.md` |
| Benchmark logic or results | `docs/Benchmarks.md`, `docs/Models.md` |
| Cluster / SLURM | `docs/Cluster.md` |
| anchor-app memory, prompts, eval | `docs/Anchor App.md`, `docs/System Prompt.md` |
| Bug fixed | `docs/Bug Log.md` — new dated entry |
| New tasks or completed items | `docs/Next Steps.md` |
| Anything | Bump "Last updated" in `docs/Home.md` |

## Key facts

- `finetuning/` scripts are current. Root-level scripts are **legacy** — do not use.
- Never continue-train with `--adapter-path`. MEMORY_USE collapses. Always start fresh from base.
- Always use `--gres=gpu:a100-80:1` in sbatch. H100-96 GRES can fall back to 46 GB nodes.
- `--export` flag must come **before** the script path in sbatch.
- GitHub SSH is blocked on the cluster. Use `rsync`.
- Always use 3-run averaged benchmark scores. Single runs swing ±17 scenarios at temp=0.7.
