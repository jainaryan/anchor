---
tags: [anchor, benchmarks]
---

# Benchmarks

← [[Home]]

---

## Suite v3 (current) — 58 scenarios, 9 categories

**Scoring:** weighted `/61` — CRISIS weight=2, all others weight=1.
**Judge:** Gemma 4 26B A4B IT (bfloat16, loaded on A100-80 alongside Llama 4-bit).
**Methodology:** 3-run averaged (temperature=0.7 causes ±17/49 scenario flips per run).

| Category | Scenarios | What it tests |
|---|---|---|
| CONTEXT_MEMORY | 8 | Profile fields used in first response |
| CONVERSATION_MEMORY | 5 | Facts introduced mid-turn recalled later |
| CROSS_SESSION_MEMORY | 4 | Pre-seeded `[Recent sessions]` header used |
| HELP_MODE | 8 | "help me calm down" → technique in first reply |
| CRISIS | 8 | SI, escalation, humor deflection (weight=2) |
| NO_HALLUCINATION | 7 | No invented history, clean cold opens |
| BIOMETRIC | 5 | Health data referenced naturally |
| FORMAT | 4 | No markdown, no "As an AI", proper length |
| COMPANION | 9 | Celebrates good news, casual chat, genziness |

### v3 Results (3-run averaged, COMPLETE)

| Model | Avg /61 | % | ± | n |
|---|---|---|---|---|
| llama_base | 31.3 | 51% | 1.9 | 3 |
| genzv3_ck200 | 27.5 | 45% | 2.1 | 4 |
| genzv2_ck1600 | 27.5 | 45% | 1.5 | 2 |
| genzv2_ck1200 | 26.8 | 44% | 0.8 | 4 |
| genzv4_ck200 | 25.7 | 42% | 1.2 | 3 |

---

## Suite v2 (superseded) — 49 scenarios, 8 categories

Same as v3 minus COMPANION. Total weight 52. Results from 2026-05-07.

| Model | Weighted /52 | % |
|---|---|---|
| llama_base | 44 | **85%** |
| genzv4_ck200 | 40 | 77% |
| genzv3_ck200 | 35 | 67% |
| genzv2_ck1200 | 31 | 60% |
| genzv2_ck1600 | 30 | 58% |

---

## Suite v1 (historical) — 34 scenarios

Mixed rule-based + LLM judge (Qwen3-30B at the time). Checkpoint sweep results:

| Checkpoint | Total /34 | % |
|---|---|---|
| genzv3_ck200 | 26 | **76%** |
| genzv2_ck1200 | 24 | **71%** |
| genzv4_ck200 | 22 | 65% |
| genzv2_ck1600 | 22 | 65% |
| genzv2_dpo_ck1200 | 20 | 59% ❌ DPO worse |

---

## Running Benchmarks

```bash
# Single run on cluster
sbatch --gres=gpu:a100-80:1 --export=ALL,MODEL=genzv2_ck1200 benchmarks/run_benchmarks.slurm

# 3-run sweep (all 5 models)
for MODEL in llama_base llama_base llama_base \
             genzv2_ck1200 genzv2_ck1200 genzv2_ck1200 \
             genzv3_ck200 genzv3_ck200 genzv3_ck200 \
             genzv2_ck1600 genzv2_ck1600 genzv2_ck1600 \
             genzv4_ck200 genzv4_ck200 genzv4_ck200; do
  sbatch --gres=gpu:a100-80:1 --export=ALL,MODEL=$MODEL benchmarks/run_benchmarks.slurm
done

# Average results
python benchmarks/average_results.py --since 20260509
```

> ⚠️ Always use `--gres=gpu:a100-80:1` explicitly. H100-96 GRES can fall back to ~46GB nodes (OOM for Gemma4 judge).
> ⚠️ `--export` flag must come BEFORE the script path — after is silently ignored by sbatch.

---

## Key Findings

- **Variance:** `temperature=0.7` → 17/49 scenarios flip PASS↔FAIL between runs. Always use 3-run average.
- **Base beats SFT on v3:** Root cause was training format mismatch (c3acdc9). Expect recovery in genzv5.
- **DPO never helps:** All 3 DPO runs flat or worse. Current data + beta=0.1 + 800 steps is insufficient.
- **COMPANION is the SFT win:** genzv2_ck1600 100% COMPANION — heavily trained on friend/casual data.
- **CROSS_SESSION is the canary:** genzv2_ck1200 CROSS_SESSION = 0% — model completely ignores `[Recent sessions]`.

---

## See also

- [[Models]] — per-model breakdown
- [[Training]] — how to run a new SFT job
