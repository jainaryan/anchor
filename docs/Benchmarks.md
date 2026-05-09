---
tags: [anchor, benchmarks]
---

# Benchmarks

← [[Home]]

---

## Suite v3 (current) — 58 scenarios, 9 categories

**Active as of 2026-05-09. Supersedes v1 and v2.**

| Parameter | Value |
|---|---|
| Total scenarios | 58 |
| Categories | 9 |
| Weighted total | /61 (CRISIS weight=2, all others weight=1) |
| Judge | Gemma 4 26B A4B IT, bfloat16, loaded on A100-80 alongside Llama 4-bit |
| Eval model temperature | 0.7 |
| Methodology | 3-run averaged — single runs have ±17 scenario variance |
| Results location | `benchmarks/results/<label>_<timestamp>.{json,md}` |

### Categories and Scenarios

| Category | Code | Scenarios | Type | What it tests |
|---|---|---|---|---|
| CONTEXT_MEMORY | CM | 8 | single-turn | Profile fields (name, coping, biometric) referenced in first response |
| CONVERSATION_MEMORY | CoM | 5 | dynamic (5 turns) | Facts introduced mid-conversation recalled later in same session |
| CROSS_SESSION_MEMORY | XS | 4 | dynamic (3 turns) | Pre-seeded `[Recent sessions]` header used correctly |
| HELP_MODE | HM | 8 | mixed | "help me calm down" → technique in first reply, not a probe question |
| CRISIS | CR | 8 | mixed | SI, escalation, humor deflection. **Weight=2 — each counts double** |
| NO_HALLUCINATION | NH | 7 | single-turn | No invented history, no fake names, clean cold opens |
| BIOMETRIC | BIO | 5 | single-turn | Sleep/HRV/mood data referenced naturally and appropriately |
| FORMAT | FMT | 4 | single-turn | No markdown, no "As an AI", proper length (not too short, not too long) |
| COMPANION | COMP | 9 | mixed | Celebrates good news, casual chat, gen-z voice matching. Sub-tests cp_07–09: lowercase/casual voice matching, therapy-speak avoidance, high-energy calibration |

**Dynamic scenarios:** Gemma4 runs as user simulator (temperature=0.7), generating a realistic conversation. Then judge evaluates the transcript. Different user simulator runs → different transcripts → different judge verdicts. This is why variance is high.

**Weighted scoring formula:**
```
weighted_pct = sum(scenario_weight × passed) / sum(all_scenario_weights)
# CRISIS: weight=2 per scenario (8 scenarios × 2 = 16 weight points)
# All others: weight=1 (50 scenarios × 1 = 50 weight points)
# Total weight = 66... wait, actually 50 + 16 = 66? No.
# 50 non-crisis + 8 crisis × 2 = 50 + 16 = 66... but docs say /61
# Correct: 50 non-crisis weight-1 + 8 crisis weight-2 = 50 + 8 = 58 scenarios
# But total weight = (58-8)×1 + 8×2 = 50 + 16 = 66... hmm
# Actually: 9 categories, 8 crisis scenarios weight=2, rest weight=1
# (58-8)×1 + 8×2 = 50 + 16 = 66? But results show /61
# Reconcile: 58 total scenarios, CRISIS has 8 scenarios
# If all non-crisis (50) weight=1 and crisis (8) weight=2:
# Checking average_results.py: sum of all weights = /61
# Actually scenario count: 8+5+4+8+8+7+5+4+9 = 58. CRISIS=8 weight=2 → +8 extra
# total weight = 58 + 8 = 66... but benchmark shows /61
# NOTE: verify with average_results.py output — trust the code, not arithmetic
```

---

## v3 Results (3-run averaged, COMPLETE as of 2026-05-09)

```bash
python benchmarks/average_results.py --since 20260509
```

| Model | Avg /61 | % | ± | n |
|---|---|---|---|---|
| llama_base | 31.3 | **51%** | 1.9 | 3 |
| genzv3_ck200 | 27.5 | **45%** | 2.1 | 4 |
| genzv2_ck1600 | 27.5 | **45%** | 1.5 | 2 |
| genzv2_ck1200 | 26.8 | **44%** | 0.8 | 4 |
| genzv4_ck200 | 25.7 | **42%** | 1.2 | 3 |

Note: genzv2_ck1600 only n=2 because 3rd job may have timestamp-collided (two finished at same second, one overwrote the other).

---

## Running Benchmarks

### Single run (ad hoc)

```bash
# ⚠️ Always --gres=gpu:a100-80:1 — H100-96 GRES falls back to 46GB (OOMs Gemma4)
# ⚠️ --export must be BEFORE the script path
sbatch --gres=gpu:a100-80:1 --export=ALL,MODEL=genzv2_ck1200 benchmarks/run_benchmarks.slurm

# Other MODEL values: llama_base, genzv2_ck1600, genzv3_ck200, genzv4_ck200
# Single category only:
sbatch --gres=gpu:a100-80:1 --export=ALL,MODEL=genzv2_ck1200,CATEGORY=CRISIS benchmarks/run_benchmarks.slurm

# Custom adapter (not in MODEL_SHORTCUTS):
sbatch --gres=gpu:a100-80:1 --export=ALL,ADAPTER=adapters/genzv5/checkpoint-800,LABEL=genzv5_ck800 benchmarks/run_benchmarks.slurm
```

### 3-run sweep (all 5 models, 15 jobs)

```bash
for MODEL in llama_base llama_base llama_base \
             genzv2_ck1200 genzv2_ck1200 genzv2_ck1200 \
             genzv3_ck200 genzv3_ck200 genzv3_ck200 \
             genzv2_ck1600 genzv2_ck1600 genzv2_ck1600 \
             genzv4_ck200 genzv4_ck200 genzv4_ck200; do
  sbatch --gres=gpu:a100-80:1 --export=ALL,MODEL=$MODEL benchmarks/run_benchmarks.slurm
done
```

### Average results after all runs complete

```bash
# --since filters to runs from this date onward (YYYYMMDD format)
python benchmarks/average_results.py --since 20260509

# Output: per-model mean ± std weighted pass rate
# Also shows per-category breakdown when run across multiple models
```

### Benchmark script args reference

`benchmarks/run_benchmarks.py` accepts:
- `--model MODEL` — use a named MODEL_SHORTCUTS entry
- `--adapter PATH` — use a custom adapter path (relative to project root)
- `--base BASE_MODEL_ID` — base model (used with --adapter)
- `--label LABEL` — result file label
- `--category CATEGORY` — run only this category (CONTEXT_MEMORY, CONVERSATION_MEMORY, etc.)
- `--ids id1,id2` — run only these scenario IDs (e.g. `cr_01,cr_02`)
- `--no-save` — don't write results files

Both models (eval Llama + Gemma4 judge) are loaded simultaneously at startup. Eval model: 4-bit NF4. Judge: bfloat16. Both on A100-80.

---

## Benchmark Variance — Why 3 Runs Are Required

Discovered 2026-05-09. Running llama_base twice with same setup:
- **17/49 scenarios flipped PASS→FAIL** between runs
- Affected categories: CONVERSATION_MEMORY (−5), CONTEXT_MEMORY (−3), CRISIS (−2), NO_HALLUCINATION (−2), FORMAT (−2), CROSS_SESSION_MEMORY (−1), BIOMETRIC (−1), HELP_MODE (−1)

Root cause: both the eval model (generates responses) and dynamic scenario user simulator run at temperature=0.7. Different outputs → different judge verdicts.

**Rule: never rank models by a single-run score. Always use 3-run averaged results.**

Most variable scenarios: bio_02, cm_08, cp_03, cp_05, cp_09, cp_07, cv_01, hm_02, nh_04 (50–67% pass rate across runs).

---

## Suite v2 (superseded — 49 scenarios, 8 categories)

Same as v3 minus COMPANION category. Total weight = /52. Historical results from 2026-05-07.

| Model | CM /8 | CoM /5 | XS /4 | HM /8 | CR /8 | NOH /7 | BIO /5 | FMT /4 | Weighted /52 | % |
|---|---|---|---|---|---|---|---|---|---|---|
| **llama_base** | **6** | **5** | **4** | 5 | **7** | **6** | **5** | 3 | **44** | **85%** |
| genzv4_ck200 | 4 | 4 | **4** | 5 | **7** | 5 | 4 | **4** | **40** | **77%** |
| genzv3_ck200 | 3 | 4 | 3 | 3 | 5 | **6** | **5** | **4** | 35 | 67% |
| genzv2_ck1200 | 5 | 2 | 2 | 4 | 5 | 5 | 3 | 3 | 31 | 60% |
| genzv2_ck1600 | 4 | 4 | 2 | 3 | 5 | 3 | 3 | **4** | 30 | 58% |

Note: these are single-run results (before the 3-run methodology was adopted). Use for directional comparison only.

---

## Suite v1 (historical — 34 scenarios, mixed judge+rule-based)

Mixed scoring: BIOMETRIC + MEMORY_USE scored by Qwen3-30B LLM judge, CRISIS/HELP_MODE/NO_HALLUCINATION/FORMAT by rule-based checks.

Full genzv2 + genzv3 checkpoint sweep (jobs 602531–602546):

| Checkpoint | BIO/5 | CRISIS/5 | FORMAT/4 | HELP/6 | MEM/8 | NOH/6 | Total/34 | % |
|---|---|---|---|---|---|---|---|---|
| **genzv3_ck200** | 4 | 4 | 4 | 4 | 4 | 6 | **26** | **76%** |
| genzv3_ck200 (rerun) | 3 | 5 | 3 | 3 | 3 | 6 | 23 | 68% |
| genzv3_ck400 | 2 | 5 | 4 | 2 | 2 | 5 | 20 | 59% |
| genzv3_ck600 | 3 | 4 | 2 | 3 | 2 | 6 | 20 | 59% |
| genzv3_ck800 | 2 | 5 | 4 | 1 | 2 | 6 | 20 | 59% |
| genzv3_ck1000 | 2 | 4 | 4 | 1 | 1 | 5 | 17 | 50% |
| genzv3_ck1200 | 1 | 4 | 4 | 1 | 2 | 5 | 17 | 50% |
| genzv3_ck1400 | 3 | 5 | 4 | 1 | 2 | 6 | 21 | 62% |
| genzv3_ck1600 | 3 | 5 | 3 | 2 | 2 | 6 | 21 | 62% |
| genzv2_ck200 | 2 | 4 | 2 | 1 | 3 | 5 | 17 | 50% |
| genzv2_ck400 | 3 | 5 | 4 | 1 | 3 | 6 | 22 | 65% |
| genzv2_ck600 | 1 | 5 | 4 | 2 | 3 | 6 | 21 | 62% |
| genzv2_ck800 | 1 | 5 | 4 | 3 | 3 | 6 | 22 | 65% |
| genzv2_ck1000 | 1 | 5 | 4 | 3 | 3 | 6 | 22 | 65% |
| **genzv2_ck1200** | 3 | 5 | 4 | 2 | **4** | 6 | **24** | **71%** |
| genzv2_ck1400 | 3 | 4 | 4 | 2 | 2 | 6 | 21 | 62% |
| genzv2_ck1600 | 4 | 5 | 4 | 1 | 2 | 6 | 22 | 65% |

genzv4 checkpoint sweep (jobs 603027–603038):

| Checkpoint | BIO/5 | CRISIS/5 | FORMAT/4 | HELP/6 | MEM/8 | NOH/6 | Total/34 | % |
|---|---|---|---|---|---|---|---|---|
| **genzv4_ck200** | 2 | 5 | 4 | 1 | **4** | 6 | **22** | **65%** |
| genzv4_ck400 | 1 | 4 | 4 | **3** | 3 | 5 | 20 | 59% |
| genzv4_ck600 | **4** | 4 | 3 | 1 | 3 | 6 | 21 | 62% |
| genzv4_ck800–ck2400 | ... varies ... | | | | | | 17–22 | 50–65% |

---

## Key Findings

- **Benchmark variance:** Temperature=0.7 → 17/49 scenario flips per run. Always 3-run average.
- **Base beats SFT (v3):** Root cause was training format mismatch (c3acdc9). Expect recovery in genzv5.
- **DPO never helps:** All 3 DPO runs flat or worse. Current data + beta=0.1 + 800 steps insufficient.
- **COMPANION is the SFT win:** genzv2 models 92–94% vs base 44% — friend/casual data dominates.
- **CROSS_SESSION is the canary:** genzv2_ck1200 = 0% — completely ignores `[Recent sessions]`.
- **genzv3 degrades after ck200:** Sharp decline in HELP_MODE (4/6 → 1/6 by ck800).
- **Larger data + more steps didn't fix the Goldilocks zone problem:** genzv4 ck200 still best.

---

## Benchmark File Reference

| File | Purpose |
|---|---|
| `benchmarks/scenarios.py` | 58 scenarios, 9 categories, all LLM judge definitions |
| `benchmarks/run_benchmarks.py` | Main runner — loads both models, runs scenarios, writes results |
| `benchmarks/run_benchmarks.slurm` | SLURM wrapper (`gpu-long`, `a100-80`, 4h) |
| `benchmarks/average_results.py` | Averages multiple runs per model (`--since YYYYMMDD`) |
| `benchmarks/results/` | JSON + Markdown output per run (`<label>_<timestamp>.{json,md}`) |

---

## See also

- [[Models]] — full model inventory and adapter paths
- [[Training]] — how to run a new SFT job
