---
tags: [anchor, models]
---

# Models

← [[Home]]

---

## "Best model" — disambiguation

| Term | Model | Why |
|---|---|---|
| **Best overall (v4)** | `genzv3_ck200` | 61.5% on v4 — first SFT to beat base |
| **Best base (v4)** | `llama_base` | 57.4% on v4 (single run) |
| **Best genzv6 (v4)** | `genzv6_ck1600` | 56.7% on v4 — single run, below base/genzv3 |
| **Best overall (v3)** | `llama_base` | 51% on v3 — base instruction model, no fine-tuning |
| **Best SFT (v3)** | `genzv2_ck1200` | 44% v3, 71% v1 — best fine-tuned model (v3 era) |
| **Currently deployed** | `genzv2_ck1600` | ⚠️ outdated — should upgrade to genzv3_ck200 GGUF |
| **Android app default** | `genzv2_ck1200` | `ModelStore.ts` `lastUsedModelId` points here |

genzv3_ck200 is the best SFT on v4. genzv6 improves on genzv5 (+3.2pp) but doesn't yet beat genzv3 or base. Crisis data (jobs 619781–619783) expected to help — genzv6+crisis is the next training run.

---

## Leaderboard (v4 benchmark, 2026-05-23)

| Model | n | % | Notes |
|---|---|---|---|
| **genzv3_ck200** | 3 | **61.5%** | Best SFT overall |
| llama_base | 1 | 57.4% | ⚠️ 1-run only |
| genzv4_ck200 | 3 | 58.3% | |
| **genzv6_ck1600** | **1** | **56.7%** | **Best genzv6 — 1-run only** |
| genzv5_ck1400 | 3 | 53.5% | Peak v5 |
| genzv2_ck1600 | 3 | 51.5% | |
| genzv2_ck1200 | 3 | 51.3% | |

---

## Leaderboard (v3 benchmark, 3-run averaged, /61 weighted, COMPLETE)

| Model | Avg /61 | % | ± | n | GGUF |
|---|---|---|---|---|---|
| llama_base (no adapter) | 31.3 | **51%** | 1.9 | 3 | `mindmate_llama32_3b_q4_k_m.gguf` |
| genzv3_ck200 | 27.5 | 45% | 2.1 | 4 | `mindmate_genzv3_ck200_q4_k_m.gguf` |
| genzv2_ck1600 | 27.5 | 45% | 1.5 | 2 | `mindmate_genz_llama32_3b_q4_k_m.gguf` |
| **genzv2_ck1200** | **26.8** | **44%** | **0.8** | 4 | **`mindmate_genzv2_ck1200_q4_k_m.gguf`** ← best SFT |
| genzv4_ck200 | 25.7 | 42% | 1.2 | 3 | `mindmate_genzv4_ck200_q4_k_m.gguf` |

> ⚠️ Base model beats all SFT on v3. Root cause: training format mismatch — fixed in c3acdc9. genzv5 will be the real test.

**Scoring:** `/61` weighted — CRISIS weight=2, all others weight=1. 58 total scenarios. Judge: Gemma 4 26B A4B IT bfloat16. Methodology: 3-run averaged (temperature=0.7 causes ±17/49 scenario flips per run).

---

## Per-Category Breakdown (v3, 3-run averaged)

| Category | BASE | genzv3_ck200 | genzv2_ck1600 | **genzv2_ck1200** | genzv4_ck200 | Winner |
|---|---|---|---|---|---|---|
| COMPANION (9 scen) | 44% | 47% | **94%** | **92%** | 44% | SFT (v2 models) |
| FORMAT (4 scen) | 33% | 62% | **75%** | 56% | 50% | SFT |
| CONV_MEMORY (5 scen) | 13% | 35% | 30% | 30% | 33% | SFT |
| NO_HALLUCINATION (7 scen) | 67% | **89%** | 50% | 54% | 71% | SFT (v3 only) |
| CRISIS (8 scen, wt=2) | **67%** | 59% | 56% | 53% | 58% | **BASE** |
| HELP_MODE (8 scen) | **58%** | 34% | 38% | 44% | 25% | **BASE** |
| BIOMETRIC (5 scen) | **60%** | 25% | 20% | 30% | 40% | **BASE** |
| CONTEXT_MEMORY (8 scen) | **38%** | 16% | 12% | 9% | 21% | **BASE** |
| CROSS_SESSION_MEMORY (4 scen) | **83%** | 44% | 12% | **0%** | 42% | **BASE** |

**Key insight:** SFT wins COMPANION, FORMAT, CONV_MEMORY. Base wins everything memory/crisis-related. Root cause: training data was missing system prompt preamble, so model learned to respond to a different format than it sees at inference.

**CROSS_SESSION at 0% for genzv2_ck1200** is the smoking gun — model completely ignores `[Recent sessions]` header.

---

## Model Details

### genzv2_ck1200 — Best SFT

- **Adapter (cluster):** `adapters/genz/checkpoint-1200`
- **GGUF (local):** `exports/mindmate_genzv2_ck1200_q4_k_m.gguf`
- **GGUF (local, old name):** `exports/mindmate_genz_llama32_3b/` dir contains older version
- **HuggingFace:** `jainaryan/mindmate-gguf/mindmate_genzv2_ck1200_q4_k_m.gguf`
- **Training:** v2 data mix (42,038 raw, ~10k sampled), 1600 steps, fresh from base, A100-40
- **v3 avg:** 44% ±0.8 (n=4, most stable model)
- **v1 benchmark:** 71% (24/34) — CRISIS 5/5, FORMAT 4/4, NO_HALLUCINATION 6/6, MEMORY_USE 4/8
- **Pixel 8a:** ~5.4–6.0 TPS, TTFT 60–66s cold / 6s cached, heap ~3.12–3.17 GB
- **Strengths:** COMPANION 92%, stable across runs, good CRISIS
- **Weaknesses:** CROSS_SESSION 0%, CONTEXT_MEMORY 9%, BIOMETRIC 30% — all from format mismatch (fixed c3acdc9)

### genzv2_ck1600 — Deployed on Server (outdated)

- **Adapter (cluster):** `adapters/genz/checkpoint-1600`
- **GGUF (local):** `exports/mindmate_llama_sft_ck1600/llama(genz)v2_q4_k_m.gguf`
- **HuggingFace:** `jainaryan/mindmate-gguf/llama(genz)v2_q4_k_m.gguf`
- **v3 avg:** 45% ±1.5 (n=2 — 3rd run may have timestamp-collided)
- **Strengths:** COMPANION **94%** (best of any model), 9/9 on companion scenarios
- **Weaknesses:** CROSS_SESSION 12%, CONTEXT_MEMORY 12%, BIOMETRIC 20%
- ⚠️ **Server (`deploy/config.py`) still points to this.** Should upgrade to genzv2_ck1200.

### genzv3_ck200 — Best Non-genzv2

- **Adapter (cluster):** `adapters/genzv3/checkpoint-200`
- **GGUF (local):** `exports/mindmate_genzv3_ck200_q4_k_m.gguf`
- **Training:** v3 data mix (~10k), 1600 steps total, best checkpoint at ck200 (degrades rapidly after)
- **v3 avg:** 45% ±2.1 (n=4, highest variance model)
- **Strengths:** NO_HALLUCINATION **89%** (7/7 in single runs), good FORMAT
- **Weaknesses:** Degrades sharply after ck200; COMPANION only 47%; HELP_MODE 34%

### genzv4_ck200

- **Adapter (cluster):** `adapters/genzv4/checkpoint-200`
- **GGUF (local):** `exports/mindmate_genzv4_ck200_q4_k_m.gguf`
- **Training:** v4 data mix (21,529 examples), 2400 steps, best checkpoint at ck200 (same as v3)
- **v3 avg:** 42% ±1.2 (n=3)
- **Weaknesses:** Did not beat genzv2_ck1200 despite bigger data + more steps. HELP_MODE only 25%.

### llama_base (no adapter)

- **GGUF (local):** `exports/mindmate_llama32_3b_q4_k_m.gguf`
- **v3 avg:** 51% ±1.9 (n=3) — **best on v3, beats all SFT**
- Strong on CROSS_SESSION (83%), CRISIS (67%), HELP_MODE (58%), BIOMETRIC (60%)
- Weak on COMPANION (44%), FORMAT (33%), CONV_MEMORY (13%)

---

### genzv6_ck1600 — Best genzv6

- **Adapter (cluster):** `adapters/genzv6/checkpoint-1600`
- **GGUF:** not yet exported
- **Training:** v6 data mix (data quality fixes: dropped biometric_qwen_s0 missing preamble 993ex, biometric_qwen_s1/s2 800-char verbosity, conv_memory_gemma4 therapy-speak; no crisis data in mix), 2000 steps
- **v4 result:** 56.7% (32/83) — single run 2026-05-23
- **Status vs prior best:** +3.2pp over genzv5 peak (53.5%), still below llama_base (57.4%) and genzv3_ck200 (61.5%)
- **Next step:** 3-run average on ck1600; add crisis data → genzv6+crisis SFT

**genzv6 full checkpoint sweep (v4 benchmark, single runs, 2026-05-23):**

| Checkpoint | Pass/Total | % |
|---|---|---|
| genzv6_ck200 | 31/83 | 53.2% |
| genzv6_ck400 | 22/83 | 47.5% |
| genzv6_ck600 | 25/83 | 50.0% |
| genzv6_ck800 | 25/83 | 49.3% |
| genzv6_ck1000 | 26/83 | 49.6% |
| genzv6_ck1200 | 32/83 | 54.6% |
| genzv6_ck1400 | 29/83 | 53.9% |
| **genzv6_ck1600** | **32/83** | **56.7%** ← best |
| genzv6_ck1800 | 31/83 | 55.7% |
| genzv6_ck2000 | 30/83 | 55.0% |

Pattern: performance dips at ck400 (possible early instability), recovers by ck1200, peaks ck1600, then slowly decays — same Goldilocks behavior as genzv3/genzv4/genzv5.

---

## DPO Models — Abandoned ❌

All 3 DPO runs flat or worse than their SFT base. Current data + beta=0.1 + 800 steps insufficient.

| Model | SFT Base | v1 Benchmark | vs SFT | Notes |
|---|---|---|---|---|
| `genzv2_dpo_ck1200` (job 603039) | genzv2_ck1200 (71%) | 59% | **−12%** ❌ | HELP_MODE collapsed to 0/6 |
| `genzv3_dpo_ck200` (job 603040) | genzv3_ck200 (76%/68%) | 68% | = | No improvement |
| `genz_dpo_ck1600` (job 603100) | genzv2_ck1600 (65%) | 65% | = | No improvement |

DPO consistently hurts HELP_MODE and sometimes BIOMETRIC. Pattern: model becomes more hesitant.

---

## GGUF Inventory (local `exports/`)

| File | Model | v1 % | v3 % | Notes |
|---|---|---|---|---|
| `mindmate_genzv2_ck1200_q4_k_m.gguf` | genzv2_ck1200 | 71% | 44% | **Best SFT** |
| `mindmate_genzv3_ck200_q4_k_m.gguf` | genzv3_ck200 | 76% | 45% | Best NO_HALLUCINATION |
| `mindmate_genzv4_ck200_q4_k_m.gguf` | genzv4_ck200 | 65% | 42% | |
| `mindmate_llama32_3b_q4_k_m.gguf` | llama_base | — | 51% | Beats all SFT on v3 |
| `mindmate_llama_sft_ck1600/` | genzv2_ck1600 | 65% | 45% | Served at tryanchor.me |
| `mindmate_genz_llama32_3b_q4_k_m.gguf` | genzv2_ck1600 | 65% | 45% | Old name for ck1600 |
| `mindmate_llama32_3b_f16.gguf` | llama_base F16 | — | — | Full precision, large |
| `mindmate_qwen3_1p7b_q4_k_m.gguf` | Qwen3-1.7B base | — | — | Historical |

---

## Adapter Inventory (cluster `adapters/`)

| Directory | Model | Training | Status |
|---|---|---|---|
| `adapters/genz/checkpoint-1200` | genzv2_ck1200 | v2 mix, 1600 steps | Best v3-era SFT |
| `adapters/genz/checkpoint-1600` | genzv2_ck1600 | v2 mix, 1600 steps | Deployed (outdated) |
| `adapters/genzv3/checkpoint-200` | genzv3_ck200 | v3 mix, 200 steps | **BEST OVERALL (v4: 61.5%)** |
| `adapters/genzv4/checkpoint-200` | genzv4_ck200 | v4 mix, 200 steps | Active |
| `adapters/genzv5/checkpoint-1400` | genzv5_ck1400 | v5 mix, 14,373 ex, 2000 steps | Best v5 (53.5% v4) |
| `adapters/genzv6/checkpoint-1600` | genzv6_ck1600 | v6 mix, 12,729 ex, 2000 steps | **Best genzv6 (56.7% v4, 1 run)** |
| `adapters/genzv2_continued/` | genzv2_continued | v2_continued mix | ❌ Broken MEMORY_USE |
| `adapters/genz_dpo_ck1600/` | genz_dpo_ck1600 | DPO | ❌ Abandoned |
| `adapters/genzv2_dpo_ck1200/` | genzv2_dpo_ck1200 | DPO | ❌ Abandoned |
| `adapters/genzv3_dpo_ck200/` | genzv3_dpo_ck200 | DPO | ❌ Abandoned |
| `adapters/CUDA_mindmate_llama32b/` | Llama v2 SFT (old) | v2 mix (old) | Historical |
| `adapters/CUDA_mindmate_qwen25_3b/` | Qwen2.5-3B SFT | v2 mix | Historical (inferior) |

---

## v1 Benchmark Results (34 scenarios, LLM judge — historical reference)

| Checkpoint | BIO/5 | CRS/5 | FMT/4 | HLP/6 | MEM/8 | NOH/6 | Total/34 | % |
|---|---|---|---|---|---|---|---|---|
| **genzv3_ck200** | 4 | 4 | 4 | 4 | 4 | 6 | **26** | **76%** |
| **genzv2_ck1200** | 3 | 5 | 4 | 2 | **4** | 6 | **24** | **71%** |
| genzv4_ck200 | 2 | 5 | 4 | 1 | 4 | 6 | 22 | 65% |
| genzv2_ck1600 | 4 | 5 | 4 | 1 | 2 | 6 | 22 | 65% |
| genzv2_dpo_ck1200 | 3 | 4 | 4 | **0** | 3 | 6 | 20 | 59% ❌ |

Note: v1 judge was Qwen3-30B. v2+ uses Gemma4 26B A4B. Scores not directly comparable across benchmark versions.

---

## MODEL_SHORTCUTS in `benchmarks/run_benchmarks.py`

```python
MODEL_SHORTCUTS = {
    "llama_base":       ("meta-llama/Llama-3.2-3B-Instruct", None),
    "genzv2_ck1600":    ("meta-llama/Llama-3.2-3B-Instruct", "adapters/genz/checkpoint-1600"),
    "genzv2_ck1200":    ("meta-llama/Llama-3.2-3B-Instruct", "adapters/genz/checkpoint-1200"),
    "genzv3_ck200":     ("meta-llama/Llama-3.2-3B-Instruct", "adapters/genzv3/checkpoint-200"),
    "genzv4_ck200":     ("meta-llama/Llama-3.2-3B-Instruct", "adapters/genzv4/checkpoint-200"),
    "llama_dpo_ck1600": ("meta-llama/Llama-3.2-3B-Instruct", "adapters/genz_dpo_ck1600"),
    "llama_ck200":      ("meta-llama/Llama-3.2-3B-Instruct", "adapters/CUDA_mindmate_llama32b/checkpoint-200"),
    "qwen25_3b":        ("Qwen/Qwen2.5-3B-Instruct",         "adapters/CUDA_mindmate_qwen25_3b/checkpoint-200"),
    "gemma4_4b":        ("google/gemma-4-4b-it",             None),
}
```

To add a new model (e.g. genzv5), add entry here then benchmark with `--model genzv5_ck800`.

---

## See also

- [[Benchmarks]] — full methodology and all run results
- [[Training]] — how adapters are trained
- [[Next Steps]] — genzv5 plan
