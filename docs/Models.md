---
tags: [anchor, models]
---

# Models

← [[Home]]

---

## Leaderboard (v3 benchmark, 3-run averaged, /61 weighted)

| Model | Avg % | ± | n | GGUF |
|---|---|---|---|---|
| llama_base (no adapter) | **51%** | 1.9 | 3 | `mindmate_llama32_3b_q4_k_m.gguf` |
| genzv3_ck200 | 45% | 2.1 | 4 | `mindmate_genzv3_ck200_q4_k_m.gguf` |
| genzv2_ck1600 | 45% | 1.5 | 2 | `mindmate_genz_llama32_3b_q4_k_m.gguf` |
| **genzv2_ck1200** | **44%** | 0.8 | 4 | **`mindmate_genzv2_ck1200_q4_k_m.gguf`** ← best |
| genzv4_ck200 | 42% | 1.2 | 3 | `mindmate_genzv4_ck200_q4_k_m.gguf` |

> ⚠️ Base still beats all SFT on v3. Root cause: training format mismatch — fixed in c3acdc9. genzv5 will be the real test.

### v1 benchmark (34 scenarios, LLM judge — historical)

| Model | % | Notes |
|---|---|---|
| genzv3_ck200 | 76% | Best SFT on v1 — early over-fit after ck200 |
| **genzv2_ck1200** | **71%** | Most stable; best MEMORY_USE (4/8) |
| genzv4_ck200 | 65% | Doesn't beat ck1200 despite bigger data |
| genzv2_ck1600 | 65% | Better companion; worse memory |
| *DPO models* | 59–68% | Flat or worse than SFT base — DPO abandoned |

---

## genzv2 ck1200 — Best SFT

- **Adapter:** `adapters/genz/checkpoint-1200`
- **GGUF:** `exports/mindmate_genzv2_ck1200_q4_k_m.gguf`
- **HF:** `jainaryan/mindmate-gguf/mindmate_genzv2_ck1200_q4_k_m.gguf`
- **Pixel 8a:** ~5.4–6.0 TPS, TTFT 6s cached / 60–66s cold, heap ~3.15 GB
- **Strengths:** CRISIS 5/5, FORMAT 4/4, NO_HALLUCINATION 6/6, MEMORY_USE 4/8
- **Weaknesses:** CROSS_SESSION_MEMORY 0% (fixed by c3acdc9 — needs re-eval on genzv5)

## genzv3 ck200

- **Adapter:** `adapters/genzv3/checkpoint-200`
- **GGUF:** `exports/mindmate_genzv3_ck200_q4_k_m.gguf`
- **Strengths:** Best NO_HALLUCINATION (7/7 on v3), good FORMAT
- **Weaknesses:** Degrades rapidly after ck200; COMPANION lower than genzv2

## genzv2 ck1600 — Deployed on Server

- **Adapter:** `adapters/genz/checkpoint-1600`
- **GGUF:** `exports/mindmate_llama_sft_ck1600/` (served at tryanchor.me)
- ⚠️ Should upgrade server to genzv2_ck1200 GGUF

## DPO Models — Abandoned ❌

All 3 runs (genzv2_dpo_ck1200, genzv3_dpo_ck200, genzv2_dpo_ck1600) were flat or worse than SFT base. DPO with current data + beta=0.1 + 800 steps does not work for this task.

---

## Per-Category Breakdown (v3, averaged)

| Category | BASE | genzv3 | genzv2_ck1600 | **genzv2_ck1200** | genzv4 | Winner |
|---|---|---|---|---|---|---|
| COMPANION | 44% | 47% | **94%** | **92%** | 44% | SFT (v2) |
| FORMAT | 33% | 62% | **75%** | 56% | 50% | SFT |
| CONV_MEMORY | 13% | 35% | 30% | 30% | 33% | SFT |
| NO_HALLUCINATION | 67% | **89%** | 50% | 54% | 71% | SFT (v3) |
| CRISIS | **67%** | 59% | 56% | 53% | 58% | **BASE** |
| HELP_MODE | **58%** | 34% | 38% | 44% | 25% | **BASE** |
| BIOMETRIC | **60%** | 25% | 20% | 30% | 40% | **BASE** |
| CONTEXT_MEMORY | **38%** | 16% | 12% | 9% | 21% | **BASE** |
| CROSS_SESSION | **83%** | 44% | 12% | **0%** | 42% | **BASE** |

CROSS_SESSION_MEMORY at 0% for genzv2_ck1200 is the smoking gun — confirmed root cause was training format mismatch (c3acdc9 fix).

---

## See also

- [[Benchmarks]] — full methodology and all run results
- [[Training]] — how adapters are trained
- [[Next Steps]] — genzv5 plan
