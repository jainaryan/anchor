---
tags: [anchor, next-steps]
---

# Next Steps

← [[Home]]

---

## What just landed (2026-05-22)

✅ **v4 benchmark complete** for all models (genzv2–genzv5, llama_base). Results in [[Bug Log]] → 2026-05-22. Key finding: genzv3_ck200 (61.5%) beats llama_base (57.4%) — SFT can beat base. But genzv5 peaks at 53.5%, well below genzv3. Root causes fully diagnosed via data quality audit — see [[Data]] → "Data Quality Audit".

✅ **Data quality audit complete** — all Qwen pipeline data dropped (uniformly too long: 529–800 char avg vs 100–250 target). v6 preset defined. See [[Data]] → "v6 data mix".

✅ **genzv6 SFT submitted** — job 619894, A100-80, 2000 steps, 12,729 examples. v3 base proportions + help_mode_qwen (markdown-stripped). Crisis slots in preset, waiting on jobs 619781–619783.

✅ **Crisis heuristic fixed** — 3 shards running (619781–619783). Previous job produced only 5 examples from 48h; fix unlocks ~hundreds per shard. See [[Bug Log]] 2026-05-22.

✅ **llama_base v4 benchmark 3-run average** — runs 2+3 submitted (619895–619896). 1 run already complete (57.4%). Proper average needed before ranking vs genzv6.

---

## What just landed (2026-05-16)

✅ **v4 benchmark infrastructure complete** across both repos. Outstanding work is now data + training, not infrastructure. See [[Benchmarks]] → "Suite v4".

What this changes for genzv5:
- Use `benchmarks/run_benchmarks_v4.slurm` (not the legacy `run_benchmarks.slurm`)
- Scripted scenarios are deterministic at temp=0 — no need for 3-run averaging on those
- Per-scenario diff (`benchmarks/diff_results.py --regressions-only`) is the release gate
- Test the actual GGUF before shipping (`--backend gguf`) — surfaces quantization regressions
- Mobile re-run on genzv2_ck1200 should use the new `EvalRunnerV4` + `scripts/sync_mobile_eval.sh all`

---

## Immediate — Waiting on Cluster

### 1. genzv5 data mix status (2026-05-17)

**Done — ready to include:**

| File | Rows | Notes |
|---|---|---|
| `synthetic_train_targeted_fix.jsonl` | 13,524 | help_mode + memory_recall, normalized |
| `synthetic_train_friend_1.jsonl` | 7,380 | friend tone, normalized |
| `synthetic_train_transition.jsonl` | 6,184 | casual→emotional pivot, normalized |
| `synthetic_train_casual.jsonl` | 5,000 | non-distress casual, normalized |
| `synthetic_train.jsonl` | 5,565 | grief/loss, normalized |
| `synthetic_train_therapist_.jsonl` | 2,637 | therapeutic, normalized |
| `synthetic_train_targeted_fixes.jsonl` | 181 | gold examples, 100% inclusion |
| `synthetic_train_conv_memory_qwen.jsonl` (pre-shard) | 299 | ready |
| `synthetic_train_conv_memory_qwen_s0.jsonl` | 335 | ready |
| `synthetic_train_conv_memory_qwen_s1.jsonl` | 316 | ready |
| `synthetic_train_conv_memory_qwen_s2.jsonl` | 324 | ready |
| `synthetic_train_biometric_qwen_s0.jsonl` | 993 | ready |
| `synthetic_train_biometric_qwen_s1.jsonl` | 80 | ready (short — less wall time) |
| `synthetic_train_biometric_qwen_s2.jsonl` | 78 | ready (short — less wall time) |
| `synthetic_train_biometric.jsonl` (old Gemma4) | 2,348 | keep — additive to Qwen biometric |
| `synthetic_train_conv_memory.jsonl` (old Gemma4) | 167 | keep — additive to Qwen conv-memory |
| **Conv-memory total (done)** | **1,441** | 1,274 Qwen + 167 Gemma4 |
| **Biometric total (done)** | **3,499** | 1,151 Qwen + 2,348 Gemma4 |

**Generating now (active cluster jobs):**

| Jobs | What | Expected rows | ETA |
|---|---|---|---|
| 615485–615490 | conv-memory new-pool shards 3–8 (`PROFILE_SET=new`) | ~1,800–3,000 | ~72h from 2026-05-17 |
| 615491–615493 | biometric shards 3–5 | ~2,900 (s0 throughput × 3) | ~72h from 2026-05-17 |
| **615518** | help_mode pipeline (HF backend, xgpi2) | ~205 | 48h from 2026-05-19 |
| **616643** | crisis pipeline (HF backend, xgpi17 H100-47) | TBD | 48h from 2026-05-19 |
| ~~617977~~ | genzv5 SFT — FAILED (root cause unknown at time): cu130 not replaced | — | — |
| ~~618080~~ | genzv5 SFT — FAILED: same; pip went to miniconda3 not venv | — | — |
| ~~618086~~ | genzv5 SFT — FAILED: shim got further; manual_seed cb hit empty default_generators | — | — |
| ~~618199~~ | genzv5 SFT — cancelled before audit | — | — |
| ~~618200~~ | genzv5 SFT — FAILED: comprehensive shim got past Trainer.init, crashed at `model._apply` → `t.to(device)` (cu130 still active — pip bug) | — | — |
| ~~618207~~ | genzv5 SFT — FAILED instantly: `import pip` ModuleNotFoundError (venv has no pip!) | — | — |
| **618208** | genzv5 SFT — added `python -m ensurepip --upgrade` to bootstrap pip into venv first | adapters/genzv5 | 24h from 2026-05-21 |

Note: vLLM is incompatible with the cluster's CUDA driver (12.0.90). All data-gen jobs use HF backend (`USE_VLLM=0`). Crisis stall bug fixed — Qwen3 was refusing SI content with the generic datagen system message; now uses `_CRISIS_DATAGEN_SYSTEM` research-context override. See Bug Log 2026-05-19.

⚠️ **Crisis heuristic issue (job 616643) — 511+ attempts, only 5 examples pass (2026-05-20):** The heuristic is blocking virtually all output. The conversations generate fine (safety-refusal fix works) but fail checks for: clinical language absence, direct acknowledgement, not opening with deflection. Qwen3's style likely doesn't match the expected patterns. **Action needed:** loosen the heuristic thresholds, or inspect a raw failing example to understand what's being rejected. The job has ~16h left — if still ~0% pass rate, cancel and relaunch with a fixed heuristic.

**Conv-memory new-pool shards (615485–615490) — finishing ~4h, very low yield:**
- s3–s8: 166 total examples (vs 320+/shard for s0–s2). Likely cause: HF backend sequential generation is much slower; `PROFILE_SET=new` uses 28 profiles with 47 facts (more complex). Don't relaunch — merge what we have into v5 mix as additive signal.

**Biometric shards (615491–615493) — finishing ~4h:**
- s3–s5: 199 total examples (89+52+58). Same throughput issue. Merge as additive to s0–s2.

**On hold (do not include in genzv5 yet):**

| File | Rows | Reason |
|---|---|---|
| `synthetic_train_conv_memory_overref_qwen_s*.jsonl` | 29 (~12 false pos + 17 genuine) | See Bug Log 2026-05-17 — revisit as DPO after genzv5 |
| `synthetic_train_conv_memory_overref_qwen_s*.jsonl` | 29 (~12 false pos + 17 genuine) | See Bug Log 2026-05-17 — revisit as DPO after genzv5 |

### 2. ✅ Crisis + help_mode data generation — now running

Jobs 616643 (crisis, xgpi17) and 615518 (help_mode, xgpi2) are running with HF backend (`USE_VLLM=0`). Crisis had a stall bug (Qwen3 safety refusal on SI content) — fixed in 2026-05-19 commit. No action needed until they finish.

### 3. Merge shards once done

```bash
# Conv-memory (all shards including new-pool)
cat data/synthetic_train_conv_memory_qwen*.jsonl > data/synthetic_train_conv_memory_qwen_merged.jsonl

# Biometric (all shards)
cat data/synthetic_train_biometric_qwen_s*.jsonl > data/synthetic_train_biometric_qwen_merged.jsonl
```

---

## genzv5 — The Priority Run

genzv5 will be the **first training run with correct system prompt format** (c3acdc9 applied). This is the real test of whether SFT can beat base model on memory categories.

### ✅ Step 1: Add v5 preset to `finetuning/build_dataset.py` — DONE

v5 preset is in `DATA_MIX_PRESETS`. See [[Training]] → "v5" for exact file/count/weight table. 13,988 examples, 2000 steps.

### ✅ Step 2: Create v5 SLURM script — DONE

`finetuning/run_sft_v5.slurm` exists. A100-80, 24h, 2000 steps → `adapters/genzv5`.

### ✅ Step 3: Sync to cluster and submit — DONE (early run)

**Job 617977** submitted 2026-05-20, PENDING on A100-80 (`gpu-long` partition, 24h). H200 was incompatible — driver 575 (CUDA 12.9) vs cu130 runtime (CUDA 13.0). A100-80 has driver 580 (CUDA 13.0), fully compatible. Full CUDA shim applied (see Bug Log 2026-05-20). This is the early run with 13,988 examples; full v5 re-run to follow once conv-memory/biometric shards finish (~24h from 2026-05-20).

### Step 4: Benchmark on v4 suite

genzv5 shortcuts are already wired into `benchmarks/run_benchmarks_v4.py::MODEL_SHORTCUTS` (ck200/400/600/800). For other checkpoints, edit `MODEL_SHORTCUTS` first.

**v4 uses production sampling (temp=0.7, top_p=0.95, top_k=40, min_p=0.05) so every scenario type is non-deterministic. Always do 3-run averaging for leaderboard rankings.**

```bash
# 3 runs for stable averaging across scripted + dynamic scenarios
for i in 1 2 3; do
  sbatch --gres=gpu:a100-80:1 \
      --export=ALL,MODEL=genzv5_ck200,LABEL=genzv5_ck200_run${i} \
      benchmarks/run_benchmarks_v4.slurm
done

# Quick single-run sanity check (variance is real — don't rank on this)
sbatch --gres=gpu:a100-80:1 \
    --export=ALL,MODEL=genzv5_ck200 \
    benchmarks/run_benchmarks_v4.slurm

# Per-scenario diff vs current best SFT — this is the release gate
python -m benchmarks.diff_results \
    benchmarks/results/cluster_genzv2_ck1200_*/judged.json \
    benchmarks/results/cluster_genzv5_ck200_*/judged.json \
    --regressions-only

# Unified leaderboard across all checkpoints
python -m benchmarks.leaderboard --since 20260516
```

Run a checkpoint sweep: genzv5_ck200, ck400, ck600, ck800, ck1000, ck1200 — find the Goldilocks zone (v3 and v4 both peaked at ck200).

### Step 5: Test the GGUF before shipping

Export ck of choice → Q4_K_M GGUF → benchmark the GGUF (not the NF4 adapter):

```bash
# Export
python scripts/export_gguf_cuda.py --adapter adapters/genzv5/checkpoint-200 \
    --out exports/mindmate_genzv5_ck200_q4_k_m.gguf

# Benchmark the actual shipping artifact
sbatch --gres=gpu:a100-80:1 \
    --export=ALL,GGUF_PATH=exports/mindmate_genzv5_ck200_q4_k_m.gguf,LABEL=genzv5_ck200_gguf \
    benchmarks/run_benchmarks_v4.slurm

# Compare NF4 vs GGUF on the same scenarios — flags any quantization regression
python -m benchmarks.leaderboard --diverge-only
```

### Expected outcomes

| Category | Current best SFT | Base | genzv5 target | Why |
|---|---|---|---|---|
| CROSS_SESSION_MEMORY | 0% (ck1200) | 83% | >60% | Conv-memory data trains explicit `[Recent sessions]` use |
| CONTEXT_MEMORY | 9% (ck1200) | 38% | >25% | Conv-memory + format fix teach profile field reference |
| BIOMETRIC | 30% (ck1200) | 60% | >45% | Biometric SFT data (generating now) |
| COMPANION | 92% (ck1200) | 44% | >85% | Friend data still in mix |
| FORMAT | 56% (ck1200) | 33% | >60% | Format data stays in mix |
| CONVERSATION_MEMORY | 30% (ck1200) | 13% | >35% | Conv-memory data reinforces in-session recall |
| **CRISIS** | **48% (ck1200)** | **67%** | **>55%** | ⚠️ Risk — see gap analysis below |
| **HELP_MODE** | **44% (ck1200)** | **58%** | **>50%** | ⚠️ Gap — see below |
| **NO_HALLUCINATION** | **54% (ck1200)** | **67%** | **>70%** | ⚠️ Need anti-hallucination data in mix |

### Known gaps in the current genzv5 data plan

These are **not addressed** by the conv-memory + biometric pipeline. Add before finalising the v5 data mix:

**1. CRISIS — actively degraded by SFT (67% → 48%)**
The friend-voice training overrides safety tone. Current genzv5 plan has no crisis-specific examples.
- Add 20–30 crisis SFT examples to the mix: passive SI, active SI, humor deflection, escalation across turns
- Format: must use production system prompt preamble + have the model respond as Anchor (not as a therapist)
- Priority: `cr_03` ('Better off without me') and `cr_07` (humor deflection) are coin-flip at 50% — most dangerous

**2. HELP_MODE — probe pattern entrenched (58% → 25–44%)**
Model learned to probe ("what's the worst part?") from therapeutic data instead of giving immediate technique.
- The 12 existing `help_cold_open` gold examples helped ck1200 to 44% but still 14pp below base
- Add 20+ more HELP_MODE examples — specifically cases where user opens with explicit help request
- Vary: "help me breathe", "i need to calm down", "what do i do", "i'm panicking"
- Constraint: technique must be offered in the FIRST model turn, not after a probe question

**3. NO_HALLUCINATION — data-mix-dependent (genzv3 89%, genzv2 54%)**
The 8 anti-hallucination gold examples landed only in genzv3's mix. genzv2/genzv4 omit them → 54% (worse than base 67%).
- Include `synthetic_train_targeted_fixes.jsonl` explicitly at 100% in v5 mix (it's only 181 examples)
- Do not downsample the gold examples — they're already small and high-signal

**4. Mobile eval re-run — stale since April 2026**
The in-app eval (`EVAL_RESULTS.md`) was run on genzv2_ck1600 before c3acdc9 and before all targeted fix data. Results are not representative of genzv2_ck1200.
- **Action**: re-run the in-app eval against genzv2_ck1200 Q4_K_M on Pixel 8a
- Expect improvements on: memory_recall_wedding, hallucination scenarios, coping_from_profile
- May still fail: cross_session_coping_outcome_followup (CROSS_SESSION 0% on cluster)
- Critical: re-run crisis scenarios — cluster shows 48% for ck1200, need to verify on-device

---

## Production Upgrade

Still serving genzv2_ck1600 at tryanchor.me. Should upgrade to genzv2_ck1200 (currently best).

See [[Production]] → "Upgrade Model to genzv2_ck1200" for exact commands.

---

## Open Questions

### 1. Will the format fix alone close the gap?

The base model passes CROSS_SESSION at 83% with zero training on it — instruction-following is intact. genzv5 will have conv_memory data that trains on cross-session context use, but it's unclear if 20% of conv_memory is enough. If memory categories don't recover, consider:
- Increasing conv_memory weight to 30-40%
- Generating more conv_memory data (rerun the pipeline for another 72h)

### 2. Optimal checkpoint for genzv5

v3 peaked at ck200 (10k examples). v4 peaked at ck200 (21k examples). More data did not push the Goldilocks zone later. genzv5 may also peak early. Run the full sweep and don't assume a higher checkpoint is better.

### 3. DPO worth reconsidering?

DPO was tried with corrected-format SFT data (pre-c3acdc9). If genzv5 SFT recovers base categories, DPO might be more useful. Would need: format-corrected DPO data, longer training (1200+ steps), possibly higher beta. Not a priority until genzv5 results are in.

Specific DPO candidate: crisis scenarios. If genzv5 SFT still drops CRISIS below base (67%), a DPO pass with chosen=crisis-safe / rejected=probe-or-deflect responses could rebalance without hurting COMPANION. The overref filter files (`synthetic_train_conv_memory_overref_qwen_s*.jsonl`) may also be useful rejected examples for a memory-specific DPO pass.

### 4. Per-category targeted data

Currently the model learns crisis/help/biometric from style examples. Could generate:
- More multi-turn conv_memory examples with `[Recent sessions]` explicitly referenced
- More biometric SFT examples that combine health context with emotional support
- HELP_MODE examples where the user explicitly says "help me calm down" in turn 2+ (not just turn 1)

### 5. Server upgrade + performance

DigitalOcean c-4 does CPU inference only. If tryanchor.me traffic grows, consider upgrading to a GPU droplet or switching to HuggingFace Spaces for inference.

---

---

## anchor-app Code Quality (session 5 — partially done)

### Done (2026-05-10)

- ✅ **Panic detection tiered** — `'urgent'` blocks, `'watch'` notifies only
- ✅ **Session-extraction race condition** — snapshot at schedule time
- ✅ **Person-name casing bug** — intro regex now matches regardless of case
- ✅ **Memory retrieval** — summary text included in search, recency bonus
- ✅ **Prompt token budgeting** — per-tier char caps in contextBuilder
- ✅ **Eval prompt drift** — EvalRunner defaults to production prompt

### Still to do

- ⬜ **Unify profile storage** — `profileStorage.ts` (AsyncStorage) and `MemoryRepository.ts` (WatermelonDB) are separate stores. Profile UI writes to AsyncStorage; inference memory reads from WatermelonDB. These can diverge. Migrate to one canonical store.
- ⬜ **Unit tests for critical memory path** — `contextBuilder`, `sessionExtractor`, `panicDetection`, `profileExtractor` have no focused tests. Add unit tests + regression fixtures. These directly affect user safety.
- ⬜ **Privacy hardening** — Profile and diary stored in plaintext local storage. Consider encrypted at-rest option or keychain-backed "privacy lock" mode.

### UX revamps to implement

- ⬜ **Calmer chat home** — move secondary controls (mode pills, thinking toggle, profile suggestion actions) behind a collapsed `Session tools` drawer. First-use/default view should show only mood check-in, input, and safe empty-state.
- ⬜ **Passive diary auto-capture** — replace persistent "Save to diary" button flow with session-end auto-generated reflection draft + lightweight `Review & Save` entry point into `DiaryEditor`.
- ⬜ **Progressive profile onboarding** — split `ProfileSetupScreen` into `2-minute essentials` and `optional deep profile` so first-run completion is faster and lower friction.

---

## See also

- [[Training]] — how to run SFT
- [[Benchmarks]] — how to run the benchmark suite
- [[Data]] — data files and conv_memory pipeline status
- [[Cluster]] — SLURM and sync commands
