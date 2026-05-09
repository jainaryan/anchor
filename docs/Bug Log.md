---
tags: [anchor, bugs]
---

# Bug Log

← [[Home]]

Chronological record of bugs found and fixed. Most recent first.

---

## 2026-05-09 (Session 4)

**Training format mismatch — root cause of base-beats-SFT**
All 42,038 training examples were missing the production system prompt preamble. `targeted_fix` + `biometric` had truncated `[User]`/`[Recent sessions]`-only system prompts; all other files had no system message. Model learned to respond to the wrong format. Fixed: commit c3acdc9.

**Benchmark variance discovered**
`temperature=0.7` on eval model → 17/49 scenarios flip PASS↔FAIL between runs. Fixed: adopted 3-run averaged methodology. Use `benchmarks/average_results.py --since YYYYMMDD`.

**H100-96 GRES falls back to ~46GB nodes**
Jobs 609078–609083 OOMed: `GPU 0 has a total capacity of 46.38 GiB` — Gemma4 needs ~52GB. Fix: always use `--gres=gpu:a100-80:1` explicitly.

**GitHub SSH blocked from cluster**
`git pull` fails with `ssh.github.com port 443: Connection timed out`. Fix: use rsync for all cluster↔local file sync.

**`average_results.py` KeyError `'weighted_pass'`**
Result JSON uses nested `overall.weighted_pass` not top-level. Fixed: access `d['overall']['weighted_pass']` with fallback.

**`llama_ck1600` shortcut name misleading**
It is the genzv2 SFT adapter. Fixed: renamed to `genzv2_ck1600` in MODEL_SHORTCUTS and docs.

**`conversation_memory_pipeline.py` v1 was not anchor-aligned**
Teacher generated full conversation JSON without being constrained by anchor prompt. Training signal was generic Gemma4 style. Fixed: refactored to teacher-as-Anchor two-phase approach.

**Timestamp collision in benchmark results**
Two genzv4_ck200 jobs finished at identical second → same output filename, one overwrote the other. Non-critical (one run captured).

---

## 2026-05-07 (Session 3)

**`sbatch --export` placed after script path**
`sbatch script.sh --export=MODEL=foo` silently treats `--export` as a script argument. 5 jobs (607691–607695) all used wrong model. Fix: always place flags BEFORE script path: `sbatch --export=ALL,MODEL=foo script.sh`.

---

## 2026-05-02–03 (Session 2/3)

**`CUDNN_STATUS_NOT_INITIALIZED` on H200/H100 nodes**
During `scaled_dot_product_attention` → benchmark Phase 1 crashed on jobs 602438–602452. Fix: added `attn_implementation="eager"` to eval model load in `run_benchmarks.py`.

**A100-40 OOM loading Gemma4 judge**
After eval model `del` + `empty_cache`, VRAM still nearly full (4.5MB free of 39.49GB). Fix: added `model.cpu(); base.cpu()` before `del` to force VRAM release; switched to A100-80.

**Remote ssh sbatch variable expansion**
`ssh host "sbatch --export=ADAPTER=${ck}..."` — `$ck` expands locally (empty). Fix: use single quotes for remote command.

**`--wrap` uses `/bin/sh`**
`source mindmatenv/activate` fails with "source: not found". Fix: `--wrap="bash -c \"source mindmatenv/...\""`.

**`DPOConfig.__init__()` unexpected argument `max_prompt_length`**
Removed from TRL's DPOConfig on cluster. Fix: removed `max_prompt_length=1024`.

**`argparse` missing `genzv2_ck1600` choice in DPO trainer**
Fix: added `genzv2_ck1600` to argparse choices and CONFIGS dict.

---

## 2026-05-01 (Session 2)

**A100-80 node xgpj0 torch import fails**
`libtorch_global_deps.so: No such file or directory` — CUDA libs not in LD_LIBRARY_PATH on that specific node. Fix: exclude xgpj0, use other A100-80 nodes.

**`ends_question` check too strict**
Failed when model appended trailing non-question clause after the `?`. Fix: check `"?" in response` instead of `response.rstrip().endswith("?")`.

**MEMORY_USE + BIOMETRIC checks were keyword-based**
Failed on semantically correct responses that used different wording. Fix: replaced with LLM judge (Qwen3-30B → later Gemma4 26B A4B).

---

## 2026-04-26

**`synthetic/utils.py` missing `import random` and `import re`**
`randomize_health_context()` crashed on every call → biometric pipeline (job 595713) generated 0 pairs in 22h. Fix: added both imports.

**DPO `format_messages()` format mismatch**
Used `tokenizer.apply_chat_template` for `llama_ck1600` → format mismatch with SFT training → all DPO rewards/margins negative (job 595679 wasted). Fix: added `"llama_ck1600"` to manual Llama 3 format branch.

**`run_dpo_targeted_fix.slurm` relative log path**
`logs/...` → no logs generated for job 595714. Fix: use absolute `/home/a/aryanj/logs/`.

---

## 2026-04-22

**SLURM log path typo**
Job 595563 instant fail: `/home/aryanj/logs/` → correct path is `/home/a/aryanj/logs/`.

**DPO targeted fix prompt had unreplaced template vars**
`{system_prompt}` / `{memory_context}` not substituted. Fix: `generate_pair()` now replaces all 4 vars.

---

## 2026-04-14

**`contextBuilder.ts` header too passive**
Model was reactive, not proactive about context use. Fix: added 3 directive lines (mood trend, health pattern, unhelpful coping).

---

## 2026-04-09

**TRL DPO `tokenizer=` arg removed in newer version**
Fix: use `processing_class=tokenizer`.

---

## 2026-04-07

**DPO datagen 0 pairs — Qwen3 `<think>` blocks corrupt JSON**
Fix: `enable_thinking=False` + strip `<think>` in `parse_json_robust`.

**SLURM log files not found**
Fix: always use `~/logs/` (home dir), not `~/projects/mindmate/logs/`.

---

## See also

- [[Training]] — current pipeline state
- [[Benchmarks]] — key findings that exposed these bugs
