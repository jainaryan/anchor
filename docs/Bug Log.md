---
tags: [anchor, bugs]
---

# Bug Log

← [[Home]]

Chronological record of bugs found and fixed. Use this to understand what has already been tried and why. Most recent first.

---

## 2026-05-09 (Session 4)

### Training format mismatch — root cause of base-beats-SFT ❗

**Bug:** All 42,038 training examples were missing the production system prompt preamble.
- `targeted_fix.jsonl` + `biometric.jsonl`: had `[User]\n...\n[Recent sessions]\n...` only — no "You are Anchor..." preamble, no "ABOUT THIS USER" instruction header
- All other 6 files: had **no system message at all**

Model was trained on a format completely different from what it sees at inference. This is why CROSS_SESSION dropped to 0%, CONTEXT_MEMORY to 9%, BIOMETRIC to 30% — the model never learned to use the production context format.

**Fix:** Commit c3acdc9. Normalization script updated all 42,038 examples:
- `targeted_fix` + `biometric`: prepended `_APP_BASE_PROMPT + MEMORY_HEADER` before existing memory blocks
- `biometric`: 179 examples with raw session notes → wrapped with full format
- `friend_1`, `therapist_`, `transition`, `casual`, `grief`: injected base prompt as system message
- `targeted_fixes` (gold): old abbreviated format → prepended base prompt
- Normalized `messages` key → `conversations` key throughout

---

### Benchmark variance — 17/49 scenarios flip between runs ❗

**Bug:** Running same model twice with temperature=0.7 gave substantially different scores. llama_base run 1 vs run 2: 17/49 scenarios flipped PASS→FAIL. Initially looked like a regression but was variance.

**Root cause:** Both the eval model (generates responses) and dynamic scenario user simulator run at temperature=0.7. Different outputs → different transcripts → different judge verdicts.

**Fix:** Adopted 3-run averaged methodology. Use `python benchmarks/average_results.py --since YYYYMMDD`. Never rank models on single-run scores.

---

### H100-96 GRES falls back to ~46GB nodes

**Bug:** Jobs 609078–609083 submitted with H100-96 GRES ran on nodes with only ~46GB VRAM. Gemma4 judge requires ~52GB → OOM.

**Fix:** Always use `--gres=gpu:a100-80:1` explicitly in the `#SBATCH` header. Do not use H100-96 GRES for any job that loads Gemma4.

---

### GitHub SSH blocked from cluster

**Bug:** `git pull` on cluster fails: `ssh.github.com port 443: Connection timed out`. Port 443 outbound SSH blocked.

**Fix:** Use rsync from local machine to push files to cluster. No git operations possible on cluster.

---

### `average_results.py` KeyError `'weighted_pass'`

**Bug:** Result JSON structure changed — uses nested `overall.weighted_pass`, not top-level `weighted_pass`. `average_results.py` crashed on the new format.

**Fix:** Changed access to `d['overall']['weighted_pass']` with fallback for both flat and nested formats.

---

### `llama_ck1600` shortcut name misleading

**Bug:** MODEL_SHORTCUTS had `llama_ck1600` pointing to the genzv2 SFT adapter. The name implied it was the base Llama model, causing confusion when comparing with base.

**Fix:** Renamed to `genzv2_ck1600` in MODEL_SHORTCUTS and all docs. Adapter path unchanged: `adapters/genz/checkpoint-1600`.

---

### `conversation_memory_pipeline.py` v1 was not anchor-aligned

**Bug:** Teacher model (Gemma4) generated full conversation JSON using a meta-prompt that described what kind of conversation to write. The teacher was never given the production Anchor system prompt — its outputs reflected generic Gemma4 style, not Anchor's voice.

**Fix:** Refactored to teacher-as-Anchor two-phase approach. Phase 1: user simulator. Phase 2: Gemma4 given production Anchor prompt as its actual system message. Training data now aligned with inference distribution by construction.

---

### Timestamp collision in benchmark results

**Bug:** Two genzv4_ck200 jobs finished at identical second → same output filename (`genzv4_ck200_20260509_0348.json`) → one overwrote the other.

**Status:** Non-critical. One run was captured. The averaged result for genzv4_ck200 used 2 new runs + 1 earlier May 9 run (n=3 total).

---

## 2026-05-07 (Session 3)

### `sbatch --export` placed after script path — silently ignored

**Bug:** `sbatch script.slurm --export=ALL,MODEL=genzv2_ck1200` treats `--export` as a script argument, not an sbatch flag. All 5 jobs (607691–607695) silently used the default model (`llama_ck1600` at the time).

**Fix:** Always place sbatch flags BEFORE the script path: `sbatch --export=ALL,MODEL=genzv2_ck1200 script.slurm`. Wasted 5 jobs; now checked in pre-submission checklist.

---

## 2026-05-02–03 (Session 2/3)

### `CUDNN_STATUS_NOT_INITIALIZED` on H200/H100 nodes

**Bug:** Benchmark Phase 1 crashed on jobs 602438–602452 during `scaled_dot_product_attention`. H200/H100 nodes on that partition had cuDNN/flash-attention issues with 4-bit models.

**Fix:** Added `attn_implementation="eager"` to eval model load in `run_benchmarks.py`. Bypasses flash-attention entirely; slower but stable.

---

### A100-40 OOM loading Gemma4 judge

**Bug:** After eval model `del` + `empty_cache`, VRAM still nearly full (4.5MB free of 39.49GB). Gemma4 bfloat16 (~52GB) couldn't fit on A100-40 (40GB).

**Fix:** Added `model.cpu(); base.cpu()` before `del` to force VRAM release before judge loads. Switched default GPU to A100-80 in `run_benchmarks.slurm`.

---

### Remote ssh sbatch variable expansion (local shell eats the variable)

**Bug:** `ssh nus-student-cluster "sbatch --export=ALL,ADAPTER=${ck} script.slurm"` — `$ck` expands in local shell (empty string). All submitted jobs had empty ADAPTER.

**Fix:** Use single quotes for remote command: `ssh nus-student-cluster 'for ck in 200 400; do sbatch --export=ALL,ADAPTER=${ck} script.slurm; done'`

---

### `--wrap` uses `/bin/sh` — `source` not found

**Bug:** `sbatch --wrap="source mindmatenv/bin/activate && python ..."` fails with "source: not found" because `--wrap` executes via `/bin/sh`, not bash.

**Fix:** Wrap with `bash -c`: `--wrap="bash -c \"source mindmatenv/bin/activate && python ...\""`

---

### `DPOConfig.__init__()` unexpected keyword argument `max_prompt_length`

**Bug:** `max_prompt_length` was removed from TRL's `DPOConfig` in the cluster's installed version. Jobs 602946–602948 failed.

**Fix:** Removed `max_prompt_length=1024` from `DPOConfig` in `finetuning/CUDA_train_dpo.py`.

---

### `genzv2_ck1600` missing from argparse choices in DPO trainer

**Bug:** `CUDA_train_dpo.py: error: argument --model: invalid choice: 'genzv2_ck1600'` — alias not added to argparse. Job 603041 failed.

**Fix:** Added `genzv2_ck1600` to argparse choices and CONFIGS dict (maps to `adapters/genz/checkpoint-1600`).

---

## 2026-05-01 (Session 2)

### A100-80 node xgpj0 torch import fails

**Bug:** `libtorch_global_deps.so: No such file or directory` on node xgpj0. CUDA libs not in LD_LIBRARY_PATH on that specific node.

**Fix:** Node xgpj0 has a broken environment. Switched `run_benchmarks.slurm` to A100-40 at the time (later switched back to A100-80 for Gemma4 VRAM reasons, but jobs avoid xgpj0 naturally via scheduler).

---

### `ends_question` check too strict

**Bug:** Crisis scenario check `response.rstrip().endswith("?")` failed when model appended a trailing non-question clause after the question. "Are you safe? I'm here for you." → failed the check.

**Fix:** Changed to `"?" in response` — any question mark anywhere in response satisfies the criterion.

---

### MEMORY_USE + BIOMETRIC checks were keyword-based

**Bug:** Rule-based checks failed on semantically correct responses that used different wording. A good response about sleep patterns would fail because it didn't contain the exact expected keywords.

**Fix:** Replaced with LLM judge (Qwen3-30B 4-bit at the time, later migrated to Gemma4 26B A4B bfloat16). Binary YES/NO per criterion. Results more reliable but require A100-80 for Gemma4.

---

## 2026-04-26

### `synthetic/utils.py` missing `import random` and `import re`

**Bug:** `randomize_health_context()` in `utils.py` called `random.choice()` and `re.sub()` without importing these modules. Crashed on every call. Biometric pipeline (job 595713) generated 0 pairs in 22h of runtime.

**Fix:** Added `import random` and `import re` to `synthetic/utils.py`.

---

### DPO `format_messages()` format mismatch (job 595679 wasted)

**Bug:** `CUDA_train_dpo.py format_messages()` used `tokenizer.apply_chat_template` for the `llama_ck1600` model shortcut. But `genzv2_ck1600` SFT training used a manual Llama 3 format. Chat template ≠ manual format → all DPO rewards and margins were negative → DPO diverged.

**Fix:** Added `"llama_ck1600"` (and later `"genzv2_ck1600"`) to the manual Llama 3 format branch in `CUDA_train_dpo.py`.

---

### `run_dpo_targeted_fix.slurm` relative log path

**Bug:** `logs/...` relative path in SLURM script → no logs generated for job 595714. Couldn't diagnose DPO failure.

**Fix:** Changed to absolute path `/home/a/aryanj/logs/...`.

---

## 2026-04-22

### SLURM log path typo (wrong home subdirectory)

**Bug:** Job 595563 instant fail. `#SBATCH --output=/home/aryanj/logs/...` → wrong path. Correct path is `/home/a/aryanj/logs/`. The username directory is nested under `/a/`.

**Fix:** All SLURM scripts now use `/home/a/aryanj/logs/`. This is now standard in all `.slurm` files.

---

### DPO targeted fix prompt had unreplaced template variables

**Bug:** `dpo_targeted_fix_preference.txt` had `{system_prompt}` and `{memory_context}` placeholders that were never substituted. `generate_pair()` in `dpo_targeted_fix_pipeline.py` didn't replace all 4 variables.

**Fix:** Updated `generate_pair()` to replace all 4 template variables. Local file is correct; just needed syncing to cluster.

---

## 2026-04-14

### `contextBuilder.ts` directives too passive — model reactive, not proactive

**Bug:** The memory header told the model what context was available but didn't instruct it to use the context proactively. Model would reference memory only when directly asked.

**Fix:** Added 5 explicit directive lines to the memory header:
1. Reference names/events/strategies the user has mentioned
2. Suggest ONE coping strategy by name when asked for help
3. Declining mood trend in Recent sessions → acknowledge in first response
4. Health/sleep pattern in Recent sessions → connect when user describes something similar
5. ✗-marked unhelpful strategy → do NOT suggest it

---

## 2026-04-09

### TRL DPO `tokenizer=` argument removed in newer TRL version

**Bug:** `DPOTrainer(tokenizer=tokenizer, ...)` raised `unexpected keyword argument 'tokenizer'`. TRL updated its API.

**Fix:** Changed to `DPOTrainer(processing_class=tokenizer, ...)`.

---

## 2026-04-07

### DPO datagen generated 0 pairs — Qwen3 `<think>` blocks corrupted JSON

**Bug:** Qwen3 generates internal `<think>...</think>` blocks before its response. The JSON parser received the `<think>` block + the JSON response as one string → invalid JSON → 0 pairs generated.

**Fix:** Two-part fix in `synthetic/utils.py`:
1. `enable_thinking=False` in Qwen3 generation config (where applicable)
2. `parse_json_robust()` strips any `<think>...</think>` blocks as fallback before JSON parse

---

### SLURM log files not created (wrong log directory)

**Bug:** SLURM scripts used `~/projects/mindmate/logs/` as the log path. This directory doesn't exist, and SLURM silently fails to create it → no log files written.

**Fix:** Changed all SLURM scripts to use `~/logs/` (home directory). This directory exists and is writable. Absolute path: `/home/a/aryanj/logs/`.

---

## See also

- [[Training]] — current pipeline state
- [[Benchmarks]] — key findings that exposed these bugs
- [[System Prompt]] — format mismatch root cause details
