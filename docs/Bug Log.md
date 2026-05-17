---
tags: [anchor, bugs]
---

# Bug Log

← [[Home]]

Chronological record of bugs found and fixed. Use this to understand what has already been tried and why. Most recent first.

## 2026-05-17

### Over-reference filter: 12 false positives held as future DPO negatives
- **File:** `data/synthetic_train_conv_memory_overref_*.jsonl` (29 total examples)
- **Split:** ~12 false positives (seed `new_fact = "They had a really good therapy session today."` — Anchor correctly using "therapist/therapy" in response to a user who mentioned their session), ~17 genuine overrefs (Anchor injecting clinical framing into casual chat unprompted)
- **Action:** The 12 false positives are **held** — do not include in genzv5 training data or DPO now. Revisit after genzv5 ships as potential DPO negatives with a corrected filter (whitelist user-mentioned therapy seeds).
- **Genuine overrefs:** Also held for now (post-genzv5 DPO). Both sets remain in `_overref_*` files.

### vLLM fails on H100-96 nodes (CUDA fork error)
- **Jobs:** 614822–614827 (conv-memory new-pool shards 3–8, first H100 attempt)
- **Error:** `RuntimeError: Cannot re-initialize CUDA in forked subprocess` — vLLM's EngineCore uses `fork` start method, which breaks CUDA on H100 nodes
- **Fix:** Set `USE_VLLM=0` in `finetuning/conv_memory_pipeline_qwen.slurm` — falls back to HuggingFace backend (proven working on biometric jobs 613218–613220)
- **Resubmitted:** jobs 614942–614947 on H100-96 with HF backend, loading successfully

### New-pool conv-memory shards OOM on A100-80
- **Jobs:** 612903–612905, 613115–613117 (shards 3–8, first attempt on A100-80)
- **Error:** 612903 got 10 convs then OOM'd in inference loop for 2+ days; 612904+ OOM-killed at ~55% weight load
- **Root cause:** `USE_VLLM=1` causes vLLM to pre-allocate KV cache on top of 30B model weights, exceeding 80GB
- **Fix:** Moved to H100-96 (96GB) + disabled vLLM

---

## 2026-05-14 (Sessions 7–8)

### Conv-memory pipeline: 65/35 mix was companion-biased — fixed to 50/50

**Bug:** `pick_mode()` used `random.random() < 0.65` for clinical profiles, giving ~65% casual outputs overall. The intent was to teach Anchor when memory is *and* isn't needed, but 65% casual skewed training toward the "not needed" case.

**Fix:** Flipped to `random.random() < 0.35`, achieving ~51.5%/48.5% casual/clinical (validated over 10k trials). Clinical USER_MODE_INSTRUCTIONS also tightened to explicitly forbid user simulator from mentioning therapy/coping unprompted in casual mode.

---

### Over-reference filter: initially dropped flagged examples — changed to route

**Bug:** First implementation of the over-reference heuristic returned `False` to drop examples where Anchor shoehorned therapy/coping keywords (≥2 hits) into casual-mode turns. Dropping loses signal — those examples are potentially useful as DPO negatives.

**Fix:** Changed `heuristic_check` return type from `bool` → `str` (`"keep"` / `"overref"` / `"drop"`). Overref examples now routed to `synthetic_train_conv_memory_overref_qwen_s{N}.jsonl` for inspection. 22 examples caught across 3 shards (~8% of output) as of 2026-05-14.

---

### Job 611376: ran pre-patch code — pipeline had loaded old module at startup

**Bug:** Job 611376 was submitted, then the pipeline script was patched (50/50 fix, overref filter, shard support). The running job had already imported the old module in memory — patches to the file on disk had no effect on the live process.

**Fix:** `scancel 611376`. Resubmitted as 611377, which loaded the patched file at startup.

---

### Biometric pipeline: output paths were hardcoded and undescriptive

**Bug:** `biometric_sft_pipeline.py` wrote to `synthetic_train_biometric.jsonl` regardless of teacher model or shard. If multiple jobs ran in parallel they would stomp each other. No way to tell from filename which teacher or run config produced the file.

**Fix:** Added `SHARD_IDX` + `OUT_LABEL` env var support (same pattern as conv-memory pipeline). Output is now `synthetic_train_biometric_qwen_s{N}.jsonl` — teacher and shard index both encoded in filename. Launcher `finetuning/launch_biometric_shards.sh` submits N parallel shards.

---

### Biometric pipeline (v1/v2): single-call generation — teacher played both user and Anchor roles

**Bug:** `biometric_sft_pipeline.py` generated entire conversations in one LLM call by asking the model to produce a JSON script of both user and Anchor turns. This means:
1. Anchor's responses were NOT grounded by the production system prompt — the teacher was imagining what Anchor would say, not being constrained to say it.
2. The training-inference distribution was misaligned: at inference time Anchor has the real system prompt; during data generation it didn't.
3. The "intelligence" of knowing when to reference biometric data was up to the teacher's imagination, not learned from real constrained responses.

**Fix:** Rewrote as v3 two-phase teacher-as-Anchor (same approach as conv-memory):
- Phase 1: user simulator generates user turns for a given mode (relevant/irrelevant/adjacent/trend)
- Phase 2: teacher constrained by the exact production Anchor system prompt generates responses one turn at a time
- Mode weights: irrelevant 40%, adjacent 25%, relevant 25%, trend 10% (model learns NOT to inject by default)
- Added vLLM backend to `utils.py` (`USE_VLLM=1` env var) — PagedAttention + FlashAttention2
- Cancelled 612894–612896, relaunched as 613120–613122

Note: v3 rate ~10-15/hr per shard (was 180/hr for v2) — quality trade-off is worth it.

---

### Conv-memory shards 3-5: first relaunch (612900-612902) used profile pool that was too small — cancelled, pool expanded

**Bug:** Jobs 612900-612902 launched with `NEW_ONLY_PROFILES = 8 clinical + 4 companion = 12 profiles` total. With 3 parallel shards each running 72h, the 12-profile pool would produce heavily repeated characters. Also, the fact pool hadn't been expanded to match the new clinical diversity.

**Fix:** Expanded `NEW_CLINICAL_PROFILES` from 8 → 18 (added: Laila, Stefan, Mei, Patrick, Adaeze, Tom, Haruto, Freya, Luca, Sangita). Expanded `NEW_COMPANION_PROFILES` from 4 → 10 (added: Aiden, Nour, Felix, Zoe, Raj, Ines). `NEW_FACTS` expanded from 35 → 47. Cancelled 612900-612902, relaunched as 612903-612905 with the same `PROFILE_SET=new` flag.

---

### vLLM install on cluster: pip goes to miniconda3, not mindmatenv — fixed to use uv pip

**Bug:** The SLURM scripts ran `pip install -q vllm 2>&1 | tail -1` after activating the uv-managed mindmatenv. Despite activation, `which pip` resolved to `/home/a/aryanj/miniconda3/bin/pip` (Python 3.13 global conda env), not mindmatenv's pip. So vllm was installed in miniconda3 but not importable from mindmatenv (Python 3.12). Job failed with `ModuleNotFoundError: No module named 'vllm'`.

**Fix:** Changed all SLURM scripts to use `uv pip install vllm 2>&1 | tail -3`. `uv` correctly resolves the active venv and installs into it. After fix, `uv pip install` printed `"Using Python 3.12.3 environment at: mindmatenv"` and vllm became importable.

---

### vLLM: `tokenizer_kwargs` not valid in installed version — removed

**Bug:** `utils.py _load_vllm()` passed `tokenizer_kwargs={"token": hf_token}` to `LLM()`. The installed vllm version (0.20.2) doesn't support this kwarg. Raised `TypeError: EngineArgs.__init__() got an unexpected keyword argument 'tokenizer_kwargs'`.

**Fix:** Removed `tokenizer_kwargs` from `LLM()` constructor. vLLM reads `HF_TOKEN` from the environment automatically — no need to pass it explicitly.

---

### vLLM V1 engine: CUDA fork error — fixed with spawn start method

**Bug:** vLLM 0.20.2 uses the V1 engine which spawns a subprocess for the EngineCore. With default `fork` multiprocessing, this raised `RuntimeError: Cannot re-initialize CUDA in forked subprocess`.

**Fix:** Added `export VLLM_WORKER_MULTIPROC_METHOD=spawn` to biometric_pipeline_qwen.slurm. This tells Python to use `spawn` (clean subprocess) instead of `fork`, avoiding the CUDA re-init issue.

---

### xgph[10-18] GPU: GRES label `gpu:a100-40:2` is misleading — node has one A100-80

**Discovery:** When requesting `--gres=gpu:a100-40:2` on xgph[10-18] nodes, nvidia-smi showed a single "NVIDIA A100 80GB PCIe, 81920 MiB". Only 1 GPU visible. Setting `tensor_parallel_size=2` failed with "World size (2) is larger than the number of available GPUs (1)".

**Conclusion:** The GRES label name doesn't reflect physical hardware — these nodes have one A100-80 per node (not two A100-40s). Use `tensor_parallel_size=1` when targeting these nodes. The `gpu:a100-40:2` GRES is useful for getting exclusive access to the node.

---

## 2026-05-15 (Session 11)

### Panic detection — 4 issues surfaced by new unit tests

Added `src/utils/__tests__/panicDetection.test.ts` (anchor-app, 61 cases).
Tests are documented via `test.failing` blocks — they pass while broken, will alert in CI when fixed.

**Issue 1: bare `cutting` in URGENT pattern fires on benign nouns**
- Pattern `/\b(self[\s-]?harm|hurt myself|cut myself|cutting)\b/i`
- Matches: `"i've been cutting carbs"`, `"i'm cutting flowers from the garden"`
- Fix: tighten the bare `cutting` alternation — require object phrase or remove it (the other patterns already cover self-harm intent: `self-harm`, `hurt myself`, `cut myself`).

**Issue 2: "i wish i could just disappear" doesn't match URGENT**
- Pattern `/\b(i (?:want|wish|feel like) (?:to )?(?:die|disappear|not exist))\b/i`
- Requires zero filler words between `wish` and `disappear`
- Real users write `"i wish i could disappear"` / `"i wish i could just disappear"`
- Fix: allow up to 3 filler words between verb and noun, e.g. `(?:to )?(?:\w+\s+){0,3}(?:die|disappear|not exist)`

**Issue 3: quoted speech triggers URGENT**
- `"my friend said 'i want to die' the other day as a joke"` → URGENT
- Detector has no concept of attribution. Acceptable to over-trigger for safety, but worth documenting.

**Issue 4: detector covers `i need help` as WATCH (intentional, but test surfaces it)**
- `"i need help understanding this"` → WATCH
- Per Bug Log 2026-05-10, this is the correct WATCH behavior — the callback fires but the message still sends. My initial test expected NONE — corrected.

---

## 2026-05-14 (Session 9 — continued)

### vLLM 0.20.2: `VLLM_USE_V1=0` removed — env var silently ignored

**Bug:** Attempted to force vLLM V0 engine via `VLLM_USE_V1=0` env var to avoid the subprocess CUDA crash. vLLM 0.20.2 printed `"Unknown vLLM environment variable detected: VLLM_USE_V1"` and ignored it. V0 engine no longer exists in 0.20.2.

**Conclusion:** No way to force V0 in 0.20.2. The CUDA driver incompatibility on xgph[10-18] is fundamental, not configurable.

---

### PyTorch+cu130 vs CUDA driver 575.57.08: Python-level patching insufficient

**Bug:** PyTorch 2.11.0+cu130 (built for CUDA 13.0 runtime) was installed in mindmatenv but xgph[10-18] nodes have CUDA driver 575.57.08 (max CUDA ~12.x). vLLM spawn subprocess called `torch._C._cuda_init()`, which raised `RuntimeError: NVIDIA driver is too old`.

**Attempted fix:** Patched `torch/cuda/__init__.py` on cluster to wrap `torch._C._cuda_init()` in try-except, downgrading the hard error to a warning. This fixed the immediate crash but exposed a cascade failure: `_check_capability()` calls `_get_device_properties` which is a C-extension only registered after successful CUDA init → `NameError: name '_get_device_properties' is not defined`.

**Conclusion:** Python patching can't bridge a C-extension registration failure. PyTorch+cu130 and CUDA driver 12.0.90 are fundamentally incompatible. Using HF backend (`USE_VLLM=0`) is the only workable option on these nodes.

---

### flash-attn install: CUDA version mismatch with PyTorch+cu130

**Bug:** Attempted `uv pip install flash-attn` on xgph[10-18] nodes. flash-attn wheel detects CUDA 12.0 on the node but PyTorch is built for CUDA 13.0 → version mismatch. Install failed or produced an unusable build. `import flash_attn` raised `ModuleNotFoundError`.

**Conclusion:** flash-attn incompatible with this PyTorch/CUDA combination on PCIe nodes. Dropped.

---

### HF backend throughput improvement: sdpa + torch.compile

**Context:** Biometric PCIe A100 jobs running at ~1.4 ex/hr (vs ~5.3 ex/hr on SXM A100 with vLLM). Since vLLM is unworkable on PCIe nodes, applied two HF backend speedups:

1. **`attn_implementation="sdpa"`** (default, was `"eager"`): PyTorch's built-in fused scaled-dot-product attention. No extra packages. ~15% faster than eager.
2. **`torch.compile(mode="reduce-overhead")`** (`USE_TORCH_COMPILE=1`): Captures CUDA computation graph after warmup. ~15-25% additional throughput. First call is slow (compilation), subsequent calls faster.

**Fix:** Updated `synthetic/utils.py`:
- Default attention changed from `"eager"` → `"sdpa"`
- Added `USE_TORCH_COMPILE` env var; if `"1"`, calls `torch.compile(model, mode="reduce-overhead")` after `model.eval()`
- Updated `finetuning/biometric_pipeline_qwen.slurm` to set `USE_TORCH_COMPILE=1`

Note: these improvements don't apply to currently running jobs (613218-613220). Will take effect on new biometric shards launched after ~May 15 on SXM nodes (which can also use vLLM).

---

## 2026-05-10 (Session 5)

### Panic detection false positives — "help me" blocked normal messages ❗

**Bug:** `detectPanic()` returned a single boolean. The pattern `/\b(help me\b|i need help\b|please help\b)/i` fired on phrases like "help me understand this" or "i need help with my homework", routing those messages to the panic/crisis flow and blocking them from being sent at all.

**Fix:** `detectPanic()` now returns `PanicRiskLevel = 'none' | 'watch' | 'urgent'`. Broad phrases moved to `WATCH_PATTERNS`; only patterns with explicit self-harm intent remain in `URGENT_PATTERNS`. `useChatSession` only hard-returns on `'urgent'`; `'watch'` calls the callback but lets the message through.

---

### Session-extraction race condition — memory written to wrong session ❗

**Bug:** In `useChatSession.ts`, the extraction timer scheduled at completion time looked up `chatSessionStore.activeSessionId` and `currentSessionMessages` at **fire time** (5 minutes later). If the user switched sessions during that window, the extracted memory was saved to whichever session happened to be active at fire time, not the session where the conversation happened.

**Fix:** `scheduleSessionExtraction()` now captures `sessionId` and a `toJS()` snapshot of `messages` immediately when called. The timer closure only uses those captured values.

---

### Person-name extraction never triggered — lowercase mismatch ❗

**Bug:** `getUserMessages()` returned all message text `.toLowerCase()`. The intro regex in `extractPeople()` was `/\bmy (?:friend|...)\s+([A-Z][a-z]+)/g` — the `[A-Z]` pattern requires an uppercase first letter, which was never present after lowercasing. The "new person introduction" path (`"my friend John"` → save "John") silently produced zero results for every session.

**Fix:** `getUserMessages()` now preserves original casing. `extractTopics()` and `detectMilestone()` lowercase internally. The intro regex changed to `[A-Za-z]` and Title-cases the captured name before storing.

---

### Memory retrieval missed narrative content — summary text not searched

**Bug:** `searchByKeywords()` in `MemoryRepository.ts` matched only `topics` and `peopleMentioned` arrays. Session summaries (the actual free-text narrative, and the only thing shown to the LLM) were not part of the search. A user mentioning "breakup" in a new session would miss an older session whose summary said "user described going through a difficult breakup" but whose `topics` array only had `["relationship"]`.

**Fix:** Summary text is now split into words and included in the haystack. Also added a recency bonus: memories within the last 14 days score up to +1.0 on top of keyword hits, so recent relevant memories rank above older ones with the same keyword count.

---

### Prompt token budget missing — memory block could silently grow unbounded

**Bug:** `buildEnhancedSystemPrompt()` assembled all tiers without any size limit. A user with 3 years of episodic memories and detailed health data could generate a system prompt thousands of tokens over the model's context window, silently truncating the chat history instead.

**Fix:** Added `TIER_BUDGET` character limits (tier1: 2400, tier2: 3200, tier3: 1600, biometric: 1600) and a `truncateToBudget()` helper that cuts at the nearest line boundary with a `[…]` marker. Applied to each tier before `assemblePrompt()`.

---

### Eval/production prompt drift — EvalRunner used legacy structured prompt by default

**Bug:** `EvalRunner.ts` always used `MINDMATE_SYSTEM_PROMPT` (the ~150-line legacy prompt with old "MindMate" name) for all eval runs. Production chat uses the 6-line `getAnchorSystemPrompt()`. Eval scores were measuring behavior on a completely different prompt from what users experience, making them poor proxies for real-world quality.

**Fix:** `runScenario()` now accepts an optional `systemPrompt` parameter that defaults to `getAnchorSystemPrompt('reflect')`. Historical runs can still use the old prompt by passing `MINDMATE_SYSTEM_PROMPT` explicitly. **Note:** all eval scores before 2026-05-10 were generated with the legacy prompt.

---

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
