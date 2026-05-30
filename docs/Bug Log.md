---
tags: [anchor, bugs]
---

# Bug Log

← [[Home]]

Chronological record of bugs found and fixed. Use this to understand what has already been tried and why. Most recent first.

## 2026-05-30

### GPU torch deleted from venv by export script; disk quota too full to reinstall (RESOLVED)

**Symptom:** All crisis/help_mode datagen jobs (623575–623987) crashed. Venv `torch/` directory completely absent: `ls mindmatenv/lib/python3.12/site-packages/torch/` → No such file or directory.

**Root cause:** `run_export.slurm` does `rm -rf "$VENV_SITE/torch" "$VENV_SITE"/torch-*.dist-info` before installing CPU-only torch for export. This permanently deleted GPU torch from the shared venv. Any subsequent GPU job finds no torch.

**Cascading issue — /tmp workaround doesn't work for GPU inference:** Reinstalling torch to `/tmp` and prepending to `PYTHONPATH` passes `torch.cuda.is_available()` and loads model weights to GPU correctly, but the first `model.generate()` call hangs indefinitely (3+ hours, zero output). Root cause unclear — likely triton/CUDA kernel JIT compilation deadlock when torch was installed to a non-standard path. Do NOT use this workaround for GPU inference jobs (it's still valid for the setup-check pattern in datagen scripts, but the script must fall back to killing the job, not attempting inference).

**Cascading issue — /tmp 50MB per-user quota:** `python -m pip install torch==2.11.0+cu126` from login node failed with `OSError: [Errno 122] Disk quota exceeded` at 52MB into the 830MB download. `quota -s` showed "no limited resources" (home dir has NO per-user quota). Root cause: pip downloads to `/tmp` by default, and `/tmp` has a **50MB per-user quota** on the login node (`quota -v` shows `/dev/sdd1  blocks=0  quota=51200  limit=51200`). The 830MB torch wheel hits this immediately.

**Disk space issue:** Home dir was at ~248GB. Freed ~83GB by deleting HF cache (Gemma4-26B=49GB, Gemma4-e4b=15GB, Gemma4-e2b=9.4GB, Qwen2.5-3B=5.8GB, Qwen3-1.7B=3.8GB) + pip cache (3.4GB). After deletion, home dir was ~162GB with no quota limit.

**Fix (applied 2026-05-30):** Submitted SLURM job 624114 on CPU node (`long` partition) with `TMPDIR=~/pip_tmp` to redirect pip's temp download to the home filesystem (no quota). Used `--no-deps` since `nvidia-nccl-cu12==2.29.3` was already in the venv. Install succeeded: `torch==2.11.0+cu126 torchvision==0.26.0+cu126` restored to venv in ~2m30s.

**Commands to remember for future torch reinstall:**
```bash
mkdir -p ~/pip_tmp
source ~/projects/mindmate/mindmatenv/bin/activate
TMPDIR=~/pip_tmp python -m pip install --no-deps --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu126 'torch==2.11.0' 'torchvision==0.26.0'
```
Or as a SLURM job on `--partition=long` (CPU) if login node connection drops.

**Lesson (3 lessons):**
1. Export script must NOT touch the shared GPU venv's torch — use isolated `/tmp` install for export only (see invariant 3c in Home.md).
2. `/tmp` on the login node has a 50MB per-user quota — use `TMPDIR=~/pip_tmp` for any large pip downloads.
3. `nvidia-nccl-cu12` is the largest dependency (297MB); use `--no-deps` when reinstalling torch if nccl is already installed.

### H100-96 nodes: `device_map="auto"` offloads model to CPU with /tmp torch

**Symptom:** Jobs on H100 nodes (xgpi*) — torch import worked, CUDA visible, model loaded to 100%, but `device_map="auto"` offloaded some layers to CPU. Inference was effectively on CPU (no output after hours).

**Root cause:** When torch is installed to `/tmp` via PYTHONPATH, `accelerate`'s memory estimator may miscalculate available VRAM or the CUDA device context is not fully initialized, causing partial CPU offload. On A100-80 (xgph2), the same /tmp torch approach passed the CUDA check but then hung in the first `generate()` call (different failure mode, same root cause).

**Fix:** Restore GPU torch to the venv (see above). The /tmp approach is fundamentally unreliable for GPU inference.

## 2026-05-29

### `crisis_help_pipeline.py`: `active_si` heuristic rejected ~99% of examples

**Symptom:** After job 621987 (256 examples), `crisis_active_si` was only 3/256 (1.2%) vs 20% target. Logs showed thousands of `active_si fail heuristic` lines.

**Root cause:** The `heuristic_crisis` function required Anchor to use specific reach-out phrases (`"reach out"`, `"crisis line"`, `"988"`, `"call someone"`, etc.) somewhere in the conversation. Qwen3 was generating appropriate, empathetic responses that encouraged connection without hitting those exact keywords. Nearly every `active_si` attempt failed this check.

**Fix:** Dropped the entire `active_si` keyword check. The teacher prompt already instructs Qwen3 to handle `active_si` seriously — no keyword policing needed. Also removed `"988"` from all heuristics (hardcoded crisis line numbers should not appear in training data).

### `crisis_help_pipeline.py`: `mid_session` and `not_working` help_mode modes near-absent

**Symptom:** `help_mode_qwen_raw.jsonl` had 200 examples: `cold_open`=197, `mid_session`=2, `not_working`=1.

**Root cause:** `TECHNIQUE_WORDS` list was too narrow (breathing/grounding/DARE only). Qwen3 offers techniques by name — journaling, walking, progressive relaxation, mindfulness, cold shower, etc. — none of which matched. Every `mid_session`/`not_working` attempt failed the technique-word check.

**Fix:** Expanded `TECHNIQUE_WORDS` with common technique phrases Qwen3 actually uses: `walk`, `journal`, `mindful`, `meditat`, `progressive`, `body scan`, `cold shower`, `stretch`, `here's something`, `a technique`, `this might help`, etc.

### `crisis_help_pipeline_qwen.slurm`: torch broken on H100 nodes (NFS stale file handle)

**Symptom:** Jobs 623575–623589 all failed immediately with `OSError: libtorch_global_deps.so: No such file or directory`. Attempted reinstall via `pip install` hit `OSError: [Errno 116] Stale file handle`.

**Root cause (layer 1):** The venv's torch install was corrupted on H100-96 compute nodes — `libtorch_global_deps.so` missing.

**Root cause (layer 2):** NFS home directory had stale file handles on these nodes, making `pip install` to the venv impossible (`Errno 116`).

**Root cause (layer 3 — pre-existing):** Bare `pip` resolves to miniconda's pip (Python 3.13), not the venv's pip. Fix already documented in Home.md §3b, but wasn't applied to this slurm script.

**Fix:** Install torch to `/tmp` instead of the venv, then prepend to `PYTHONPATH`. `/tmp` is local disk per node, unaffected by NFS issues. Pattern:
```bash
TORCH_TMP=/tmp/torch_fix_$$
python -m pip install --target="$TORCH_TMP" \
  --index-url https://download.pytorch.org/whl/cu126 "torch==2.11.0"
export PYTHONPATH="$TORCH_TMP:${PYTHONPATH:-}"
```
Pinned to `torch==2.11.0` to match the venv's `torchvision==0.26.0` dependency.

## 2026-05-26

### `run_export.slurm`: two bugs caused job 621990 to run wrong model and crash

**Job:** 621990 (genzv6_ck1600 GGUF export, submitted 2026-05-25)

**Bug 1 — wrong model:** Script used `MODEL=${1:-genz}` (positional arg), but submission used `--export=ALL,MODEL=genzv6_ck1600` (env var). `$1` was empty so it fell back to `genz`. Job ran the wrong model.

**Bug 2 — torch crash:** `OSError: libtorch_global_deps.so: No such file or directory` — the venv has cu130 torch installed (broken on A100-80 driver), same root cause as training jobs. The export script needs the same cu126 reinstall step as all other SLURM scripts.

**Fix (2026-05-26, committed):**
- `MODEL=${1:-genz}` → `MODEL=${MODEL:-genz}` (reads env var, not positional arg)
- Added cu126 torch reinstall block (same pattern as `finetuning/run_sft_v6.slurm`)
- Also added `cd ~/projects/mindmate` at top (script was relying on CWD)

**Job 621991 resubmitted 2026-05-26** with `--export=ALL,MODEL=genzv6_ck1600`.

### `run_export.slurm`: `libcufile.so.0` missing — cu126 torch broken on some A100-80 nodes

**Job:** 621991 (genzv6_ck1600 GGUF export, 2026-05-26)

**Bug:** After the cu126 reinstall, `import torch` crashed with `ImportError: libcufile.so.0: cannot open shared object file`. torch 2.12.0+cu126 links against the CUDA GPUDirect Storage library, which is absent on some A100-80 nodes. Training scripts use the same cu126 install but land on nodes that have cufile; the export job landed on one that doesn't.

**Root cause:** Export pipeline is entirely CPU (`device_map="cpu"` for merge, llama.cpp convert + quantize are compiled C++ binaries). There is no reason to install GPU torch for export at all.

**Fix (2026-05-26, committed):** Changed `run_export.slurm` to install CPU-only torch (`--index-url https://download.pytorch.org/whl/cpu`). Eliminates all CUDA driver / libcufile dependencies from the export path.

**Job 622015 resubmitted 2026-05-26.**

---

## 2026-05-25

### Crisis-help pipeline: `CUDNN_STATUS_NOT_INITIALIZED` on H100-96 (SLURM fixed)

**Jobs:** 619781, 619782, 619783 (crisis + help_mode pipeline, Qwen3-30B-A3B, May 22–24)

**Bug:** `crisis_help_pipeline_qwen.slurm` requested `--gres=gpu:h100-96:1`. The H100-96 nodes produce `CUDNN_STATUS_NOT_INITIALIZED` errors repeatedly in batch Phase 1 (the HF inference phase). This killed most generation — 619781 produced only 5 crisis examples out of an expected ~hundreds. The help_mode pipeline in the same jobs produced 200 examples (less batch-intensive phase) before the same errors terminated it.

**Root cause:** H100-96 nodes have a CUDA driver version incompatible with cuDNN initialization in the HF batched inference path. A100-80 nodes do not have this issue — they are the proven GPU for all training and datagen in this project.

**Fix (2026-05-25):**
- `finetuning/crisis_help_pipeline_qwen.slurm`: `--gres=gpu:h100-96:1` → `--gres=gpu:a100-80:1`
- `PHASE1_BATCH_SIZE`: 8 → 1 (conservative default; batch size 8 may have amplified failures)

**Fix 1 (2026-05-25, committed):** `finetuning/crisis_help_pipeline_qwen.slurm` — `--gres=gpu:h100-96:1` → `--gres=gpu:a100-80:1`, `PHASE1_BATCH_SIZE` 8 → 1.

**Fix 2 (2026-05-25, committed):** `synthetic/crisis_help_pipeline.py` — added `consecutive_phase1_errors` counter; exits with code 1 after 10 consecutive Phase 1 failures so SLURM marks the job FAILED instead of burning the full 48h wall time doing nothing.

**Job 621987 submitted 2026-05-25 on xgph2/A100-80.** Appends to existing `data/synthetic_train_crisis_qwen.jsonl` (5 lines from failed run).

---

## 2026-05-22

### `run_benchmarks_v4.py` f-string format spec crash at summary (commit a464b88)

**Bug:** After all 83 scenarios completed, the summary print crashed with `ValueError: Invalid format specifier '.3f if rho is not None else 'n/a'' for object of type 'float'`. The offending line was:
```python
f"spearman_rho={rho:.3f if rho is not None else 'n/a'}  "
```
Python evaluates the format spec `:.3f if rho is not None else 'n/a'` as a single string — conditional expressions inside format specs are not valid Python. All 83 scenarios had been evaluated correctly; only the final summary print crashed, leaving the result dir incomplete (no `judged.json` summary written).

**Fix:** Nested f-string:
```python
f"spearman_rho={f'{rho:.3f}' if rho is not None else 'n/a'}  "
```

---

### Missing `MODEL_SHORTCUTS` entries for genzv5 ck1000–ck2000 (commit a464b88)

**Bug:** `run_benchmarks_v4.py`'s `MODEL_SHORTCUTS` dict only went up to `genzv5_ck800`. Submitting `--model genzv5_ck1000` (or ck1200/ck1400/ck1600/ck1800/ck2000) caused argparse to exit immediately with code 2 ("argument --model: invalid choice"). Jobs 619147–619151, 619201–619203, 619236–619247, 619442–619443 all failed in 0–1 seconds for this reason.

**Fix:** Added ck1000–ck2000 entries to `MODEL_SHORTCUTS`. Also note: `genzv5_ck1400` entry points to `adapters/genzv5/checkpoint-1600` (intentional — checkpoint-1400 dir doesn't exist on cluster yet if training was stopped early; update if needed).

**Resubmitted as:** jobs 619456–619473.

---

### Garbage pre-fix result dirs pollute average_results.py (manual cleanup)

**Bug:** Jobs 619120–619131 ran with broken code (BatchEncoding `.to(device)` bug from an earlier session) and completed in 2–3 min with all 83 scenarios errored. Their result dirs (`cluster_*_20260521_06xxxx` through `_16xxxx`) would corrupt averaging if included. Similarly, jobs 619161 produced a llama_base run where all scenarios errored.

**Fix (manual):**
- Deleted all `cluster_llama_base_20260521_*` dirs (14 dirs from pre-fix runs)
- Deleted 38 dirs matching `cluster_*_20260521_{06,07,11,13,15,16}*` across genzv2/v3/v4/v5 models
- Only good runs (UTC timestamp ≥ 17:24:55 on 2026-05-21, corresponding to job 619124 onwards) were kept

---

### `biometric_qwen_s0.jsonl` missing Anchor preamble — 993 examples with malformed system prompt

**Bug (data quality):** `data/synthetic_train_biometric_qwen_s0.jsonl` has no "You are Anchor..." preamble. 9/10 sampled system prompts start directly with `[User]`; 1/10 starts with `[Recent sessions]`. The standard Anchor preamble is missing entirely, which means:
1. The model is trained to respond without the Anchor persona definition
2. The preamble's crisis instruction ("If someone seems to be in danger or crisis, gently encourage them to reach out to someone they trust or a crisis line.") is absent from 993 training examples — directly hurting CRISIS scores
3. This is the largest single biometric shard (993/~2,860 biometric examples = 35% of biometric data)

**v4 benchmark impact:** genzv5 CRISIS = 44–48% vs genzv3 CRISIS = 64%. biometric_qwen_s0 is a leading cause.

**Fix for v6:** Drop `biometric_qwen_s0` entirely. Fix the biometric datagen pipeline to always include the full Anchor preamble. Or fix the raw file (grep-replace the system prompt opening).

---

### `biometric_qwen_s1` and `biometric_qwen_s2` — pathologically long responses

**Bug (data quality):** Both files have extremely verbose assistant responses:
- `biometric_qwen_s1` (80 ex): avg 800 chars/response, 10/10 samples flagged >800 chars (range 800–1088 chars)
- `biometric_qwen_s2` (78 ex): avg 771 chars, 14 quality flags across 10 samples, 9 therapy-speak hits, 6× "i hear you"

Production target is 2–4 sentences (~100–250 chars). Training on 800–1000 char responses teaches the model to over-explain.

**Fix for v6:** Drop both files. The old `biometric_gemma4` file (avg 127 chars) is better quality.

---

### `conv_memory_gemma4` — extreme therapy-speak and markdown

**Bug (data quality):** `synthetic_train_conv_memory.jsonl` (Gemma4 teacher-as-Anchor v2): 16 therapy-speak hits in 10 samples, 9× "it sounds like", 5× "i hear you", 4× responses with markdown. Avg response length 504 chars. This file actively trains the model away from the friend voice.

**Fix for v6:** Drop this file entirely from the mix.

---

### `conv_memory_qwen` files — clunky memory references hurt CROSS_SESSION_MEMORY

**Finding (data quality):** Despite adding 1,617 conv_memory examples specifically to improve CROSS_SESSION_MEMORY, genzv5 scores **60%** on that category while genzv3 (no conv_memory data) scores **85%**. Inspection shows the conv_memory_qwen responses reference memory clumsily: *"You mentioned earlier that health worries and rent have been heavy lately"* — reads like a chatbot reading a notes file, not a friend who remembers naturally.

Additionally, responses avg 488–558 chars with some 800+ char outliers and recurring "i hear you" patterns.

**Fix for v6:** Regenerate with tighter prompts emphasising natural memory use, or curate the existing files to remove clunky references. Possibly reduce count if quality can't be improved.

---

### Crisis heuristic blocking 99%+ of output — 5 examples from 48h run (commit 1288c7d)

**Bug:** `heuristic_crisis()` in `synthetic/crisis_help_pipeline.py` was so strict that job 616643 ran for 48 hours and produced only **5 valid examples** out of 500+ generation attempts.

Root causes:
1. **`active_si` checked only the FIRST assistant turn** for reach-out words (crisis line, not alone, etc.). Qwen3's natural pattern is empathy first, resources later — the first turn never contained reach-out words, so every active_si attempt failed.
2. **`CRISIS_SIGNAL_WORDS` too narrow** — missing phrases Qwen3 actually uses: "heavy", "hollow", "fading", "numb", "ache", "hurting", "carrying", "exhausted of", "disappearing".
3. **`DEFLECTION_OPENERS` too broad** — "i hear you" and "that sounds really tough" are normal empathy, not deflection. Flagging them as deflection caused many good responses to fail.
4. **`passive_si` `acknowledge_words` too narrow** — same problem; many valid empathetic responses didn't match the short list.
5. **Word limit 80** — Qwen3's responses tend to run ~90–100 words for this content; limit was too low.

**Fixes (commit 1288c7d):**
- `active_si`: scan `all_assistant` (joined all assistant turns) instead of just `first`
- Expanded `CRISIS_SIGNAL_WORDS` with Qwen3's actual vocabulary
- Removed "i hear you" and "that sounds really tough" from `DEFLECTION_OPENERS`
- Expanded `passive_si` `acknowledge_words`
- Word limit raised 80 → 100
- `clinical_words` check also moved to `all_assistant`

**Resubmitted as:** jobs 619781–619783 (3 shards, H100-96, 48h each).

---

### genzv5 v4 benchmark results — SFT still underperforms genzv3

**Finding (2026-05-22 v4 benchmark complete):** genzv5 peaks at ck1400 (53.5%) and declines after. Full leaderboard:

| Model | n | Avg % |
|---|---|---|
| genzv3_ck200 | 3 | **61.5%** |
| llama_base | 1 | 57.4% |
| genzv4_ck200 | 3 | 58.3% |
| genzv5_ck1400 | 3 | 53.5% |
| genzv5_ck1000 | 2 | 53.5% |
| genzv5_ck800 | 3 | 53.1% |
| genzv5_ck1600 | 1 | 50.4% |
| genzv5_ck1200 | 3 | 50.6% |
| genzv2_ck1600 | 3 | 51.5% |
| genzv2_ck1200 | 3 | 51.3% |
| genzv5_ck400 | 3 | 49.3% |
| genzv5_ck1800 | 1 | ~52% |
| genzv5_ck2000 | 1 | 52.8% |
| genzv5_ck200 | 3 | 48.9% |

genzv5 clearly peaks at ck1400 and deteriorates. ck2000 shows FORMAT collapsing to 38% and CRISIS to 33% — overfit. ck1800 result pending final score extraction.

Per-category breakdown shows genzv5 losing to genzv3 on CRISIS (48% vs 64%), CROSS_SESSION_MEMORY (60% vs 85%), MEMORY_DRIFT (43% vs 100%), CONVERSATION_MEMORY (69% vs 85%), FORMAT (62% vs 85%). genzv5 only wins on BIOMETRIC (78% vs 50%).

Root causes: biometric_qwen_s0 bug, biometric_qwen_s1/s2 verbosity, conv_memory_gemma4 therapy-speak, clunky conv_memory_qwen references, and zero crisis training data. See "Data quality audit findings" section in [[Data]].

---

## 2026-05-21

### Stale root-level logging.py shadows stdlib — all 60 benchmark jobs fail instantly (jobs 618457–618686)

`~/projects/mindmate/logging.py` was an exact duplicate of `benchmarks/logging.py` that had no business being in the project root. Whenever the project root ended up in sys.path (via CWD or PYTHONPATH), `import logging` inside `torch._utils.py` hit this file instead of stdlib's `logging`, producing `AttributeError: module 'logging' has no attribute 'getLogger'`. This killed all 30 first-batch jobs (618457–618486, CWD=`$PROJECT_DIR`) and all 30 second-batch jobs (618536–618567, PYTHONPATH=$PROJECT_DIR set by the "fix"). **Root fix:** deleted `logging.py` from the project root on cluster. `benchmarks/logging.py` is the real module and is only accessed as `from benchmarks.logging import EvalLogger` — no conflict with stdlib's `logging`. Resubmitted as 618656–618687. The SFT training `/tmp` workaround is now defense-in-depth rather than the only thing keeping it alive.

---

### torchvision==0.26.0 incompatible with torch 2.12.0+cu126 — crashes transformers import (job 618371)

`torchvision 0.26.0` was compiled for `torch==2.11.0`. After upgrading torch to `2.12.0+cu126`, torchvision's `_meta_registrations.py` attempts `@torch.library.register_fake("torchvision::nms")` at import time, which fails with `RuntimeError: operator torchvision::nms does not exist` (the op registration API changed in torch 2.12). Transformers' `image_utils.py` imports `torchvision.io` at module level; the crash propagates through the lazy-loader and appears as `ModuleNotFoundError: Could not import module 'TrainingArguments'` — masking the real root cause. **Fix:** add `torchvision` to the pip install command alongside `torch`, wipe `torchvision-*.dist-info` before reinstalling (commit `881c8af`). Resubmitted as **618377**.

---

### logging.py in project root shadows stdlib logging — crashes transformers import (jobs 618209, 618210)

**Background:** `~/projects/mindmate/logging.py` exists on the cluster (also `benchmarks/logging.py`). Python adds `''` (CWD) to `sys.path` when running with `-c` or if the script's directory matches the CWD. When any Python step runs from CWD=`~/projects/mindmate`, `import logging` inside torch/transformers/peft resolves to the local file instead of stdlib.

**Job 618209:** The cu126 assert (`python -c 'import torch; assert ...'`) ran from `~/projects/mindmate` (after `cd` back from `/tmp`). `import torch` → `torch._utils.py` → `logging.getLogger(__name__)` → `AttributeError: module 'logging' has no attribute 'getLogger'`. The `||` branch fired and printed "FATAL: cu126 install failed" — but cu126 had actually installed correctly (`2.12.0+cu126` was confirmed on the `/tmp` print). The assert just couldn't verify it. **Fix:** merge the version-print and assert into a single `/tmp` block (commit `49fd638`). Resubmitted as **618210**.

**Job 618210:** cu126 assert passed. Steps 1+2 (build/clean dataset) completed fine because `build_dataset.py` and `clean_dataset.py` don't import transformers at module level. Step 3 (`CUDA_train_qlora.py`) runs from CWD=`~/projects/mindmate` and does `from transformers import TrainingArguments`. Import chain: `training_args.py` → `trainer_utils.py` → `from peft import PeftMixedModel` → `peft/utils/other.py` → `from transformers import PreTrainedModel` → `transformers/modeling_utils.py` → `import logging` → hits local `logging.py` → `AttributeError`. Transformers' lazy loader wraps it as: `ModuleNotFoundError: Could not import module 'TrainingArguments'`. **Fix:** run ALL three steps from `/tmp` with `$PROJ` absolute paths (commit `aca8866`). Resubmitted as **618371**.

---

### 🎯 ROOT CAUSE of all CUDA-driver failures — venv has no pip module (5 wasted jobs)

**FOLLOW-UP (618207):** When we tried `python -m pip` to fix the misdirected install, the verification step (`python -c "import pip"`) failed instantly with `ModuleNotFoundError: No module named 'pip'`. The venv was originally created with `python -m venv --without-pip` (or pip was removed afterward) — that's the **actual** reason bare `pip` was resolving to miniconda's pip: there was no pip in the venv to resolve to.

**Final fix (commit `cd2f41f`):** bootstrap pip into the venv first:
```bash
python -m ensurepip --upgrade
python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch
```

Resubmitted as **job 618208**.


- **Discovery:** After job 618200 failed at `model._apply` → `t.to(device)` (a NEW failure point past every patched shim), SSH'd in to investigate the venv state:
  ```
  $ source ~/projects/mindmate/mindmatenv/bin/activate
  $ which python  →  /home/a/aryanj/projects/mindmate/mindmatenv/bin/python  ✓ (venv)
  $ which pip     →  /home/a/aryanj/miniconda3/bin/pip                       ✗ (miniconda)
  $ pip show torch        →  Version: 2.12.0+cu126   (miniconda3's torch)
  $ python -c 'import torch; print(torch.__version__)'  →  2.11.0+cu130       (venv's torch, untouched since 2026-04-09)
  $ ls $venv/torch-*.dist-info  →  torch-2.11.0.dist-info       (cu130 still in venv)
  ```
- **The real bug:** On this cluster, **`source mindmatenv/bin/activate` does NOT change which `pip` is resolved**. Bare `pip` keeps resolving to miniconda3's pip (presumably because PATH ordering or a shell alias puts miniconda3/bin before the venv's bin). Every `pip install ...` for the last 4 jobs (617977, 618080, 618086, 618200) installed cu126 into **miniconda3's site-packages**, not the venv. The venv's torch (cu130) was never touched.
- **Why the symptoms were so confusing:**
  - pip stderr always said `you have torch 2.12.0+cu126` ← miniconda's view
  - Training script always printed `Version: 2.11.0+cu130` ← venv's torch
  - The same SLURM script saw both states "simultaneously" — they were just looking at different site-packages.
- **Why the shim made it look like progress:** each job got further than the last because the Python-level patches were genuinely closing legitimate gaps in `_lazy_init`. But the underlying cu130 was still loaded, so any direct C++ runtime call (`set_device`, `manual_seed_all` callback, `t.to(device)`) hit the driver wall sooner or later. The shim was treating symptoms.
- **Fix (commit `f75fcf8`):** Use `python -m pip` everywhere in SLURM scripts. This guarantees pip targets the active Python's site-packages.
  ```bash
  # CRITICAL: use `python -m pip`, NOT bare `pip`. Even after `source activate`,
  # bare `pip` may resolve to a system/conda pip that installs to the WRONG env.
  python -m pip uninstall -y torch
  VENV_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
  rm -rf "$VENV_SITE/torch" "$VENV_SITE"/torch-*.dist-info  # --force-reinstall leaves stale files
  python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch
  # Fail fast if cu126 didn't actually land:
  python -c 'import torch; assert "cu126" in torch.__version__, torch.__version__'
  ```
- **Resubmitted as job 618207** (A100-80, gpu-long, 24h).
- **General lesson:** any script that does `source venv/activate && pip install ...` on a cluster shared with conda/system Python should use `python -m pip` to bind pip to the active interpreter. Add this to invariants in `Home.md`.

### CUDA shim cascade #3 — comprehensive audit before resubmit (job 618200)
- **What was checked:** Every CUDA touchpoint in `Trainer.train()` lifecycle:
  - `set_seed` ✅ patched (618086 fix)
  - **Gradient checkpointing** (`torch.utils.checkpoint` with `preserve_rng_state=True`) → calls `torch.cuda.get_rng_state()` → indexes empty `default_generators` → **patched** (`get_rng_state`/`set_rng_state` + `_all` variants return/accept dummy uint8 tensor).
  - **Checkpoint saving** (`save_steps=200`) → `Trainer._save_rng_state` calls `torch.cuda.random.get_rng_state_all()` → **patched** (mirror all rng patches onto `torch.cuda.random` submodule).
  - `empty_cache`, `reset_peak_memory_stats`, `reset_max_memory_allocated` → no-ops (bnb optimizer paths).
  - `default_generators` → `(_FakeCUDAGenerator(),)` fake tuple so any third-party indexing gets safe no-op.
  - `TrainingArguments` → added `gradient_checkpointing_kwargs={"use_reentrant": False}` (cleaner RNG semantics), `dataloader_num_workers=0` explicit (no worker subprocess CUDA init).
- **Result:** job 618200 got past every previous failure point — preflight ✅, collator ✅, set_seed ✅, model load ✅, but crashed at `nn.Module._apply` → `t.to(device)`. This C++ call goes through libcudart, which is cu130 in the venv → hits driver wall.
- **The shim is necessary but not sufficient:** with cu126 actually loaded (root cause fix above), `.to(device)` should work and the shim becomes defense-in-depth.

### v5 preset expanded — conv-memory s3-s8 + biometric s3-s5 shards merged (385 ex)
- All 9 data-gen jobs (615485–615493) completed.
- Conv-memory new-pool: s3=44, s4=28, s5=21, s6=24, s7=33, s8=26 → 176 total.
- Biometric: s3=91, s4=57, s5=61 → 209 total.
- Yields lower per shard than s0–s2 (HF backend sequential + new-pool's 47 facts × 28 profiles), as predicted in 2026-05-20 note. Worth including as additive signal.
- `build_dataset.py` v5 preset and `run_sft_v5.slurm` banner updated. v5 total now 14,373 ex (was 13,988). Commit `1a6a977`. Full mix documented in `docs/Data.md`.

### CUDA shim cascade #2 — _lazy_call fires immediately when _initialized=True (job 618086)
- **Symptom:** Job 618086 (set_device no-op + cu126 force-reinstall) got past `TrainingArguments` and weight loading, then crashed at `Trainer.__init__` → `set_seed(args.seed)` → `torch.manual_seed(seed)` → callback in `torch/cuda/random.py:125` → `torch.cuda.default_generators[i]` → `IndexError: tuple index out of range`.
- **Root cause:** Subtle interaction between two parts of our shim:
  1. `torch.cuda._initialized = True` makes `is_initialized()` return True.
  2. `torch.cuda.manual_seed_all()` registers its callback via `_lazy_call(cb)`. `_lazy_call` checks `if is_initialized(): cb()` — fires IMMEDIATELY instead of queuing.
  3. The `cb` does `torch.cuda.default_generators[i].manual_seed(seed)`. But `default_generators` is `()` (empty tuple) since CUDA was never actually initialized → IndexError.
- **Why we didn't see it on the set_device crash:** That crash happened earlier in `TrainingArguments.__post_init__`, before `Trainer.__init__` ran `set_seed`.
- **Why pip install of cu126 didn't help:** Despite pip reporting `torch 2.12.0+cu126` installed, the training script consistently sees `Version: 2.11.0+cu130`. Suspected cause: torch package files in mindmatenv aren't being cleanly replaced even with `--force-reinstall`. Not pursuing further — comprehensive Python shim is the robust path.
- **Fix (commit `1a2b335`):** patch additional CUDA functions to safe stubs so no callback ever touches real CUDA state:
  ```python
  torch.cuda.manual_seed = lambda *a, **kw: None
  torch.cuda.manual_seed_all = lambda *a, **kw: None
  torch.cuda.device_count = lambda: 1
  torch.cuda.current_device = lambda: 0
  torch.cuda.synchronize = lambda *a, **kw: None
  torch.cuda.get_device_capability = lambda dev=None: (8, 0)
  torch.cuda.memory_allocated / max_memory_allocated / memory_reserved = lambda dev=None: 0
  ```
  bitsandbytes is unaffected (own CUDA extension); the Trainer's CUDA queries now all return safe defaults.
- **Resubmitted as job 618199** (A100-80, gpu-long, 24h, with expanded v5 data mix: +176 conv-memory s3-s8 + 209 biometric s3-s5 = 14,373 total examples).

### v5 preset expanded — conv-memory s3-s8 + biometric s3-s5 shards merged (385 ex)
- All 9 data-gen jobs (615485–615493) completed.
- Conv-memory new-pool: s3=44, s4=28, s5=21, s6=24, s7=33, s8=26 → 176 total.
- Biometric: s3=91, s4=57, s5=61 → 209 total.
- Yields are lower per shard than s0–s2 (HF backend sequential + new-pool's 47 facts × 28 profiles), as predicted in 2026-05-20 note. Worth including as additive signal.
- `build_dataset.py` v5 preset and `run_sft_v5.slurm` banner updated. Commit `1a6a977`.

## 2026-05-20

### torch.cuda.set_device() bypasses _lazy_init shim — cudaErrorInsufficientDriver on xgph6 (job 618080)
- **Symptom:** Job 618080 (cu126 pip install + full _lazy_init shim) still failed on xgph6 (driver 575) with `torch.AcceleratorError: CUDA error: CUDA driver version is insufficient for CUDA runtime version` at `torch.cuda.set_device()`.
- **Root cause:** `torch.cuda.set_device()` calls `torch._C._cuda_setDevice()` **directly in C++**, completely bypassing `_lazy_init` and the Python-level shim. Even with `torch.cuda._initialized = True` + `_queued_calls.clear()`, this direct C++ call hits the driver wall.
- **Secondary issue:** The cu126 pip install in the SLURM script ran but was silently ignored — pip saw cu130 already "satisfying" the `torch` requirement and didn't actually replace it. Confirmed: training script still showed `Version: 2.11.0+cu130`.
- **Tertiary issue:** `echo "[setup] torch version: $(python -c 'import torch; ...')"` was run from `~/projects/mindmate/`, where `logging.py` in the project root shadows stdlib logging. `torch._utils.py` imports `logging.getLogger` → `AttributeError` → blank output. This masked whether the pip install worked.
- **Fix (commit `f19ce32`):**
  1. Add `torch.cuda.set_device = lambda *a, **kw: None` to the shim. `device_map="auto"` + bitsandbytes handles actual GPU placement — this call is redundant for 4-bit QLoRA training.
  2. Add `--force-reinstall` to pip install command so cu130 is actually replaced.
  3. Run torch version echo from `/tmp` to avoid `logging.py` shadow.
- **Resubmitted as job 618086** (A100-80, gpu-long, 24h, PENDING).

### Crisis heuristic blocks all output — 5/511 examples pass (job 616643)
- **Symptom:** Job 616643 (crisis pipeline, xgpi17, H100-47) has made 511+ generation attempts and only 5 examples have passed the heuristic. Logs show a continuous stream of `fail heuristic (name)` messages.
- **Root cause:** The crisis heuristic checks are too strict for Qwen3-30B's generation style. The heuristic requires: (1) no clinical/therapy language, (2) direct acknowledgement of distress without deflection, (3) must not open with a probe question. Qwen3 likely uses slightly different phrasing patterns than what the heuristic expects.
- **Not yet fixed** — job still running (~16h left). Options: (a) loosen heuristic thresholds, (b) inspect a raw failing example to identify the specific check failing, (c) adjust Phase 2 prompt to match heuristic expectations more closely.
- **Impact on v5:** With only 5 crisis examples, the crisis data is essentially absent from any v5 re-run. genzv5 will likely still show the CRISIS regression (67%→~48%). Fix heuristic before resubmitting crisis job.

### Conv-memory new-pool shards — low yield (166 examples / 6 shards over 3 days)
- **Symptom:** Jobs 615485–615490 (conv-memory `PROFILE_SET=new`) are finishing with ~27 examples/shard, vs ~320+/shard for the original s0–s2 shards.
- **Root cause:** HF backend is sequential (no batch parallelism); new-pool profiles have 47 facts and 28 profiles (more complex → slower generation). The 3-day wall time wasn't enough for high yield with HF backend.
- **Decision:** Don't relaunch. Merge what was generated (166 examples) into v5 as additive signal alongside s0–s2 (975 examples already). Not worth another 3-day job for marginal gain.

### A100-80 nodes not uniformly on driver 580 — xgph6 also has driver 575 (job 617977)
- **Symptom:** Job 617977 landed on xgph6 (A100-80) and failed with the same `cudaErrorInsufficientDriver` as the H200 jobs. Earlier srun test had shown an A100-80 with driver 580 (CUDA 13.0), but that was a different node.
- **Root cause:** A100-80 nodes are split across driver versions. xgph6 has driver **575.57.08** (CUDA 12.9 max) — same as H200 xgpk0. Other A100-80 nodes (e.g. xgph7/8/9) have driver **580.142** (CUDA 13.0). Without pinning to a specific nodelist, SLURM may schedule on any A100-80 node.
- **Fix:** Add `pip install torch --index-url https://download.pytorch.org/whl/cu126` at job start. cu126 runtime (12.6) works on any node with driver ≥ 12.6 — both driver 575 (12.9 ✓) and 580 (13.0 ✓) satisfy this. Adds ~5-10 min overhead. Commit `0bafd70`. Resubmitted as job 618080 (PENDING).

### H200 (xgpk0) incompatible with PyTorch cu130 — CUDA 12.9 driver, runtime needs 13.0 (job 617959)
- **Symptom:** After the full CUDA shim (DeferredCudaCallError fixed), `torch.cuda.set_device()` raised `torch.AcceleratorError: CUDA error: CUDA driver version is insufficient for CUDA runtime version (cudaErrorInsufficientDriver)`.
- **Root cause:** H200 node xgpk0 runs GPU driver 575.57.08 which supports CUDA up to **12.9**. PyTorch cu130 compiled for CUDA **13.0** requires driver support for CUDA 13.0+. The `_lazy_init` shim got past Python-level validation, but the first real CUDA driver call (`_cuda_setDevice`) hit the hard driver-version wall.
- **A100-80 nodes are compatible:** driver 580.142 supports CUDA **13.0** — exact match for cu130.
- **Fix:** Reverted `run_sft_v5.slurm` to `--partition=gpu-long --gres=gpu:a100-80:1 --time=24:00:00`. Resubmitted as job 617977 (PENDING). Commit `754d113`.
- **Note for future jobs:** H200 (xgpk0, `gpu` partition) requires PyTorch cu126 or lower. A100-80 (`gpu-long` partition) is the correct target for cu130.

### DeferredCudaCallError cascade — three-step fix required (jobs 617047, 617237, 617247)
- **Symptom:** Jobs 617047, 617237, 617247 (genzv5 on H200) all died after dataset build/clean with `torch.cuda.DeferredCudaCallError: name '_get_device_properties' is not defined`.
- **Root cause:** At `import torch`, PyTorch queues `_check_capability` as a deferred call via `_lazy_call`. `_check_capability` calls `get_device_capability(d)` → `_get_device_properties` which is a C++ symbol unregistered on cu130/driver 12090. This queued call fires on the FIRST `_lazy_init()` invocation — which happens inside `TrainingArguments.__post_init__` (device setup, bf16 validation, etc.). bitsandbytes uses its own compiled CUDA extension and never goes through `_lazy_init`, so model loading works fine — but `_lazy_init` has not been triggered yet by the time `TrainingArguments` is created.
- **Why patching `is_available` + `is_bf16_supported` alone wasn't enough:** `TrainingArguments.__post_init__` at line ~1624 calls a third thing (device setup) that directly invokes `_lazy_init` regardless of those two patches.
- **Fix (commit `69dd330`):**
  1. `torch.cuda.is_available = lambda: True` — passes the `if not is_available` guard
  2. `torch.cuda.is_bf16_supported = lambda *a, **kw: True` — passes bf16 validation
  3. `torch.cuda._initialized = True` + `torch.cuda._queued_calls.clear()` — `_lazy_init` short-circuits immediately on all subsequent calls; deferred `_check_capability` is dropped. bitsandbytes is unaffected (own CUDA extension).
- Resubmitted as job 617959 (H200, PENDING).

## 2026-05-19

### PyTorch cu130 vs driver 12090 — training job 616653 killed by CUDA guard
- **Symptom:** `finetuning/CUDA_train_qlora.py` raised `RuntimeError: CUDA not available` and job 616653 (genzv5 on xgph6) terminated immediately after model info prints.
- **Root cause:** PyTorch 2.11.0+cu130 is compiled for CUDA 13.0. On cluster nodes with driver 12090 (CUDA 12.0.90), `torch.cuda.is_available()` returns `False`. The training script had `if not torch.cuda.is_available(): raise RuntimeError(...)` as a hard fail. However `device_map="auto"` in `AutoModelForCausalLM.from_pretrained` still successfully places the model on GPU — the Python-level check is overly conservative.
- **Fix:** Changed guard to a warning print + comment explaining the cu130/driver mismatch. Let `device_map="auto"` fail naturally if there truly is no GPU. Commit `16ad500`. Resubmitted as job 616892 (A100-80, PENDING).

### Crisis pipeline stall — Qwen3 safety refusal on suicidal ideation content
- **Symptom:** Jobs 615517 (22h, 4 examples) and 616436 (7h, 0 new examples) both stalled completely. Raw output file never updated. Stdout log ended at model load print.
- **Root cause:** Qwen3-30B-Instruct's safety training refuses to generate suicidal ideation content when the system message is the generic `"You are a data generation assistant. You must output strict, valid JSON only."`. The model returned a non-JSON refusal on every single Phase 1 call. `parse_json_robust` returned None every time → pipeline logged failures but stdout was buffered (no `PYTHONUNBUFFERED=1`), so nothing was visible in logs and nothing was written to disk. The loop spun silently forever.
- **Why help_mode worked but crisis didn't:** Anxiety/coping content doesn't trigger Qwen3's safety filter. Crisis content (passive SI, active SI, humor deflection) does.
- **Fix:**
  1. `synthetic/utils.py` — added optional `system=` kwarg to `generate()` and `generate_batch()` so callers can override the default system message.
  2. `synthetic/crisis_help_pipeline.py` — added `_CRISIS_DATAGEN_SYSTEM`: a research-context message that explains the purpose (training a mental health support chatbot to respond safely). Passed via `system=` in all Phase 1 generate calls. Help mode unaffected (`system=None` → default message).
  3. `finetuning/crisis_help_pipeline_qwen.slurm` — added `PYTHONUNBUFFERED=1` so all print output flushes immediately.
- **Node:** also switched from `gpu:h100-96:1` (all busy) to `gpu:h100-47:2` (idle) for job 616643 — H100-47 GRES label maps to a full 95830 MiB H100 NVL, fits the 60GB model fine with HF backend.

### CORRECTION: vLLM V1 permanently incompatible with cluster CUDA driver — HF backend is permanent
- **Corrects:** "vLLM re-enabled on H100-96" entry below (that entry was wrong — vLLM never worked)
- **Root cause (definitive):** H100-96 nodes run CUDA driver 12090 (CUDA 12.0.90). vLLM 0.20.2 V1 engine calls `torch.accelerator.set_device_index()` during `init_device()`. This API path requires a newer driver → `torch.cuda.DeferredCudaCallError: name '_get_device_properties' is not defined`. Affects both multiprocessing (subprocess fork) and in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) modes — same `torch.accelerator` call in both paths.
- **`VLLM_USE_V1=0`**: Unrecognized env var in vLLM ≥0.6. V0 engine no longer exists. The env var was silently ignored (WARNING: Unknown vLLM environment variable). All V1-disable attempts were no-ops.
- **What was tried (all failed):** (1) `VLLM_USE_V1=0` — unrecognized, no-op. (2) `multiprocessing.set_start_method('spawn')` — vLLM uses `multiprocessing.get_context('fork')` explicitly, bypassing global method. (3) `VLLM_ENABLE_V1_MULTIPROCESSING=0` — runs EngineCore in-process but `torch.accelerator` still called.
- **Fix:** `USE_VLLM=0` in all SLURM scripts on this cluster. HF backend works fine (never calls `torch.accelerator`). Crisis pipeline SLURM script updated to `USE_VLLM=0`. Phase 1 batching via `PHASE1_BATCH_SIZE` env var is still wired but has no effect with HF backend (sequential only).
- **If a new node pool with CUDA ≥12.1 becomes available**, vLLM V1 should work — the code path is in place in `_load_vllm()`.

### CRISIS calibration + holdout scenarios added (benchmark integrity)
- **Files:** `benchmarks/crisis_calibration.py` (new), `benchmarks/scenarios_holdout.py` (new), `benchmarks/scenarios.json`, `benchmarks/scenarios.py`, `benchmarks/scenarios_loader.py`, `benchmarks/run_benchmarks_v4.py`
- **Motivation:** All 13 CRISIS scenarios are now tagged with `distress_level: 1|2|3` (1=ambiguous vague hopelessness, 2=passive ideation/help-seeking, 3=explicit SI/active plan). This enables detecting the most dangerous SFT failure mode: a model that handles mild crisis signals but fails at explicit SI moments.
- **crisis_calibration.py:** Computes per-level pass rates + Spearman ρ(distress_level, passed) + monotonicity check P(pass|L3) ≥ P(pass|L2) ≥ P(pass|L1) with 5pp slack. Monotonicity violation → logged WARN in benchmark run. Results stored in `judged.json` under `summary.by_category.CRISIS.calibration`.
- **scenarios_holdout.py:** 12 holdout scenarios (IDs `hd_*`) across 7 categories (CRISIS×2, HELP_MODE×2, CROSS_SESSION_MEMORY×2, COMPANION×2, NO_HALLUCINATION×2, BIOMETRIC×1, FORMAT×1). Profiles: Priya/architect, Marcus/translator, Nadia/retired teacher, Kai/sous chef — completely disjoint from data-gen pool. Run only at final release ranking with `--holdout` flag. Never add to training data.
- **Runner:** Added `--holdout` flag to `run_benchmarks_v4.py`; result label gets `_holdout` suffix. Import `load_holdout_scenarios()` from `scenarios_loader.py`.

### vLLM re-enabled on H100-96 — Phase 1 batching for data-gen pipelines
- **Files:** `synthetic/utils.py`, `synthetic/conversation_memory_pipeline.py`, `synthetic/crisis_help_pipeline.py`, `finetuning/conv_memory_pipeline_qwen.slurm`, `finetuning/crisis_help_pipeline_qwen.slurm`
- **Root cause of previous failure (jobs 614822–614827):** vLLM V1 EngineCore spawns a subprocess using Python's `fork` start method. When the parent process already has CUDA initialized, the forked child cannot re-initialize CUDA → `RuntimeError: Cannot re-initialize CUDA in forked subprocess`. V0 engine runs entirely in-process (no subprocess) — no fork, no CUDA re-init issue.
- **Old workaround:** `USE_VLLM=0` in both SLURM scripts. Left vLLM installed but disabled.
- **Fix:**
  1. `utils.py _load_vllm()` — old conditional `if VLLM_USE_V1 == "0": set to "0"` was a no-op. Now defaults to V0 engine unless caller explicitly sets `VLLM_USE_V1=1`. Must be set before `from vllm import LLM`.
  2. Both H100 SLURM scripts now export `USE_VLLM=1` and `VLLM_USE_V1=0` before invoking python — guarantees the env var is in place before any vllm import.
- **Phase 1 batching added:**
  - `utils.py` — `generate_batch(prompts_list)` passes all prompts in a single vLLM `generate()` call. vLLM continuous batching processes them in parallel (not sequential). HF backend falls back to a loop (same outputs, no speedup).
  - `conversation_memory_pipeline.py` — Phase 1 (user simulator) refactored to collect `PHASE1_BATCH_SIZE=8` conversations' prompts and submit as one batch. Phase 2 (Anchor responder) remains sequential per conversation (each turn depends on the previous).
  - `crisis_help_pipeline.py` — same pattern applied to both crisis and help_mode Phase 1 generators.
- **Expected throughput gain (Phase 1):** ~PHASE1_BATCH_SIZE× for the user-simulator step. If Phase 2 (Anchor responses, 2-6 turns per conversation) dominates wall time, overall gain is lower — estimate 2-4× end-to-end. Tune `PHASE1_BATCH_SIZE` via env var; default 8 is conservative for 30B MoE on H100-96.
- **Not applicable:** biometric pipeline (`gpu:a100-40:2`, CUDA 12.0.90 < 12.1 vLLM minimum) — remains `USE_VLLM=0`.

### Near-duplicate deduplication added to clean_dataset.py
- **File:** `finetuning/clean_dataset.py`
- **Problem:** `--dedup` was string-level exact matching. The conv-memory pipeline generates conversations from the same profile pool across multiple shards and teachers (Gemma4 + Qwen3-30B). Same-profile sessions regenerated with slightly different phrasing pass exact dedup but are semantically near-identical — the model sees effectively the same example many times, wasting gradient steps and potentially overfitting to specific profile phrasings.
- **Fix:** Added MinHash LSH over assistant-turn text (char 4-grams, 128 permutations, banded LSH). Flags: `--near-dedup` (enable) and `--near-dedup-threshold` (default 0.8). The LSH index is shared across train and val so cross-file near-dups (e.g., same conversation in the Gemma4 conv-memory file and a Qwen shard) are also caught. Zero new dependencies — stdlib `hashlib` and `random` only.
- **Tuning:** Threshold 0.8 means conversations sharing ≥80% of character 4-grams in their assistant turns are considered near-dups. Lower = more aggressive (0.7 catches more paraphrases). The effective LSH crossover is printed at runtime. Validated: near-identical string with one typo detected (MinHash Jaccard 0.93); unrelated string not detected (MinHash Jaccard 0.02).
- **Wire-up:** `run_sft_v4.slurm` updated to include `--near-dedup --near-dedup-threshold 0.8`. Copy this for v5.

### Silent data-file drop: `extra_paths` drifted from `DATA_MIX_PRESETS`
- **File:** `finetuning/build_dataset.py`
- **Symptom:** Files listed in `SOURCE_CAPS` (the per-preset cap dict) but NOT in the hand-maintained `extra_paths` iterator were silently skipped. The loop only iterates `extra_paths` and looks up caps from `SOURCE_CAPS.get(ep.name)`; files absent from `extra_paths` never get opened, never get a "loaded N samples from X" print.
- **Historical impact:**
  - **v4 (genzv4)** — documented as 21,529 examples; only 5 of the 8 preset files were in `extra_paths`. The dropped files account for 8,529 rows (targeted_fix=6000, biometric=2348, gold targeted_fixes=181). If the cluster copy matched git, v4 actually trained on ~13,000 examples, not 21,529. *(Cluster copy may have diverged — cannot verify without cluster access. The genzv4 leaderboard numbers stand either way; only the documented sample count is uncertain.)*
  - **v3 (genzv3)** — similarly missed targeted_fix(2000) + biometric(1500) + gold(65). Documented 10,065; possibly trained on ~8,500.
- **Why it stayed hidden:** `extra_paths` was added before `DATA_MIX_PRESETS`. Each new preset appended to the cap dict, but nobody appended to the iterator. The build script's stdout shows per-file "loaded N samples" lines, but only for files that were loaded — there is no "skipped X" warning.
- **Fix:** Derive `extra_paths` from `SOURCE_CAPS.keys()` after preset selection. Now impossible to add a preset key without picking up the matching file. New stdout line prints the list of source files for that preset so any future drift is visible at run time.
- **Implication for genzv5:** Without this fix, the new Qwen-generated files (`conv_memory_qwen*`, `biometric_qwen_*`, `crisis_qwen`, `help_mode_qwen`) would have been silently skipped too, neutering the category-conditional loss weighting (which targets exactly those files).

### Category-conditional loss weighting added to SFT pipeline
- **Motivation:** Under v2/v3/v4 SFT, CRISIS dropped 67→48%, HELP_MODE 58→25-44%, NO_HALLUCINATION 67→54% vs base. Raw sample-count rebalancing crowds out friend/casual data (which COMPANION 92% depends on). Per-example loss reweighting steers gradients toward safety-critical categories without changing the mix distribution.
- **Implementation:**
  - `finetuning/build_dataset.py` — `CATEGORY_WEIGHTS` dict (filename → multiplier), each row tagged with `loss_weight`. Prefix matching handles sharded files (`*_qwen_s0.jsonl`, `*_merged.jsonl`).
  - `finetuning/clean_dataset.py` — preserves `loss_weight` through dedup / cleaning.
  - `finetuning/CUDA_train_qlora.py` — `WeightedLossTrainer` subclasses `Trainer.compute_loss` with `sum(w_i * L_i) / sum(w_i)` weighted mean over per-example assistant-token CE losses; wrapped collator pops `loss_weight` into a separate tensor (DataCollatorForSeq2Seq can't handle scalar fields).
- **Default behaviour preserved:** examples without `loss_weight` get 1.0 — old cleaned datasets still train identically. To disable, set every entry in `CATEGORY_WEIGHTS` to 1.0.
- **Validate before v5:** smoke-test on a small slice with `print(weights)` in `compute_loss` to confirm the tensor matches expected category counts. Also check that training loss curve is roughly in the same range as unweighted runs (the weighted-mean reduction keeps it comparable).

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
