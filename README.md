# Anchor

A finetuned, local mental-health companion that runs on Android. Models are trained with QLoRA on a CUDA cluster, exported to GGUF, and deployed in a React Native Android app via a llama.cpp JNI bridge.

**Production webapp:** https://tryanchor.me (FastAPI + SSE, DigitalOcean)
**Android app:** `anchor-app/` (React Native, package `com.pocketpal`)

---

## Best Model

**`adapters/genz/checkpoint-1200`** — Llama 3.2 3B SFT (genzv2_ck1200)
- Best SFT model overall (71% on v1 benchmark, stable across runs)
- DPO abandoned — all 3 DPO runs flat or worse than SFT
- GGUF: `exports/mindmate_genzv2_ck1200/`
- On device at `/sdcard/Download/mindmate.gguf` (Pixel 8a)
- Gen: ~5.4–6.0 TPS, TTFT: 6s (cached) / 60–66s (cold 1100-tok prompt)

**⚠️ Base model currently beats all SFT on v3 benchmark (51% vs 45%)** — root cause identified, fix running (see below).

---

## Stack

| Layer | Tool |
|---|---|
| Base model | Llama 3.2 3B Instruct |
| SFT | QLoRA (4-bit NF4, PEFT), `CUDA_train_qlora.py` |
| Teacher (datagen) | Gemma 4 26B A4B IT (bfloat16, ~52GB, A100-80) |
| Export | GGUF Q4_K_M via llama.cpp, `scripts/export_gguf_cuda.py` |
| Android runtime | JNI → llama.cpp (`android/llama/src/main/cpp/llama-android.cpp`) |
| Webapp | FastAPI + SSE (`deploy/server.py`) |
| Cluster | NUS SoC SLURM, `gpu-long` partition, A100-80 |

---

## Training Pipeline

```
Synthetic Data (Gemma 4 26B A4B IT teacher — teacher-as-Anchor mode)
        ↓
  SFT: CUDA_run_pipeline.py → CUDA_train_qlora.py
        ↓
  adapters/genz/checkpoint-1200  ← best SFT checkpoint
        ↓
  GGUF export: scripts/export_gguf_cuda.py
        ↓
  Android app (anchor-app/) or webapp (deploy/)
```

### Running SFT on cluster
```bash
cd ~/projects/mindmate
sbatch finetuning/run_sft_v4.slurm
```

### Exporting to GGUF
```bash
sbatch scripts/run_export_top3.slurm
# or directly:
python scripts/export_gguf_cuda.py --model genzv2_ck1200
```

---

## Data

### Active SFT training data

| File | Examples | Content |
|---|---|---|
| `data/synthetic_train_targeted_fix.jsonl` | 13,524 | help_mode + memory_recall |
| `data/synthetic_train_friend_1.jsonl` | 7,380 | Casual friend-style support |
| `data/synthetic_train_therapist_.jsonl` | 6,347 | Therapeutic dialogue |
| `data/synthetic_train_transition.jsonl` | 6,184 | Casual→emotional pivot |
| `data/synthetic_train.jsonl` | 5,565 | Grief/loss |
| `data/synthetic_train_casual.jsonl` | 5,000 | Non-distress casual |
| `data/synthetic_train_biometric.jsonl` | 2,348 | Biometric health context |
| `data/synthetic_train_targeted_fixes.jsonl` | 181 | Hand-crafted gold examples |
| `data/synthetic_train_conv_memory.jsonl` | generating | Multi-turn + memory (teacher-as-Anchor) |

---

## Benchmarks

58 scenarios, 9 categories, LLM judge (Gemma 4 26B A4B). **3-run averaged methodology** (temperature=0.7 causes variance).

```bash
# Single run
sbatch --gres=gpu:a100-80:1 --export=ALL,MODEL=genzv2_ck1200 benchmarks/run_benchmarks.slurm

# Average across runs
python benchmarks/average_results.py --since 20260509
```

**v3 averaged results (partial, /61 weighted):**

| Model | Avg % | n |
|---|---|---|
| llama_base | 51% | 3 |
| genzv3_ck200 | 45% | 4 |
| genzv4_ck200 | 42% | 3 |
| genzv2_ck1200 | ⏳ rerunning | — |
| genzv2_ck1600 | ⏳ rerunning | — |

---

## Cluster Access

```bash
ssh nus-student-cluster   # requires NUS VPN
squeue -u aryanj          # check jobs

# Push files to cluster (git pull blocked — GitHub SSH not allowed)
rsync -az -e "ssh -o LogLevel=QUIET" file.py nus-student-cluster:~/projects/mindmate/path/

# Sync results back
rsync -az -e "ssh -o LogLevel=QUIET" nus-student-cluster:~/projects/mindmate/benchmarks/results/ benchmarks/results/
```

**Partitions:** `gpu` (max 3h), `gpu-long` (max 3 days). All training on `gpu-long` with `a100-80`.
**Logs:** `~/logs/` on cluster (NOT `~/projects/mindmate/logs/`).
**GPU note:** Always use `--gres=gpu:a100-80:1` explicitly. H100-96 GRES unreliable — has fallen back to ~46GB nodes.

---

## Android App

App: `anchor-app/` (React Native, NOT `mindmate_app/` which is a stale scratch fork)

```bash
cd anchor-app
yarn typecheck
cd android && ./gradlew assembleDebug
```

---

## Project Structure

```
mindmate/
├── adapters/               # LoRA checkpoints
│   ├── genz/checkpoint-1200/   ← BEST MODEL (genzv2_ck1200)
│   ├── genzv3/checkpoint-200/  ← best genzv3
│   └── genzv4/checkpoint-200/  ← best genzv4
├── benchmarks/             # Benchmark suite + runner + results
│   ├── scenarios.py        # 58 scenarios, 9 categories
│   ├── run_benchmarks.py   # model loader + judge runner
│   ├── average_results.py  # 3-run averaging
│   └── results/            # JSON + MD per run
├── data/                   # Training data (JSONL)
├── exports/                # GGUF exports
├── finetuning/             # Training scripts
├── synthetic/              # Data generation pipelines
│   ├── conversation_memory_pipeline.py  ← active (teacher-as-Anchor)
│   └── utils.py            # TeacherModel.generate() + .chat()
├── deploy/                 # Production FastAPI server
└── scripts/                # Export + utility scripts
```
