#!/usr/bin/env python3
"""
Run the full MindMate QLoRA pipeline for Gemma 4 E2B IT.
Uses the same build/clean steps but calls CUDA_train_qlora_gemma4_e2b.py.
"""

import os
import sys
import subprocess
import shlex
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINETUNING_DIR = PROJECT_ROOT / "finetuning"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "conversations_raw"
CLEAN_DATA_DIR = DATA_DIR / "conversations_cleaned"

RAW_TRAIN = RAW_DATA_DIR / "mindmate_train.jsonl"
RAW_VAL = RAW_DATA_DIR / "mindmate_val.jsonl"

CLEAN_TRAIN = CLEAN_DATA_DIR / "mindmate_train.jsonl"
CLEAN_VAL = CLEAN_DATA_DIR / "mindmate_val.jsonl"

BASE_MODEL_ID = "google/gemma-4-e2b-it"
ITERS = 1600
ADAPTER_PATH = PROJECT_ROOT / "adapters" / "CUDA_mindmate_gemma4_e2b"


def run(cmd, cwd=None, env=None):
    if isinstance(cmd, (list, tuple)):
        printable = " ".join(shlex.quote(str(c)) for c in cmd)
    else:
        printable = cmd
    print(f"\n>>> Running: {printable}\n")
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def script_path(name):
    p = FINETUNING_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Expected script {p} but it does not exist")
    return p


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def check_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — training would run on CPU")
    return f"CUDA available: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}"


def step_build_dataset():
    print("Building dataset...")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    script = script_path("build_dataset.py")
    run([sys.executable, str(script), "--model", "gemma4_e2b"], cwd=PROJECT_ROOT)

    if not RAW_TRAIN.exists() or not RAW_VAL.exists():
        raise FileNotFoundError(
            f"[build_dataset] Expected {RAW_TRAIN} and {RAW_VAL} but did not find them."
        )

    train_count = count_lines(RAW_TRAIN)
    val_count = count_lines(RAW_VAL)
    print(f"[build_dataset] Done. Raw files:\n  {RAW_TRAIN}\n  {RAW_VAL}")
    print(f"[build_dataset] #train samples: {train_count}")
    print(f"[build_dataset] #val samples  : {val_count}")


def step_clean_dataset():
    print("Cleaning dataset...")
    CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    script = script_path("clean_dataset.py")
    cmd = [
        sys.executable, str(script),
        "--train-in", str(RAW_TRAIN),
        "--val-in", str(RAW_VAL),
        "--train-out", str(CLEAN_TRAIN),
        "--val-out", str(CLEAN_VAL),
        "--ensure-assistant-last",
        "--drop-min-turns", "4",
        "--dedup",
    ]
    run(cmd, cwd=PROJECT_ROOT)

    if not CLEAN_TRAIN.exists() or not CLEAN_VAL.exists():
        raise FileNotFoundError(
            f"[clean_dataset] Expected {CLEAN_TRAIN} and {CLEAN_VAL} but did not find them."
        )

    train_count = count_lines(CLEAN_TRAIN)
    val_count = count_lines(CLEAN_VAL)
    print(f"[clean_dataset] Done. Clean files:\n  {CLEAN_TRAIN}\n  {CLEAN_VAL}")
    print(f"[clean_dataset] #clean train samples: {train_count}")
    print(f"[clean_dataset] #clean val samples  : {val_count}")


def step_train_qlora():
    run(
        [sys.executable, str(FINETUNING_DIR / "CUDA_train_qlora_gemma4_e2b.py"), "--iters", str(ITERS)],
        cwd=PROJECT_ROOT,
    )


def write_metadata():
    from datetime import datetime
    adapter_dir = PROJECT_ROOT / "adapters" / "CUDA_mindmate_gemma4_e2b"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    meta = adapter_dir / "training_info.txt"
    with open(meta, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MindMate — Gemma 4 E2B IT SFT Adapter\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Date trained     : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Base model       : google/gemma-4-e2b-it\n")
        f.write(f"Output dir       : adapters/CUDA_mindmate_gemma4_e2b/\n")
        f.write(f"Inference script : inference/CUDA_chat_gemma4_e2b.py\n\n")
        f.write("--- Hyperparameters ---\n")
        f.write(f"Method           : QLoRA (4-bit NF4 + PEFT)\n")
        f.write(f"lora_r           : 8\n")
        f.write(f"lora_alpha       : 16\n")
        f.write(f"lora_dropout     : 0.05\n")
        f.write(f"learning_rate    : 1e-5\n")
        f.write(f"max_steps        : 1600\n")
        f.write(f"batch_size       : 4 (per device) x 2 (grad accum) = 8 effective\n")
        f.write(f"lr_scheduler     : cosine\n")
        f.write(f"warmup_ratio     : 0.03\n")
        f.write(f"max_seq_len      : 2048\n")
        f.write(f"compute_dtype    : bfloat16\n")
        f.write(f"attn_impl        : eager (required for Gemma 4)\n\n")
        f.write("--- Data Mix (preset: gemma4_e2b) ---\n")
        f.write("  Source                            Samples    %\n")
        f.write("  -------------------------------------------- \n")
        f.write("  synthetic_train_transition.jsonl   3000     30%\n")
        f.write("  synthetic_train_therapist_.jsonl   2500     25%\n")
        f.write("  synthetic_train_casual.jsonl       2000     20%\n")
        f.write("  synthetic_train.jsonl              1500     15%\n")
        f.write("  synthetic_train_friend_1.jsonl     1000     10%\n")
        f.write("  -------------------------------------------- \n")
        f.write("  TOTAL                             10000    100%\n\n")
        f.write("--- Loss masking ---\n")
        f.write("Only model turns supervised. Boundaries:\n")
        f.write("  start : <start_of_turn>model\\n\n")
        f.write("  end   : <end_of_turn>\n\n")
        f.write("--- Notes ---\n")
        f.write("eager attention required — Gemma 4 SDPA can cause NaN gradients in 4-bit.\n")
        f.write("No trust_remote_code needed (native transformers support).\n")
    print(f"[metadata] Written to {meta}")


def main():
    print("=== MindMate QLoRA Pipeline — Gemma 4 E2B IT ===")
    print(check_cuda())
    print(f"Project root   : {PROJECT_ROOT}")
    print(f"Data dir       : {DATA_DIR}")
    print(f"Base model     : {BASE_MODEL_ID}\n")

    step_build_dataset()
    step_clean_dataset()
    step_train_qlora()
    write_metadata()

    print("\nAll steps completed successfully.\n")


if __name__ == "__main__":
    main()
