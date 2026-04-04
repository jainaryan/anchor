#!/usr/bin/env python3
"""
Run the full MindMate QLoRA pipeline for Qwen3-1.7B.
Uses the same build/clean steps but calls CUDA_train_qlora_qwen.py.
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

BASE_MODEL_ID = "Qwen/Qwen3-1.7B"
ITERS = 1600
ADAPTER_PATH = PROJECT_ROOT / "adapters" / "CUDA_mindmate_qwen3_1p7b"


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
    run([sys.executable, str(script)], cwd=PROJECT_ROOT)

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
        [sys.executable, str(FINETUNING_DIR / "CUDA_train_qlora_qwen.py"), "--iters", str(ITERS)],
        cwd=PROJECT_ROOT,
    )


def main():
    print("=== MindMate QLoRA Pipeline — Qwen3-1.7B ===")
    print(check_cuda())
    print(f"Project root   : {PROJECT_ROOT}")
    print(f"Data dir       : {DATA_DIR}")
    print(f"Base model     : {BASE_MODEL_ID}\n")

    step_build_dataset()
    step_clean_dataset()
    step_train_qlora()

    print("\nAll steps completed successfully.\n")


if __name__ == "__main__":
    main()
