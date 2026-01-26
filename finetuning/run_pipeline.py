#!/usr/bin/env python3
"""
Run the full MindMate QLoRA pipeline.

Layout this script expects (based on your current project):

mindmate/
├── adapters/
├── data/
│   ├── raw_data/
│   ├── cleaned_data/
│   ├── new_raw_data/              (unused here, ok to keep)
│   └── additional_training_samples.jsonl
├── mlx_llama32_3b/                # base Llama 3.2 3B
└── finetuning/
    ├── build_dataset.py
    ├── clean_dataset.py
    ├── chunk.py
    ├── run_pipeline.py   (this file)
"""

import os
import sys
import subprocess
import shlex
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ======================
# CONFIG – EDIT IF NEEDED
# ======================

# This file lives in mindmate/finetuning/, so project root is one level up
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Scripts are now in the same directory as this file
SCRIPTS_DIR = Path(__file__).resolve().parent

# Data dirs
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "new_raw_data"
CLEAN_DATA_DIR = DATA_DIR / "cleaned_data"
CHUNK_DATA_DIR = CLEAN_DATA_DIR / "chunked_3072"

# Raw data filenames (output of build_dataset.py)
RAW_TRAIN = RAW_DATA_DIR / "mindmate_train.jsonl"
RAW_VAL = RAW_DATA_DIR / "mindmate_val.jsonl"

# Clean data filenames (output of clean_dataset.py)
CLEAN_TRAIN = CLEAN_DATA_DIR / "mindmate_train_clean.jsonl"
CLEAN_VAL = CLEAN_DATA_DIR / "mindmate_val_clean.jsonl"

# Chunked files (output of chunk.py)
CHUNK_TRAIN = CHUNK_DATA_DIR / "train.jsonl"
CHUNK_VAL   = CHUNK_DATA_DIR / "valid.jsonl"

# Base model dir for tokenizer + MLX
BASE_MODEL_DIR = PROJECT_ROOT / "mlx_llama32_3b"

# Path to mlx_lm.lora binary
MLX_LORA_BIN = Path("/Users/aryanjain/miniconda3/envs/mindmatenv/bin/mlx_lm.lora")

# QLoRA / training config
MAX_LEN = 3072
OVERLAP = 256
ITERS = 1500
LR = 3e-5
BATCH_SIZE = 1
NUM_LAYERS = 10
ADAPTER_PATH = PROJECT_ROOT / "adapters" / "mindmate_llama32_3b_qlora_nl10_3072_lr3e5"
LOG_FILE = "run_qlora_nl10_3072_lr3e5.log"


# ======================
# Helpers
# ======================

def run(cmd, cwd=None, env=None):
    """Run a subprocess with pretty printing."""
    if isinstance(cmd, (list, tuple)):
        printable = " ".join(shlex.quote(str(c)) for c in cmd)
    else:
        printable = cmd
    print(f"\n>>> Running: {printable}\n")
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def script_path(name):
    p = SCRIPTS_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Expected script {p} but it does not exist")
    return p


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# ======================
# Steps
# ======================
def step_build_dataset():
    """
    Step 1: build raw dataset into data/raw_data/
    Assumes build_dataset.py writes mindmate_train.jsonl & mindmate_val.jsonl there.
    """
    print("Building dataset…")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    script = script_path("build_dataset.py")
    run([sys.executable, str(script)], cwd=PROJECT_ROOT)

    if not RAW_TRAIN.exists() or not RAW_VAL.exists():
        raise FileNotFoundError(
            f"[build_dataset] Expected {RAW_TRAIN} and {RAW_VAL} but did not find them. "
            "Check build_dataset.py output paths."
        )

    train_count = count_lines(RAW_TRAIN)
    val_count = count_lines(RAW_VAL)

    print(f"[build_dataset] Done. Raw files:\n  {RAW_TRAIN}\n  {RAW_VAL}")
    print(f"[build_dataset] #train samples: {train_count}")
    print(f"[build_dataset] #val samples  : {val_count}")


def step_clean_dataset():
    """
    Step 2: clean raw dataset into data/cleaned_data/
    clean_dataset.py must accept --train-in/--val-in/--train-out/--val-out.
    """
    print("Cleaning dataset…")
    CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)

    script = script_path("clean_dataset.py")
    cmd = [
        sys.executable,
        str(script),
        "--train-in",
        str(RAW_TRAIN),
        "--val-in",
        str(RAW_VAL),
        "--train-out",
        str(CLEAN_TRAIN),
        "--val-out",
        str(CLEAN_VAL),
        "--ensure-assistant-last",
        "--drop-min-turns",
        "4",
        "--dedup",
        "--merge-consecutive-user",
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


def step_chunk_dataset():
    """
    Step 3: chunk cleaned JSONL into 3072-token windows,
    stored under data/cleaned_data/chunked_3072/.
    """
    CHUNK_DATA_DIR.mkdir(parents=True, exist_ok=True)

    script = script_path("chunk.py")

    cmd_train = [
        sys.executable,
        str(script),
        "--in",
        str(CLEAN_TRAIN),
        "--out",
        str(CHUNK_TRAIN),
        "--model-dir",
        str(BASE_MODEL_DIR),
        "--max-len",
        str(MAX_LEN),
        "--overlap",
        str(OVERLAP),
    ]
    cmd_val = [
        sys.executable,
        str(script),
        "--in",
        str(CLEAN_VAL),
        "--out",
        str(CHUNK_VAL),
        "--model-dir",
        str(BASE_MODEL_DIR),
        "--max-len",
        str(MAX_LEN),
        "--overlap",
        str(OVERLAP),
    ]

    run(cmd_train, cwd=PROJECT_ROOT)
    run(cmd_val, cwd=PROJECT_ROOT)

    if not CHUNK_TRAIN.exists() or not CHUNK_VAL.exists():
        raise FileNotFoundError(
            f"[chunk_dataset] Expected {CHUNK_TRAIN} and {CHUNK_VAL} but did not find them."
        )
    print(
        "[chunk_dataset] Done. Chunked files:\n"
        f"  {CHUNK_TRAIN}\n"
        f"  {CHUNK_VAL}"
    )


def step_train_qlora():
    """
    Step 4: run QLoRA training via mlx_lm.lora on the chunked data,
    wrapped with nice + caffeinate and resource limits so the Mac stays usable.
    """

    if not MLX_LORA_BIN.exists():
        raise FileNotFoundError(f"mlx_lm.lora binary not found at {MLX_LORA_BIN}")

    ADAPTER_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Base MLX QLoRA command
    base_cmd = [
        str(MLX_LORA_BIN),
        "--model", "./mlx_llama32_3b",
        "--train",
        "--data", str(CHUNK_DATA_DIR),
        "--batch-size", str(BATCH_SIZE),
        "--iters", str(ITERS),
        "--save-every", "200",
        "--max-seq-length", str(MAX_LEN),
        "--num-layers", str(NUM_LAYERS),
        "--learning-rate", str(LR),
        "--grad-checkpoint",
        "--steps-per-eval", "200",
        "--steps-per-report", "50",
        "--adapter-path", str(ADAPTER_PATH),
    ]

    # Shell string with tee for logging
    train_part = " ".join(shlex.quote(c) for c in base_cmd)
    full_shell = f"{train_part} 2>&1 | tee {LOG_FILE}"

    # Wrap with nice + caffeinate
    caffeinated_cmd = [
        "nice", "-n", "15",
        "caffeinate", "-dims",
        "bash", "-o", "pipefail", "-c",
        full_shell,
    ]

    # Environment with resource limits
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    env["MLX_NUM_THREADS"] = "2"

    print("\n[train_qlora] Starting QLoRA under caffeinate with resource limits…")
    print(f"[train_qlora] OMP_NUM_THREADS={env['OMP_NUM_THREADS']}, "
          f"MLX_NUM_THREADS={env['MLX_NUM_THREADS']}, nice -n 10\n")

    run(caffeinated_cmd, cwd=PROJECT_ROOT, env=env)

    print(f"[train_qlora] Done. Adapters saved to {ADAPTER_PATH}")
    print(f"[train_qlora] Log file: {PROJECT_ROOT / LOG_FILE}")

def main():
    print("=== MindMate QLoRA Pipeline ===")
    print(f"Project root   : {PROJECT_ROOT}")
    print(f"Data dir       : {DATA_DIR}")
    print(f"Base model dir : {BASE_MODEL_DIR}\n")

    step_build_dataset()
    step_clean_dataset()
    step_chunk_dataset()
    step_train_qlora()

    print("\nAll steps completed successfully.\n")


if __name__ == "__main__":
    main()