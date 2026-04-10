"""
MindMate GGUF Export Script (CUDA cluster version)
Merges a LoRA adapter into its base model, converts to GGUF F16,
then quantizes to Q4_K_M.

Usage:
    python scripts/export_gguf_cuda.py --model llama_sft_ck200
    python scripts/export_gguf_cuda.py --model llama_dpo_ck200
    python scripts/export_gguf_cuda.py --model qwen25_dpo_ck200
    python scripts/export_gguf_cuda.py --model genz
    python scripts/export_gguf_cuda.py --model qwen

Outputs land in exports/<model_name>/
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORTS_DIR = PROJECT_ROOT / "exports"
LLAMA_CPP_DIR = PROJECT_ROOT / "llama.cpp"
CONVERT_SCRIPT = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
QUANTIZE_BIN = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"

MODELS = {
    # SFT baseline — Llama 3.2 3B checkpoint-200
    "llama_sft_ck200": {
        "base": "meta-llama/Llama-3.2-3B-Instruct",
        "adapter": PROJECT_ROOT / "adapters" / "CUDA_mindmate_llama32b" / "checkpoint-200",
        "out_name": "mindmate_llama_sft_ck200",
    },
    # DPO — Llama 3.2 3B: merge SFT ck200 first, then DPO LoRA on top
    "llama_dpo_ck200": {
        "base": "meta-llama/Llama-3.2-3B-Instruct",
        "sft_adapter": PROJECT_ROOT / "adapters" / "CUDA_mindmate_llama32b" / "checkpoint-200",
        "adapter": PROJECT_ROOT / "adapters" / "CUDA_mindmate_llama32b_dpo_ck200",
        "out_name": "mindmate_llama_dpo_ck200",
    },
    # DPO — Qwen2.5-3B: merge SFT ck200 first, then DPO LoRA on top
    "qwen25_dpo_ck200": {
        "base": "Qwen/Qwen2.5-3B-Instruct",
        "sft_adapter": PROJECT_ROOT / "adapters" / "CUDA_mindmate_qwen25_3b" / "checkpoint-200",
        "adapter": PROJECT_ROOT / "adapters" / "CUDA_mindmate_qwen25_3b_dpo_ck200",
        "out_name": "mindmate_qwen25_dpo_ck200",
    },
    # Legacy
    "genz": {
        "base": "meta-llama/Llama-3.2-3B-Instruct",
        "adapter": PROJECT_ROOT / "adapters" / "genz",
        "out_name": "mindmate_genz_llama32_3b",
    },
    "qwen": {
        "base": "Qwen/Qwen3-1.7B",
        "adapter": PROJECT_ROOT / "adapters" / "CUDA_mindmate_qwen3_1p7b",
        "out_name": "mindmate_qwen3_1p7b",
    },
}


def step_merge(cfg: dict, merged_dir: Path):
    has_sft = "sft_adapter" in cfg

    print(f"\n[STEP 1] Merging adapter(s) into base model...")
    print(f"  Base:    {cfg['base']}")
    if has_sft:
        print(f"  SFT adapter: {cfg['sft_adapter']}")
    print(f"  Adapter: {cfg['adapter']}")
    print(f"  Output:  {merged_dir}")

    if has_sft and not Path(cfg["sft_adapter"]).exists():
        raise FileNotFoundError(f"SFT adapter not found: {cfg['sft_adapter']}")
    if not cfg["adapter"].exists():
        raise FileNotFoundError(f"Adapter not found: {cfg['adapter']}")

    merged_dir.mkdir(parents=True, exist_ok=True)

    print("  Loading base model in bfloat16...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base"],
        torch_dtype=torch.bfloat16,
        device_map="cpu",          # CPU merge — avoids VRAM pressure
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if has_sft:
        print("  Loading SFT adapter and merging...")
        model = PeftModel.from_pretrained(model, str(cfg["sft_adapter"]))
        model = model.merge_and_unload()
        print("  SFT merged. Loading DPO adapter...")
    else:
        print("  Loading adapter...")

    model = PeftModel.from_pretrained(model, str(cfg["adapter"]))

    print("  Merging and unloading LoRA weights...")
    model = model.merge_and_unload()

    print("  Saving merged model...")
    model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    print(f"  [OK] Merged model saved to {merged_dir}")


def step_convert(merged_dir: Path, f16_gguf: Path):
    print(f"\n[STEP 2] Converting to GGUF F16...")

    if not CONVERT_SCRIPT.exists():
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found at {CONVERT_SCRIPT}. "
            "Make sure llama.cpp is in the project root."
        )

    f16_gguf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(CONVERT_SCRIPT),
        str(merged_dir),
        "--outfile", str(f16_gguf),
        "--outtype", "f16",
    ]
    print(f"  CMD: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    size_gb = f16_gguf.stat().st_size / (1024 ** 3)
    print(f"  [OK] F16 GGUF saved: {f16_gguf} ({size_gb:.2f} GB)")


def step_quantize(f16_gguf: Path, q4_gguf: Path):
    print(f"\n[STEP 3] Quantizing to Q4_K_M...")

    if not QUANTIZE_BIN.exists():
        print(f"  [WARN] llama-quantize binary not found at {QUANTIZE_BIN}")
        print("  Compile it first:")
        print("    cd llama.cpp && cmake -B build && cmake --build build --config Release -j$(nproc)")
        print("  Skipping quantization — F16 GGUF is still usable.")
        return

    cmd = [str(QUANTIZE_BIN), str(f16_gguf), str(q4_gguf), "Q4_K_M"]
    print(f"  CMD: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    size_gb = q4_gguf.stat().st_size / (1024 ** 3)
    print(f"  [OK] Q4_K_M GGUF saved: {q4_gguf} ({size_gb:.2f} GB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, choices=list(MODELS.keys()),
        help="Which model to export"
    )
    parser.add_argument(
        "--skip-merge", action="store_true",
        help="Skip merge step if merged model already exists"
    )
    args = parser.parse_args()

    cfg = MODELS[args.model]
    out_dir = EXPORTS_DIR / cfg["out_name"]
    merged_dir = out_dir / "merged_hf"
    f16_gguf = out_dir / f"{cfg['out_name']}_f16.gguf"
    q4_gguf = out_dir / f"{cfg['out_name']}_q4_k_m.gguf"

    print(f"=== MindMate GGUF Export: {args.model} ===")

    if args.skip_merge and merged_dir.exists():
        print(f"[SKIP] Merge step skipped — using existing {merged_dir}")
    else:
        step_merge(cfg, merged_dir)

    step_convert(merged_dir, f16_gguf)
    step_quantize(f16_gguf, q4_gguf)

    print(f"\n=== Done ===")
    print(f"  F16 GGUF : {f16_gguf}")
    print(f"  Q4_K_M   : {q4_gguf}")


if __name__ == "__main__":
    main()
