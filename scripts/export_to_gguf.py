
import os
import subprocess
import sys
import shutil
from pathlib import Path
from mlx_lm import fuse

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "mlx_llama32_3b"
ADAPTER_DIR = PROJECT_ROOT / "adapters" / "mindmate_llama32_3b_qlora_nl10_3072_lr3e5"
FUSED_MODEL_DIR = PROJECT_ROOT / "mlx_export" / "mindmate_fused"
GGUF_MODEL_PATH = PROJECT_ROOT / "exports" / "mindmate_llama32_3b_q4_k_m.gguf"
EXPORTS_DIR = PROJECT_ROOT / "exports"

# Internal checks
def check_dependencies():
    try:
        import mlx_lm
        print("[INFO] mlx_lm is installed.")
    except ImportError:
        print("[ERROR] mlx_lm is not installed. Please run: pip install mlx-lm")
        sys.exit(1)

    try:
        import huggingface_hub
        print("[INFO] huggingface_hub is installed.")
    except ImportError:
        print("[ERROR] huggingface_hub is not installed. Please run: pip install huggingface_hub")
        sys.exit(1)

def fuse_model():
    print(f"\n[STEP 1] Fusing base model and adapter...")
    print(f"Base: {MODEL_PATH}")
    print(f"Adapter: {ADAPTER_DIR}")
    
    if not ADAPTER_DIR.exists():
        print(f"[ERROR] Adapter directory not found: {ADAPTER_DIR}")
        sys.exit(1)
        
    FUSED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # We use the fuse_model CLI equivalent logic from mlx_lm
    # Since mlx_lm.fuse is a module, we can use the library function if available, 
    # but the simplest reliable way across versions is via CLI or subprocess if the python API is unstable.
    # However, mlx_lm 0.10+ has a CLI Entry point. Let's try subprocess to be safe and avoid API mismatches.
    
    cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", str(MODEL_PATH),
        "--adapter-path", str(ADAPTER_DIR),
        "--save-path", str(FUSED_MODEL_DIR)
    ]
    
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print("[ERROR] Fusing failed.")
        sys.exit(result.returncode)
    
    print("[INFO] Fusing complete.")

def convert_to_gguf():
    print(f"\n[STEP 2] Converting to GGUF...")
    
    # Download convert_hf_to_gguf.py if not present
    convert_script = PROJECT_ROOT / "scripts" / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print("[INFO] Downloading convert_hf_to_gguf.py from llama.cpp...")
        url = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"
        subprocess.run(["curl", "-o", str(convert_script), url], check=True)
    
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Quantization type: q4_k_m is a good balance for mobile/desktop
    # NOTE: The python script does NOT quantize. It converts to F16 or F32 GGUF.
    # To quantize, we need the `llama-quantize` binary which is compiled C++.
    # However, converting to GGUF F16 is the first step.
    # Wait, 'convert_hf_to_gguf.py' produces unquantized GGUF.
    # For Windows/Android ease, we usually want quantized.
    # If the user doesn't have llama.cpp compiled, we can only give them the F16 GGUF 
    # and they can quantize it in LM Studio, or we leave it as F16 (large but works).
    # 
    # Actually, Llama 3.2 3B is small (6GB in F16), so F16 might be acceptable for modern devices.
    # Let's produce the F16 GGUF first.
    
    f16_gguf_path = EXPORTS_DIR / "mindmate_llama32_3b_f16.gguf"
    
    cmd = [
        sys.executable, str(convert_script),
        str(FUSED_MODEL_DIR),
        "--outfile", str(f16_gguf_path),
        "--outtype", "f16"
    ]
    
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        print("[ERROR] Conversion failed.")
        sys.exit(result.returncode)
        
    print(f"[SUCCESS] Model exported to: {f16_gguf_path}")
    print(f"Size: {f16_gguf_path.stat().st_size / (1024**3):.2f} GB")

def main():
    check_dependencies()
    fuse_model()
    convert_to_gguf()
    print("\n[DONE] Export process finished.")

if __name__ == "__main__":
    main()
