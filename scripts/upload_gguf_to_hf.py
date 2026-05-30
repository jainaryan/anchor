"""
Upload MindMate GGUF models to HuggingFace Hub.
Creates the repo if it doesn't exist.

Usage:
    python scripts/upload_gguf_to_hf.py                        # upload all
    python scripts/upload_gguf_to_hf.py --model llama_sft_ck1600  # upload one
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

HF_REPO_ID = "jainaryan/mindmate-gguf"
HF_TOKEN   = os.environ.get("HF_TOKEN")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORTS_DIR  = PROJECT_ROOT / "exports"

# All available Q4_K_M GGUFs
ALL_MODELS = {
    # Best SFT (v4 benchmark 2026-05-30)
    "genzv6_ck1600":      EXPORTS_DIR / "mindmate_genzv6_ck1600"      / "mindmate_genzv6_ck1600_q4_k_m.gguf",
    # Top-3 SFT checkpoints (benchmark 2026-05-03)
    "genzv3_ck200":       EXPORTS_DIR / "mindmate_genzv3_ck200"       / "mindmate_genzv3_ck200_q4_k_m.gguf",
    "genzv2_ck1200":      EXPORTS_DIR / "mindmate_genzv2_ck1200"      / "mindmate_genzv2_ck1200_q4_k_m.gguf",
    "genzv4_ck200":       EXPORTS_DIR / "mindmate_genzv4_ck200"       / "mindmate_genzv4_ck200_q4_k_m.gguf",
    # Legacy
    "llama_sft_ck200":    EXPORTS_DIR / "mindmate_llama_sft_ck200"    / "mindmate_llama_sft_ck200_q4_k_m.gguf",
    "llama_dpo_ck200":    EXPORTS_DIR / "mindmate_llama_dpo_ck200"    / "mindmate_llama_dpo_ck200_q4_k_m.gguf",
    "llama_sft_ck1600":   EXPORTS_DIR / "mindmate_llama_sft_ck1600"   / "mindmate_llama_sft_ck1600_q4_k_m.gguf",
    "qwen25_dpo_ck200":   EXPORTS_DIR / "mindmate_qwen25_dpo_ck200"   / "mindmate_qwen25_dpo_ck200_q4_k_m.gguf",
    "gemma4_e2b":         EXPORTS_DIR / "mindmate_gemma4_e2b"         / "mindmate_gemma4_e2b_q4_k_m.gguf",
    "gemma4_e4b":         EXPORTS_DIR / "mindmate_gemma4_e4b"         / "mindmate_gemma4_e4b_q4_k_m.gguf",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(ALL_MODELS.keys()), default=None,
                        help="Upload a single model. Omit to upload all.")
    args = parser.parse_args()

    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN not set. Run: export HF_TOKEN=your_token")

    api = HfApi(token=HF_TOKEN)

    print(f"Creating/checking repo: {HF_REPO_ID}")
    create_repo(repo_id=HF_REPO_ID, repo_type="model", private=True, exist_ok=True, token=HF_TOKEN)
    print(f"Repo ready: https://huggingface.co/{HF_REPO_ID}")

    to_upload = {args.model: ALL_MODELS[args.model]} if args.model else ALL_MODELS

    for name, filepath in to_upload.items():
        if not filepath.exists():
            print(f"[SKIP] Not found: {filepath}")
            continue

        size_gb = filepath.stat().st_size / (1024 ** 3)
        print(f"\nUploading {filepath.name} ({size_gb:.2f} GB)...")

        api.upload_file(
            path_or_fileobj=str(filepath),
            path_in_repo=filepath.name,
            repo_id=HF_REPO_ID,
            repo_type="model",
            token=HF_TOKEN,
        )
        print(f"[OK] {filepath.name} uploaded.")

    print(f"\nAll done! View at: https://huggingface.co/{HF_REPO_ID}")

if __name__ == "__main__":
    main()
