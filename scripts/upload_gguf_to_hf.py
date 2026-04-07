"""
Upload MindMate GGUF models to HuggingFace Hub.
Creates the repo if it doesn't exist.

Usage:
    python scripts/upload_gguf_to_hf.py
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

HF_REPO_ID = "jainaryan/mindmate-gguf"
HF_TOKEN = os.environ.get("HF_TOKEN")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILES_TO_UPLOAD = [
    PROJECT_ROOT / "exports" / "mindmate_genz_llama32_3b" / "mindmate_genz_llama32_3b_q4_k_m.gguf",
    PROJECT_ROOT / "exports" / "mindmate_qwen3_1p7b" / "mindmate_qwen3_1p7b_q4_k_m.gguf",
]

def main():
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN not set. Run: export HF_TOKEN=your_token")

    api = HfApi(token=HF_TOKEN)

    # Create repo if it doesn't exist
    print(f"Creating/checking repo: {HF_REPO_ID}")
    create_repo(
        repo_id=HF_REPO_ID,
        repo_type="model",
        private=True,
        exist_ok=True,
        token=HF_TOKEN,
    )
    print(f"Repo ready: https://huggingface.co/{HF_REPO_ID}")

    for filepath in FILES_TO_UPLOAD:
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
