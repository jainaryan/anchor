from pathlib import Path
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

HF_USERNAME = "jainaryan"
REPO_ID = f"{HF_USERNAME}/mindmate-llama32-3b-instruct"

ROOT = Path(__file__).resolve().parent

FUSED_MODEL_DIR = ROOT / "mlx_export" / "mindmate_llama32_3b_fused"
EXECUTORCH_DIR = ROOT / "mlx_export" / "mindmate_llama32_3b_executorch"
PTE_PATH = EXECUTORCH_DIR / "mindmate_llama32_3b.pte"

print("\n=== PATH CHECK ===")
print("Project root:", ROOT)
print("Fused model dir:", FUSED_MODEL_DIR)
print("ExecuTorch dir:", EXECUTORCH_DIR)
print("PTE path:", PTE_PATH)

# Ensure repo exists
create_repo(REPO_ID, repo_type="model", exist_ok=True)

api = HfApi()

# Upload fused HF model folder
print("\n=== Uploading fused HF model ===")
api.upload_folder(
    folder_path=str(FUSED_MODEL_DIR),
    repo_id=REPO_ID,
    repo_type="model",
)
print("✅ Fused model uploaded.")

# Upload ExecuTorch .pte file
print("\n=== Uploading ExecuTorch PTE ===")
api.upload_file(
    path_or_fileobj=str(PTE_PATH),
    path_in_repo="executorch/mindmate_llama32_3b.pte",
    repo_id=REPO_ID,
    repo_type="model",
)
print("✅ ExecuTorch file uploaded.")

print("\n🎉 All uploads finished successfully!")
print(f"👉 View your repo at: https://huggingface.co/{REPO_ID}\n")
