import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_here)

# ---------------------------------------------------------------------------
# Model definitions — update this list tomorrow once models are confirmed
# ---------------------------------------------------------------------------
MODELS = [
    {
        "id": "mindmate-llama-sft-ck1600-q4",
        "name": "default",
        "path": os.path.join(_repo_root, "exports", "mindmate_llama_sft_ck1600", "llama(genz)v2_q4_k_m.gguf"),
        "chat_format": "llama-3",
    },
]

# ---------------------------------------------------------------------------
# Inference settings
# ---------------------------------------------------------------------------
N_GPU_LAYERS = 0        # 0 = CPU-only; set to -1 for full GPU offload
N_CTX = 4096            # context window per session
MAX_TOKENS = 512        # max tokens per response
TEMPERATURE = 0.75
INFERENCE_TIMEOUT = 30  # seconds to wait for next token before aborting

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_prompt_path = os.path.join(_repo_root, "inference", "system_prompt.txt")
with open(_prompt_path) as f:
    SYSTEM_PROMPT = f.read().strip()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(_here, "logs")
