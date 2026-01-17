# chat_mindmate.py
# Usage:
#   conda activate mindmatenv
#   python scripts/chat_mindmate.py

import json
import shutil
from datetime import datetime
from pathlib import Path

from mlx_lm import load, generate

# ========= CONFIG: EDIT HERE IF NEEDED =========

# This file lives in .../mindmate/scripts/chat_mindmate.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Base model (your converted Llama 3.2 3B)
MODEL_PATH = PROJECT_ROOT / "mlx_llama32_3b"

# QLoRA adapter directory + checkpoint to use
ADAPTER_DIR = PROJECT_ROOT / "adapters" / "mindmate_llama32_3b_qlora_nl10_3072_lr3e5"
BEST_CHECKPOINT = ADAPTER_DIR / "0001000_adapters.safetensors"   # 1000-iter model
ACTIVE_ADAPTER_FILE = ADAPTER_DIR / "adapters.safetensors"       # what mlx_lm.load expects

# System prompt
PROMPT_PATH = PROJECT_ROOT / "system_prompt.txt"

# Generation settings
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 1024

# Conversation logs directory
LOG_DIR = PROJECT_ROOT / "logs"

# ========= END CONFIG =========


def ensure_best_adapter_active():
    """
    Make sure the 'active' adapter file corresponds to the 1000-iter checkpoint.

    mlx_lm.load(..., adapter_path=ADAPTER_DIR) expects to find something like
    'adapters.safetensors' inside ADAPTER_DIR. To force it to use the 1000-iter
    checkpoint, we copy that file to adapters.safetensors.
    """
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(f"Adapter directory not found: {ADAPTER_DIR}")

    if not BEST_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Expected checkpoint not found: {BEST_CHECKPOINT}\n"
            f"Make sure adapters.safetensors exists in {ADAPTER_DIR}"
        )

    # Copy 0001000_adapters.safetensors -> adapters.safetensors
    shutil.copy2(BEST_CHECKPOINT, ACTIVE_ADAPTER_FILE)
    print(f"[info] set active adapter to: {BEST_CHECKPOINT.name}")


def load_system_prompt():
    try:
        with PROMPT_PATH.open("r", encoding="utf-8") as f:
            system = f.read().strip()
        print(f"[info] loaded system prompt from {PROMPT_PATH}")
        return system
    except FileNotFoundError:
        print(f"[warn] system prompt not found at {PROMPT_PATH}.")
        return ""


def save_conversation(history: list, log_dir: Path):
    """
    Save the conversation history to a timestamped JSON file.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"chat_{timestamp}.json"
    
    with log_file.open("w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "messages": history
        }, f, indent=2, ensure_ascii=False)
    
    print(f"[info] conversation saved to {log_file}")
    return log_file


def main():
    print(f"[info] project root: {PROJECT_ROOT}")

    # Ensure the chosen checkpoint is the active adapter
    ensure_best_adapter_active()

    # Load model + adapter
    print(f"[info] loading model from: {MODEL_PATH}")
    print(f"[info] loading adapter from: {ADAPTER_DIR}")

    model, tokenizer = load(
        str(MODEL_PATH),
        adapter_path=str(ADAPTER_DIR),   # must be a directory, not a file
    )

    system = load_system_prompt()
    history = [{"role": "system", "content": system}]

    print("\nType your message. Ctrl+C to exit.\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            # Save conversation on exit
            if len(history) > 1:  # More than just system prompt
                save_conversation(history, LOG_DIR)
            return

        if not user:
            continue

        history.append({"role": "user", "content": user})

        # Build chat-style prompt using tokenizer chat template
        prompt = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )

        out = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=MAX_NEW_TOKENS,
            # temp=TEMPERATURE,
            # top_p=TOP_P,
        )

        text = out["text"] if isinstance(out, dict) and "text" in out else str(out)
        reply = text[len(prompt):] if text.startswith(prompt) else text
        reply = reply.strip()

        print(f"Mindmate: {reply}\n")
        history.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()
