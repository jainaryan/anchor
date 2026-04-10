import json
import torch
import os
import time
import argparse
from datetime import datetime
from pathlib import Path
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from peft import PeftModel

# ========= ARGS =========

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint",
    type=str,
    default="checkpoint-200",
    help="Checkpoint subfolder to load, e.g. 'checkpoint-600'. Defaults to checkpoint-200."
)
args = parser.parse_args()

# ========= CONFIG: EDIT HERE IF NEEDED =========

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = "meta-llama/Llama-3.2-3B-Instruct"

_BASE_ADAPTER = PROJECT_ROOT / "adapters" / "CUDA_mindmate_llama32b"
ADAPTER_DIR = _BASE_ADAPTER / args.checkpoint if args.checkpoint else _BASE_ADAPTER

PROMPT_PATH = PROJECT_ROOT / "system_prompt.txt"

TEMPERATURE = 0.75
TOP_P = 0.9
TOP_K = 50
MAX_NEW_TOKENS = 256
REPETITION_PENALTY = 1.25

LOG_DIR = PROJECT_ROOT / "logs"

# ========= END CONFIG =========

def load_system_prompt():
    encodings = ["utf-8-sig", "utf-16", "utf-8", "cp1252"]
    for enc in encodings:
        try:
            with PROMPT_PATH.open("r", encoding=enc) as f:
                system = f.read().strip()
            print(f"[info] loaded system prompt from {PROMPT_PATH} (encoding: {enc})")
            return system
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    print(f"[warn] system prompt not found at {PROMPT_PATH}. Using fallback.")
    return "You are MindMate, a helpful and empathetic AI assistant specialized in emotional support and conversation."

def save_conversation(history: list, log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"chat_llama_{timestamp}.json"
    with log_file.open("w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "messages": history}, f, indent=2, ensure_ascii=False)
    print(f"[info] conversation saved to {log_file}")
    return log_file

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] project root: {PROJECT_ROOT}")
    print(f"[info] adapter: {ADAPTER_DIR}")
    print(f"[info] loading tokenizer and model (4-bit NF4)...")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR),
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.config.use_cache = True

    print(f"[info] loading adapters from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    model.eval()

    system = load_system_prompt()
    history_list = [{"role": "system", "content": system}] if system else []

    print("\n--- MindMate Interactive (Llama 3.2 3B) ---")
    print("Type your message. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            if len(history_list) > 1:
                save_conversation(history_list, LOG_DIR)
            return

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break

        history_list.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(
            history_list,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        generate_kwargs = dict(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            do_sample=True,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=REPETITION_PENALTY,
            use_cache=True,
            streamer=streamer,
        )

        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()

        print("\nMindmate: ", end="", flush=True)
        response = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            response += new_text
        print("\n")

        history_list.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
