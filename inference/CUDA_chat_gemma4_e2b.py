import json
import torch
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
    default=None,
    help="Checkpoint subfolder to load, e.g. 'checkpoint-1600'. Defaults to final adapter."
)
args = parser.parse_args()

# ========= CONFIG =========

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_MODEL = "google/gemma-4-e2b-it"

_BASE_ADAPTER = PROJECT_ROOT / "adapters" / "CUDA_mindmate_gemma4_e2b"
ADAPTER_DIR = _BASE_ADAPTER / args.checkpoint if args.checkpoint else _BASE_ADAPTER

PROMPT_PATH = PROJECT_ROOT / "system_prompt.txt"
LOG_DIR = PROJECT_ROOT / "logs"

TEMPERATURE = 0.75
TOP_P = 0.9
TOP_K = 50
MAX_NEW_TOKENS = 256
REPETITION_PENALTY = 1.15

# ========= END CONFIG =========


def load_system_prompt():
    for enc in ["utf-8-sig", "utf-8", "utf-16", "cp1252"]:
        try:
            with PROMPT_PATH.open("r", encoding=enc) as f:
                system = f.read().strip()
            print(f"[info] loaded system prompt ({enc})")
            return system
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    print("[warn] system prompt not found, using fallback")
    return "You are Anchor, a warm and caring AI companion — like a close friend who genuinely listens."


def save_conversation(history: list):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"chat_gemma4_e2b_{timestamp}.json"
    with log_file.open("w", encoding="utf-8") as f:
        json.dump({"timestamp": timestamp, "messages": history}, f, indent=2, ensure_ascii=False)
    print(f"[info] conversation saved to {log_file}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] base model : {BASE_MODEL}")
    print(f"[info] adapter    : {ADAPTER_DIR}")
    print(f"[info] device     : {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",  # required for Gemma 4
    )
    model.config.use_cache = True

    # Gemma 4 wraps linear layers in Gemma4ClippableLinear which PEFT doesn't recognise.
    # Replace each wrapper with its inner Linear4bit before injecting the LoRA adapter.
    def unwrap_gemma4_clippable_linears(m):
        for parent in list(m.modules()):
            for attr_name, child in list(parent.named_children()):
                if type(child).__name__ == "Gemma4ClippableLinear" and hasattr(child, "linear"):
                    setattr(parent, attr_name, child.linear)
        return m

    model = unwrap_gemma4_clippable_linears(model)
    print("[gemma4-fix] Unwrapped Gemma4ClippableLinear → inner Linear4bit for PEFT compatibility.")

    print(f"[info] loading adapter from {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
    model.eval()

    system = load_system_prompt()
    history = [{"role": "system", "content": system}]

    # Gemma 4 eot token id
    eot_token_id = tokenizer.convert_tokens_to_ids("<turn|>")
    eos_token_id = tokenizer.eos_token_id
    stop_ids = list({eot_token_id, eos_token_id} - {None, -1})

    print("\n--- MindMate / Anchor (Gemma 4 E2B) ---")
    print("Type your message. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            if len(history) > 1:
                save_conversation(history)
            return

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break

        history.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

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
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=REPETITION_PENALTY,
            use_cache=True,
            streamer=streamer,
        )

        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()

        print("\nAnchor: ", end="", flush=True)
        response = ""
        for new_text in streamer:
            print(new_text, end="", flush=True)
            response += new_text
        print("\n")
        thread.join()

        history.append({"role": "assistant", "content": response})

    if len(history) > 1:
        save_conversation(history)


if __name__ == "__main__":
    main()
