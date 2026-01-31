import json
import torch
import os
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ========= CONFIG: EDIT HERE IF NEEDED =========

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Base model (your Llama 3.2 3B)
MODEL_DIR = PROJECT_ROOT / "models" / "CUDA_llama-3.2-3b-instruct"

# QLoRA adapter directory (newly trained windows version)
ADAPTER_DIR = PROJECT_ROOT / "adapters" / "CUDA_mindmate_llama32b"

# System prompt
PROMPT_PATH = PROJECT_ROOT / "system_prompt.txt"

# Generation settings
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 256

# Conversation logs directory
LOG_DIR = PROJECT_ROOT / "logs"

# ========= END CONFIG =========

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] project root: {PROJECT_ROOT}")
    print(f"[info] loading tokenizer and model (4-bit NF4)...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), 
        local_files_only=True,
        fix_mistral_regex=True
    )
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
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    model.config.use_cache = True
    
    print(f"[info] loading adapters from: {ADAPTER_DIR}")
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR), adapter_name="mindmate")
    model.eval()

    system = load_system_prompt()
    # Initial prompt build. Note: your model uses <|user|> tags.
    history_str = f"<|system|> {system}\n" if system else ""
    history_list = [{"role": "system", "content": system}] if system else []

    print("\n--- MindMate Interactive (Windows Port) ---")
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
        
        # Use template for inference
        prompt = tokenizer.apply_chat_template(
            history_list, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        import time
        t0 = time.time()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        t1 = time.time()
        print(f"[debug] Tokenization took: {t1 - t0:.4f}s")
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
        t2 = time.time()
        print(f"[debug] Generation took: {t2 - t1:.4f}s")

        new_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        print(f"\nMindmate: {response}\n")
        history_list.append({"role": "assistant", "content": response})
        
        # History string is no longer needed as we use history_list directly
        history_str = "" 

if __name__ == "__main__":
    main()