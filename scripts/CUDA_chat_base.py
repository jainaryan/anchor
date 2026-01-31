import json
import torch
import os
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ========= CONFIG: EDIT HERE IF NEEDED =========

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Base model (your Llama 3.2 3B)
MODEL_DIR = PROJECT_ROOT / "models" / "CUDA_llama-3.2-3b-instruct"

# Generation settings
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 512

# Conversation logs directory
LOG_DIR = PROJECT_ROOT / "logs"

# ========= END CONFIG =========

def save_conversation(history: list, log_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"base_chat_{timestamp}.json"
    
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
    print(f"[info] loading tokenizer and base model (4-bit NF4)...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR), 
        local_files_only=True,
        fix_mistral_regex=True
    )
    
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
    model.eval()

    # Initial prompt build.
    history_str = ""
    history_list = []

    print("\n--- Base Model Interactive (Windows Port) ---")
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
        
        # Build prompt: history + current input
        # NOTE: Using the same prompt format as MindMate to compare apples-to-apples performance.
        prompt = history_str + f"<|user|> {user_input}\n<|assistant|>"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # Prevent runaway generation
        if "<|user|>" in response:
            response = response.split("<|user|>")[0].strip()
            
        print(f"\nBaseModel: {response}\n")
        history_list.append({"role": "assistant", "content": response})
        
        # Update history string for context
        history_str += f"<|user|> {user_input}\n<|assistant|> {response}<|end_of_text|>\n"

if __name__ == "__main__":
    main()
