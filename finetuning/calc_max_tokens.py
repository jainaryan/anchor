import json
import re
from pathlib import Path
from transformers import AutoTokenizer

# Config
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_DIR = PROJECT_ROOT / "models" / "CUDA_llama-3.2-3b-instruct"
TRAIN_FILE = PROJECT_ROOT / "data" / "cleaned_data" / "mindmate_train_clean.jsonl"
VAL_FILE = PROJECT_ROOT / "data" / "cleaned_data" / "mindmate_val_clean.jsonl"

def main():
    if not BASE_MODEL_DIR.exists():
        print(f"Error: Base model not found at {BASE_MODEL_DIR}")
        return

    print(f"Loading tokenizer from {BASE_MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(str(BASE_MODEL_DIR), use_fast=False)

    files_to_check = [TRAIN_FILE, VAL_FILE]
    
    for data_file in files_to_check:
        if not data_file.exists():
            print(f"Warning: File {data_file} not found. Skipping.")
            continue
            
        print(f"\nAnalyzing {data_file.name}...")
        max_tokens = 0
        total_tokens = 0
        count = 0
        lengths = []

        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                text = ex.get("text", "")
                
                # Logic from CUDA_train_qlora.py
                conversation = []
                parts = re.split(r"(<\|user\|>|<\|assistant\|>)", text)
                role = None
                for p in parts:
                    p = p.strip()
                    if p == "<|user|>":
                        role = "user"
                    elif p == "<|assistant|>":
                        role = "assistant"
                    elif role and p:
                        conversation.append({"role": role, "content": p})
                
                # Apply template
                formatted = tokenizer.apply_chat_template(
                    conversation, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                
                # Use tokenizer to get IDs
                tokens = tokenizer.encode(formatted, add_special_tokens=False)
                length = len(tokens)
                
                lengths.append(length)
                if length > max_tokens:
                    max_tokens = length
                total_tokens += length
                count += 1

        if count > 0:
            avg_tokens = total_tokens / count
            print(f"  Total examples: {count}")
            print(f"  Max tokens:     {max_tokens}")
            print(f"  Avg tokens:     {avg_tokens:.2f}")
            # Optional: count how many are over 2048
            over_2048 = sum(1 for l in lengths if l > 2048)
            print(f"  Examples > 2048 tokens: {over_2048}")

if __name__ == "__main__":
    main()
