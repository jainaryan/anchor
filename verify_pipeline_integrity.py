import json
import os
import sys
import torch
import re
from transformers import AutoTokenizer

# Paths
BASE_MODEL_DIR = "models/CUDA_llama-3.2-3b-instruct"
CHUNK_SCRIPT = "scripts/CUDA_chunk.py"
TEMP_IN = "temp_audit_in.jsonl"
TEMP_OUT = "temp_audit_out.jsonl"

def test_chunking():
    print(">>> 1. Testing Chunking Logic (CUDA_chunk.py)...")
    
    # Create dummy data with tags
    sample_data = {
        "text": "<|user|> WARNING: Do not strip me! <|assistant|> I won't!",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
    }
    
    with open(TEMP_IN, "w", encoding="utf-8") as f:
        f.write(json.dumps(sample_data) + "\n")
        
    # Run the chunk script via subprocess
    cmd = f"{sys.executable} {CHUNK_SCRIPT} --in {TEMP_IN} --out {TEMP_OUT} --tokenizer {BASE_MODEL_DIR} --max-len 1024 --overlap 0"
    ret = os.system(cmd)
    
    if ret != 0:
        print("❌ Chunking script failed to run.")
        return False
        
    with open(TEMP_OUT, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    if not lines:
        print("❌ Chunking output is empty.")
        return False
        
    data = json.loads(lines[0])
    text = data["text"]
    print(f"    Chunked Output: {text!r}")
    
    # Check 1: Did it strip special tokens?
    if "<|user|>" not in text or "<|assistant|>" not in text:
        print("❌ CRITICAL FAIL: Special tokens were stripped!")
        return False
        
    print("✅ Chunking preserved special tokens.")
    return True

def test_masking():
    print("\n>>> 2. Testing Masking Logic (from CUDA_train_qlora.py)...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_DIR, 
        local_files_only=True,
        fix_mistral_regex=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Sample multi-turn text
    raw_text = "<|begin_of_text|><|user|> User Turn 1 <|assistant|> Assistant Turn 1 <|user|> User Turn 2 <|assistant|> Assistant Turn 2"
    
    # --- LOGIC COPIED FROM CUDA_train_qlora.py ---
    batch = {"text": [raw_text]}
    tokenized = tokenizer(
        batch["text"],
        truncation=True,
        max_length=2048,
        padding=False,
        return_offsets_mapping=True,
    )
    
    input_ids = tokenized["input_ids"][0]
    offsets = tokenized["offset_mapping"][0]
    labels = list(input_ids)
    
    # Find assistant ranges
    assistant_ranges = []
    for m in re.finditer(r"<\|assistant\|>(.*?)(?=<\|user\|>|$|(?=<\|assistant\|>))", raw_text, re.DOTALL):
        assistant_ranges.append((m.start(1), m.end(1)))
        
    # Mask default
    for j in range(len(labels)):
        labels[j] = -100
        
    # Unmask assistant
    for j, (start, end) in enumerate(offsets):
         for a_start, a_end in assistant_ranges:
            if start >= a_start and end <= a_end and start < end:
                labels[j] = input_ids[j]
                break
    # ---------------------------------------------
    
    # Verification
    print(f"    Raw Text: {raw_text}")
    print("    Token Analysis:")
    
    sub_fail = False
    
    # Helper to check if a token is part of a string
    def is_part_of(token_id, target_str):
        s = tokenizer.decode([token_id])
        return s.strip() in target_str
        
    for idx, (t_id, l_id) in enumerate(zip(input_ids, labels)):
        token_str = tokenizer.decode([t_id])
        status = "✅ Masked" if l_id == -100 else "✅ TRAIN"
        
        # Check correctness
        if "User" in token_str and l_id != -100:
            status = "❌ FAIL (User token not masked)"
            sub_fail = True
        elif "Assistant" in token_str and l_id == -100:
             # Note: The tag properties might mean 'Assistant' word itself is part of the tag or content?
             # In raw text: "<|assistant|> Assistant Turn 1"
             # The regex captures " Assistant Turn 1".
             # So "Assistant" (the word content) SHOULD be trained.
             if "<|assistant|>" not in token_str: # ignore the tag itself
                 status = "❌ FAIL (Assistant content masked)"
                 sub_fail = True
                 
        print(f"      [{idx:02d}] {token_str!r:<15} | Label: {l_id:<6} | {status}")

    if sub_fail:
        print("❌ Masking logic failed verification.")
        return False
        
    print("✅ Masking logic is PERFECT.")
    return True

if __name__ == "__main__":
    if test_chunking() and test_masking():
        print("\n\n🎉 ALL SYSTEMS GREEN. PIPELINE IS READY.")
    else:
        print("\n\n⛔ AUDIT FAILED. DO NOT RUN PIPELINE.")
