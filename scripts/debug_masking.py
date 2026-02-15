import torch
from transformers import AutoTokenizer
import os

# Use the same model as the pipeline
BASE_MODEL_DIR = "models/CUDA_llama-3.2-3b-instruct"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR)

# Sample Multi-turn Conversation
sample_conv = [
    {"role": "user", "content": "Hello, I am feeling sad today."},
    {"role": "assistant", "content": "I'm so sorry to hear that. What's been on your mind?"},
    {"role": "user", "content": "Just a lot of work stress."},
    {"role": "assistant", "content": "Work can be really overwhelming. Can you tell me more about it?"}
]

# Apply Template
formatted = tokenizer.apply_chat_template(
    sample_conv, 
    tokenize=False, 
    add_generation_prompt=False
)

print("--- Formatted Text ---")
print(formatted)
print("-" * 20)

# Tokenize
tokenized = tokenizer(
    formatted,
    truncation=True,
    max_length=2048,
    padding=False,
    return_offsets_mapping=True,
    add_special_tokens=False
)

input_ids = tokenized["input_ids"]
offsets = tokenized["offset_mapping"]
labels = [-100] * len(input_ids)

header_token = "<|start_header_id|>assistant<|end_header_id|>\n\n"
eot_token = "<|eot_id|>"

search_start = 0
while True:
    start_idx = formatted.find(header_token, search_start)
    if start_idx == -1:
        break
    
    content_start = start_idx + len(header_token)
    
    end_idx = formatted.find(eot_token, content_start)
    if end_idx == -1:
        content_end = len(formatted)
    else:
        content_end = end_idx + len(eot_token)
    
    for j, (tok_start, tok_end) in enumerate(offsets):
        if tok_start >= content_start and tok_end <= content_end:
            if tok_start < tok_end:
                labels[j] = input_ids[j]
    
    search_start = content_end
    if end_idx == -1: break

# Print Verification
print("\n--- Masking Audit Results ---")
for i, (tid, label) in enumerate(zip(input_ids, labels)):
    token_str = tokenizer.decode([tid])
    status = "UNMASKED" if label != -100 else "MASKED"
    # Escaping special chars for display
    display_token = token_str.replace("\n", "\\n")
    print(f"Token {i:3}: ID {tid:6} | Label {label:6} | {status:8} | '{display_token}'")

print("-" * 20)
print(f"Total Tokens: {len(input_ids)}")
print(f"Unmasked Tokens: {sum(1 for l in labels if l != -100)}")
