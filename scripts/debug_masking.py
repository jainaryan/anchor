
import torch
import re
from transformers import AutoTokenizer

BASE_MODEL_DIR = "models/CUDA_llama-3.2-3b-instruct"

def debug_masking():
    print(f"Loading tokenizer from {BASE_MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_DIR, 
        local_files_only=True, 
        fix_mistral_regex=True
    )
    
    # 1. Sample input representing Chunker output
    raw_text = "<|user|> Help me <|assistant|> Sure thing <|user|> Thanks <|assistant|> You're welcome"
    print(f"\n[Input Raw Text]:\n{raw_text!r}\n")
    
    # --- LOGIC FROM CUDA_train_qlora.py ---
    
    formatted_texts = []
    
    # Parse back to messages
    conversation = []
    parts = re.split(r"(<\|user\|>|<\|assistant\|>)", raw_text)
    role = None
    for p in parts:
        p = p.strip()
        if p == "<|user|>":
            role = "user"
        elif p == "<|assistant|>":
            role = "assistant"
        elif role and p:
            conversation.append({"role": role, "content": p})
            
    print("[Parsed Conversation]:")
    for msg in conversation:
        print(f"  {msg}")
        
    # Apply Template
    formatted = tokenizer.apply_chat_template(
        conversation, 
        tokenize=False, 
        add_generation_prompt=False
    )
    print(f"\n[Templated Text]:\n{formatted!r}\n")
    
    # Tokenize
    tokenized = tokenizer(
        [formatted],
        truncation=True,
        max_length=2048,
        padding=False,
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    
    input_ids = tokenized["input_ids"][0]
    offsets = tokenized["offset_mapping"][0]
    labels = list(input_ids)
    
    # Mask Init
    for j in range(len(labels)):
        labels[j] = -100
        
    # Pattern Match
    # Matches: <|start_header_id|>assistant<|end_header_id|>\n\n(AGENTS RESPONSE)<|eot_id|>
    # Capture group 1 is content.
    # We want to unmask content + EOT.
    # regex matches header...content...eot.
    # m.start(1) is start of content.
    # m.end(0) is end of the match (which includes EOT).
    pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)(?:<\|eot_id\|>|$)"
    
    assistant_ranges = []
    for m in re.finditer(pattern, formatted, re.DOTALL):
        # Unmask from start of content to end of EOT
        assistant_ranges.append((m.start(1), m.end(0)))
        print(f"  [Found Turn]: {m.group(1)!r} (Length: {m.end(0)-m.start(1)})")

    # Apply Logic
    for j, (start, end) in enumerate(offsets):
        for a_start, a_end in assistant_ranges:
            if start >= a_start and end <= a_end and start < end:
                labels[j] = input_ids[j]
                break
                
    # EOS Check logic
    if labels[-1] == -100:
         if input_ids[-1] == tokenizer.eos_token_id or input_ids[-1] == 128009:
             labels[-1] = input_ids[-1]
             print("  [EOS Logic]: Forcing unmask of last token (EOS)")

    # --- VERIFICATION ---
    print("\n[Token Analysis]:")
    for idx, (tid, lid) in enumerate(zip(input_ids, labels)):
        token_str = tokenizer.decode([tid])
        lbl_str = tokenizer.decode([lid]) if lid != -100 else "<MASKED>"
        
        # Check integrity
        status = "✅"
        if "Sure thing" in token_str and lid == -100: status = "❌ MISSED CONTENT"
        if "Help me" in token_str and lid != -100: status = "❌ LEAKED USER"
        if ("<|eot_id|>" in token_str) and lid == -100: status = "⚠️ MASKED EOT (Check if intended)"
        
        print(f"{idx:3} | {tid:6} | {token_str:20} | {lbl_str:20} | {status}")

if __name__ == "__main__":
    debug_masking()
