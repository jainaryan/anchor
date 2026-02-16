"""
chunk.py

Prepares data for MLX training by converting tagged text back into structured `messages` format
and applying "smart truncation" (removing oldest User+Assistant pairs) to fit within context length.

Input: JSONL with {"text": "<|user|> ... <|assistant|> ..."} (from clean_dataset.py)
Output: JSONL with {"messages": [{"role": "user", "content": "..."}, ...]}
"""

import json
import argparse
import sys
from transformers import AutoTokenizer

TAG_USER = "<|user|>"
TAG_ASSIST = "<|assistant|>"

def parse_tagged_text(text: str):
    """
    Parses a string like "<|user|> Hi <|assistant|> Hello" into a list of dicts.
    Assumes standard tags.
    """
    messages = []
    # Split by tags, keeping the delimiter to know which role it is
    # This is a simple parser assuming the file is well-formed from clean_dataset.py
    
    # We'll just look for the tags manually to be robust
    parts = []
    curr_role = None
    curr_text = []
    
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line: continue
        
        if line.startswith(TAG_USER):
            if curr_role:
                messages.append({"role": curr_role, "content": "\n".join(curr_text).strip()})
            curr_role = "user"
            curr_text = [line[len(TAG_USER):].strip()]
        elif line.startswith(TAG_ASSIST):
            if curr_role:
                messages.append({"role": curr_role, "content": "\n".join(curr_text).strip()})
            curr_role = "assistant"
            curr_text = [line[len(TAG_ASSIST):].strip()]
        else:
            # Continuation of previous turn
            if curr_role:
                curr_text.append(line)
            else:
                # Untagged text at start? Treat as system or ignore. 
                # For this dataset, we expect tags immediately.
                pass
                
    if curr_role and curr_text:
        messages.append({"role": curr_role, "content": "\n".join(curr_text).strip()})
        
    return messages

def apply_chat_template(messages, tokenizer):
    """
    Applies the tokenizer's chat template to get the full formatted string length.
    """
    return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)

def truncate_pairs(messages, tokenizer, max_len):
    """
    Truncates the conversation to fit within max_len tokens.
    Strategy:
    1. Keep System prompt (if exists).
    2. If too long, remove the oldest (User, Assistant) pair.
    3. Repeat until it fits.
    """
    
    # Check if first message is system
    has_system = messages[0]["role"] == "system" if messages else False
    system_msg = [messages[0]] if has_system else []
    conversation = messages[1:] if has_system else messages
    
    # Initial check
    full_ids = apply_chat_template(system_msg + conversation, tokenizer)
    if len(full_ids) <= max_len:
        return system_msg + conversation

    # Iterative removal
    # conversation is expected to look like [u, a, u, a, u, a...]
    # We remove from the front (oldest)
    
    while len(conversation) >= 2:
        # Check current length
        full_ids = apply_chat_template(system_msg + conversation, tokenizer)
        if len(full_ids) <= max_len:
            return system_msg + conversation
            
        # Remove oldest pair (User + Assistant)
        # We assume standard turns. If the first is user, remove it and the next.
        # If the structure is weird, we just remove the first two.
        conversation = conversation[2:]
        
    # Final check just in case we stripped everything
    if len(apply_chat_template(system_msg + conversation, tokenizer)) > max_len:
         # If purely system prompt is too long or single last turn is too long,
         # we might have to just return what fits or empty.
         # For safety, let's return whatever remains, usually the last turn.
         pass
         
    return system_msg + conversation

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--model-dir", required=True, help="Path to your converted model dir")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--overlap", type=int, default=128) # Unused now, but kept for script compat
    
    args = ap.parse_args()
    if args.overlap < 0 or args.overlap >= args.max_len:
        raise ValueError(f"--overlap must be in [0, {args.max_len - 1}]")
    
    if not args.tokenizer and not args.model_dir:
        raise ValueError("You must pass either --tokenizer or --model-dir")
        
    tok_src = args.tokenizer or args.model_dir
    tok = AutoTokenizer.from_pretrained(
    tok_src,
)

    total = overs = written = 0

    with open(args.inp, "r", encoding="utf-8") as f_in, open(args.out, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line: continue
            
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # 1. Parse content
            # If it's already "messages", great. If "text" with tags, parse it.
            messages = []
            if "messages" in ex:
                messages = ex["messages"]
            elif "text" in ex:
                messages = parse_tagged_text(ex["text"])
            
            if not messages:
                continue
                
            # 2. Truncate
            # Note: We rely on the tokenizer's chat template to estimate length
            final_messages = truncate_pairs(messages, tokenizer, args.max_len)
            
            if len(final_messages) < len(messages):
                truncated += 1
                
            # 3. Write output
            # Output format: {"messages": [...]}
            # MLX will handle the formatting and masking automatically if we pass this structure
            out_obj = {"messages": final_messages}
            f_out.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            kept += 1

    print(f"Done. Total: {kept+truncated}, Kept: {kept}, Truncated: {truncated}")

if __name__ == "__main__":
    main()
