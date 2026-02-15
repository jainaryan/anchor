"""
import json, argparse, os, sys
from transformers import AutoTokenizer

def get_text_from_example(ex):
    # Supports several common SFT schemas. Extend if needed.
    if "text" in ex and isinstance(ex["text"], str):
        return ex["text"], "text"
    if "prompt" in ex and "completion" in ex:
        return (ex.get("prompt","") + ex.get("completion","")), "prompt_completion"
    if "messages" in ex and isinstance(ex["messages"], list):
        # Flatten chat into a single string. Adjust the template to match training/inference.
        parts = []
        for m in ex["messages"]:
            role = m.get("role", "")
            content = m.get("content", "")
            parts.append(f"{role.capitalize()}: {content}")
        return "\n".join(parts), "messages"
    # Fallback: stringify whole object (last resort)
    return json.dumps(ex, ensure_ascii=False), "unknown"

def set_text_into_example(ex, text, schema, chunk_id):
    ex = dict(ex)  # shallow copy
    if schema == "text":
        ex["text"] = text
    elif schema == "prompt_completion":
        # Put everything into "text" to keep it simple/consistent for trainer
        ex = {"text": text, **{k:v for k,v in ex.items() if k not in ("prompt","completion")}}
    elif schema == "messages":
        ex = {"text": text, **{k:v for k,v in ex.items() if k != "messages"}}
    else:
        ex = {"text": text}  # fallback
    ex["chunk_id"] = chunk_id
    return ex

def chunk_ids(ids, max_len, overlap):
    i = 0
    step = max_len - overlap
    while i < len(ids):
        yield ids[i:i+max_len]
        i += step

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--model-dir", required=True, help="Path to your converted model dir (e.g., ./mlx_llama32_3b)")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--overlap", type=int, default=128)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    total = 0
    overs = 0
    written = 0

    with open(args.inp, "r") as f_in, open(args.out, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line: continue
            total += 1
            ex = json.loads(line)
            text, schema = get_text_from_example(ex)
            ids = tok.encode(text, add_special_tokens=False)
            if len(ids) <= args.max_len:
                f_out.write(json.dumps(set_text_into_example(ex, text, schema, 0), ensure_ascii=False) + "\n")
                written += 1
                continue

            overs += 1
            for ci, chunk in enumerate(chunk_ids(ids, args.max_len, args.overlap)):
                ctext = tok.decode(chunk)
                out_ex = set_text_into_example(ex, ctext, schema, ci)
                f_out.write(json.dumps(out_ex, ensure_ascii=False) + "\n")
                written += 1

    print(f"[split] read={total}, oversize={overs}, written={written}, out={args.out}")

if __name__ == "__main__":
    main()
"""
#%%
# chunk.py
# Usage:
#   python finetuning/chunk.py --in ./data/cleaned_data/mindmate_train.cleaned.jsonl --out ./data/chunked_3072/train.split.jsonl \
#     --model-dir ./mlx_llama32_3b --tokenizer meta-llama/Llama-3.2-3B-Instruct \
#     --max-len 2048 --overlap 128

import json, argparse
from transformers import AutoTokenizer

def get_text_from_example(ex):
    if "text" in ex and isinstance(ex["text"], str):
        return ex["text"], "text"
    if "prompt" in ex and "completion" in ex:
        return (ex.get("prompt","") + ex.get("completion","")), "prompt_completion"
    if "messages" in ex and isinstance(ex["messages"], list):
        parts = []
        for m in ex["messages"]:
            role = m.get("role", "")
            content = m.get("content", "")
            parts.append(f"{role.capitalize()}: {content}")
        return "\n".join(parts), "messages"
    return json.dumps(ex, ensure_ascii=False), "unknown"

def set_text_into_example(ex, text, schema, chunk_id):
    # FORCE unification to {"text": ...} + carry chunk_id only
    return {"text": text, "chunk_id": chunk_id}

def chunk_ids(ids, max_len, overlap):
    assert max_len > 0, "--max-len must be > 0"
    assert 0 <= overlap < max_len, "--overlap must be in [0, max_len-1]"
    i = 0
    step = max_len - overlap
    while i < len(ids):
        yield ids[i:i+max_len]
        i += step

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--model-dir", default=None, help="Path to your MLX model dir (used only if --tokenizer not set)")
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer name/path (recommended), e.g. meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--overlap", type=int, default=128)
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
            if not line:
                continue
            total += 1
            ex = json.loads(line)
            text, schema = get_text_from_example(ex)
            ids = tok.encode(text, add_special_tokens=False)
            ids.append(tok.eos_token_id)
            if len(ids) <= args.max_len:
                f_out.write(json.dumps(set_text_into_example(ex, text, schema, 0), ensure_ascii=False) + "\n")
                written += 1
                continue

            overs += 1
            for ci, chunk in enumerate(chunk_ids(ids, args.max_len, args.overlap)):
                ctext = tok.decode(chunk, skip_special_tokens=True)
                out_ex = set_text_into_example(ex, ctext, schema, ci)
                f_out.write(json.dumps(out_ex, ensure_ascii=False) + "\n")
                written += 1

    print(f"[split] read={total}, oversize={overs}, written={written}, out={args.out}")

if __name__ == "__main__":
    main()

# %%
