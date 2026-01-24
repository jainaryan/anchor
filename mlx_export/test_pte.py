from pathlib import Path

import torch
from executorch.runtime import Runtime, Verification
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent

PTE_PATH = ROOT / "mindmate_llama32_3b_executorch" / "mindmate_llama32_3b.pte"
HF_DIR = ROOT / "mindmate_llama32_3b_fused"  # tokenizer dir

SEQ_LEN = 64  # fixed by your ExecuTorch graph


def load_tokenizer():
    print("Loading tokenizer from:", HF_DIR)
    tokenizer = AutoTokenizer.from_pretrained(HF_DIR)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def pad_truncate(ids: torch.Tensor, seq_len: int, pad_id: int) -> torch.Tensor:
    """
    ids: [1, L] Long tensor
    returns: [1, seq_len]
    """
    L = ids.size(1)
    if L > seq_len:
        # keep last seq_len tokens
        ids = ids[:, -seq_len:]
    elif L < seq_len:
        pad = torch.full((1, seq_len - L), pad_id, dtype=ids.dtype)
        ids = torch.cat([ids, pad], dim=1)
    return ids


def build_inputs(tokenizer, all_token_ids):
    """
    all_token_ids: list[int] of full (prompt + generated) so far.
    Returns:
      input_ids [1, SEQ_LEN], attention_mask [1, SEQ_LEN], orig_len (before pad/trunc)
    """
    ids = torch.tensor(all_token_ids, dtype=torch.long).unsqueeze(0)  # [1, L]
    orig_len = ids.size(1)

    pad_id = tokenizer.pad_token_id
    input_ids = pad_truncate(ids, SEQ_LEN, pad_id)

    attention_mask = (input_ids != pad_id).long()

    return input_ids, attention_mask, orig_len


def load_executorch_program():
    print("Loading ExecuTorch program from:", PTE_PATH)
    if not PTE_PATH.exists():
        raise FileNotFoundError(f".pte file not found at: {PTE_PATH}")
    et_runtime: Runtime = Runtime.get()
    program = et_runtime.load_program(
        PTE_PATH,
        verification=Verification.Minimal,
    )
    print("Program methods:", program.method_names)
    forward = program.load_method("forward")
    meta = program.metadata("forward")
    print("Method metadata:", meta)
    return forward


def generate_reply(forward, tokenizer, prompt, max_new_tokens=16):
    # Tokenize prompt
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    prompt_ids = enc["input_ids"][0].tolist()  # list[int]
    eos_id = tokenizer.eos_token_id

    all_ids = prompt_ids.copy()
    print(f"Prompt token length: {len(prompt_ids)}")

    for step in range(max_new_tokens):
        input_ids, attention_mask, orig_len = build_inputs(tokenizer, all_ids)

        inputs = (input_ids, attention_mask)
        outputs = forward.execute(inputs)
        logits = outputs[0]  # [1, SEQ_LEN, vocab]

        # ---- FIXED last_pos computation ----
        # count how many non-pad tokens we have in this window, use last one
        num_real_tokens = int(attention_mask[0].sum().item())
        last_pos = num_real_tokens - 1  # index of last real token

        next_token_logits = logits[0, last_pos, :]
        next_token_id = int(next_token_logits.argmax(dim=-1))
        all_ids.append(next_token_id)

        print(f"Step {step+1}: next_token_id={next_token_id}")
        if next_token_id == eos_id:
            print(f"Stopped on EOS at step {step+1}")
            break

    # Decode only the new part (excluding original prompt)
    gen_ids = all_ids[len(prompt_ids):]
    reply = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return reply


def main():
    tokenizer = load_tokenizer()
    forward = load_executorch_program()

    prompt = "You: I feel very stressed about exams.\nMindmate:"
    print("\n=== PROMPT ===")
    print(prompt)

    reply = generate_reply(forward, tokenizer, prompt, max_new_tokens=16)

    print("\n=== MODEL REPLY ===")
    print(repr(reply))
    print(reply)


if __name__ == "__main__":
    main()
