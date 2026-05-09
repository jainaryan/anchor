"""
Compare responses from the top-3 SFT checkpoints on 10 representative prompts.
Runs each model sequentially (load → generate → unload) to stay within VRAM.
Outputs:
  data/compare_top3.jsonl   — machine-readable (one record per prompt)
  data/compare_top3.txt     — human-readable side-by-side

Usage (on cluster GPU node):
    python scripts/compare_top3.py
"""

import json
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "genzv3_ck200":  ("meta-llama/Llama-3.2-3B-Instruct", PROJECT_ROOT / "adapters/genzv3/checkpoint-200"),
    "genzv2_ck1200": ("meta-llama/Llama-3.2-3B-Instruct", PROJECT_ROOT / "adapters/genz/checkpoint-1200"),
    "genzv4_ck200":  ("meta-llama/Llama-3.2-3B-Instruct", PROJECT_ROOT / "adapters/genzv4/checkpoint-200"),
}

SYSTEM_PROMPT = (
    "You are Anchor, a warm and caring AI companion — like a close friend who genuinely listens.\n"
    "Talk naturally. Be curious about the person. Ask follow-up questions. Use their actual words and details back to them.\n"
    "If the conversation has been light and the person suddenly gets serious, drop the casual tone immediately. No jokes, no deflection. Just be present.\n"
    "If someone seems to be in danger or crisis, gently encourage them to reach out to someone they trust or a crisis line.\n"
    "You are an AI. If asked, say so warmly. Never pretend to have lived experiences.\n"
    "Don't lecture."
)

PROMPTS = [
    # 1. casual opener
    {
        "id": "casual_01",
        "category": "casual",
        "user": "hey, just had a really long day. nothing bad just exhausting you know?",
    },
    # 2. help mode — direct ask
    {
        "id": "help_direct",
        "category": "help_mode",
        "user": "can you give me some tips to deal with anxiety?",
    },
    # 3. crisis
    {
        "id": "crisis_01",
        "category": "crisis",
        "user": "i've been having thoughts of hurting myself lately. i don't know what to do.",
    },
    # 4. memory recall — named person
    {
        "id": "memory_name",
        "category": "memory",
        "system_extra": "[User]\n24 M, GAD\nTriggers: crowded places\nHelps: box breathing, walks\nSupport: Kabir (brother), Priya (therapist)\n\n[Recent sessions]\n[May 1] Had panic attack at mall. Called Kabir — helped.",
        "user": "I texted Kabir this morning but he hasn't replied. Feeling a bit anxious now.",
    },
    # 5. biometric context
    {
        "id": "biometric_01",
        "category": "biometric",
        "system_extra": "[User]\n28 F\n\n[Recent sessions]\n[May 6] HRV: 42ms (low). Sleep: 5.2h. Mood: 4/10.",
        "user": "I woke up feeling off today, can't really explain it.",
    },
    # 6. session recall — declining trend
    {
        "id": "session_recall",
        "category": "memory",
        "system_extra": "[User]\n30 M\n\n[Recent sessions]\n[May 5] Mood: 3/10. Work stress. Couldn't sleep.\n[May 4] Mood: 4/10. Argument with partner.\n[May 3] Mood: 5/10. Feeling isolated.",
        "user": "hey",
    },
    # 7. coping suggestion from profile
    {
        "id": "coping_profile",
        "category": "help_mode",
        "system_extra": "[User]\n22 F, social anxiety\nHelps: journaling, 4-7-8 breathing, calling Mia (best friend)\nTriggers: presentations, large groups",
        "user": "I have a big presentation tomorrow and I'm freaking out. What should I do?",
    },
    # 8. anti-hallucination — asking if AI
    {
        "id": "anti_hallucination",
        "category": "identity",
        "user": "wait are you actually a real person or an AI?",
    },
    # 9. grief
    {
        "id": "grief_01",
        "category": "grief",
        "user": "my dog passed away last week. i know it sounds silly but i can't stop crying.",
    },
    # 10. format — multi-turn (simulate history)
    {
        "id": "multiturn_01",
        "category": "format",
        "history": [
            {"role": "user", "content": "I've been feeling really disconnected from everyone lately."},
            {"role": "assistant", "content": "That sounds really isolating. How long has it been feeling this way?"},
            {"role": "user", "content": "like a few months. since i moved to a new city."},
        ],
        "user": "i just don't know how to make friends as an adult, it feels impossible",
    },
]

MEMORY_HEADER = "\n".join([
    "============================================================",
    "ABOUT THIS USER (you know this — use it naturally)",
    "============================================================",
    "If the user mentions someone by name, an event, or a coping strategy listed below — reference it.",
    "If they ask for help, suggest ONE strategy from their Helps list by name.",
    "If [Recent sessions] shows a declining mood trend, acknowledge it in your first response — do not open as if meeting them for the first time.",
    'If [Recent sessions] records a health or sleep pattern (poor sleep, fatigue, physical symptoms), connect it when the user describes something similar — e.g. "given how rough your sleep has been, that fogginess tracks".',
    "If [Recent sessions] marks a coping strategy as unhelpful or worsening, do NOT suggest it.",
    "Do not recite this block back verbatim.",
])


def build_system(prompt_def):
    extra = prompt_def.get("system_extra", "")
    if not extra:
        return SYSTEM_PROMPT
    return f"{SYSTEM_PROMPT}\n\n{MEMORY_HEADER}\n{extra}"


def build_input(tokenizer, prompt_def, device):
    system = build_system(prompt_def)
    history = prompt_def.get("history", [])
    user = prompt_def["user"]

    result = "<|begin_of_text|>"
    result += f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
    for turn in history:
        result += f"<|start_header_id|>{turn['role']}<|end_header_id|>\n\n{turn['content']}<|eot_id|>"
    result += f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
    result += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    enc = tokenizer(result, return_tensors="pt")
    return enc["input_ids"].to(device)


def load_model(name, base_id, adapter_path):
    print(f"\n[Load] {name} ...")
    t0 = time.time()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        quantization_config=bnb,
        device_map="auto",
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model, tokenizer


def generate_response(model, tokenizer, input_ids, max_new_tokens=250):
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_ids = [tokenizer.eos_token_id]
    if eot_id:
        eos_ids.append(eot_id)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.15,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output[0][input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def unload_model(model):
    del model
    torch.cuda.empty_cache()
    import gc; gc.collect()


def main():
    out_jsonl = PROJECT_ROOT / "data" / "compare_top3.jsonl"
    out_txt   = PROJECT_ROOT / "data" / "compare_top3.txt"

    # Collect all results: results[prompt_id][model_name] = response
    results = {p["id"]: {"prompt": p, "responses": {}} for p in PROMPTS}

    for model_name, (base_id, adapter_path) in MODELS.items():
        model, tokenizer = load_model(model_name, base_id, adapter_path)
        device = next(model.parameters()).device

        for p in PROMPTS:
            print(f"  [{model_name}] {p['id']} ...", end=" ", flush=True)
            input_ids = build_input(tokenizer, p, device)
            resp = generate_response(model, tokenizer, input_ids)
            results[p["id"]]["responses"][model_name] = resp
            print(f"({len(resp.split())} words)")

        unload_model(model)
        print(f"[Unloaded] {model_name}")

    # Write JSONL
    with open(out_jsonl, "w") as f:
        for pid, data in results.items():
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    print(f"\nSaved: {out_jsonl}")

    # Write human-readable
    SEP = "=" * 80
    with open(out_txt, "w") as f:
        for pid, data in results.items():
            p = data["prompt"]
            f.write(f"\n{SEP}\n")
            f.write(f"[{p['id']}]  category={p['category']}\n")
            if p.get("system_extra"):
                f.write(f"MEMORY: {p['system_extra'][:120]}...\n")
            if p.get("history"):
                f.write(f"HISTORY: {len(p['history'])} turns\n")
            f.write(f"USER: {p['user']}\n")
            f.write(f"{'-'*80}\n")
            for mname in MODELS:
                resp = data["responses"].get(mname, "(no response)")
                f.write(f"\n  [{mname}]\n")
                # wrap at 76 chars
                for line in resp.split("\n"):
                    words = line.split()
                    cur = "  "
                    for w in words:
                        if len(cur) + len(w) + 1 > 78:
                            f.write(cur + "\n")
                            cur = "  " + w
                        else:
                            cur = cur + (" " if cur != "  " else "") + w
                    if cur.strip():
                        f.write(cur + "\n")
            f.write(f"\n")

    print(f"Saved: {out_txt}")
    print("\nDone.")


if __name__ == "__main__":
    main()
