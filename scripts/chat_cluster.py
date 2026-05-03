"""
Interactive chat with MindMate on cluster GPU.

Implements the full production stack:
  - Anchor system prompt (matches anchorSystemPrompt.ts)
  - Memory engine (matches contextBuilder.ts assemblePrompt())
  - Llama 3 chat format

Usage (interactive srun session):
    srun --partition=gpu-long --gres=gpu:a100-80:1 --pty bash
    source ~/projects/mindmate/mindmatenv/bin/activate
    cd ~/projects/mindmate
    python scripts/chat_cluster.py
    python scripts/chat_cluster.py --adapter adapters/genzv3/checkpoint-200
    python scripts/chat_cluster.py --adapter adapters/genz/checkpoint-1200 --label genzv2_ck1200

Memory setup (optional, prompted at start):
    Profile  — age/gender/diagnoses/triggers/coping/support
    Sessions — recent session summaries seeded into context
"""

import argparse
import sys
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--adapter", default="adapters/genzv3/checkpoint-200",
                    help="Adapter path relative to project root")
parser.add_argument("--base", default="meta-llama/Llama-3.2-3B-Instruct",
                    help="Base model ID")
parser.add_argument("--label", default=None,
                    help="Display name (defaults to adapter basename)")
parser.add_argument("--no-memory", action="store_true",
                    help="Skip memory setup, use bare system prompt")
parser.add_argument("--max-new-tokens", type=int, default=300)
parser.add_argument("--temperature", type=float, default=0.7)
args = parser.parse_args()

adapter_path = PROJECT_ROOT / args.adapter
label = args.label or Path(args.adapter).name

# ── Anchor base prompt (matches anchorSystemPrompt.ts exactly) ────────────────

_ANCHOR_PROMPT = (
    "You are Anchor, a warm and caring AI companion — like a close friend who genuinely listens.\n"
    "Talk naturally. Be curious about the person. Ask follow-up questions. Use their actual words and details back to them.\n"
    "If the conversation has been light and the person suddenly gets serious, drop the casual tone immediately. No jokes, no deflection. Just be present.\n"
    "If someone seems to be in danger or crisis, gently encourage them to reach out to someone they trust or a crisis line.\n"
    "You are an AI. If asked, say so warmly. Never pretend to have lived experiences.\n"
    "Don't lecture."
)

# ── Memory engine (matches contextBuilder.ts assemblePrompt() exactly) ────────

_MEMORY_HEADER = "\n".join([
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


def build_system_prompt(profile: str = "", sessions: str = "") -> str:
    """Port of contextBuilder.ts assemblePrompt()."""
    if not profile and not sessions:
        return _ANCHOR_PROMPT
    blocks = []
    if profile:
        blocks.append(f"[User]\n{profile}")
    if sessions:
        blocks.append(f"[Recent sessions]\n{sessions}")
    return f"{_ANCHOR_PROMPT}\n\n{_MEMORY_HEADER}\n" + "\n\n".join(blocks)


def build_prompt(system: str, history: list[dict], user_msg: str) -> str:
    """Llama 3 chat format — matches build_prompt() in run_benchmarks.py."""
    result = "<|begin_of_text|>"
    result += f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
    for turn in history:
        result += f"<|start_header_id|>{turn['role']}<|end_header_id|>\n\n{turn['content']}<|eot_id|>"
    result += f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
    result += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return result


# ── Memory setup ──────────────────────────────────────────────────────────────

def collect_memory() -> tuple[str, str]:
    print()
    print("─" * 60)
    print("  MEMORY SETUP (press Enter to skip)")
    print("─" * 60)
    print("Profile line (e.g. '24 M, GAD\\nTriggers: crowds\\nHelps: box breathing\\nSupport: Kabir (brother)')")
    print("  → Enter multiple lines, type END on its own line when done, or just Enter to skip")

    profile_lines = []
    while True:
        line = input()
        if line.strip().upper() == "END" or (line == "" and not profile_lines):
            break
        if line == "" and profile_lines:
            break
        profile_lines.append(line)
    profile = "\n".join(profile_lines).strip()

    print()
    print("Recent sessions (e.g. '[May 1] Panic attack at work. Tried box breathing — helped.')")
    print("  → Enter multiple lines (one per session), type END when done, or just Enter to skip")
    session_lines = []
    while True:
        line = input()
        if line.strip().upper() == "END" or (line == "" and not session_lines):
            break
        if line == "" and session_lines:
            break
        session_lines.append(line)
    sessions = "\n".join(session_lines).strip()

    return profile, sessions


# ── Load model ────────────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print(f"  MindMate Chat — {label}")
print(f"  Adapter: {adapter_path}")
print(f"{'=' * 60}\n")

if not adapter_path.exists():
    print(f"ERROR: adapter not found: {adapter_path}")
    sys.exit(1)

if not torch.cuda.is_available():
    print("ERROR: CUDA not available — run this on a GPU node")
    sys.exit(1)

print("Loading model...")
t0 = time.time()

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(args.base)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

base_model = AutoModelForCausalLM.from_pretrained(
    args.base,
    quantization_config=bnb,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
)
model = PeftModel.from_pretrained(base_model, str(adapter_path))
model.eval()

device = next(model.parameters()).device
print(f"Model loaded in {time.time() - t0:.1f}s on {device}\n")

eot_id  = tokenizer.convert_tokens_to_ids("<|eot_id|>")
eos_ids = [tokenizer.eos_token_id, eot_id] if eot_id else [tokenizer.eos_token_id]

# ── Memory setup ──────────────────────────────────────────────────────────────

if args.no_memory:
    profile, sessions = "", ""
else:
    profile, sessions = collect_memory()

system_prompt = build_system_prompt(profile, sessions)

print()
print("─" * 60)
print("  System prompt preview (first 300 chars):")
print(f"  {system_prompt[:300].replace(chr(10), chr(10) + '  ')}")
print("─" * 60)
print()
print("  Type your message and press Enter. Commands:")
print("  /memory  — show current system prompt")
print("  /reset   — clear conversation history")
print("  /quit    — exit")
print("─" * 60)
print()

# ── Chat loop ─────────────────────────────────────────────────────────────────

history: list[dict] = []
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        break

    if not user_input:
        continue

    if user_input == "/quit":
        print("Bye.")
        break
    elif user_input == "/reset":
        history = []
        print("[History cleared]\n")
        continue
    elif user_input == "/memory":
        print(f"\n{system_prompt}\n")
        continue

    prompt = build_prompt(system_prompt, history, user_input)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("Anchor: ", end="", flush=True)
    t_start = time.time()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=0.9,
            top_k=50,
            do_sample=True,
            repetition_penalty=1.15,
            streamer=streamer,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    elapsed = time.time() - t_start
    tps = len(new_tokens) / elapsed
    print(f"\n  [{len(new_tokens)} tokens, {tps:.1f} tok/s]\n")

    history.append({"role": "user",    "content": user_input})
    history.append({"role": "assistant", "content": response})
