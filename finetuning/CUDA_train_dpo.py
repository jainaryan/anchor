"""
MindMate DPO Training Script

Architecture:
  - Policy model : base (4-bit) + SFT adapter (trainable DPO LoRA on top)
  - Reference model: base (4-bit) + SFT adapter (fully frozen)
  precompute_ref_log_probs=True means TRL runs one forward pass over the whole
  dataset with ref_model upfront, caches the logprobs, then discards ref_model
  before training begins → peak memory = two 4-bit models only during precompute.

Usage:
    python finetuning/CUDA_train_dpo.py --model llama_ck200
    python finetuning/CUDA_train_dpo.py --model qwen25_3b
"""

import os
import json
import argparse
import torch
from pathlib import Path
from datasets import load_dataset, Features, Value
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import DPOTrainer, DPOConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Argument parsing ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=["llama", "qwen", "qwen25_3b", "llama_ck200", "llama_ck1600",
             "genzv2_ck1200", "genzv3_ck200", "genzv2_ck1600"],
    default="llama",
)
parser.add_argument("--steps", type=int, default=800)
parser.add_argument("--data", choices=["v1", "v2"], default="v2")
args = parser.parse_args()

# ── Per-model config ─────────────────────────────────────────────────────────
CONFIGS = {
    "llama": {
        "base_model": "meta-llama/Llama-3.2-3B-Instruct",
        "sft_adapter": "adapters/CUDA_mindmate_llama32b",
        "dpo_out": "adapters/CUDA_mindmate_llama32b_dpo",
        "trust_remote_code": False,
        "thinking": False,
    },
    "qwen": {
        "base_model": "Qwen/Qwen3-1.7B",
        "sft_adapter": "adapters/CUDA_mindmate_qwen3_1p7b",
        "dpo_out": "adapters/CUDA_mindmate_qwen3_1p7b_dpo",
        "trust_remote_code": True,
        "thinking": True,
    },
    "qwen25_3b": {
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "sft_adapter": "adapters/CUDA_mindmate_qwen25_3b/checkpoint-200",
        "dpo_out": "adapters/CUDA_mindmate_qwen25_3b_dpo_ck200",
        "trust_remote_code": True,
        "thinking": False,
    },
    "llama_ck200": {
        "base_model": "meta-llama/Llama-3.2-3B-Instruct",
        "sft_adapter": "adapters/CUDA_mindmate_llama32b/checkpoint-200",
        "dpo_out": "adapters/CUDA_mindmate_llama32b_dpo_ck200",
        "trust_remote_code": False,
        "thinking": False,
    },
    "llama_ck1600": {
        "base_model": "meta-llama/Llama-3.2-3B-Instruct",
        "sft_adapter": "adapters/genz/checkpoint-1600",
        "dpo_out": "adapters/genz_dpo_ck1600",
        "trust_remote_code": False,
        "thinking": False,
    },
    "genzv2_ck1200": {
        "base_model": "meta-llama/Llama-3.2-3B-Instruct",
        "sft_adapter": "adapters/genz/checkpoint-1200",
        "dpo_out": "adapters/genzv2_dpo_ck1200",
        "trust_remote_code": False,
        "thinking": False,
    },
    "genzv3_ck200": {
        "base_model": "meta-llama/Llama-3.2-3B-Instruct",
        "sft_adapter": "adapters/genzv3/checkpoint-200",
        "dpo_out": "adapters/genzv3_dpo_ck200",
        "trust_remote_code": False,
        "thinking": False,
    },
    "genzv2_ck1600": {
        "base_model": "meta-llama/Llama-3.2-3B-Instruct",
        "sft_adapter": "adapters/genz/checkpoint-1600",
        "dpo_out": "adapters/genz_dpo_ck1600",
        "trust_remote_code": False,
        "thinking": False,
    },
}

cfg = CONFIGS[args.model]
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_MODEL  = cfg["base_model"]
SFT_ADAPTER = PROJECT_ROOT / cfg["sft_adapter"]
DPO_OUT     = PROJECT_ROOT / cfg["dpo_out"]
_suffix     = "_v2" if args.data == "v2" else ""
TRAIN_FILE  = PROJECT_ROOT / "data" / f"dpo_train{_suffix}.jsonl"
VAL_FILE    = PROJECT_ROOT / "data" / f"dpo_val{_suffix}.jsonl"

print("=" * 60)
print(f"  MindMate DPO Training — {args.model.upper()}")
print("=" * 60)
print(f"  Base model : {BASE_MODEL}")
print(f"  SFT adapter: {SFT_ADAPTER}")
print(f"  DPO output : {DPO_OUT}")
print(f"  Steps      : {args.steps}")
print()

print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

for p in (TRAIN_FILE, VAL_FILE):
    if not p.exists():
        raise FileNotFoundError(f"DPO data not found: {p}")

# ── Tokenizer ────────────────────────────────────────────────────────────────
print("[1/5] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL, trust_remote_code=cfg["trust_remote_code"]
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# ── Shared BnB config ────────────────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

def load_base():
    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=cfg["trust_remote_code"],
    )

# ── Policy model: SFT adapter + new trainable DPO LoRA ──────────────────────
print("[2/5] Loading policy model (SFT + DPO LoRA)...")
policy_model = PeftModel.from_pretrained(load_base(), str(SFT_ADAPTER), is_trainable=True)
dpo_lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
policy_model.add_adapter("dpo", dpo_lora_config)
policy_model.set_adapter("dpo")
policy_model.enable_input_require_grads()

# ── Reference model: SFT adapter, fully frozen ───────────────────────────────
print("[3/5] Loading reference model (frozen SFT)...")
ref_model = PeftModel.from_pretrained(load_base(), str(SFT_ADAPTER), is_trainable=False)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

# ── Dataset ──────────────────────────────────────────────────────────────────
print("[4/5] Loading and preprocessing DPO dataset...")
dataset = load_dataset(
    "json",
    data_files={"train": str(TRAIN_FILE), "validation": str(VAL_FILE)},
)
print(f"  Train: {len(dataset['train'])} pairs | Val: {len(dataset['validation'])} pairs")


def format_messages(messages) -> str:
    if args.model in ("llama", "llama_ck200", "llama_ck1600", "genzv2_ck1200", "genzv3_ck200", "genzv2_ck1600"):
        # Manually construct Llama 3 prompt — bypasses Jinja2 template issues.
        # Must match the format used in SFT training (CUDA_train_qlora.py).
        result = "<|begin_of_text|>"
        for msg in messages:
            result += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
        return result
    if cfg["thinking"]:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def to_row_major(val):
    """Convert message data from any HF/Arrow format → plain list of {"role", "content"} dicts.
    Handles column-major dicts, row-major lists, and non-serializable Arrow types."""
    try:
        raw = json.loads(json.dumps(val, default=str))  # default=str handles Arrow scalars
    except Exception:
        raw = val

    if isinstance(raw, dict) and "role" in raw and isinstance(raw.get("role"), list):
        # Column-major: {"role": [...], "content": [...]}
        return [{"role": str(r), "content": str(c)}
                for r, c in zip(raw["role"], raw["content"])]
    if isinstance(raw, list):
        result = []
        for m in raw:
            if isinstance(m, dict):
                result.append({"role": str(m.get("role", "")), "content": str(m.get("content", ""))})
            else:
                try:
                    result.append({"role": str(m["role"]), "content": str(m["content"])})
                except (KeyError, TypeError):
                    try:
                        result.append({"role": str(m.role), "content": str(m.content)})
                    except AttributeError:
                        result.append({"role": str(m), "content": ""})
        return result
    if isinstance(raw, dict):
        return [{"role": str(raw.get("role", "")), "content": str(raw.get("content", ""))}]
    return []


def preprocess(example):
    prompt   = to_row_major(example["prompt"])
    chosen   = to_row_major(example["chosen"])
    rejected = to_row_major(example["rejected"])
    prompt_str       = format_messages(prompt)
    chosen_content   = chosen[0]["content"]   if chosen   else ""
    rejected_content = rejected[0]["content"] if rejected else ""
    return {"prompt": prompt_str, "chosen": chosen_content, "rejected": rejected_content}


out_features = Features({"prompt": Value("string"), "chosen": Value("string"), "rejected": Value("string")})
dataset = dataset.map(preprocess, batched=False, remove_columns=dataset["train"].column_names, features=out_features)

sample = dataset["train"][0]
assert isinstance(sample["prompt"], str),  f"prompt is not str: {type(sample['prompt'])}"
assert isinstance(sample["chosen"], str),  f"chosen is not str: {type(sample['chosen'])}"
assert isinstance(sample["rejected"], str),f"rejected is not str: {type(sample['rejected'])}"
print(f"[Sanity] prompt[:80] : {sample['prompt'][:80]!r}")
print(f"[Sanity] chosen[:80] : {sample['chosen'][:80]!r}")
print(f"[Sanity] rejected[:80]: {sample['rejected'][:80]!r}")

# ── DPO Training ─────────────────────────────────────────────────────────────
print("[5/5] Starting DPO training...")
DPO_OUT.mkdir(parents=True, exist_ok=True)

dpo_config = DPOConfig(
    output_dir=str(DPO_OUT),
    beta=0.1,
    loss_type="sigmoid",
    precompute_ref_log_probs=True,  # cache ref logprobs upfront, then free ref_model
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,
    max_steps=args.steps,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    optim="paged_adamw_32bit",
    max_length=1536,
    logging_steps=10,
    save_steps=200,
    eval_strategy="steps",
    eval_steps=200,
    bf16=True,
    gradient_checkpointing=True,
    report_to="none",
    remove_unused_columns=False,
)

trainer = DPOTrainer(
    model=policy_model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
)

trainer.train()

policy_model.set_adapter("dpo")
policy_model.save_pretrained(str(DPO_OUT))
tokenizer.save_pretrained(str(DPO_OUT))

print(f"\nDPO adapter saved to: {DPO_OUT}")
print("Done.")
