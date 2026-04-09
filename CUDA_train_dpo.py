"""
MindMate DPO Training Script
Trains a DPO adapter on top of the existing SFT adapter.

Architecture:
  - Base model (4-bit, frozen)
  - SFT adapter (frozen) → acts as reference policy
  - DPO adapter (trainable, stacked on top)

Usage:
    python finetuning/CUDA_train_dpo.py --model llama
    python finetuning/CUDA_train_dpo.py --model qwen
"""

import os
import argparse
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Argument parsing ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=["llama", "qwen"],
    default="llama",
    help="Which base model to train (llama = Llama-3.2-3B, qwen = Qwen3-1.7B)",
)
parser.add_argument("--steps", type=int, default=800)
args = parser.parse_args()

# ── Per-model config ────────────────────────────────────────────────────────
CONFIGS = {
    "llama": {
        "base_model": "meta-llama/Llama-3.2-3B-Instruct",
        "sft_adapter": "adapters/CUDA_mindmate_llama32b",
        "dpo_out": "adapters/CUDA_mindmate_llama32b_dpo",
        "trust_remote_code": False,
    },
    "qwen": {
        "base_model": "Qwen/Qwen3-1.7B",
        "sft_adapter": "adapters/CUDA_mindmate_qwen3_1p7b",
        "dpo_out": "adapters/CUDA_mindmate_qwen3_1p7b_dpo",
        "trust_remote_code": True,
    },
}

cfg = CONFIGS[args.model]
PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_MODEL = cfg["base_model"]
SFT_ADAPTER = PROJECT_ROOT / cfg["sft_adapter"]
DPO_OUT = PROJECT_ROOT / cfg["dpo_out"]
TRAIN_FILE = PROJECT_ROOT / "data" / "dpo_train.jsonl"
VAL_FILE = PROJECT_ROOT / "data" / "dpo_val.jsonl"

print("=" * 60)
print(f"  MindMate DPO Training — {args.model.upper()}")
print("=" * 60)
print(f"  Base model : {BASE_MODEL}")
print(f"  SFT adapter: {SFT_ADAPTER}")
print(f"  DPO output : {DPO_OUT}")
print(f"  Train file : {TRAIN_FILE}")
print(f"  Steps      : {args.steps}")
print()

print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

for p in (TRAIN_FILE, VAL_FILE):
    if not p.exists():
        raise FileNotFoundError(
            f"DPO data not found: {p}\n"
            "Run synthetic/dpo_pipeline.py first to generate data."
        )

# ── Tokenizer ───────────────────────────────────────────────────────────────
print("[1/4] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=cfg["trust_remote_code"],
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # DPO needs left-padding

# ── Base model in 4-bit ─────────────────────────────────────────────────────
print("[2/4] Loading base model (4-bit NF4)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    trust_remote_code=cfg["trust_remote_code"],
)

# ── Load SFT adapter (frozen) as reference policy ───────────────────────────
print("[3/4] Loading SFT adapter (frozen, reference policy)...")
model = PeftModel.from_pretrained(
    base_model,
    str(SFT_ADAPTER),
    is_trainable=False,
    adapter_name="reference",
)

# ── Add a new trainable DPO LoRA adapter on top ─────────────────────────────
print("[4/4] Adding trainable DPO LoRA adapter...")
dpo_lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)
model.add_adapter("policy", dpo_lora_config)
model.set_adapter("policy")
model.enable_input_require_grads()

# ── Dataset ─────────────────────────────────────────────────────────────────
print("\nLoading DPO dataset...")
dataset = load_dataset(
    "json",
    data_files={
        "train": str(TRAIN_FILE),
        "validation": str(VAL_FILE),
    },
)
print(f"  Train: {len(dataset['train'])} pairs")
print(f"  Val  : {len(dataset['validation'])} pairs")


def format_messages(messages: list) -> str:
    """Apply chat template to a message list."""
    if args.model == "qwen":
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def preprocess(batch):
    """
    Convert DPO format to TRL DPOTrainer format.
    DPOTrainer expects: prompt (str), chosen (str), rejected (str)
    where prompt is the formatted conversation up to (not including) the final assistant turn,
    and chosen/rejected are the final assistant response strings.
    """
    prompts, chosens, rejecteds = [], [], []

    for prompt_msgs, chosen_msgs, rejected_msgs in zip(
        batch["prompt"], batch["chosen"], batch["rejected"]
    ):
        # Format the prompt (conversation history without last assistant turn)
        # prompt_msgs already ends with a user message per our pipeline
        prompt_str = format_messages(prompt_msgs)

        # chosen/rejected are single-element lists: [{"role": "assistant", "content": "..."}]
        chosen_content = chosen_msgs[0]["content"] if chosen_msgs else ""
        rejected_content = rejected_msgs[0]["content"] if rejected_msgs else ""

        prompts.append(prompt_str)
        chosens.append(chosen_content)
        rejecteds.append(rejected_content)

    return {"prompt": prompts, "chosen": chosens, "rejected": rejecteds}


dataset = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

# Quick sanity check
sample = dataset["train"][0]
print(f"\n[Sanity] prompt[:200]: {sample['prompt'][:200]!r}")
print(f"[Sanity] chosen[:100]: {sample['chosen'][:100]!r}")
print(f"[Sanity] rejected[:100]: {sample['rejected'][:100]!r}")

# ── DPO Training ─────────────────────────────────────────────────────────────
DPO_OUT.mkdir(parents=True, exist_ok=True)

dpo_config = DPOConfig(
    output_dir=str(DPO_OUT),
    # DPO-specific
    beta=0.1,
    loss_type="sigmoid",
    precompute_ref_log_probs=True,        # compute ref logprobs once to avoid dual-model OOM
    ref_adapter_name="reference",         # TRL uses the frozen "reference" adapter as ref policy
    # Training
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,        # effective batch = 8
    learning_rate=5e-7,
    max_steps=args.steps,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    optim="paged_adamw_32bit",
    # Sequence lengths
    max_length=1536,
    max_prompt_length=1024,
    # Logging / checkpointing
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
    model=model,
    args=dpo_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
)

print("\n[DPO] Starting training...")
trainer.train()

# Save only the DPO (policy) adapter
model.set_adapter("policy")
model.save_pretrained(str(DPO_OUT))
tokenizer.save_pretrained(str(DPO_OUT))

print(f"\nDPO adapter saved to: {DPO_OUT}")
print("Done.")
