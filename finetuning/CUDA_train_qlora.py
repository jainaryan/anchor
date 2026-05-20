import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, PeftModel
import bitsandbytes as bnb
import os
import argparse
from pathlib import Path
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

parser = argparse.ArgumentParser()
parser.add_argument("--iters", type=int, default=1600)
parser.add_argument("--out-dir", type=str, default=None,
                    help="Adapter output directory (default: adapters/CUDA_mindmate_llama32b_v2)")
parser.add_argument("--data-dir", type=str, default="data/conversations_cleaned",
                    help="Directory containing mindmate_train.jsonl and mindmate_val.jsonl")
parser.add_argument("--adapter-path", type=str, default=None,
                    help="Path to existing LoRA adapter for continued training (skips fresh LoRA init)")
args_parsed = parser.parse_args()

print(f'Version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')

# torch.cuda.is_available() returns False on this cluster (PyTorch cu130 vs driver 12090)
# even though device_map="auto" can still access the GPU. Skip the hard check and let
# the model load fail naturally if there truly is no GPU.
if not torch.cuda.is_available():
    print("[trainer] WARNING: torch.cuda.is_available()=False — proceeding anyway "
          "(cu130 vs driver 12090 mismatch; device_map=auto still works)")
    # ── PyTorch cu130 / driver 12090 compatibility shim ──────────────────────
    # torch._C._get_device_properties is not registered on this driver, so any
    # call that triggers _lazy_init (e.g. TrainingArguments device setup,
    # is_bf16_supported) raises DeferredCudaCallError via _check_capability.
    # bitsandbytes uses its own compiled CUDA extension and never calls _lazy_init,
    # so actual GPU operations (model load, forward/backward, optimizer) still work.
    # Fix: patch the three query functions + short-circuit _lazy_init globally.
    torch.cuda.is_available = lambda: True
    torch.cuda.is_bf16_supported = lambda *a, **kw: True
    torch.cuda._initialized = True          # _lazy_init returns immediately hereafter
    if hasattr(torch.cuda, "_queued_calls"):
        torch.cuda._queued_calls.clear()    # drop _check_capability from deferred queue
    # set_device() calls torch._C._cuda_setDevice() directly in C++ (bypasses _lazy_init).
    # device_map="auto" + bitsandbytes handles actual GPU placement; this call is redundant.
    torch.cuda.set_device = lambda *a, **kw: None
    # With _initialized=True, _lazy_call() executes its callable IMMEDIATELY instead of
    # queuing it. So torch.cuda.manual_seed_all() triggers a callback that indexes into
    # torch.cuda.default_generators (empty tuple when CUDA not actually initialized) →
    # IndexError. Patch these to no-ops; bitsandbytes handles all actual GPU RNG state.
    torch.cuda.manual_seed = lambda *a, **kw: None
    torch.cuda.manual_seed_all = lambda *a, **kw: None
    torch.cuda.device_count = lambda: 1
    torch.cuda.current_device = lambda: 0
    torch.cuda.synchronize = lambda *a, **kw: None
    torch.cuda.get_device_capability = lambda dev=None: (8, 0)  # A100-80
    torch.cuda.memory_allocated = lambda dev=None: 0
    torch.cuda.max_memory_allocated = lambda dev=None: 0
    torch.cuda.memory_reserved = lambda dev=None: 0


BASE_MODEL_DIR = "meta-llama/Llama-3.2-3B-Instruct"
DATA_DIR = args_parsed.data_dir
OUT_DIR = args_parsed.out_dir or "adapters/CUDA_mindmate_llama32b_v2"
TRAIN_FILE = Path(DATA_DIR) / "mindmate_train.jsonl"
VAL_FILE = Path(DATA_DIR) / "mindmate_val.jsonl"

print(f"[trainer] Data dir   : {DATA_DIR}")
print(f"[trainer] Output dir : {OUT_DIR}")
print(f"[trainer] Adapter    : {args_parsed.adapter_path or 'none (fresh LoRA)'}")
print(f"[trainer] Steps      : {args_parsed.iters}")

if not TRAIN_FILE.exists() or not VAL_FILE.exists():
    raise FileNotFoundError(
        "Expected cleaned dataset files were not found:\n"
        f" - {TRAIN_FILE}\n"
        f" - {VAL_FILE}\n"
        "Run finetuning/CUDA_run_pipeline.py (build + clean) before training."
    )

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_DIR,
)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    quantization_config=bnb_config,
    device_map="auto",
)

if args_parsed.adapter_path:
    # Continued training: load existing adapter weights, keep training
    print(f"[trainer] Loading existing adapter from {args_parsed.adapter_path}")
    model = PeftModel.from_pretrained(model, args_parsed.adapter_path, is_trainable=True)
    print(f"[trainer] Adapter loaded — continued training mode")
else:
    # Fresh training: initialise new LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    print(f"[trainer] Fresh LoRA initialised")

model.enable_input_require_grads()

dataset = load_dataset(
    "json",
    data_files={
        "train": str(TRAIN_FILE),
        "validation": str(VAL_FILE),
    },
)

import re

def tokenize(batch):
    # This function implements robust "Assistant-Only Loss Masking"
    # using structured 'conversations' and Llama 3 native templates.

    formatted_texts = []
    conversations = batch["conversations"]
    batch_weights = batch.get("loss_weight", [1.0] * len(conversations))
    
    for conv in conversations:
        # conv is already a list of {"role": "...", "content": "..."}
        formatted = tokenizer.apply_chat_template(
            conv,
            tokenize=False,
            add_generation_prompt=False
        )
        formatted_texts.append(formatted)

    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=2048,
        padding=False,
        return_offsets_mapping=True,
        add_special_tokens=False
    )
    
    all_labels = []
    for i, input_ids in enumerate(tokenized["input_ids"]):
        text = formatted_texts[i]
        offsets = tokenized["offset_mapping"][i]
        labels = list(input_ids)
        
        # 1. Start by masking everything
        for j in range(len(labels)):
            labels[j] = -100
        
        # 2. Identify Assistant Response Boundaries
        # Llama 3 template: <|start_header_id|>assistant<|end_header_id|>\n\n(CONTENT)<|eot_id|>
        # We want to unmask everything BETWEEN <|end_header_id|>\n\n and <|eot_id|>
        
        header_token = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        eot_token = "<|eot_id|>"
        
        search_start = 0
        while True:
            start_idx = text.find(header_token, search_start)
            if start_idx == -1:
                break
            
            # The actual content starts AFTER the header tokens
            content_start = start_idx + len(header_token)
            
            end_idx = text.find(eot_token, content_start)
            if end_idx == -1:
                # Content goes to end of string if EOT is missing
                content_end = len(text)
            else:
                # Include the EOT token in the loss calculation so model learns to stop
                content_end = end_idx + len(eot_token)
            
            # Unmask tokens whose character offsets fall within this range
            for j, (tok_start, tok_end) in enumerate(offsets):
                if tok_start >= content_start and tok_end <= content_end:
                    if tok_start < tok_end: # valid token
                        labels[j] = input_ids[j]
            
            search_start = content_end
            if end_idx == -1: break

        if len(labels) != len(input_ids):
            raise ValueError(
                f"Label/input length mismatch at sample {i}: "
                f"labels={len(labels)} input_ids={len(input_ids)}"
            )
        
        supervised = sum(1 for x in labels if x != -100)
        if supervised == 0:
            raise ValueError(
                f"No assistant-supervised tokens detected at sample {i}. "
                "Check chat template / masking boundaries."
            )

        all_labels.append(labels)
        
    tokenized["labels"] = all_labels
    tokenized["loss_weight"] = [float(w) if w is not None else 1.0 for w in batch_weights]
    del tokenized["offset_mapping"]
    return tokenized

dataset = dataset.map(tokenize, batched=True, remove_columns=["conversations"])

def validate_tokenized_split(split, split_name: str, sample_limit: int = 128):
    n = min(len(split), sample_limit)
    for i in range(n):
        ex = split[i]
        li = len(ex["input_ids"])
        la = len(ex["attention_mask"])
        ll = len(ex["labels"])
        if not (li == la == ll):
            raise ValueError(
                f"{split_name}[{i}] has inconsistent lengths: "
                f"input_ids={li}, attention_mask={la}, labels={ll}"
            )
    print(f"[preflight] {split_name}: validated {n} samples for length consistency")

validate_tokenized_split(dataset["train"], "train")
validate_tokenized_split(dataset["validation"], "validation")

args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    max_steps=args_parsed.iters,
    logging_steps=10,
    save_steps=200,
    eval_strategy="steps",
    eval_steps=200,
    bf16=True,
    gradient_checkpointing=True,
    report_to="none",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="paged_adamw_32bit",
)

_base_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8,
    label_pad_token_id=-100,
)


def collator(features):
    # Pop per-example loss weights before delegating to the seq2seq collator,
    # which doesn't know how to handle scalar fields.
    weights = []
    stripped = []
    for f in features:
        w = f.get("loss_weight", 1.0)
        weights.append(float(w) if w is not None else 1.0)
        stripped.append({k: v for k, v in f.items() if k != "loss_weight"})
    batch = _base_collator(stripped)
    batch["loss_weight"] = torch.tensor(weights, dtype=torch.float32)
    return batch

# Fail fast before launching Trainer if collation is malformed.
_smoke = [dataset["train"][j] for j in range(min(4, len(dataset["train"])))]
if _smoke:
    smoke_batch = collator(_smoke)
    if smoke_batch["input_ids"].shape != smoke_batch["labels"].shape:
        raise ValueError(
            f"Collator output mismatch: input_ids {smoke_batch['input_ids'].shape} "
            f"vs labels {smoke_batch['labels'].shape}"
        )
    print(
        "[preflight] collator smoke test passed with shape "
        f"{tuple(smoke_batch['input_ids'].shape)}"
    )

class WeightedLossTrainer(Trainer):
    """Trainer that scales each example's loss by its `loss_weight`.

    Per-example loss = mean cross-entropy over its unmasked (assistant) tokens.
    Batch loss = sum(w_i * L_i) / sum(w_i)   (weighted mean — keeps gradient scale
    comparable to unweighted training so the existing LR / cosine schedule still applies).
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        weights = inputs.pop("loss_weight", None)
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        flat = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        mask = (shift_labels != -100).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        per_example = (flat * mask).sum(dim=1) / denom

        if weights is not None:
            w = weights.to(per_example.device).float()
            loss = (per_example * w).sum() / w.sum().clamp(min=1e-6)
        else:
            loss = per_example.mean()

        return (loss, outputs) if return_outputs else loss


trainer = WeightedLossTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=collator,
)

trainer.train()

model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
