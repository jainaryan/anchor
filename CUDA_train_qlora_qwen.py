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
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
import os
import argparse
from pathlib import Path
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

parser = argparse.ArgumentParser()
parser.add_argument("--iters", type=int, default=1600)
args_parsed = parser.parse_args()

print(f'Version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

BASE_MODEL_DIR = "Qwen/Qwen3-1.7B"
DATA_DIR = "data/conversations_cleaned"
OUT_DIR = "adapters/CUDA_mindmate_qwen3_1p7b_v2"
TRAIN_FILE = Path(DATA_DIR) / "mindmate_train.jsonl"
VAL_FILE = Path(DATA_DIR) / "mindmate_val.jsonl"

if not TRAIN_FILE.exists() or not VAL_FILE.exists():
    raise FileNotFoundError(
        "Expected cleaned dataset files were not found:\n"
        f" - {TRAIN_FILE}\n"
        f" - {VAL_FILE}\n"
        "Run finetuning/CUDA_run_pipeline_qwen.py (build + clean) before training."
    )

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
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
    trust_remote_code=True,
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
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
    formatted_texts = []
    conversations = batch["conversations"]

    for conv in conversations:
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

        # Mask everything first
        for j in range(len(labels)):
            labels[j] = -100

        # Qwen3 chat template: <|im_start|>assistant\n(CONTENT)<|im_end|>
        header_token = "<|im_start|>assistant\n"
        eot_token = "<|im_end|>"

        search_start = 0
        while True:
            start_idx = text.find(header_token, search_start)
            if start_idx == -1:
                break

            content_start = start_idx + len(header_token)
            end_idx = text.find(eot_token, content_start)
            if end_idx == -1:
                content_end = len(text)
            else:
                content_end = end_idx + len(eot_token)

            for j, (tok_start, tok_end) in enumerate(offsets):
                if tok_start >= content_start and tok_end <= content_end:
                    if tok_start < tok_end:
                        labels[j] = input_ids[j]

            search_start = content_end
            if end_idx == -1:
                break

        if len(labels) != len(input_ids):
            raise ValueError(
                f"Label/input length mismatch at sample {i}: "
                f"labels={len(labels)} input_ids={len(input_ids)}"
            )

        supervised = sum(1 for x in labels if x != -100)
        if supervised == 0:
            raise ValueError(
                f"No assistant-supervised tokens at sample {i}. "
                "Check Qwen3 chat template / masking boundaries."
            )

        all_labels.append(labels)

    tokenized["labels"] = all_labels
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

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8,
    label_pad_token_id=-100,
)

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

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=collator,
)

trainer.train()

model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
