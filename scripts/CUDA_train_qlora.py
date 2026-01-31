import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
import os
import argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

parser = argparse.ArgumentParser()
parser.add_argument("--iters", type=int, default=1500)
args_parsed = parser.parse_args()

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

BASE_MODEL_DIR = "models/CUDA_llama-3.2-3b-instruct"
DATA_DIR = "data/cleaned_data/chunked_3072"
OUT_DIR = "adapters/CUDA_mindmate_llama32b"

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_DIR,
    use_fast=False,
    fix_mistral_regex=True,
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.enable_input_require_grads()

dataset = load_dataset(
    "json",
    data_files={
        "train": f"{DATA_DIR}/train.jsonl",
        "validation": f"{DATA_DIR}/valid.jsonl",
    },
)

import re

def tokenize(batch):
    # This function implements "Prompt-Loss Masking" for multi-turn conversations
    # using Llama 3 native chat templates to prevent hallucinations.
    
    formatted_texts = []
    
    # 1. Convert text to chat format and apply template
    for text in batch["text"]:
        # Parse our flat text back into a conversation list
        conversation = []
        parts = re.split(r"(<\|user\|>|<\|assistant\|>)", text)
        role = None
        for p in parts:
            p = p.strip()
            if p == "<|user|>":
                role = "user"
            elif p == "<|assistant|>":
                role = "assistant"
            elif role and p:
                conversation.append({"role": role, "content": p})
        
        # Apply Llama 3 template securely
        formatted = tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=False
        )
        formatted_texts.append(formatted)

    # 2. Tokenize the NEW formatted text
    tokenized = tokenizer(
        formatted_texts,
        truncation=True,
        max_length=2048,
        padding=False,
        return_offsets_mapping=True,
        add_special_tokens=False # Template adds BOS already
    )
    
    all_labels = []
    for i, input_ids in enumerate(tokenized["input_ids"]):
        text = formatted_texts[i]
        offsets = tokenized["offset_mapping"][i]
        labels = list(input_ids)
        
        # Mask everything by default
        for j in range(len(labels)):
            labels[j] = -100
        
        # Unmask assistant turns using Llama 3 header structure
        # Matches: <|start_header_id|>assistant<|end_header_id|>\n\n(AGENTS RESPONSE)<|eot_id|>
        pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)(?:<\|eot_id\|>|$)"
        
        assistant_ranges = []
        for m in re.finditer(pattern, text, re.DOTALL):
            assistant_ranges.append((m.start(1), m.end(1)))

        # Unmask tokens that fall within assistant ranges
        for j, (start, end) in enumerate(offsets):
            for a_start, a_end in assistant_ranges:
                if start >= a_start and end <= a_end and start < end:
                    labels[j] = input_ids[j]
                    break
        
        # ALWAYS Train on the EOS token (last token in a sequence should be EOS)
        if labels[-1] == -100:
             # Check if last token is indeed EOS
             if input_ids[-1] == tokenizer.eos_token_id or input_ids[-1] == 128009:
                 labels[-1] = input_ids[-1]

        all_labels.append(labels)
        
    tokenized["labels"] = all_labels
    tokenized["input_ids"] = tokenized["input_ids"] # ensure list of lists
    del tokenized["offset_mapping"]
    return tokenized

dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

args = TrainingArguments(
    output_dir=OUT_DIR,
    overwrite_output_dir=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=3e-5,
    max_steps=args_parsed.iters,
    logging_steps=10,
    save_steps=300,
    eval_strategy="steps",
    eval_steps=300,
    fp16=True,
    gradient_checkpointing=True,
    report_to="none",
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()

model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
