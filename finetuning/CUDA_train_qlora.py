import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb
import os
import argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

parser = argparse.ArgumentParser()
parser.add_argument("--iters", type=int, default=1500)
args_parsed = parser.parse_args()

print(f'Version: {torch.__version__}') 
print(f'CUDA available: {torch.cuda.is_available()}') 
print(f'CUDA version: {torch.version.cuda}')

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")
    

BASE_MODEL_DIR = "models/CUDA_llama-3.2-3b-instruct"
DATA_DIR = "data/conversations_cleaned"
OUT_DIR = "adapters/CUDA_mindmate_llama32b"

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

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.enable_input_require_grads()

dataset = load_dataset(
    "json",
    data_files={
        "train": f"{DATA_DIR}/mindmate_train.jsonl",
        "validation": f"{DATA_DIR}/mindmate_val.jsonl",
    },
)

import re

def tokenize(batch):
    # This function implements robust "Assistant-Only Loss Masking"
    # using structured 'conversations' and Llama 3 native templates.
    
    formatted_texts = []
    conversations = batch["conversations"]
    
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

        all_labels.append(labels)
        
    tokenized["labels"] = all_labels
    del tokenized["offset_mapping"]
    return tokenized

dataset = dataset.map(tokenize, batched=True, remove_columns=["conversations"])

args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
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
