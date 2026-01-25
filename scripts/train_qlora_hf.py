import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

BASE_MODEL_DIR = "models/llama-3.2-3b-instruct"
DATA_DIR = "data/cleaned_data/chunked_3072"
OUT_DIR = "adapters/mindmate_llama32_3b_qlora_hf"

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_DIR,
    use_fast=False,
    fix_mistral_regex=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_DIR,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=20,
    lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

dataset = load_dataset(
    "json",
    data_files={
        "train": f"{DATA_DIR}/train.jsonl",
        "validation": f"{DATA_DIR}/valid.jsonl",
    },
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=3072,
        padding=False,
    )

dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=3e-5,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=200,
    eval_strategy="steps",
    eval_steps=200,
    fp16=True,
    report_to="none",
)


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
)

trainer.train()

model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
