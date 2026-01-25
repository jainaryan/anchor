
#%%
import sys
import torch

print("Python:", sys.executable)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))




#%%


from pathlib import Path
from mlx_lm import load, generate

# === Paths relative to this file ===
HERE = Path(__file__).resolve().parent

# Base MLX model directory (same as your training)
MODEL_DIR = HERE.parent / "mlx_llama32_3b"

# Adapter directory we created inside mvp/adapters/
ADAPTER_DIR = HERE / "adapters" / "mindmate_llama32_3b_step1000"

print("Model dir:   ", MODEL_DIR)
print("Adapter dir: ", ADAPTER_DIR)

# === Load base model + 1000-step LoRA adapter ===
model, tokenizer = load(
    str(MODEL_DIR),
    adapter_path=str(ADAPTER_DIR),
)

print("✅ Loaded MLX LLaMA 3.2 with 1000-step MindMate adapter")

# === Test prompt ===
prompt = "You: hey! uni is so tough im struggling\nMindmate:"

output = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=80,
    # temp=0.7,
    # top_p=0.9,
)

print("\n--- Model output ---\n")
print(output)
print("\n--------------------")

# %%
from pathlib import Path
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel

# === Paths relative to this file ===
HERE = Path(__file__).resolve().parent

# Base model directory (exported in HF format)
MODEL_DIR = HERE.parent / "mlx_llama32_3b"

# LoRA adapter directory
ADAPTER_DIR = HERE / "adapters" / "mindmate_llama32_3b_step1000"

print("Model dir:   ", MODEL_DIR)
print("Adapter dir: ", ADAPTER_DIR)

# === Device ===
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR),
    use_fast=False,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_DIR),
    torch_dtype=dtype,
    device_map="auto",
    local_files_only=True,
)

model = PeftModel.from_pretrained(
    model,
    str(ADAPTER_DIR),
    local_files_only=True,
)

model.eval()


print("✅ Loaded LLaMA 3.2 + MindMate LoRA adapter (Windows)")

# === Test prompt ===
prompt = "You: hey! uni is so tough im struggling\nMindmate:"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
    )

output = tokenizer.decode(
    output_ids[0],
    skip_special_tokens=True,
)

print("\n--- Model output ---\n")
print(output)
print("\n--------------------")

# %%
