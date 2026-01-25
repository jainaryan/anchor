#%%

from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent 
# ✅ Correct base model
MODEL_DIR = ROOT / "models" / "llama-3.2-3b-instruct"

# ✅ Pick ONE adapter checkpoint
ADAPTER_DIR = ROOT / "adapters" / "mindmate_llama32_3b_lora"

ADAPTER_WEIGHTS = ADAPTER_DIR / "adapters.safetensors"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True,
    fix_mistral_regex=True,
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
    adapter_name="mindmate",
    is_trainable=False,
)

# Manually load the checkpoint you want
model.load_adapter(
    str(ADAPTER_WEIGHTS)
)

model.eval()

prompt = "You: hey! uni is so tough im struggling\nMindmate:"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=80,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

# %%
