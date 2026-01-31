#%%

from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent 
# ✅ Correct base model
MODEL_DIR = ROOT / "models" / "CUDA_llama-3.2-3b-instruct"

# ✅ Pick ONE adapter checkpoint
ADAPTER_DIR = ROOT / "adapters" / "CUDA_mindmate_llama32b"

ADAPTER_WEIGHTS = ADAPTER_DIR / "adapter_model.safetensors"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True,
    fix_mistral_regex=True,
)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_DIR),
    quantization_config=quant_config,
    device_map={"": 0},
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    attn_implementation="sdpa",
    local_files_only=True,
)
model.config.use_cache = True

model = PeftModel.from_pretrained(
    model,
    str(ADAPTER_DIR),
    adapter_name="mindmate",
)

model.eval()

messages = [
    {"role": "user", "content": "hey! uni is so tough im struggling"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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
