import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os

MODEL_DIR = "models/CUDA_llama-3.2-3b-instruct"
ADAPTER_DIR = "adapters/CUDA_mindmate_llama32b"

def test():
    print(f"[info] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    print(f"[info] Loading model (4-bit)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        quantization_config=quant_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
    )
    
    print(f"[info] Loading adapters...")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    
    messages = [{"role": "user", "content": "Hello"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"[debug] Prompt: {repr(prompt)}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    print(f"[info] Generating...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False, # Deterministic for debugging
        )
    
    generated_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    print(f"[debug] Generated Token IDs: {generated_tokens.tolist()}")
    
    response = tokenizer.decode(generated_tokens, skip_special_tokens=False)
    print(f"[debug] Decoded Response: {repr(response)}")

if __name__ == "__main__":
    test()
