import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import time

# Configuration
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
VLLM_URL = "http://localhost:8000/v1"
USE_VLLM = False # Set to True for A100 deployment

# Set USE_4BIT=1 in environment to load in 4-bit (fits on H100-96/smaller GPUs)
# Default: bfloat16 full precision (higher quality, requires ~60GB VRAM e.g. A100-80)
USE_4BIT = os.environ.get("USE_4BIT", "0") == "1"

class TeacherModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

        if not USE_VLLM:
            self._load_local_model()

    def _load_local_model(self):
        print(f"[Teacher] Loading model from HuggingFace: {MODEL_ID}...")
        print(f"[Teacher] Quantization: {'4-bit NF4' if USE_4BIT else 'bfloat16 full precision'}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if USE_4BIT:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                quantization_config=quant_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )

        self.model.eval()
        print("[Teacher] Model loaded.")

    def generate(self, prompt: str, max_new_tokens: int = 1000, temperature: float = 0.7) -> str:
        """
        Generates text using the loaded model or vLLM API.
        """
        if USE_VLLM:
            # Placeholder for vLLM API call
            # import openai
            # client = openai.Client(base_url=VLLM_URL, api_key="EMPTY")
            # ...
            raise NotImplementedError("vLLM integration not yet enabled.")
        else:
            return self._generate_local(prompt, max_new_tokens, temperature)

    def _generate_local(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        messages = [
            {"role": "system", "content": "You are a data generation assistant. You must output strict, valid JSON only. Do not output markdown blocks or conversational text."},
            {"role": "user", "content": prompt}
        ]
        outputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
        
        # Handle BatchEncoding vs Tensor output
        input_ids = outputs
        if hasattr(outputs, "input_ids"):
            input_ids = outputs.input_ids
            
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids])
            
        if not isinstance(input_ids, torch.Tensor):
             # Fallback: maybe it's a BatchEncoding that behaves like a dict but didn't have input_ids attr?
             if hasattr(input_ids, "to"):
                 pass 
             else:
                 input_ids = torch.tensor(input_ids)

        input_ids = input_ids.to(self.device)
        
        if input_ids.dim() == 1:
             input_ids = input_ids.unsqueeze(0)
        
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
	 # Filter out None values (e.g. <|eot_id|> doesn't exist in Qwen3)
        terminators = [t for t in terminators if t is not None]


        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                eos_token_id=terminators,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_ids = output_ids[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response

import re

def parse_json_robust(response: str, expected_keys: list = None):
    """
    Robustly extracts and parses JSON from model output.
    Handles:
    - Markdown code blocks
    - Raw JSON strings
    - Truncated JSON (missing closing braces)
    - Field-level regex extraction as final fallback
    """
    # 1. Clean response (remove some common artifacts)
    resp_clean = response.strip()
    
    # 2. Try Standard JSON parsing from block or raw
    try:
        data = None
        if "```json" in resp_clean:
            data = json.loads(resp_clean.split("```json")[1].split("```")[0].strip())
        elif resp_clean.startswith("{"):
             # Try to find the matching closing brace for a raw object
             match = re.search(r"\{.*\}", resp_clean, re.DOTALL)
             if match:
                 data = json.loads(match.group(0))
        
        if data:
            if expected_keys:
                if all(k in data for k in expected_keys):
                    return data
            else:
                return data
    except:
        pass

    # 3. Try fixing truncated JSON (common with Llama 3 smaller models)
    try:
        # If it ends with a number or quote but no brace
        fixed = resp_clean
        if not fixed.rstrip().endswith("}"):
            fixed = fixed.rstrip() + "}"
        
        match = re.search(r"\{.*\}", fixed, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            if data and (not expected_keys or all(k in data for k in expected_keys)):
                return data
    except:
        pass

    # 4. Regex Fallback (Last Resort)
    if expected_keys:
        try:
            extracted = {}
            for key in expected_keys:
                # Basic string extractor
                p = rf'"{key}":\s*"(.*?)"'
                m = re.search(p, resp_clean)
                if m:
                    extracted[key] = m.group(1)
                else:
                    # Try int extractor
                    p_int = rf'"{key}":\s*(\d+)'
                    m_int = re.search(p_int, resp_clean)
                    if m_int:
                        extracted[key] = int(m_int.group(1))
            
            # Special case for "conversations" list (harder via regex, but let's try a simple one)
            if "conversations" in expected_keys and "conversations" not in extracted:
                # If we need conversations, regex is risky. Let's not fake it too much.
                pass
            
            if len(extracted) >= 2: # At least core info
                return extracted
        except:
             pass

def calculate_similarity(s1: str, s2: str) -> float:
    """Calculates Jaccard similarity based on word tokens."""
    words1 = set(re.findall(r'\w+', s1.lower()))
    words2 = set(re.findall(r'\w+', s2.lower()))
    if not words1 or not words2: return 0.0
    
    # Remove common stop words for better comparison
    stop_words = {"a", "the", "and", "or", "in", "with", "to", "for", "of", "on", "at"}
    words1 = words1 - stop_words
    words2 = words2 - stop_words
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union) if union else 0.0

def load_prompt(filename: str) -> str:
    path = Path(__file__).resolve().parent / "prompts" / filename
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def save_jsonl(data: List[Dict], filename: str):
    path = Path(__file__).resolve().parent / "outputs" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

