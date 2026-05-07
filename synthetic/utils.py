import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Optional
import time

# Configuration
# TEACHER_MODEL env var selects the teacher:
#   "30b"     → Qwen3-30B-A3B MoE (default, ~60GB VRAM bfloat16, A100-80)
#   "72b"     → Qwen3-72B dense (~40GB in 4-bit, H100-96 or H200-141)
#   "gemma4"  → Gemma 4 26B A4B MoE (~52GB bfloat16, A100-80, ~4B active params)
_TEACHER = os.environ.get("TEACHER_MODEL", "gemma4").lower()
if _TEACHER == "72b":
    MODEL_ID = "Qwen/Qwen3-72B-Instruct"
elif _TEACHER == "gemma4":
    MODEL_ID = "google/gemma-4-26B-A4B-it"
else:
    MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"

VLLM_URL = "http://localhost:8000/v1"
USE_VLLM = False # Set to True for vLLM deployment

# Set USE_4BIT=1 in environment to load in 4-bit (required for 72B on H100/H200)
# Default: bfloat16 full precision (higher quality, requires ~60GB VRAM e.g. A100-80)
# Note: 72B always forces 4-bit; gemma4 and 30b run bfloat16 on A100-80
USE_4BIT = os.environ.get("USE_4BIT", "0") == "1" or _TEACHER == "72b"

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
        hf_token = os.environ.get("HF_TOKEN", None)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
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
                token=hf_token,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                token=hf_token,
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
        # enable_thinking=False: Qwen3 is a thinking model — without this it prepends
        # <think>...</think> blocks that break JSON extraction downstream.
        template_kwargs = {"return_tensors": "pt", "add_generation_prompt": True}
        try:
            outputs = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **template_kwargs
            )
        except TypeError:
            # Fallback for non-Qwen3 tokenizers that don't support enable_thinking
            outputs = self.tokenizer.apply_chat_template(
                messages, **template_kwargs
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
    - Qwen3 <think>...</think> blocks
    - Markdown code blocks (```json ... ```)
    - Raw JSON strings
    - Truncated JSON (missing closing braces)
    - Field-level regex extraction as final fallback
    """
    # 1. Strip Qwen3 thinking blocks <think>...</think>
    resp_clean = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    
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

def randomize_health_context(profile: dict) -> dict:
    """
    Return a shallow copy of the profile with the health_context string
    randomized — numbers, dates, and qualitative descriptors are varied
    so the teacher model never sees the same seed twice.

    This prevents the model from memorizing specific strings like
    "Mood 4/10 — noticeably lower than last session (was 7/10)" and instead
    learns the general skill of referencing any health data in the context block.
    """
    import copy
    p = copy.copy(profile)
    ctx = p["health_context"]
    health_type = p["health_type"]

    # ── Date randomization ─────────────────────────────────────────────────────
    # Shift all dates together so multi-session ordering is preserved
    DATE_POOL = [
        "Mar 28", "Mar 29", "Mar 30", "Mar 31",
        "Apr 2",  "Apr 3",  "Apr 4",  "Apr 5",  "Apr 6",  "Apr 7",
        "Apr 8",  "Apr 9",  "Apr 10", "Apr 11", "Apr 12", "Apr 13",
        "Apr 14", "Apr 15", "Apr 16", "Apr 17", "Apr 18", "Apr 19",
        "Apr 20", "Apr 21", "Apr 22", "Apr 23", "Apr 24", "Apr 25",
        "May 1",  "May 2",  "May 3",  "May 4",  "May 5",
    ]
    existing_dates = re.findall(r'\[([A-Z][a-z]{2} \d+)\]', ctx)
    if existing_dates:
        max_offset = max(0, len(DATE_POOL) - len(existing_dates))
        offset = random.randint(0, max_offset)
        new_dates = DATE_POOL[offset: offset + len(existing_dates)]
        for old, new in zip(existing_dates, new_dates):
            ctx = ctx.replace(f'[{old}]', f'[{new}]', 1)

    # ── Sleep randomization ────────────────────────────────────────────────────
    if health_type == "sleep":
        hrs_lo = random.randint(2, 5)
        hrs_hi = min(hrs_lo + random.randint(0, 2), 6)
        ctx = re.sub(r'\d-\d hrs', f'{hrs_lo}-{hrs_hi} hrs', ctx)
        ctx = re.sub(r'\b\d hrs\b',   f'{random.randint(2, 5)} hrs',   ctx)
        ctx = re.sub(r'\b\d hours\b', f'{random.randint(2, 5)} hours', ctx)
        ctx = re.sub(r'wok(e|ing) \d+-?\d* times',
                     f'wok\\1 {random.randint(2, 6)} times', ctx)
        for q in ["very poor", "severely disrupted", "extremely poor",
                  "very disrupted", "badly disrupted", "terrible"]:
            if q in ctx:
                ctx = ctx.replace(q, random.choice([
                    "very poor", "severely disrupted", "extremely poor",
                    "very disrupted", "badly disrupted", "really terrible",
                ]), 1)
                break
        ctx = re.sub(
            r'about a week|a week\b|ten days|over a week|nearly two weeks'
            r'|almost two weeks|several days|more than a week',
            random.choice([
                "about a week", "nearly two weeks", "several days",
                "over a week", "more than a week", "almost ten days",
            ]), ctx, count=1)

    # ── Mood score randomization ───────────────────────────────────────────────
    elif health_type == "mood_trend":
        n_scores = len(re.findall(r'\d+/10', ctx))
        if n_scores >= 2:
            # Multi-session: generate a coherent declining sequence
            start = random.randint(6, 9)
            scores = [start]
            for _ in range(n_scores - 1):
                scores.append(max(1, scores[-1] - random.randint(1, 3)))
            idx = [0]
            def _replace_score(m):
                s = scores[min(idx[0], n_scores - 1)]
                idx[0] += 1
                return f'{s}/10'
            ctx = re.sub(r'\d+/10', _replace_score, ctx)
            # Also fix "was X/10" back-references (appear in parentheses)
            # These are already replaced above in the same pass — fine.
        elif n_scores == 1:
            score = random.randint(2, 6)
            ctx = re.sub(r'\d+/10', f'{score}/10', ctx, count=1)
            # Fix "was X/10" to be higher than current score
            prev = min(score + random.randint(2, 4), 9)
            ctx = re.sub(r'was \d+/10', f'was {prev}/10', ctx)
        for low_desc in ["noticeably lower", "significantly lower", "much lower",
                         "considerably worse", "markedly lower"]:
            if low_desc in ctx:
                ctx = ctx.replace(low_desc, random.choice([
                    "noticeably lower", "significantly lower", "much lower",
                    "considerably worse", "markedly lower",
                ]), 1)
                break
        for bad_desc in ["very bad", "extremely low", "really struggling",
                         "at a low point", "rock bottom"]:
            if bad_desc in ctx:
                ctx = ctx.replace(bad_desc, random.choice([
                    "very bad", "extremely low", "really struggling",
                    "at a low point", "at its lowest",
                ]), 1)
                break

    # ── Anxiety intensity randomization ───────────────────────────────────────
    elif health_type == "anxiety_intensity":
        # Replace "Anxiety X/10" with a varied score in the moderate-high range
        ctx = re.sub(r'Anxiety \d+/10',
                     lambda m: f'Anxiety {random.randint(5, 9)}/10', ctx)
        # "always braced" / "constant background" descriptors
        for phrase in ["always braced for something bad", "constantly braced"]:
            if phrase in ctx:
                ctx = ctx.replace(phrase, random.choice([
                    "always braced for something bad",
                    "constantly waiting for the next problem",
                    "permanently on guard",
                    "braced for the worst",
                    "never fully off alert",
                ]), 1)
                break

    # ── Energy randomization ───────────────────────────────────────────────────
    elif health_type == "energy":
        for phrase in ["running on empty", "running on fumes", "nearly empty",
                       "nearly depleted", "very low", "half capacity"]:
            if phrase in ctx:
                ctx = ctx.replace(phrase, random.choice([
                    "running on empty", "running on fumes", "nearly depleted",
                    "half capacity", "very low", "at rock bottom energy-wise",
                    "almost nothing left",
                ]), 1)
                break
        # Randomize "hitting a wall by Xpm"
        ctx = re.sub(r'by \d+(am|pm)',
                     f'by {random.choice([11, 12, 1, 2, 3])}{"am" if random.random() < 0.1 else "pm"}',
                     ctx)

    # ── Physical symptoms randomization ───────────────────────────────────────
    elif health_type == "physical_symptoms":
        for phrase in ["high-stress moments", "high-stress days",
                       "stressful periods", "stressful days"]:
            if phrase in ctx:
                ctx = ctx.replace(phrase, random.choice([
                    "high-stress moments", "high-stress days",
                    "stressful periods", "stressful days",
                    "pressure-filled situations", "overwhelming days",
                ]), 1)
                break

    # ── Social withdrawal randomization ───────────────────────────────────────
    elif health_type == "social_withdrawal":
        n = random.randint(2, 5)
        ctx = re.sub(r'declined? (two|three|four|five|\d+) invitations?',
                     f'declined {n} invitation{"s" if n > 1 else ""}', ctx)
        ctx = re.sub(r"hasn't seen anyone in person for (a week|two weeks|\d+ days)",
                     f"hasn't seen anyone in person for {random.choice(['a week', 'ten days', 'over a week', 'nearly two weeks'])}",
                     ctx)

    p = dict(p)
    p["health_context"] = ctx
    return p


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

