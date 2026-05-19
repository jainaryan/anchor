import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Optional
import time

# Configuration
# TEACHER_MODEL env var selects the teacher:
#   "30b"     → Qwen3-30B-A3B MoE (bfloat16, ~60GB VRAM, A100-80)
#   "72b"     → Qwen3-72B dense (bfloat16, H100-96 or H200-141)
#   "gemma4"  → Gemma 4 26B A4B MoE (~52GB bfloat16, A100-80, ~4B active params)
_TEACHER = os.environ.get("TEACHER_MODEL", "gemma4").lower()
if _TEACHER == "72b":
    MODEL_ID = "Qwen/Qwen3-72B-Instruct"
elif _TEACHER == "gemma4":
    MODEL_ID = "google/gemma-4-26B-A4B-it"
else:
    MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# USE_VLLM=1 env var enables vLLM backend (PagedAttention + FlashAttention2).
# ~1.5-2x faster inference vs HuggingFace for sequential generation.
# Requires: pip install vllm  (already available on cluster mindmatenv)
# Note: vLLM 0.20.2 V1 engine requires CUDA ≥12.1. Set VLLM_USE_V1=0 to force
# V0 engine on nodes with older drivers (e.g. xgph[10-18] have CUDA 12.0.90).
USE_VLLM = os.environ.get("USE_VLLM", "0") == "1"

# USE_FLASH_ATTN=1 enables FlashAttention 2 in the HF backend (~2x attention speedup).
# Requires flash-attn installed. Works on CUDA 12.0+. Falls back silently if unavailable.
USE_FLASH_ATTN = os.environ.get("USE_FLASH_ATTN", "0") == "1"

# USE_TORCH_COMPILE=1 wraps the HF model with torch.compile(mode="reduce-overhead").
# Captures CUDA graphs after first call, giving ~15-25% throughput improvement.
# Safe on any CUDA version — no new packages needed. Default off to avoid first-call delay.
USE_TORCH_COMPILE = os.environ.get("USE_TORCH_COMPILE", "0") == "1"


class TeacherModel:
    def __init__(self):
        self.use_vllm = USE_VLLM
        self.tokenizer = None
        self.model = None          # HF model (None when using vLLM)
        self.vllm_engine = None    # vLLM LLM (None when using HF)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if USE_VLLM:
            self._load_vllm()
        else:
            self._load_hf()

    # ── HuggingFace backend ────────────────────────────────────────────────────

    def _load_hf(self):
        hf_token = os.environ.get("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Pick attention implementation (best available):
        #   flash_attention_2 > sdpa > eager
        # sdpa = PyTorch's built-in fused attention (no extra package, ~15% faster than eager).
        # flash_attention_2 requires flash-attn package (USE_FLASH_ATTN=1).
        attn_impl = "sdpa"
        if USE_FLASH_ATTN:
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
            except ImportError:
                print("[Teacher] flash-attn not installed — using sdpa")

        print(f"[Teacher] Backend: HuggingFace  model={MODEL_ID}  attn={attn_impl}")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            token=hf_token,
            attn_implementation=attn_impl,
        )
        self.model.eval()

        if USE_TORCH_COMPILE:
            print("[Teacher] Compiling model with torch.compile(reduce-overhead) …")
            self.model = torch.compile(self.model, mode="reduce-overhead")
            print("[Teacher] torch.compile done.")

        print("[Teacher] HF model loaded.")

    def _hf_call(self, messages: list, max_new_tokens: int, temperature: float) -> str:
        template_kwargs = {"return_tensors": "pt", "add_generation_prompt": True}
        try:
            outputs = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **template_kwargs
            )
        except TypeError:
            outputs = self.tokenizer.apply_chat_template(messages, **template_kwargs)

        input_ids = outputs
        if hasattr(outputs, "input_ids"):
            input_ids = outputs.input_ids
        if isinstance(input_ids, list):
            input_ids = torch.tensor([input_ids])
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        input_ids = input_ids.to(self.device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
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
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    # ── vLLM backend ───────────────────────────────────────────────────────────

    def _load_vllm(self):
        """
        vLLM offline engine — PagedAttention + FlashAttention2.
        Qwen3-30B-A3B-Instruct-2507 (MoE) is supported via Qwen3MoeForCausalLM.
        GPU memory: ~60GB model + ~10GB KV cache on A100-80 with util=0.90.

        CLUSTER NOTE (2026-05-19): H100-96 nodes run CUDA driver 12090 (12.0.90).
        vLLM V1 calls torch.accelerator.set_device_index() which requires a newer
        driver → DeferredCudaCallError. Both subprocess and in-process modes fail.
        Use USE_VLLM=0 (HF backend) on this cluster. vLLM may work on newer nodes.
        """
        print("[Teacher] vLLM V1 engine — loading")

        try:
            from vllm import LLM
        except ImportError:
            raise RuntimeError(
                "[Teacher] USE_VLLM=1 but vllm is not installed. "
                "Run: uv pip install vllm"
            )
        tp = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
        print(f"[Teacher] Backend: vLLM  model={MODEL_ID}  tensor_parallel_size={tp}")
        # HF_TOKEN is picked up from the environment by vllm automatically.
        # tokenizer_kwargs is not supported in all vllm versions, so omit it.
        self.vllm_engine = LLM(
            model=MODEL_ID,
            dtype="bfloat16",
            trust_remote_code=True,
            gpu_memory_utilization=0.90,
            max_model_len=16384,          # sufficient for all our prompts
            enforce_eager=False,           # allow CUDA graphs for speed
            tokenizer_mode="auto",
            tensor_parallel_size=tp,      # 1 = A100-80, 2 = 2×A100-40
        )
        self.tokenizer = self.vllm_engine.get_tokenizer()
        print("[Teacher] vLLM engine ready.")

    def _vllm_call(self, messages: list, max_new_tokens: int, temperature: float) -> str:
        return self._vllm_batch_call([messages], max_new_tokens, temperature)[0]

    def _vllm_batch_call(self, messages_list: list, max_new_tokens: int, temperature: float) -> list:
        """Submit all prompts in one vLLM request — continuous batching handles them in parallel."""
        from vllm import SamplingParams

        prompts = []
        for messages in messages_list:
            try:
                p = self.tokenizer.apply_chat_template(
                    messages, enable_thinking=False,
                    tokenize=False, add_generation_prompt=True,
                )
            except TypeError:
                p = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            prompts.append(p)

        params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_new_tokens,
        )
        outputs = self.vllm_engine.generate(prompts, params)
        return [o.outputs[0].text for o in outputs]

    # ── Public API (same interface regardless of backend) ──────────────────────

    _DEFAULT_GEN_SYSTEM = (
        "You are a data generation assistant. "
        "You must output strict, valid JSON only. "
        "Do not output markdown blocks or conversational text."
    )

    def generate(self, prompt: str, max_new_tokens: int = 1000, temperature: float = 0.7,
                 system: str = None) -> str:
        """
        Structured data generation: wraps prompt in a JSON-assistant meta-system message.
        Use this for generating JSON outputs (user turn lists, scenario data, etc.).
        Pass system= to override the default system message (e.g. for sensitive content
        that the default message doesn't give enough context for).
        """
        messages = [
            {"role": "system", "content": system or self._DEFAULT_GEN_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        return self._call(messages, max_new_tokens, temperature)

    def generate_batch(self, prompts: list, max_new_tokens: int = 1000, temperature: float = 0.7,
                       system: str = None) -> list:
        """
        Batch version of generate(). Passes all prompts in a single vLLM call so they
        are processed in parallel via continuous batching. HF backend falls back to a
        sequential loop (same result, no speedup, but identical API).

        Returns a list of response strings in the same order as prompts.
        """
        _sys = system or self._DEFAULT_GEN_SYSTEM
        messages_list = [
            [{"role": "system", "content": _sys}, {"role": "user", "content": p}]
            for p in prompts
        ]
        if self.use_vllm:
            return self._vllm_batch_call(messages_list, max_new_tokens, temperature)
        return [self._hf_call(msgs, max_new_tokens, temperature) for msgs in messages_list]

    def chat(self, messages: list, max_new_tokens: int = 400, temperature: float = 0.82) -> str:
        """
        Generate the next assistant turn given a full messages list (system + history).

        Teacher-as-Anchor mode: caller sets messages[0]["role"]="system" to the
        production anchor prompt + injected memory. The teacher is constrained by
        that prompt exactly as the student model will be at inference time.

        Returns raw assistant text (not JSON).
        """
        return self._call(messages, max_new_tokens, temperature)

    def _call(self, messages: list, max_new_tokens: int, temperature: float) -> str:
        """Route to vLLM or HF backend."""
        if self.use_vllm:
            return self._vllm_call(messages, max_new_tokens, temperature)
        return self._hf_call(messages, max_new_tokens, temperature)


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

    # 2. Try standard JSON parsing from block or raw
    try:
        data = None
        if "```json" in resp_clean:
            data = json.loads(resp_clean.split("```json")[1].split("```")[0].strip())
        elif resp_clean.startswith("{"):
            match = re.search(r"\{.*\}", resp_clean, re.DOTALL)
            if match:
                data = json.loads(match.group(0))

        if data:
            if expected_keys:
                if all(k in data for k in expected_keys):
                    return data
            else:
                return data
    except Exception:
        pass

    # 3. Try fixing truncated JSON (common with smaller models)
    try:
        fixed = resp_clean
        if not fixed.rstrip().endswith("}"):
            fixed = fixed.rstrip() + "}"
        match = re.search(r"\{.*\}", fixed, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            if data and (not expected_keys or all(k in data for k in expected_keys)):
                return data
    except Exception:
        pass

    # 4. Regex fallback (last resort)
    if expected_keys:
        try:
            extracted = {}
            for key in expected_keys:
                p = rf'"{key}":\s*"(.*?)"'
                m = re.search(p, resp_clean)
                if m:
                    extracted[key] = m.group(1)
                else:
                    p_int = rf'"{key}":\s*(\d+)'
                    m_int = re.search(p_int, resp_clean)
                    if m_int:
                        extracted[key] = int(m_int.group(1))
            if len(extracted) >= 2:
                return extracted
        except Exception:
            pass

    return None


def calculate_similarity(s1: str, s2: str) -> float:
    """Calculates Jaccard similarity based on word tokens."""
    words1 = set(re.findall(r"\w+", s1.lower()))
    words2 = set(re.findall(r"\w+", s2.lower()))
    if not words1 or not words2:
        return 0.0
    stop_words = {"a", "the", "and", "or", "in", "with", "to", "for", "of", "on", "at"}
    words1 -= stop_words
    words2 -= stop_words
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
    DATE_POOL = [
        "Mar 28", "Mar 29", "Mar 30", "Mar 31",
        "Apr 2",  "Apr 3",  "Apr 4",  "Apr 5",  "Apr 6",  "Apr 7",
        "Apr 8",  "Apr 9",  "Apr 10", "Apr 11", "Apr 12", "Apr 13",
        "Apr 14", "Apr 15", "Apr 16", "Apr 17", "Apr 18", "Apr 19",
        "Apr 20", "Apr 21", "Apr 22", "Apr 23", "Apr 24", "Apr 25",
        "May 1",  "May 2",  "May 3",  "May 4",  "May 5",
    ]
    existing_dates = re.findall(r"\[([A-Z][a-z]{2} \d+)\]", ctx)
    if existing_dates:
        max_offset = max(0, len(DATE_POOL) - len(existing_dates))
        offset = random.randint(0, max_offset)
        new_dates = DATE_POOL[offset: offset + len(existing_dates)]
        for old, new in zip(existing_dates, new_dates):
            ctx = ctx.replace(f"[{old}]", f"[{new}]", 1)

    # ── Sleep randomization ────────────────────────────────────────────────────
    if health_type == "sleep":
        hrs_lo = random.randint(2, 5)
        hrs_hi = min(hrs_lo + random.randint(0, 2), 6)
        ctx = re.sub(r"\d-\d hrs", f"{hrs_lo}-{hrs_hi} hrs", ctx)
        ctx = re.sub(r"\b\d hrs\b", f"{random.randint(2, 5)} hrs", ctx)
        ctx = re.sub(r"\b\d hours\b", f"{random.randint(2, 5)} hours", ctx)
        ctx = re.sub(r"wok(e|ing) \d+-?\d* times", f"wok\\1 {random.randint(2, 6)} times", ctx)
        for q in ["very poor", "severely disrupted", "extremely poor",
                  "very disrupted", "badly disrupted", "terrible"]:
            if q in ctx:
                ctx = ctx.replace(q, random.choice([
                    "very poor", "severely disrupted", "extremely poor",
                    "very disrupted", "badly disrupted", "really terrible",
                ]), 1)
                break
        ctx = re.sub(
            r"about a week|a week\b|ten days|over a week|nearly two weeks"
            r"|almost two weeks|several days|more than a week",
            random.choice([
                "about a week", "nearly two weeks", "several days",
                "over a week", "more than a week", "almost ten days",
            ]), ctx, count=1)

    # ── Mood score randomization ───────────────────────────────────────────────
    elif health_type == "mood_trend":
        n_scores = len(re.findall(r"\d+/10", ctx))
        if n_scores >= 2:
            start = random.randint(6, 9)
            scores = [start]
            for _ in range(n_scores - 1):
                scores.append(max(1, scores[-1] - random.randint(1, 3)))
            idx = [0]

            def _replace_score(m):
                s = scores[min(idx[0], n_scores - 1)]
                idx[0] += 1
                return f"{s}/10"

            ctx = re.sub(r"\d+/10", _replace_score, ctx)
        elif n_scores == 1:
            score = random.randint(2, 6)
            ctx = re.sub(r"\d+/10", f"{score}/10", ctx, count=1)
            prev = min(score + random.randint(2, 4), 9)
            ctx = re.sub(r"was \d+/10", f"was {prev}/10", ctx)
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
        ctx = re.sub(r"Anxiety \d+/10",
                     lambda m: f"Anxiety {random.randint(5, 9)}/10", ctx)
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
        ctx = re.sub(
            r"by \d+(am|pm)",
            f"by {random.choice([11, 12, 1, 2, 3])}{'am' if random.random() < 0.1 else 'pm'}",
            ctx,
        )

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
        ctx = re.sub(
            r"declined? (two|three|four|five|\d+) invitations?",
            f"declined {n} invitation{'s' if n > 1 else ''}",
            ctx,
        )
        ctx = re.sub(
            r"hasn't seen anyone in person for (a week|two weeks|\d+ days)",
            f"hasn't seen anyone in person for {random.choice(['a week', 'ten days', 'over a week', 'nearly two weeks'])}",
            ctx,
        )

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
