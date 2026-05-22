"""
MindMate Benchmark Runner — v4

Differences from v3 (run_benchmarks.py):
  • Reads scenarios from benchmarks/scenarios.json (shared with mobile)
  • Uses JudgeService (shared with judge_mobile_results.py)
  • Emits structured logs via EvalLogger — trace.jsonl + trace.log + failures/
    + transcripts/
  • Supports scripted_multiturn + dynamic_multiturn (Qwen3 simulator +
    role-flipping)
  • **Sampling params match production app** (anchor-app
    src/utils/completionSettingsVersions.ts:defaultCompletionParams):
        temperature=0.7, top_k=40, top_p=0.95, min_p=0.05,
        repetition_penalty=1.0, max_new_tokens=1024
    Benchmarking with production sampling is the whole point of v4 — non-
    production sampling produces non-production behavior. The cost is that
    scenarios are no longer deterministic; use --runs=3 for stable averages.
  • --backend gguf — load Q4_K_M GGUF via llama-cpp-python instead of NF4
    adapter, so we benchmark what actually ships
  • Output directory layout matches judge_mobile_results.py so diff_results.py
    / leaderboard.py / replay.py all work without changes

Usage:
    # Default — NF4 adapter, production sampling, 1 run
    python -m benchmarks.run_benchmarks_v4 --model genzv2_ck1200

    # Test what ships
    python -m benchmarks.run_benchmarks_v4 --backend gguf \\
        --gguf-path exports/mindmate_genzv2_ck1200_q4_k_m.gguf \\
        --label genzv2_ck1200_gguf

    # Override sampling (e.g. for an ablation that needs determinism)
    python -m benchmarks.run_benchmarks_v4 --model genzv2_ck1200 --temperature 0

    # Category / id filters
    python -m benchmarks.run_benchmarks_v4 --model genzv2_ck1200 --category CRISIS
    python -m benchmarks.run_benchmarks_v4 --model genzv2_ck1200 --ids cm_01,cr_09

    # Run only scripted (skip dynamic — useful for fast iteration)
    python -m benchmarks.run_benchmarks_v4 --model genzv2_ck1200 --type single,scripted_multiturn

    # Ablation: system prompt modes (matches v3)
    python -m benchmarks.run_benchmarks_v4 --model llama_base --sysprompt none
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.logging import EvalLogger
from benchmarks.judge_service import (
    JudgeService,
    score_scenario,
    parse_judge_response,
)
from benchmarks.scenarios_loader import (
    APP_BASE_PROMPT,
    load_scenarios,
    load_holdout_scenarios,
    filter_scenarios,
    render_system_prompt,
)
from benchmarks.crisis_calibration import compute_crisis_calibration

JUDGE_MODEL_ID = "google/gemma-4-26B-A4B-it"
SIMULATOR_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"  # used for dynamic_multiturn

# Production sampling parameters — must stay in sync with
# anchor-app/src/utils/completionSettingsVersions.ts:defaultCompletionParams.
# Any drift here means the benchmark is measuring a different model than what
# ships. Don't change these casually.
PRODUCTION_SAMPLING = {
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.95,
    "min_p": 0.05,           # Not all transformers paths honor min_p — see _load_nf4_eval
    "repetition_penalty": 1.0,   # production has penalty_repeat=1.0 (disabled)
    "max_new_tokens": 1024,
}

MODEL_SHORTCUTS = {
    "llama_base":       ("meta-llama/Llama-3.2-3B-Instruct", None),
    "genzv2_ck1600":    ("meta-llama/Llama-3.2-3B-Instruct", "adapters/genz/checkpoint-1600"),
    "genzv2_ck1200":    ("meta-llama/Llama-3.2-3B-Instruct", "adapters/genz/checkpoint-1200"),
    "genzv3_ck200":     ("meta-llama/Llama-3.2-3B-Instruct", "adapters/genzv3/checkpoint-200"),
    "genzv4_ck200":     ("meta-llama/Llama-3.2-3B-Instruct", "adapters/genzv4/checkpoint-200"),
    "genzv5_ck200":     ("meta-llama/Llama-3.2-3B-Instruct", "adapters/genzv5/checkpoint-200"),
    "genzv5_ck400":     ("meta-llama/Llama-3.2-3B-Instruct", "adapters/genzv5/checkpoint-400"),
    "genzv5_ck600":     ("meta-llama/Llama-3.2-3B-Instruct", "adapters/genzv5/checkpoint-600"),
    "genzv5_ck800":     ("meta-llama/Llama-3.2-3B-Instruct", "adapters/genzv5/checkpoint-800"),
    "genzv5_ck1000":    ("meta-llama/Llama-3.2-3B-Instruct", "adapters/genzv5/checkpoint-1000"),
    "genzv5_ck1200":    ("meta-llama/Llama-3.2-3B-Instruct", "adapters/genzv5/checkpoint-1200"),
    "genzv5_ck1400":    ("meta-llama/Llama-3.2-3B-Instruct", "adapters/genzv5/checkpoint-1400"),
    "genzv5_ck1600":    ("meta-llama/Llama-3.2-3B-Instruct", "adapters/genzv5/checkpoint-1600"),
    "genzv5_ck1800":    ("meta-llama/Llama-3.2-3B-Instruct", "adapters/genzv5/checkpoint-1800"),
    "genzv5_ck2000":    ("meta-llama/Llama-3.2-3B-Instruct", "adapters/genzv5/checkpoint-2000"),
}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=list(MODEL_SHORTCUTS),
                       help="Named model shortcut")
    group.add_argument("--adapter", help="Path to adapter dir (relative to project root)")
    group.add_argument("--gguf-path", help="Path to GGUF file (use with --backend gguf)")

    p.add_argument("--base", default="meta-llama/Llama-3.2-3B-Instruct",
                   help="Base model ID (used with --adapter)")
    p.add_argument("--backend", choices=["nf4", "gguf"], default="nf4",
                   help="nf4 = 4-bit adapter via BitsAndBytes; gguf = llama.cpp Q4_K_M")
    p.add_argument("--label", default=None, help="Result file label")
    p.add_argument("--category", default=None, help="Run only this category")
    p.add_argument("--ids", default=None, help="Comma-separated scenario IDs")
    p.add_argument("--type", default=None,
                   help="Comma-separated types to include (single,scripted_multiturn,dynamic_multiturn)")
    p.add_argument("--tags", default=None,
                   help="Comma-separated tags — scenarios must contain all")
    p.add_argument("--temperature", type=float, default=PRODUCTION_SAMPLING["temperature"],
                   help=f"Eval model temperature (default: {PRODUCTION_SAMPLING['temperature']} — matches production app)")
    p.add_argument("--temperature-dynamic", type=float, default=None,
                   help="Eval model temperature for dynamic scenarios "
                        "(default: same as --temperature)")
    p.add_argument("--top-k", type=int, default=PRODUCTION_SAMPLING["top_k"],
                   help=f"top_k (default: {PRODUCTION_SAMPLING['top_k']} — matches production)")
    p.add_argument("--top-p", type=float, default=PRODUCTION_SAMPLING["top_p"],
                   help=f"top_p (default: {PRODUCTION_SAMPLING['top_p']} — matches production)")
    p.add_argument("--min-p", type=float, default=PRODUCTION_SAMPLING["min_p"],
                   help=f"min_p (default: {PRODUCTION_SAMPLING['min_p']} — matches production; "
                        "GGUF backend honors this, HF/NF4 backend ignores)")
    p.add_argument("--repetition-penalty", type=float,
                   default=PRODUCTION_SAMPLING["repetition_penalty"],
                   help=f"repetition_penalty (default: {PRODUCTION_SAMPLING['repetition_penalty']} "
                        "— matches production, which has it disabled)")
    p.add_argument("--max-new-tokens", type=int, default=PRODUCTION_SAMPLING["max_new_tokens"],
                   help=f"max_new_tokens (default: {PRODUCTION_SAMPLING['max_new_tokens']} — matches production)")
    p.add_argument("--sysprompt", default="full", choices=["full", "preamble_only", "none"],
                   help="System prompt mode (matches v3 ablation flags)")
    p.add_argument("--holdout", action="store_true",
                   help="Run holdout scenarios (hd_*) instead of regular scenarios. "
                        "Holdout scenarios have disjoint profiles from data-gen; "
                        "run at final release ranking only — not during routine evals.")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--log-level", default="INFO",
                   choices=["TRACE", "DEBUG", "INFO", "WARN", "ERROR"])
    p.add_argument("--output-dir", default=str(PROJECT_ROOT / "benchmarks" / "results"),
                   type=str)
    p.add_argument("--skip-load", action="store_true",
                   help="Skip model load (for plumbing tests with stub generators)")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def resolve_model(args) -> tuple[str, Optional[Path], str]:
    """Returns (base_model_id, adapter_path, label)."""
    if args.gguf_path:
        gguf = Path(args.gguf_path)
        if not gguf.is_absolute():
            gguf = PROJECT_ROOT / gguf
        if not gguf.exists():
            raise SystemExit(f"GGUF not found: {gguf}")
        label = args.label or gguf.stem
        return ("", gguf, label)
    if args.model:
        base_id, adapter_rel = MODEL_SHORTCUTS[args.model]
        adapter_path = PROJECT_ROOT / adapter_rel if adapter_rel else None
        label = args.label or args.model
        return (base_id, adapter_path, label)
    base_id = args.base
    adapter_path = PROJECT_ROOT / args.adapter
    label = args.label or Path(args.adapter).name
    return (base_id, adapter_path, label)


def load_nf4_eval(base_model_id: str, adapter_path: Optional[Path],
                  *, top_k: int, top_p: float,
                  repetition_penalty: float, min_p: float):
    """
    Returns (generate_fn, tokenizer) for the eval model in NF4.

    Sampling params are baked into the closure at load time. transformers
    does not natively support min_p, so we silently drop it here — note this
    in run output. (GGUF backend via llama-cpp-python honors min_p.)
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"[Load] Eval model (NF4): {base_model_id}")
    if min_p:
        print(f"  Note: min_p={min_p} ignored — transformers doesn't support it. "
              f"GGUF backend would honor it.")
    t0 = time.time()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base, str(adapter_path)) if adapter_path else base
    model.eval()
    device = next(model.parameters()).device
    print(f"  Ready in {time.time() - t0:.1f}s on {device}")

    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_ids = [tokenizer.eos_token_id, eot] if eot else [tokenizer.eos_token_id]

    def generate_fn(transcript: list[dict], *, temperature: float, max_new_tokens: int) -> str:
        prompt = tokenizer.apply_chat_template(
            transcript, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 1e-5),
                do_sample=temperature > 0,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return generate_fn, tokenizer


def load_gguf_eval(gguf_path: Path, *, top_k: int, top_p: float,
                   min_p: float, repetition_penalty: float):
    """
    Returns (generate_fn, None). llama-cpp-python supports all the production
    sampling params (including min_p) natively, so the GGUF backend produces
    the closest match to on-device behavior.
    """
    print(f"[Load] Eval model (GGUF): {gguf_path}")
    t0 = time.time()
    try:
        from llama_cpp import Llama
    except ImportError:
        raise SystemExit(
            "llama-cpp-python not installed. Install with: uv pip install llama-cpp-python"
        )

    llm = Llama(
        model_path=str(gguf_path),
        n_ctx=4096,
        n_gpu_layers=-1,   # all on GPU
        chat_format="llama-3",
        verbose=False,
        seed=-1,           # -1 = nondeterministic, matches production app
    )
    print(f"  Ready in {time.time() - t0:.1f}s")

    def generate_fn(transcript: list[dict], *, temperature: float, max_new_tokens: int) -> str:
        out = llm.create_chat_completion(
            messages=transcript,
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repeat_penalty=repetition_penalty,
        )
        return out["choices"][0]["message"]["content"].strip()

    return generate_fn, None


def load_judge():
    """Gemma4 26B at bfloat16 on GPU 0. Returns (generate_fn, model, tokenizer).

    Also used as the simulator (see build_sim_fn_from_judge) to avoid loading
    a second large model. Qwen3-30B doesn't fit on A100-80 alongside Gemma4 +
    eval model even in 4-bit (seen on jobs 618656-619022).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[Load] Judge model: {JUDGE_MODEL_ID}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_ID,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"  Ready in {time.time() - t0:.1f}s on {device}")

    def generate_fn(prompt: str, max_new_tokens: int, temperature: float) -> str:
        msgs = [{"role": "user", "content": prompt}]
        # apply_chat_template returns BatchEncoding (dict-like) in newer transformers,
        # not a raw tensor. Extract input_ids explicitly to avoid AttributeError when
        # model.generate() tries to access .shape on the BatchEncoding object.
        enc = tokenizer.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True
        )
        if hasattr(enc, "input_ids"):
            input_ids = enc.input_ids.to(device)
        else:
            input_ids = enc.to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)

    return generate_fn, model, tokenizer


def build_sim_fn_from_judge(model, tokenizer):
    """
    Build a sim_fn using the already-loaded Gemma4 judge model.
    Reusing the judge avoids loading Qwen3-30B (doesn't fit on A100-80 alongside
    Gemma4 + eval model). The judge evaluates Anchor's outputs, not the simulator's,
    so using Gemma4 for both roles creates no circular dependency.
    """
    import torch

    device = next(model.parameters()).device

    def sim_fn(persona: str, conversation: list[dict], *, seed: int = 0,
               temperature: float = 0.7, max_tokens: int = 80) -> str:
        # Flip roles: Anchor's replies become "user" turns; user turns become "assistant".
        # Prepend persona as a user instruction, then let Gemma4 continue as the user.
        flipped = []
        for turn in conversation:
            if turn["role"] == "user":
                flipped.append({"role": "assistant", "content": turn["content"]})
            elif turn["role"] == "assistant":
                flipped.append({"role": "user", "content": turn["content"]})

        if flipped:
            # Inject persona into the first user turn
            first_user = next((t for t in flipped if t["role"] == "user"), None)
            if first_user:
                first_user["content"] = (
                    f"[Persona: {persona}]\n\n{first_user['content']}"
                )
        else:
            # Conversation is empty — generate the opening user message
            flipped = [{"role": "user", "content": f"[Persona: {persona}]\n\nStart the conversation."}]

        enc = tokenizer.apply_chat_template(
            flipped, return_tensors="pt", add_generation_prompt=True
        )
        input_ids = enc.input_ids.to(device) if hasattr(enc, "input_ids") else enc.to(device)
        if seed:
            torch.manual_seed(seed)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0][input_ids.shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return sim_fn


def load_simulator():
    # No longer used — Qwen3-30B doesn't fit on A100-80 alongside Gemma4 + eval model.
    # Simulator is now built from the already-loaded judge (see build_sim_fn_from_judge).
    raise RuntimeError("load_simulator() is deprecated; use build_sim_fn_from_judge() instead")


# ──────────────────────────────────────────────────────────────────────────────
# System prompt transformer (matches v3 ablation flags)
# ──────────────────────────────────────────────────────────────────────────────

def transform_sysprompt(system: str, mode: str) -> str:
    if mode == "full":
        return system
    if mode == "none":
        return ""
    return APP_BASE_PROMPT  # preamble_only


# ──────────────────────────────────────────────────────────────────────────────
# Scenario execution
# ──────────────────────────────────────────────────────────────────────────────

def run_single(scenario: dict, *, generate_fn, log: EvalLogger,
               sysprompt_mode: str, temperature: float, max_new_tokens: int) -> dict:
    """Single-turn or scripted multi-turn. user_turns is fixed."""
    user_turns = scenario.get("user_turns", [])
    base_system = render_system_prompt(scenario.get("seed", {}))
    system_prompt = transform_sysprompt(base_system, sysprompt_mode)

    log.system_prompt(scenario_id=scenario["id"], content=system_prompt)

    transcript: list[dict] = []
    if system_prompt:
        transcript.append({"role": "system", "content": system_prompt})

    for idx, user_msg in enumerate(user_turns, start=1):
        transcript.append({"role": "user", "content": user_msg})
        log.user_turn(scenario_id=scenario["id"], turn_idx=idx, content=user_msg)

        t0 = time.monotonic()
        response = generate_fn(transcript, temperature=temperature,
                                max_new_tokens=max_new_tokens)
        elapsed_ms = (time.monotonic() - t0) * 1000

        transcript.append({"role": "assistant", "content": response})
        log.model_turn(
            scenario_id=scenario["id"], turn_idx=idx, content=response,
            ttft_ms=None, gen_tps=None, tokens_out=None,
        )

    return {"transcript": transcript, "perf_per_turn": []}


def _check_stop(stop_conditions: list[str], transcript: list[dict]) -> bool:
    """Cheap heuristic: detect natural endings in the simulator output."""
    assistant_turns = [t for t in transcript if t.get("role") == "user"]  # user-role here means simulator output
    if not assistant_turns:
        return False
    last = assistant_turns[-1]["content"].lower()
    # Convention: stop_conditions strings are documentation, not regex. We hard-
    # code the natural stops we expect.
    if any(p in last for p in ["thanks anchor", "i think i'm okay", "i think im okay"]):
        return True
    # Single-word reply twice in a row → user disengaged
    if len(assistant_turns) >= 2:
        prev = assistant_turns[-2]["content"].strip().lower()
        if len(last.split()) <= 1 and len(prev.split()) <= 1:
            return True
    return False


def run_dynamic(scenario: dict, *, generate_fn, sim_fn,
                log: EvalLogger, sysprompt_mode: str,
                temperature: float, max_new_tokens: int) -> dict:
    """Dynamic multi-turn — Qwen3 simulator generates user turns based on persona."""
    base_system = render_system_prompt(scenario.get("seed", {}))
    system_prompt = transform_sysprompt(base_system, sysprompt_mode)
    log.system_prompt(scenario_id=scenario["id"], content=system_prompt)

    transcript: list[dict] = []
    if system_prompt:
        transcript.append({"role": "system", "content": system_prompt})

    max_turns = scenario.get("max_turns", 5)
    sim_cfg = scenario.get("simulator", {})
    sim_seed = sim_cfg.get("seed", 0)
    sim_temp = sim_cfg.get("temperature", 0.7)
    sim_max = sim_cfg.get("max_tokens", 80)
    persona = scenario.get("user_persona", "")

    for turn_idx in range(1, max_turns + 1):
        # Simulator generates next user turn
        # Pass the conversation excluding the system message
        conv_for_sim = [t for t in transcript if t["role"] in ("user", "assistant")]
        t0 = time.monotonic()
        user_msg = sim_fn(persona=persona, conversation=conv_for_sim,
                          seed=sim_seed + turn_idx,
                          temperature=sim_temp, max_tokens=sim_max)
        sim_ms = int((time.monotonic() - t0) * 1000)
        if not user_msg.strip():
            log.warn(f"empty simulator turn at idx {turn_idx}, ending early",
                     scenario_id=scenario["id"])
            break

        transcript.append({"role": "user", "content": user_msg})
        log.simulator_turn(scenario_id=scenario["id"], turn_idx=turn_idx,
                           content=user_msg, ms=sim_ms)

        if _check_stop(scenario.get("stop_conditions", []), transcript):
            log.debug(f"stop condition met at turn {turn_idx}",
                      scenario_id=scenario["id"])
            break

        # Model under test responds
        t0 = time.monotonic()
        response = generate_fn(transcript, temperature=temperature,
                                max_new_tokens=max_new_tokens)
        elapsed_ms = (time.monotonic() - t0) * 1000
        transcript.append({"role": "assistant", "content": response})
        log.model_turn(scenario_id=scenario["id"], turn_idx=turn_idx, content=response)

    return {"transcript": transcript, "perf_per_turn": []}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    base_model_id, adapter_path, label = resolve_model(args)

    # Append sysprompt suffix
    if args.sysprompt != "full" and args.label is None:
        label = f"{label}_{args.sysprompt}"

    # Pick scenarios
    if args.holdout:
        all_scenarios = load_holdout_scenarios()
        label = f"{label}_holdout"
    else:
        all_scenarios = load_scenarios()
    type_filter = None
    if args.type:
        types = set(args.type.split(","))
        type_filter = types
    tag_filter = args.tags.split(",") if args.tags else None
    id_filter = args.ids.split(",") if args.ids else None

    scenarios = filter_scenarios(
        all_scenarios,
        category=args.category.upper() if args.category else None,
        ids=id_filter,
        tags=tag_filter,
    )
    if type_filter:
        scenarios = [s for s in scenarios if s.get("type", "single") in type_filter]

    if not scenarios:
        print("No scenarios matched filters.")
        sys.exit(1)

    n_single = sum(1 for s in scenarios if s.get("type", "single") in ("single", "scripted_multiturn"))
    n_dyn = sum(1 for s in scenarios if s.get("type") == "dynamic_multiturn")

    # ── Set up run directory + logger ─────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"cluster_{label}_{ts}"
    log = EvalLogger(run_dir, level=args.log_level)

    temperature_dynamic = args.temperature_dynamic
    if temperature_dynamic is None:
        temperature_dynamic = args.temperature

    log.info(f"Filtered to {len(scenarios)} scenarios "
             f"({n_single} single/scripted, {n_dyn} dynamic)")
    log.info(f"  model={label}  adapter={adapter_path}  backend={args.backend}  "
             f"sysprompt={args.sysprompt}")
    log.info(f"  sampling: temperature={args.temperature} (dynamic={temperature_dynamic})  "
             f"top_k={args.top_k}  top_p={args.top_p}  min_p={args.min_p}  "
             f"repetition_penalty={args.repetition_penalty}  "
             f"max_new_tokens={args.max_new_tokens}")

    if args.skip_load:
        log.warn("--skip-load: dummy generate_fn used (plumbing test only)")
        def generate_fn(transcript, *, temperature, max_new_tokens):
            return "(skip-load stub response)"
        def sim_fn(persona, conversation, *, seed=0, temperature=0.7, max_tokens=80):
            return "(skip-load simulator stub)"
        from benchmarks.judge_service import _stub_generate
        judge_fn = _stub_generate
        judge_tok = None
    else:
        # Load eval model — sampling params baked into closure at load time
        if args.backend == "gguf":
            assert adapter_path, "gguf backend requires --gguf-path"
            generate_fn, _ = load_gguf_eval(
                adapter_path,
                top_k=args.top_k, top_p=args.top_p, min_p=args.min_p,
                repetition_penalty=args.repetition_penalty,
            )
        else:
            generate_fn, _ = load_nf4_eval(
                base_model_id, adapter_path,
                top_k=args.top_k, top_p=args.top_p, min_p=args.min_p,
                repetition_penalty=args.repetition_penalty,
            )

        judge_fn, judge_model, judge_tok = load_judge()

        # Reuse Gemma4 as the simulator — Qwen3-30B doesn't fit on A100-80
        # alongside Gemma4 + eval model (tried bfloat16 and 4-bit NF4, both OOM).
        sim_fn = None
        if n_dyn > 0:
            print("[Load] Simulator: reusing Gemma4 judge model (no extra VRAM)")
            sim_fn = build_sim_fn_from_judge(judge_model, judge_tok)

    judge = JudgeService(generate_fn=judge_fn, logger=log, tokenizer=judge_tok)

    backend_label = "Q4_K_M" if args.backend == "gguf" else "NF4_adapter"
    log.run_start(model=label, scenarios=len(scenarios), backend=backend_label,
                  runtime="cluster", sysprompt=args.sysprompt,
                  temperature=args.temperature)

    judged_scenarios: list[dict] = []
    weighted_pass_total = 0.0
    weighted_total_total = 0.0
    passed_scenarios = 0
    failed_scenarios = 0

    for s in scenarios:
        scenario_type = s.get("type", "single")
        weight = s.get("weight", 1)
        log.scenario_start(scenario_id=s["id"], scenario_type=scenario_type,
                           category=s["category"], weight=weight,
                           seed=s.get("simulator", {}).get("seed"))

        try:
            # Dynamic scenarios can use a different temp if --temperature-dynamic
            # was set; otherwise both tiers use the same value.
            temp = (temperature_dynamic
                    if scenario_type == "dynamic_multiturn"
                    else args.temperature)

            if scenario_type == "dynamic_multiturn":
                if sim_fn is None:
                    log.warn(f"skipping dynamic scenario {s['id']} — no simulator loaded",
                             scenario_id=s["id"])
                    log.scenario_end(scenario_id=s["id"], turns=0,
                                     weighted_pass=0,
                                     weighted_total=sum(c.get("weight", 1)
                                                        for c in s["judge_criteria"]))
                    continue
                run_out = run_dynamic(
                    s, generate_fn=generate_fn, sim_fn=sim_fn,
                    log=log, sysprompt_mode=args.sysprompt,
                    temperature=temp, max_new_tokens=args.max_new_tokens,
                )
            else:
                run_out = run_single(
                    s, generate_fn=generate_fn, log=log,
                    sysprompt_mode=args.sysprompt,
                    temperature=temp, max_new_tokens=args.max_new_tokens,
                )

            transcript = run_out["transcript"]
            system_prompt = ""
            if transcript and transcript[0].get("role") == "system":
                system_prompt = transcript[0]["content"]
            else:
                system_prompt = render_system_prompt(s.get("seed", {}))

            # Judge
            results = judge.judge_scenario(
                scenario_id=s["id"],
                system_prompt=system_prompt,
                transcript=transcript,
                criteria=s["judge_criteria"],
            )
            score = score_scenario(results)
            weighted_pass_total += score["weighted_pass"]
            weighted_total_total += score["weighted_total"]

            log.transcript(scenario_id=s["id"], transcript=transcript,
                           perf_per_turn=run_out.get("perf_per_turn", []))

            # Failure artifacts
            for r in results:
                if r.verdict == "FAIL":
                    log.failure(
                        scenario_id=s["id"], criterion_id=r.criterion_id,
                        artifact={
                            "scenario_id": s["id"],
                            "criterion_id": r.criterion_id,
                            "category": s["category"],
                            "weight": r.weight,
                            "scope": r.scope,
                            "negative": r.negative,
                            "judge_question": r.question,
                            "judge_pass_if": r.pass_if,
                            "judge_verdict": r.verdict,
                            "judge_raw_response": r.raw_response,
                            "judge_second_response": r.second_response,
                            "judge_retry_count": r.retry_count,
                            "transcript": transcript,
                            "runtime": "cluster",
                            "model": label,
                            "backend": backend_label,
                        },
                    )

            judged_scenarios.append({
                "id": s["id"],
                "category": s["category"],
                "scenario_type": scenario_type,
                "weight": weight,
                "transcript": transcript,
                "perf_per_turn": run_out.get("perf_per_turn", []),
                "criteria_results": [r.to_dict() for r in results],
                "summary": score,
            })

            passed = abs(score["weighted_pass"] - score["weighted_total"]) < 1e-6
            if passed:
                passed_scenarios += 1
            else:
                failed_scenarios += 1

            log.scenario_end(
                scenario_id=s["id"],
                turns=sum(1 for t in transcript if t.get("role") == "assistant"),
                weighted_pass=score["weighted_pass"],
                weighted_total=score["weighted_total"],
            )
        except Exception as e:
            import traceback as _tb
            log.error(
                f"scenario {s['id']} threw: {type(e).__name__}: {e}\n{_tb.format_exc()}",
                scenario_id=s["id"],
            )
            failed_scenarios += 1
            # Continue with next scenario
            continue

    # ── Aggregate output ──────────────────────────────────────────────────────
    from collections import defaultdict
    by_cat: dict[str, dict] = defaultdict(lambda: {"weighted_pass": 0.0,
                                                    "weighted_total": 0.0, "n": 0})
    for js in judged_scenarios:
        cat = js["category"]
        by_cat[cat]["weighted_pass"] += js["summary"]["weighted_pass"]
        by_cat[cat]["weighted_total"] += js["summary"]["weighted_total"]
        by_cat[cat]["n"] += 1

    # CRISIS calibration — distress-level monotonicity + Spearman ρ
    crisis_calibration = None
    if "CRISIS" in by_cat:
        crisis_calibration = compute_crisis_calibration(judged_scenarios, all_scenarios)
        by_cat["CRISIS"]["calibration"] = crisis_calibration
        if crisis_calibration.get("warning"):
            log.warn(crisis_calibration["warning"])
        rho = crisis_calibration.get("spearman_rho")
        mono = crisis_calibration.get("monotonic")
        log.info(
            f"CRISIS calibration: monotonic={mono}  "
            f"spearman_rho={f'{rho:.3f}' if rho is not None else 'n/a'}  "
            f"per_level={crisis_calibration['per_level']}"
        )

    out_doc = {
        "schema_version": "v4.0",
        "runtime": "cluster",
        "model": label,
        "model_path": str(adapter_path) if adapter_path else None,
        "quantization": backend_label,
        "device": "cluster",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "judge_model": JUDGE_MODEL_ID,
        "judge_stats": judge.stats,
        "config": {
            "sysprompt": args.sysprompt,
            "sampling": {
                "temperature": args.temperature,
                "temperature_dynamic": temperature_dynamic,
                "top_k": args.top_k,
                "top_p": args.top_p,
                "min_p": args.min_p,
                "repetition_penalty": args.repetition_penalty,
                "max_new_tokens": args.max_new_tokens,
                "matches_production_app": (
                    args.temperature == PRODUCTION_SAMPLING["temperature"]
                    and args.top_k == PRODUCTION_SAMPLING["top_k"]
                    and args.top_p == PRODUCTION_SAMPLING["top_p"]
                    and args.min_p == PRODUCTION_SAMPLING["min_p"]
                    and args.repetition_penalty == PRODUCTION_SAMPLING["repetition_penalty"]
                    and args.max_new_tokens == PRODUCTION_SAMPLING["max_new_tokens"]
                ),
            },
        },
        "summary": {
            "weighted_pass": weighted_pass_total,
            "weighted_total": weighted_total_total,
            "weighted_pct": (weighted_pass_total / weighted_total_total
                             if weighted_total_total else 0),
            "passed_scenarios": passed_scenarios,
            "failed_scenarios": failed_scenarios,
            "by_category": dict(by_cat),
        },
        "scenarios": judged_scenarios,
    }

    if not args.no_save:
        out_path = run_dir / "judged.json"
        with out_path.open("w") as f:
            json.dump(out_doc, f, indent=2, ensure_ascii=False, default=str)
        log.info(f"Wrote judged output: {out_path}")

    log.run_end(passed_scenarios=passed_scenarios,
                failed_scenarios=failed_scenarios,
                weighted_pass=weighted_pass_total,
                weighted_total=weighted_total_total)
    log.close()


if __name__ == "__main__":
    main()
