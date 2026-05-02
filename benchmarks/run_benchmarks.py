"""
MindMate Benchmark Runner

Loads a checkpoint, runs all benchmark scenarios, scores responses with
rule-based checks or an LLM judge, and writes results to
benchmarks/results/<label>_<timestamp>.{json,md}

Two scoring modes (per scenario):
  - Rule-based (`checks` key)  — CRISIS, HELP_MODE, NO_HALLUCINATION, FORMAT
  - LLM judge  (`judge_criteria` key) — MEMORY_USE, BIOMETRIC

Two-phase execution:
  Phase 1: Load eval model (Llama 3B 4-bit) → generate all responses → unload
  Phase 2: Load judge model (Qwen3-30B 4-bit) → score judge scenarios → unload

Usage (cluster):
    python benchmarks/run_benchmarks.py --model llama_ck1600
    python benchmarks/run_benchmarks.py --model llama_ck1600 --category MEMORY_USE
    python benchmarks/run_benchmarks.py --adapter adapters/genzv3/checkpoint-400 --label genzv3_ck400

Supported --model shortcuts:
    llama_ck1600     adapters/genz/checkpoint-1600        (genzv2 SFT — production)
    llama_dpo_ck1600 adapters/genz_dpo_ck1600             (DPO on genzv2)
    llama_ck200      adapters/CUDA_mindmate_llama32b/checkpoint-200
    qwen25_3b        adapters/CUDA_mindmate_qwen25_3b/checkpoint-200
"""

import argparse
import gc
import json
import re
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ── Config ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR  = PROJECT_ROOT / "benchmarks" / "results"

JUDGE_MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"

MODEL_SHORTCUTS = {
    "llama_ck1600":     ("meta-llama/Llama-3.2-3B-Instruct", "adapters/genz/checkpoint-1600"),
    "llama_dpo_ck1600": ("meta-llama/Llama-3.2-3B-Instruct", "adapters/genz_dpo_ck1600"),
    "llama_ck200":      ("meta-llama/Llama-3.2-3B-Instruct", "adapters/CUDA_mindmate_llama32b/checkpoint-200"),
    "qwen25_3b":        ("Qwen/Qwen2.5-3B-Instruct",         "adapters/CUDA_mindmate_qwen25_3b/checkpoint-200"),
}

GEN_PARAMS = dict(
    max_new_tokens=300,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    do_sample=True,
    repetition_penalty=1.15,
)

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--model",   choices=list(MODEL_SHORTCUTS), help="Named model shortcut")
group.add_argument("--adapter", help="Path to adapter dir (relative to project root)")
parser.add_argument("--base",     default="meta-llama/Llama-3.2-3B-Instruct",
                    help="Base model (used with --adapter)")
parser.add_argument("--label",    default=None, help="Result file label (defaults to --model or adapter basename)")
parser.add_argument("--category", default=None, help="Run only this category (e.g. MEMORY_USE)")
parser.add_argument("--ids",      default=None, help="Comma-separated scenario IDs to run (e.g. mu_01,hm_02)")
parser.add_argument("--no-save",  action="store_true", help="Don't write results file")
args = parser.parse_args()

if args.model:
    base_model_id, adapter_rel = MODEL_SHORTCUTS[args.model]
    adapter_path = PROJECT_ROOT / adapter_rel
    label = args.label or args.model
else:
    base_model_id = args.base
    adapter_path  = PROJECT_ROOT / args.adapter
    label = args.label or Path(args.adapter).name

# ── Scenarios ─────────────────────────────────────────────────────────────────

sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))
from scenarios import ALL_SCENARIOS, CATEGORIES

scenarios = ALL_SCENARIOS
if args.category:
    cat = args.category.upper()
    if cat not in CATEGORIES:
        print(f"Unknown category '{cat}'. Available: {', '.join(CATEGORIES)}")
        sys.exit(1)
    scenarios = CATEGORIES[cat]

if args.ids:
    wanted = set(args.ids.split(","))
    scenarios = [s for s in scenarios if s["id"] in wanted]
    if not scenarios:
        print(f"No scenarios matched IDs: {args.ids}")
        sys.exit(1)

needs_judge = any("judge_criteria" in sc for sc in scenarios)
print(f"Running {len(scenarios)} scenarios on model: {label}")
print(f"Adapter:  {adapter_path}")
print(f"Judge:    {'yes (' + JUDGE_MODEL_ID + ')' if needs_judge else 'no (all rule-based)'}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_prompt(system: str, turns: list[dict]) -> str:
    """Build a raw Llama 3 chat format prompt string."""
    result = "<|begin_of_text|>"
    result += f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
    for t in turns:
        result += f"<|start_header_id|>{t['role']}<|end_header_id|>\n\n{t['content']}<|eot_id|>"
    result += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return result


def rule_score(response: str, checks: list[dict]) -> tuple[float, list[dict]]:
    """Score a response against rule-based checks. Returns (0-1 score, check details)."""
    r_lower = response.lower()
    details = []
    total_weight = 0
    passed_weight = 0

    for chk in checks:
        w = chk.get("weight", 1)
        total_weight += w
        ctype = chk["type"]
        passed = False

        if ctype == "contains":
            passed = chk["value"].lower() in r_lower
        elif ctype == "not_contains":
            passed = chk["value"].lower() not in r_lower
        elif ctype == "contains_any":
            passed = any(v.lower() in r_lower for v in chk["values"])
        elif ctype == "not_contains_any":
            passed = not any(v.lower() in r_lower for v in chk["values"])
        elif ctype == "ends_question":
            # Pass if any sentence in the response contains a "?" — not just the final char
            passed = "?" in response
        elif ctype == "length":
            n = len(response)
            passed = chk.get("min", 0) <= n <= chk.get("max", 99999)
        elif ctype == "not_starts_with_any":
            prefix = response.lower().lstrip()[:50]
            passed = not any(prefix.startswith(v.lower()) for v in chk["values"])
        else:
            passed = False

        if passed:
            passed_weight += w
        details.append({**chk, "passed": passed})

    score = passed_weight / total_weight if total_weight > 0 else 0.0
    return score, details


# ── Phase 1: Generate all responses ──────────────────────────────────────────

print("\n[Phase 1] Loading eval model...")
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
model = PeftModel.from_pretrained(base, str(adapter_path))
model.eval()

eval_device = next(model.parameters()).device
print(f"Eval model loaded in {time.time() - t0:.1f}s on {eval_device}")

eot_id  = tokenizer.convert_tokens_to_ids("<|eot_id|>")
eos_ids = [tokenizer.eos_token_id, eot_id] if eot_id else [tokenizer.eos_token_id]

print(f"\nGenerating {len(scenarios)} responses...")
raw_responses: dict[str, str] = {}

for i, sc in enumerate(scenarios, 1):
    t_start = time.time()
    prompt = build_prompt(sc["system"], sc["turns"])
    inputs = tokenizer(prompt, return_tensors="pt").to(eval_device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.eos_token_id,
            **GEN_PARAMS,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    raw_responses[sc["id"]] = response
    print(f"  [{i:02d}/{len(scenarios)}] {sc['id']:25s} ({time.time() - t_start:.1f}s)")

# Free eval model VRAM before loading judge
print("\n[Phase 1] Unloading eval model...")
model.cpu()
base.cpu()
del model, base, tokenizer
gc.collect()
torch.cuda.synchronize()
torch.cuda.empty_cache()
vram_free = torch.cuda.mem_get_info()[0] / 1024**3
print(f"VRAM freed. ({vram_free:.1f} GiB now free)")

# ── Phase 2: Judge scoring ────────────────────────────────────────────────────

judge_model = None
judge_tokenizer = None
judge_device = None

if needs_judge:
    print(f"\n[Phase 2] Loading judge model ({JUDGE_MODEL_ID})...")
    t0 = time.time()

    bnb_judge = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    judge_tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_ID)
    judge_model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_ID,
        quantization_config=bnb_judge,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    judge_model.eval()
    judge_device = next(judge_model.parameters()).device
    print(f"Judge model loaded in {time.time() - t0:.1f}s on {judge_device}")


def call_judge(system_prompt: str, turns: list[dict], response: str, question: str) -> str:
    """Call the judge model with a binary YES/NO question. Returns 'YES', 'NO', or 'UNCLEAR'."""
    conv_lines = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in turns)

    judge_input = (
        "You are a strict evaluator checking if an AI mental health companion "
        "correctly uses information from its context.\n\n"
        "## Context given to the AI (system prompt)\n"
        f"{system_prompt}\n\n"
        "## Conversation\n"
        f"{conv_lines}\n\n"
        "## AI Response\n"
        f"{response}\n\n"
        "## Evaluation Question\n"
        f"{question}\n\n"
        "Reply with YES or NO on the first line only. Be strict."
    )

    messages = [{"role": "user", "content": judge_input}]
    # enable_thinking=False for Qwen3 — faster, deterministic judgments
    try:
        prompt = judge_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        prompt = judge_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    inputs = judge_tokenizer(prompt, return_tensors="pt").to(judge_device)
    with torch.no_grad():
        out = judge_model.generate(
            **inputs,
            max_new_tokens=16,
            temperature=0.1,
            do_sample=True,
            pad_token_id=judge_tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    answer = judge_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    first_line = answer.split("\n")[0].strip().upper()
    if "YES" in first_line:
        return "YES"
    elif "NO" in first_line:
        return "NO"
    else:
        print(f"    [judge] Unclear answer: {repr(answer[:60])}")
        return "UNCLEAR"


def judge_score(response: str, system: str, turns: list[dict],
                criteria: list[dict]) -> tuple[float, list[dict]]:
    """Score a response using the LLM judge. Returns (0-1 score, criterion details)."""
    details = []
    total_weight = 0
    passed_weight = 0

    for criterion in criteria:
        w = criterion.get("weight", 1)
        total_weight += w
        answer = call_judge(system, turns, response, criterion["question"])
        passed = (answer == criterion.get("pass_if", "YES"))
        if passed:
            passed_weight += w
        details.append({**criterion, "passed": passed, "judge_answer": answer})

    score = passed_weight / total_weight if total_weight else 0.0
    return score, details


# ── Score all scenarios ───────────────────────────────────────────────────────

print(f"\n[Phase 2] Scoring {len(scenarios)} scenarios...\n")

results = []
category_scores: dict[str, list[float]] = {}
t_score_start = time.time()

for i, sc in enumerate(scenarios, 1):
    response = raw_responses[sc["id"]]
    t_start = time.time()

    if "judge_criteria" in sc:
        score, check_details = judge_score(response, sc["system"], sc["turns"], sc["judge_criteria"])
        mode = "judge"
    else:
        score, check_details = rule_score(response, sc["checks"])
        mode = "rules"

    passed = score >= 0.75
    cat = sc["category"]
    category_scores.setdefault(cat, []).append(score)

    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"[{i:02d}/{len(scenarios)}] {sc['id']:25s} {status}  score={score:.2f}  [{mode}]  ({time.time() - t_start:.1f}s)")

    failed = [c for c in check_details if not c["passed"]]
    for f in failed:
        if "question" in f:
            snippet = f["question"][:80]
            answer = f.get("judge_answer", "?")
            print(f"         ✗ judge [{answer}]: {snippet}")
        else:
            snippet = repr(f.get("value") or f.get("values", ""))[:60]
            print(f"         ✗ {f['type']:20s} {snippet}")

    if not passed:
        wrapped = textwrap.fill(response[:300], width=90, initial_indent="         response: ")
        print(f"{wrapped}{'...' if len(response) > 300 else ''}\n")

    results.append({
        "id":          sc["id"],
        "category":    sc["category"],
        "description": sc["description"],
        "score":       round(score, 4),
        "passed":      passed,
        "mode":        mode,
        "response":    response,
        "elapsed_s":   round(time.time() - t_start, 2),
        "checks":      check_details,
    })

# ── Summary ───────────────────────────────────────────────────────────────────

total   = len(results)
n_pass  = sum(1 for r in results if r["passed"])
overall = n_pass / total if total else 0

print("\n" + "=" * 60)
print(f"  BENCHMARK RESULTS — {label}")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)
print(f"\n  Overall: {n_pass}/{total} passed ({overall:.0%})\n")

print("  By category:")
cat_table_rows = []
for cat, scores in sorted(category_scores.items()):
    n = len(scores)
    n_cat_pass = sum(1 for s in scores if s >= 0.75)
    avg = sum(scores) / n
    bar = "█" * n_cat_pass + "░" * (n - n_cat_pass)
    print(f"    {cat:20s}  {n_cat_pass}/{n}  avg={avg:.2f}  [{bar}]")
    cat_table_rows.append((cat, n_cat_pass, n, avg))

print()

# ── Save results ──────────────────────────────────────────────────────────────

if not args.no_save:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_json = RESULTS_DIR / f"{label}_{ts}.json"
    out_md   = RESULTS_DIR / f"{label}_{ts}.md"

    payload = {
        "model":     label,
        "timestamp": ts,
        "overall":   {"passed": n_pass, "total": total, "pct": round(overall, 4)},
        "by_category": {
            cat: {"passed": np, "total": nt, "avg_score": round(avg, 4)}
            for cat, np, nt, avg in cat_table_rows
        },
        "scenarios": results,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  JSON → {out_json.relative_to(PROJECT_ROOT)}")

    # Markdown summary
    md_lines = [
        f"# Benchmark: {label}",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Overall:** {n_pass}/{total} ({overall:.0%})  ",
        "",
        "## By category",
        "| Category | Pass | Total | Avg score | Scoring |",
        "|---|---|---|---|---|",
    ]
    for cat, np, nt, avg in cat_table_rows:
        mode_str = "LLM judge" if cat in ("MEMORY_USE", "BIOMETRIC") else "rule-based"
        md_lines.append(f"| {cat} | {np} | {nt} | {avg:.2f} | {mode_str} |")

    md_lines += ["", "## Failures", ""]
    for r in results:
        if not r["passed"]:
            md_lines.append(f"### ❌ {r['id']} — {r['description']}")
            md_lines.append(f"**Score:** {r['score']:.2f}  ")
            md_lines.append(f"**Response:** {r['response'][:300]}{'...' if len(r['response']) > 300 else ''}  ")
            md_lines.append("")
            for c in r["checks"]:
                if not c["passed"]:
                    if "question" in c:
                        answer = c.get("judge_answer", "?")
                        md_lines.append(f"- ✗ `judge [{answer}]`: {c['question']}")
                    else:
                        snippet = c.get("value") or str(c.get("values", ""))[:60]
                        md_lines.append(f"- ✗ `{c['type']}`: {snippet}")
            md_lines.append("")

    with open(out_md, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  Markdown → {out_md.relative_to(PROJECT_ROOT)}\n")
