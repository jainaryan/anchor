"""
MindMate Benchmark Runner

Loads a checkpoint, runs all benchmark scenarios, scores responses with
rule-based checks, and writes results to benchmarks/results/<model>.json
plus a Markdown summary.

Usage (cluster):
    python benchmarks/run_benchmarks.py --model llama_ck1600
    python benchmarks/run_benchmarks.py --model llama_ck1600 --category MEMORY_USE
    python benchmarks/run_benchmarks.py --adapter adapters/genz_dpo_ck1600 --label genz_dpo_ck1600

Supported --model shortcuts:
    llama_ck1600     adapters/genz/checkpoint-1600        (genzv2 SFT — production)
    llama_dpo_ck1600 adapters/genz_dpo_ck1600             (DPO on genzv2)
    llama_ck200      adapters/CUDA_mindmate_llama32b/checkpoint-200
    qwen25_3b        adapters/CUDA_mindmate_qwen25_3b/checkpoint-200
"""

import argparse
import json
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

print(f"Running {len(scenarios)} scenarios on model: {label}")
print(f"Adapter: {adapter_path}")

# ── Model loading ─────────────────────────────────────────────────────────────

print("\n[1/2] Loading model...")
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
)
model = PeftModel.from_pretrained(base, str(adapter_path))
model.eval()

device = next(model.parameters()).device
print(f"Model loaded in {time.time() - t0:.1f}s on {device}")

# ── Inference ─────────────────────────────────────────────────────────────────

eot_id  = tokenizer.convert_tokens_to_ids("<|eot_id|>")
eos_ids = [tokenizer.eos_token_id, eot_id] if eot_id else [tokenizer.eos_token_id]


def build_prompt(system: str, turns: list[dict]) -> str:
    result = "<|begin_of_text|>"
    result += f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
    for t in turns:
        result += f"<|start_header_id|>{t['role']}<|end_header_id|>\n\n{t['content']}<|eot_id|>"
    result += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return result


def generate(system: str, turns: list[dict]) -> str:
    prompt = build_prompt(system, turns)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.eos_token_id,
            **GEN_PARAMS,
        )
    # Decode only the new tokens
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# ── Scoring ───────────────────────────────────────────────────────────────────


def score_response(response: str, checks: list[dict]) -> tuple[float, list[dict]]:
    """Returns (0–1 score, list of check results)."""
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
            passed = response.rstrip().endswith("?")
        elif ctype == "length":
            n = len(response)
            passed = chk.get("min", 0) <= n <= chk.get("max", 99999)
        elif ctype == "not_starts_with_any":
            prefix = response.lower().lstrip()[:50]
            passed = not any(prefix.startswith(v.lower()) for v in chk["values"])
        else:
            passed = False  # unknown check type

        if passed:
            passed_weight += w
        details.append({**chk, "passed": passed})

    score = passed_weight / total_weight if total_weight > 0 else 0.0
    return score, details

# ── Run ───────────────────────────────────────────────────────────────────────

print(f"\n[2/2] Running {len(scenarios)} scenarios...\n")

results = []
category_scores: dict[str, list[float]] = {}

for i, sc in enumerate(scenarios, 1):
    t_start = time.time()
    response = generate(sc["system"], sc["turns"])
    elapsed  = time.time() - t_start

    score, check_details = score_response(response, sc["checks"])
    passed = score >= 0.75  # scenario passes if ≥75% of weighted checks pass

    cat = sc["category"]
    category_scores.setdefault(cat, []).append(score)

    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"[{i:02d}/{len(scenarios)}] {sc['id']:25s} {status}  score={score:.2f}  ({elapsed:.1f}s)")

    # Show which checks failed
    failed = [c for c in check_details if not c["passed"]]
    for f in failed:
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
        "response":    response,
        "elapsed_s":   round(elapsed, 2),
        "checks":      check_details,
    })

# ── Summary ───────────────────────────────────────────────────────────────────

total   = len(results)
passed  = sum(1 for r in results if r["passed"])
overall = passed / total if total else 0

print("\n" + "=" * 60)
print(f"  BENCHMARK RESULTS — {label}")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 60)
print(f"\n  Overall: {passed}/{total} passed ({overall:.0%})\n")

print("  By category:")
cat_table_rows = []
for cat, scores in sorted(category_scores.items()):
    n = len(scores)
    n_pass = sum(1 for s in scores if s >= 0.75)
    avg    = sum(scores) / n
    bar    = "█" * n_pass + "░" * (n - n_pass)
    print(f"    {cat:20s}  {n_pass}/{n}  avg={avg:.2f}  [{bar}]")
    cat_table_rows.append((cat, n_pass, n, avg))

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
        "overall":   {"passed": passed, "total": total, "pct": round(overall, 4)},
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
        f"**Overall:** {passed}/{total} ({overall:.0%})  ",
        "",
        "## By category",
        "| Category | Pass | Total | Avg score |",
        "|---|---|---|---|",
    ]
    for cat, np, nt, avg in cat_table_rows:
        md_lines.append(f"| {cat} | {np} | {nt} | {avg:.2f} |")

    md_lines += [
        "",
        "## Failures",
        "",
    ]
    for r in results:
        if not r["passed"]:
            md_lines.append(f"### ❌ {r['id']} — {r['description']}")
            md_lines.append(f"**Score:** {r['score']:.2f}  ")
            md_lines.append(f"**Response:** {r['response'][:300]}{'...' if len(r['response']) > 300 else ''}  ")
            md_lines.append("")
            for c in r["checks"]:
                if not c["passed"]:
                    snippet = c.get("value") or str(c.get("values", ""))[:60]
                    md_lines.append(f"- ✗ `{c['type']}`: {snippet}")
            md_lines.append("")

    with open(out_md, "w") as f:
        f.write("\n".join(md_lines))
    print(f"  Markdown → {out_md.relative_to(PROJECT_ROOT)}\n")
