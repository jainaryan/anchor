"""
Judge mobile-generated transcripts.

The mobile app generates raw transcripts on-device (without judging) and
writes them as JSON files. This script:

  1. Reads a raw mobile results file (transcripts only, no verdicts)
  2. Loads Gemma4 on the cluster GPU
  3. For each scenario: runs the judge against the saved transcript
  4. Writes the judged results JSON + log artifacts

Input format (from mobile EvalRunner — see src/eval/types.ts):
{
  "runtime": "mobile",
  "model": "genzv2_ck1200",
  "model_path": "exports/mindmate_genzv2_ck1200_q4_k_m.gguf",
  "quantization": "Q4_K_M",
  "device": "Pixel 8a",
  "timestamp": "2026-05-15T10:00:00Z",
  "scenarios": [
    {
      "id": "cm_01",
      "transcript": [{"role":"system","content":"..."}, ...],
      "perf_per_turn": [{...}, ...]
    },
    ...
  ]
}

Output format: same shape + per-criterion `criteria_results` + `summary`.

Usage:
    sbatch --gres=gpu:a100-80:1 \\
        --export=ALL,MOBILE_RESULTS=results_pending/mobile_ck1200_20260515.json \\
        benchmarks/judge_mobile.slurm

Or directly:
    python -m benchmarks.judge_mobile_results --input <raw_json> --output-dir results/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from benchmarks.logging import EvalLogger
from benchmarks.judge_service import JudgeService, score_scenario
from benchmarks.scenarios_loader import load_scenarios, render_system_prompt


# ──────────────────────────────────────────────────────────────────────────────
# Gemma4 loader — copied/adapted from run_benchmarks.py
# ──────────────────────────────────────────────────────────────────────────────

JUDGE_MODEL_ID = "google/gemma-4-26B-A4B-it"


def load_gemma4_judge():
    """
    Load Gemma 4 26B A4B IT at bfloat16. Returns (generate_fn, tokenizer).
    Requires ~52GB VRAM. Use --gres=gpu:a100-80:1.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[Judge] Loading {JUDGE_MODEL_ID} …")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        JUDGE_MODEL_ID,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[Judge] Loaded in {time.time() - t0:.1f}s")

    def generate_fn(prompt: str, max_new_tokens: int, temperature: float) -> str:
        msgs = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            msgs, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(
            out[0][inputs.shape[-1]:], skip_special_tokens=True,
        )
        return decoded

    return generate_fn, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def judge_mobile_file(
    *,
    input_path: Path,
    output_dir: Path,
    log_level: str = "INFO",
    skip_judge_load: bool = False,
    generate_fn_override=None,
    tokenizer_override=None,
) -> Path:
    """
    Judge a raw mobile results file. Returns path to the judged output JSON.

    `skip_judge_load=True` + `generate_fn_override` lets tests skip the heavy
    Gemma4 load and inject a stub.
    """
    # 1. Read raw input
    with input_path.open("r", encoding="utf-8") as f:
        raw_doc = json.load(f)

    runtime = raw_doc.get("runtime", "unknown")
    model = raw_doc.get("model", "unknown")
    timestamp = raw_doc.get("timestamp", datetime.now(timezone.utc).isoformat())

    # 2. Set up output directory + logger
    judge_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{runtime}_{model}_judged_{judge_ts}"

    with EvalLogger(run_dir, level=log_level) as log:
        log.info(f"Judging mobile results from {input_path}")
        log.info(f"  runtime={runtime}  model={model}  timestamp={timestamp}")
        log.info(f"  scenarios={len(raw_doc.get('scenarios', []))}")

        # 3. Load scenarios.json for criteria
        all_scenarios = {s["id"]: s for s in load_scenarios()}

        # 4. Load Gemma4
        if skip_judge_load:
            if generate_fn_override is None:
                raise ValueError("skip_judge_load requires generate_fn_override")
            generate_fn = generate_fn_override
            tokenizer = tokenizer_override
        else:
            generate_fn, tokenizer = load_gemma4_judge()

        judge = JudgeService(generate_fn=generate_fn, logger=log, tokenizer=tokenizer)

        log.run_start(model=model, scenarios=len(raw_doc["scenarios"]),
                      backend=raw_doc.get("quantization", "unknown"),
                      runtime=runtime)

        # 5. Judge each scenario
        judged_scenarios: list[dict] = []
        weighted_pass_total = 0.0
        weighted_total_total = 0.0

        for raw_scen in raw_doc["scenarios"]:
            sid = raw_scen["id"]
            scenario_def = all_scenarios.get(sid)
            if not scenario_def:
                log.warn(f"unknown scenario id: {sid} (not in scenarios.json)",
                         scenario_id=sid)
                continue

            log.scenario_start(
                scenario_id=sid,
                scenario_type=scenario_def.get("type", "single"),
                category=scenario_def["category"],
                weight=scenario_def.get("weight", 1),
            )

            transcript = raw_scen.get("transcript", [])
            system_prompt = ""
            # Extract system prompt from transcript first message
            if transcript and transcript[0].get("role") == "system":
                system_prompt = transcript[0]["content"]
            else:
                # Re-render from seed if not embedded
                system_prompt = render_system_prompt(scenario_def.get("seed", {}))

            results = judge.judge_scenario(
                scenario_id=sid,
                system_prompt=system_prompt,
                transcript=transcript,
                criteria=scenario_def["judge_criteria"],
            )

            score = score_scenario(results)
            weighted_pass_total += score["weighted_pass"]
            weighted_total_total += score["weighted_total"]

            # Save transcript + write failure artifacts
            log.transcript(
                scenario_id=sid,
                transcript=transcript,
                perf_per_turn=raw_scen.get("perf_per_turn", []),
            )

            for r in results:
                if r.verdict == "FAIL":
                    log.failure(
                        scenario_id=sid,
                        criterion_id=r.criterion_id,
                        artifact={
                            "scenario_id": sid,
                            "criterion_id": r.criterion_id,
                            "category": scenario_def["category"],
                            "weight": r.weight,
                            "negative": r.negative,
                            "scope": r.scope,
                            "judge_question": r.question,
                            "judge_pass_if": r.pass_if,
                            "judge_verdict": r.verdict,
                            "judge_raw_response": r.raw_response,
                            "judge_second_response": r.second_response,
                            "judge_retry_count": r.retry_count,
                            "transcript": transcript,
                            "perf_per_turn": raw_scen.get("perf_per_turn", []),
                            "runtime": runtime,
                            "model": model,
                            "quantization": raw_doc.get("quantization", "unknown"),
                            "device": raw_doc.get("device", "unknown"),
                        },
                    )

            judged_scenarios.append({
                **raw_scen,
                "criteria_results": [r.to_dict() for r in results],
                "summary": score,
            })

            log.scenario_end(
                scenario_id=sid,
                turns=sum(1 for m in transcript if m.get("role") == "assistant"),
                weighted_pass=score["weighted_pass"],
                weighted_total=score["weighted_total"],
            )

        # 6. Compose output doc
        passed_scenarios = sum(
            1 for s in judged_scenarios
            if s["summary"]["weighted_pass"] == s["summary"]["weighted_total"]
        )
        failed_scenarios = len(judged_scenarios) - passed_scenarios

        # By-category breakdown
        from collections import defaultdict
        by_cat: dict[str, dict] = defaultdict(lambda: {"weighted_pass": 0.0, "weighted_total": 0.0, "n": 0})
        for s in judged_scenarios:
            cat = all_scenarios[s["id"]]["category"]
            by_cat[cat]["weighted_pass"] += s["summary"]["weighted_pass"]
            by_cat[cat]["weighted_total"] += s["summary"]["weighted_total"]
            by_cat[cat]["n"] += 1

        out_doc = {
            **{k: v for k, v in raw_doc.items() if k != "scenarios"},
            "judged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "judge_model": JUDGE_MODEL_ID,
            "judge_stats": judge.stats,
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

        out_path = run_dir / "judged.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out_doc, f, indent=2, ensure_ascii=False, default=str)

        log.run_end(
            passed_scenarios=passed_scenarios,
            failed_scenarios=failed_scenarios,
            weighted_pass=weighted_pass_total,
            weighted_total=weighted_total_total,
        )

    return out_path


def main():
    p = argparse.ArgumentParser(description="Judge mobile-generated transcripts on the cluster.")
    p.add_argument("--input", required=True, type=Path,
                   help="Path to raw mobile results JSON (transcripts only)")
    p.add_argument("--output-dir", default=Path("benchmarks/results"), type=Path,
                   help="Where to write judged output (default: benchmarks/results)")
    p.add_argument("--log-level", default="INFO",
                   choices=["TRACE", "DEBUG", "INFO", "WARN", "ERROR"])
    args = p.parse_args()

    if not args.input.exists():
        print(f"ERROR: input file does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = judge_mobile_file(
        input_path=args.input,
        output_dir=args.output_dir,
        log_level=args.log_level,
    )
    print(f"\n✓ Wrote judged results to: {out}")


if __name__ == "__main__":
    main()
