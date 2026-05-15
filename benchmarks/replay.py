"""
Replay a benchmark run — re-judge saved transcripts without re-running
inference. Useful when:

  • Judge criteria changed and you want updated verdicts on old transcripts
    (avoids re-running the eval model — ~10x cheaper)
  • You want to compare two judge models against the same transcripts
    (e.g. Gemma4 vs Qwen3 as judge)
  • You want to dig into a single failure — open the transcript in $PAGER
    and re-run the judge interactively

Usage:
    python -m benchmarks.replay <run_dir> --judge-only
    python -m benchmarks.replay <run_dir> --judge-only --judge-model Qwen/Qwen3-30B-A3B-Instruct-2507
    python -m benchmarks.replay <run_dir>/failures/cm_01__crit_01.json --interactive
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from benchmarks.judge_service import JudgeService, score_scenario
from benchmarks.logging import EvalLogger
from benchmarks.scenarios_loader import load_scenarios


def _load_transcript_dir(run_dir: Path) -> dict[str, dict]:
    """Return {scenario_id: transcript_doc} from <run_dir>/transcripts/."""
    out: dict[str, dict] = {}
    tdir = run_dir / "transcripts"
    if not tdir.exists():
        # v3 result files don't have separate transcripts/; they're inline
        # in the JSON's "scenarios" array. Pull them from the summary file.
        summary = None
        for cand in (run_dir / "judged.json", run_dir / "summary.json"):
            if cand.exists():
                summary = json.load(cand.open())
                break
        if summary:
            for s in summary.get("scenarios", []):
                sid = s.get("id") or s.get("scenario_id")
                if sid:
                    out[sid] = {
                        "scenario_id": sid,
                        "transcript": s.get("transcript", []),
                        "perf_per_turn": s.get("perf_per_turn", []),
                    }
        return out

    for p in sorted(tdir.glob("*.json")):
        try:
            with p.open("r", encoding="utf-8") as f:
                doc = json.load(f)
            sid = doc.get("scenario_id") or p.stem
            out[sid] = doc
        except json.JSONDecodeError:
            continue
    return out


def _load_judge_generator(judge_model: str, stub: bool = False):
    """
    Returns (generate_fn, tokenizer). For real Gemma4 use, the import is heavy
    so we only do it on demand. `--stub` returns the test stub for fast checks.
    """
    if stub:
        from benchmarks.judge_service import _stub_generate
        return _stub_generate, None

    print(f"[replay] Loading judge model: {judge_model}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(judge_model)
    model = AutoModelForCausalLM.from_pretrained(
        judge_model,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()

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
        return tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)

    return generate_fn, tokenizer


def replay_run(
    run_dir: Path,
    *,
    judge_model: str = "google/gemma-4-26B-A4B-it",
    log_level: str = "INFO",
    stub: bool = False,
    out_subdir: str = "replay",
) -> Path:
    """Re-judge every transcript in a run; write to <run_dir>/<out_subdir>/."""
    transcripts = _load_transcript_dir(run_dir)
    if not transcripts:
        raise SystemExit(f"No transcripts found in {run_dir}/transcripts/")

    all_scenarios = {s["id"]: s for s in load_scenarios()}

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = run_dir / f"{out_subdir}_{ts}"
    log = EvalLogger(out_dir, level=log_level)

    generate_fn, tokenizer = _load_judge_generator(judge_model, stub=stub)
    judge = JudgeService(generate_fn=generate_fn, logger=log, tokenizer=tokenizer)

    log.run_start(model=f"replay({judge_model})",
                  scenarios=len(transcripts),
                  backend="replay")

    results_out = []
    for sid in sorted(transcripts):
        scenario_def = all_scenarios.get(sid)
        if not scenario_def:
            log.warn(f"unknown scenario id: {sid}", scenario_id=sid)
            continue
        tdoc = transcripts[sid]
        log.scenario_start(
            scenario_id=sid,
            scenario_type=scenario_def.get("type", "single"),
            category=scenario_def["category"],
            weight=scenario_def.get("weight", 1),
        )
        transcript = tdoc.get("transcript", [])
        system_prompt = ""
        if transcript and transcript[0].get("role") == "system":
            system_prompt = transcript[0]["content"]

        results = judge.judge_scenario(
            scenario_id=sid,
            system_prompt=system_prompt,
            transcript=transcript,
            criteria=scenario_def["judge_criteria"],
        )
        score = score_scenario(results)

        log.scenario_end(
            scenario_id=sid,
            turns=sum(1 for m in transcript if m.get("role") == "assistant"),
            weighted_pass=score["weighted_pass"],
            weighted_total=score["weighted_total"],
        )
        results_out.append({
            "id": sid,
            "criteria_results": [r.to_dict() for r in results],
            "summary": score,
        })

    log.run_end(
        passed_scenarios=sum(1 for r in results_out
                             if r["summary"]["weighted_pass"] == r["summary"]["weighted_total"]),
        failed_scenarios=sum(1 for r in results_out
                             if r["summary"]["weighted_pass"] != r["summary"]["weighted_total"]),
    )

    out_path = out_dir / "judged.json"
    with out_path.open("w") as f:
        json.dump({
            "schema_version": "v4.0",
            "judge_model": judge_model,
            "stub": stub,
            "original_run_dir": str(run_dir),
            "judged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "scenarios": results_out,
        }, f, indent=2, default=str)

    log.close()
    return out_path


def replay_failure(failure_path: Path, *, judge_model: str, stub: bool, interactive: bool):
    """Re-run a single criterion against its saved transcript."""
    with failure_path.open("r", encoding="utf-8") as f:
        art = json.load(f)

    sid = art["scenario_id"]
    transcript = art["transcript"]
    criterion = {
        "id": art["criterion_id"],
        "question": art["judge_question"],
        "pass_if": art.get("judge_pass_if", "YES"),
        "weight": art.get("weight", 1),
        "scope": art.get("scope", "any_turn"),
        "negative": art.get("negative", False),
    }

    if interactive:
        # Render the transcript in $PAGER first
        text = []
        text.append(f"Scenario: {sid}")
        text.append(f"Criterion: {criterion['id']} (w={criterion['weight']}, scope={criterion['scope']})")
        text.append(f"Question: {criterion['question']}")
        text.append("")
        for m in transcript:
            role = m.get("role", "?").upper()
            text.append(f"--- {role} ---")
            text.append(m.get("content", ""))
            text.append("")
        text.append(f"Previous verdict: {art.get('judge_verdict')}")
        text.append(f"Previous judge raw: {art.get('judge_raw_response', '')[:200]}")
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
            tf.write("\n".join(text))
            tmp_path = tf.name
        pager = os.environ.get("PAGER", "less")
        subprocess.call([pager, tmp_path])
        os.unlink(tmp_path)

    generate_fn, tokenizer = _load_judge_generator(judge_model, stub=stub)
    judge = JudgeService(generate_fn=generate_fn, tokenizer=tokenizer)

    # Reconstruct system prompt from transcript
    sys_prompt = ""
    if transcript and transcript[0].get("role") == "system":
        sys_prompt = transcript[0]["content"]

    result = judge.judge_criterion(
        scenario_id=sid,
        system_prompt=sys_prompt,
        transcript=transcript,
        criterion=criterion,
    )
    print(f"\nVerdict: {result.verdict}")
    print(f"Raw:\n  {result.raw_response}")
    if result.second_response:
        print(f"Retry raw:\n  {result.second_response}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("target", type=Path,
                   help="Run directory (with transcripts/) OR a single failure JSON")
    p.add_argument("--judge-only", action="store_true",
                   help="Replay every transcript in the run dir (no inference re-run)")
    p.add_argument("--interactive", action="store_true",
                   help="(failure target) open transcript in $PAGER before judging")
    p.add_argument("--judge-model", default="google/gemma-4-26B-A4B-it")
    p.add_argument("--stub", action="store_true",
                   help="Use stub judge (for plumbing tests, no GPU needed)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    if not args.target.exists():
        print(f"No such path: {args.target}", file=sys.stderr)
        sys.exit(1)

    if args.target.is_dir():
        if not args.judge_only:
            print("Run-directory replay requires --judge-only "
                  "(re-running inference is not supported here yet).",
                  file=sys.stderr)
            sys.exit(1)
        out = replay_run(args.target, judge_model=args.judge_model,
                         log_level=args.log_level, stub=args.stub)
        print(f"\n✓ Wrote {out}")
    else:
        replay_failure(args.target,
                       judge_model=args.judge_model,
                       stub=args.stub,
                       interactive=args.interactive)


if __name__ == "__main__":
    main()
