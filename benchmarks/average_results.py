#!/usr/bin/env python3
"""
Average benchmark results across multiple runs of the same model.

Usage:
  python benchmarks/average_results.py                    # averages all models with ≥2 runs
  python benchmarks/average_results.py --model genzv4_ck200
  python benchmarks/average_results.py --since 20260509   # only runs on/after this date
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_results(since: str = None, model_filter: str = None):
    """Load all JSON result files, grouped by model label."""
    groups = defaultdict(list)
    for f in sorted(RESULTS_DIR.glob("*.json")):
        name = f.stem  # e.g. genzv4_ck200_20260509_0205
        # Extract model label (strip trailing _YYYYMMDD_HHMM or _YYYYMMDD_HHMM_jobid)
        m = re.match(r"^(.+?)_(\d{8})_(\d{4})(?:_[a-zA-Z0-9]+)?$", name)
        if not m:
            continue
        label, date, time_ = m.group(1), m.group(2), m.group(3)

        if model_filter and label != model_filter:
            continue
        if since and date < since:
            continue

        try:
            data = json.loads(f.read_text())
        except Exception:
            continue

        groups[label].append({"file": f.name, "date": date, "time": time_, "data": data})

    return groups


def summarise(groups, min_runs: int = 2):
    rows = []

    for label, runs in sorted(groups.items()):
        if len(runs) < min_runs:
            continue

        # Per-run weighted scores
        weighted_pass = []
        weighted_total = []
        per_scenario = defaultdict(list)  # scenario_id → list of passed bools

        for run in runs:
            d = run["data"]
            # Support both flat and nested (overall.weighted_pass) result formats
            if "weighted_pass" in d:
                weighted_pass.append(d["weighted_pass"])
                weighted_total.append(d["weighted_total"])
            else:
                weighted_pass.append(d["overall"]["weighted_pass"])
                weighted_total.append(d["overall"]["weighted_total"])
            for s in d.get("scenarios", []):
                per_scenario[s["id"]].append(s["passed"])

        n = len(runs)
        avg_pass = sum(weighted_pass) / n
        total = weighted_total[0]  # same across runs
        pct = avg_pass / total * 100

        # Variance: std dev of weighted pass counts
        mean = avg_pass
        std = (sum((x - mean) ** 2 for x in weighted_pass) / n) ** 0.5

        # Scenario stability: which scenarios flip most
        unstable = []
        for sid, results in sorted(per_scenario.items()):
            if len(results) == n:
                pass_rate = sum(results) / n
                if 0 < pass_rate < 1:  # not always pass or always fail
                    unstable.append((sid, pass_rate))
        unstable.sort(key=lambda x: abs(x[1] - 0.5))  # closest to 50/50 = most unstable

        rows.append({
            "label": label,
            "n_runs": n,
            "avg_pass": avg_pass,
            "total": total,
            "pct": pct,
            "std": std,
            "runs": [f"{p}/{t} ({p/t*100:.0f}%)" for p, t in zip(weighted_pass, weighted_total)],
            "unstable": unstable[:5],
        })

    # Sort by avg score descending
    rows.sort(key=lambda r: r["pct"], reverse=True)
    return rows


def print_report(rows):
    print("\n" + "=" * 70)
    print("  AVERAGED BENCHMARK LEADERBOARD")
    print("=" * 70)
    print(f"{'Rank':<5} {'Model':<22} {'Avg score':>12}  {'±':>6}  {'Runs':<8}  Individual runs")
    print("-" * 70)
    for i, r in enumerate(rows, 1):
        runs_str = "  ".join(r["runs"])
        print(f"  {i:<4} {r['label']:<22} {r['avg_pass']:.1f}/{r['total']} ({r['pct']:.0f}%)  ±{r['std']:.1f}  (n={r['n_runs']})  {runs_str}")

    print()
    print("Most variable scenarios (flip between runs):")
    seen = set()
    for r in rows:
        for sid, rate in r["unstable"]:
            if sid not in seen:
                seen.add(sid)
                print(f"  {sid:<10} pass rate {rate*100:.0f}% across runs")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Filter to one model label")
    parser.add_argument("--since", help="Only include runs on/after YYYYMMDD")
    parser.add_argument("--min-runs", type=int, default=2)
    args = parser.parse_args()

    groups = load_results(since=args.since, model_filter=args.model)
    rows = summarise(groups, min_runs=args.min_runs)

    if not rows:
        print("No models with enough runs found.")
        return

    print_report(rows)


if __name__ == "__main__":
    main()
