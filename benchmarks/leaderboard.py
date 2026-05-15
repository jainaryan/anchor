"""
Unified leaderboard across cluster + mobile benchmark runs.

Reads all `judged.json` (or v3 `summary.json`) files in a results directory
and shows side-by-side per-category scores for every model × runtime ×
quantization combination it finds.

Goal: spot divergence between cluster_NF4 and mobile_Q4_K_M for the same
model. Big deltas (≥5pp per category) suggest a quantization or runtime
issue, not a model issue.

Usage:
    python -m benchmarks.leaderboard
    python -m benchmarks.leaderboard --root benchmarks/results --since 20260515
    python -m benchmarks.leaderboard --diverge-only       # only show cells with cross-runtime divergence
    python -m benchmarks.leaderboard --format json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional


def _load_doc(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _discover(root: Path) -> list[tuple[Path, dict]]:
    """
    Find every judged result document under root. Two layouts supported:
      - <root>/<run_dir>/judged.json     (v4 mobile + cluster)
      - <root>/<label>_<timestamp>.json  (v3 cluster summary)
    """
    out: list[tuple[Path, dict]] = []
    for p in sorted(root.rglob("judged.json")):
        d = _load_doc(p)
        if d: out.append((p, d))
    for p in sorted(root.glob("*.json")):
        if p.name == "judged.json": continue
        if "_judged_" in p.name: continue
        d = _load_doc(p)
        if d and "by_category" in d:
            out.append((p, d))
    return out


def _parse_timestamp(doc: dict) -> Optional[str]:
    """Return ISO timestamp (any of judged_at / timestamp / 'time')."""
    for k in ("judged_at", "timestamp", "time"):
        v = doc.get(k)
        if v:
            return str(v)
    return None


def _ts_after(ts: Optional[str], since_yyyymmdd: Optional[str]) -> bool:
    if not since_yyyymmdd:
        return True
    if not ts:
        return False
    # Pull leading YYYY-MM-DD or YYYYMMDD
    ts_compact = ts.replace("-", "").replace("/", "")[:8]
    return ts_compact >= since_yyyymmdd


def _category_scores(doc: dict) -> dict[str, tuple[float, float]]:
    """Return {category: (weighted_pass, weighted_total)}."""
    out: dict[str, tuple[float, float]] = {}
    # v4 layout: summary.by_category
    by_cat = doc.get("summary", {}).get("by_category") or doc.get("by_category", {})
    for cat, v in by_cat.items():
        wp = v.get("weighted_pass", v.get("passed", 0))
        wt = v.get("weighted_total", v.get("total", 0))
        out[cat] = (float(wp), float(wt))
    return out


def _runtime_label(doc: dict) -> str:
    rt = doc.get("runtime", "cluster")
    q = doc.get("quantization", "NF4")
    return f"{rt}/{q}"


def _model(doc: dict) -> str:
    return doc.get("model", "unknown")


def build_leaderboard(docs: list[dict]) -> dict:
    """Build a nested {model: {runtime_label: {category: pct, ...}}}."""
    by_model: dict[str, dict[str, dict[str, tuple[float, float]]]] = defaultdict(dict)
    for d in docs:
        model = _model(d)
        runtime = _runtime_label(d)
        cats = _category_scores(d)
        if cats:
            by_model[model][runtime] = cats
    return by_model


def _all_categories(by_model: dict) -> list[str]:
    cats: set[str] = set()
    for runtimes in by_model.values():
        for cats_data in runtimes.values():
            cats.update(cats_data.keys())
    # Sort with critical categories first for visibility
    critical = ["CRISIS", "SAFETY_AMBIGUITY"]
    rest = sorted(c for c in cats if c not in critical)
    return [c for c in critical if c in cats] + rest


def _pct(score: Optional[tuple[float, float]]) -> Optional[float]:
    if not score:
        return None
    wp, wt = score
    return 100 * wp / wt if wt else None


def _print_table(by_model: dict, diverge_only: bool, color: bool) -> None:
    cats = _all_categories(by_model)

    GREEN = "\033[32m"; RED = "\033[31m"; YELLOW = "\033[33m"; DIM = "\033[2m"
    BOLD = "\033[1m"
    def c(text, code):
        return f"{code}{text}\033[0m" if color else text

    for model in sorted(by_model):
        runtimes = by_model[model]
        runtime_keys = sorted(runtimes.keys())
        print(f"\n{c('━━ ' + model + ' ━━', BOLD)}")
        # Header
        header = "  " + f"{'Category':<24}"
        for rt in runtime_keys:
            header += f"{rt:>16}"
        print(header)

        # Rows
        for cat in cats:
            row = f"  {cat:<24}"
            scores = [_pct(runtimes[rt].get(cat)) for rt in runtime_keys]
            # Compute spread
            scores_present = [s for s in scores if s is not None]
            spread = (max(scores_present) - min(scores_present)) if len(scores_present) >= 2 else 0
            if diverge_only and spread < 5:
                continue
            for s in scores:
                if s is None:
                    row += f"{'—':>16}"
                else:
                    cell = f"{s:.0f}%"
                    if cat in ("CRISIS", "SAFETY_AMBIGUITY") and s < 55:
                        cell = c(cell, RED + BOLD)
                    elif spread >= 5:
                        cell = c(cell, YELLOW)
                    row += f"{cell:>16}"
            print(row)

        # Overall row
        row = "  " + c(f"{'TOTAL':<24}", BOLD)
        for rt in runtime_keys:
            total_wp = sum(wp for wp, _ in runtimes[rt].values())
            total_wt = sum(wt for _, wt in runtimes[rt].values())
            pct = (100 * total_wp / total_wt) if total_wt else None
            row += f"{(f'{pct:.0f}%' if pct is not None else '—'):>16}"
        print(row)

    if diverge_only:
        print(c("\n  (showing only rows with ≥5pp cross-runtime spread)", DIM))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="benchmarks/results", type=Path)
    p.add_argument("--since", help="YYYYMMDD filter on judged_at/timestamp")
    p.add_argument("--diverge-only", action="store_true",
                   help="Only show categories with ≥5pp cross-runtime spread")
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.add_argument("--no-color", action="store_true")
    args = p.parse_args()

    if not args.root.exists():
        print(f"No results dir: {args.root}", file=sys.stderr)
        sys.exit(0)

    pairs = _discover(args.root)
    docs = []
    for path, d in pairs:
        ts = _parse_timestamp(d)
        if _ts_after(ts, args.since):
            docs.append(d)

    if not docs:
        print(f"No judged results found in {args.root} (since={args.since})")
        sys.exit(0)

    by_model = build_leaderboard(docs)

    if args.format == "json":
        json.dump({
            "n_docs": len(docs),
            "by_model": {
                model: {
                    rt: {cat: list(score) for cat, score in cats.items()}
                    for rt, cats in runtimes.items()
                }
                for model, runtimes in by_model.items()
            },
        }, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    color = sys.stdout.isatty() and not args.no_color
    _print_table(by_model, args.diverge_only, color)


if __name__ == "__main__":
    main()
