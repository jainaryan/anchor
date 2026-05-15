"""
Per-scenario diff between two benchmark runs.

When genzv5 lands, you need to know which exact scenarios changed verdict
vs. genzv2_ck1200 — not just that the overall CRISIS score moved 2pp.
This script reads two `judged.json` outputs and emits a side-by-side diff.

Usage:
    python -m benchmarks.diff_results results/<A>/judged.json results/<B>/judged.json
    python -m benchmarks.diff_results <A> <B> --category CRISIS
    python -m benchmarks.diff_results <A> <B> --regressions-only
    python -m benchmarks.diff_results <A> <B> --format json > diff.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ANSI colors (skip when not tty)
def _color(text: str, code: str, enabled: bool) -> str:
    return f"{code}{text}\033[0m" if enabled else text


def _load(path: Path) -> dict:
    p = path
    # Allow passing either the run dir or the judged.json directly
    if p.is_dir():
        candidates = [p / "judged.json", p / "summary.json"]
        for c in candidates:
            if c.exists():
                p = c
                break
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _index_scenarios(doc: dict) -> dict[str, dict]:
    """
    Return {scenario_id: {category, weighted_pass, weighted_total, criteria}}.

    Works for both run_benchmarks.py output (v3 style: by_category at top)
    and judge_mobile_results.py output (v4 style: scenarios[].summary).
    """
    out: dict[str, dict] = {}
    scenarios = doc.get("scenarios", [])
    for s in scenarios:
        sid = s.get("id") or s.get("scenario_id")
        if not sid:
            continue
        summary = s.get("summary", {})
        # Each criterion gets PASS/FAIL/AMBIGUOUS
        crits = []
        for c in s.get("criteria_results", []):
            crits.append({
                "id": c.get("criterion_id"),
                "verdict": c.get("verdict"),
                "weight": c.get("weight", 1),
            })
        out[sid] = {
            "category": s.get("category") or _find_category(doc, sid),
            "weighted_pass": summary.get("weighted_pass", 0),
            "weighted_total": summary.get("weighted_total", 0),
            "criteria": crits,
        }
    return out


def _find_category(doc: dict, sid: str) -> str:
    """Fallback: scan top-level by_category if scenarios don't have category."""
    by_cat = doc.get("summary", {}).get("by_category", {})
    if not by_cat:
        by_cat = doc.get("by_category", {})
    # by_category is keyed by category name with counts inside — can't reverse
    # lookup without the scenario list having a category. Return unknown.
    return "UNKNOWN"


def _label(verdict_a: Optional[str], verdict_b: Optional[str]) -> tuple[str, str]:
    """Return (delta_symbol, classification)."""
    if verdict_a is None and verdict_b is None:
        return ("·", "missing")
    if verdict_a is None:
        return ("+", "new")
    if verdict_b is None:
        return ("-", "removed")
    if verdict_a == verdict_b:
        return ("=", "unchanged")
    if verdict_a == "FAIL" and verdict_b == "PASS":
        return ("▲", "improved")
    if verdict_a == "PASS" and verdict_b == "FAIL":
        return ("▼", "REGRESSED")
    return ("≠", "changed")


def diff_runs(doc_a: dict, doc_b: dict,
              category_filter: Optional[str] = None,
              regressions_only: bool = False) -> list[dict]:
    """Compute per-scenario diff records."""
    a_scenarios = _index_scenarios(doc_a)
    b_scenarios = _index_scenarios(doc_b)
    all_ids = sorted(set(a_scenarios.keys()) | set(b_scenarios.keys()))

    rows: list[dict] = []
    for sid in all_ids:
        a = a_scenarios.get(sid)
        b = b_scenarios.get(sid)
        category = (a or b or {}).get("category") or "UNKNOWN"
        if category_filter and category != category_filter:
            continue

        # Aggregate verdict: did the scenario "pass" overall (all criteria passed)?
        def overall(rec: Optional[dict]) -> Optional[str]:
            if rec is None: return None
            crits = rec["criteria"]
            if not crits: return None
            if all(c["verdict"] == "PASS" for c in crits): return "PASS"
            if any(c["verdict"] == "AMBIGUOUS" for c in crits): return "AMBIGUOUS"
            return "FAIL"

        va, vb = overall(a), overall(b)
        symbol, classification = _label(va, vb)
        if regressions_only and classification != "REGRESSED":
            continue

        # Per-criterion deltas — find any criteria that flipped
        crit_diffs: list[dict] = []
        ca_by_id = {c["id"]: c for c in (a["criteria"] if a else [])}
        cb_by_id = {c["id"]: c for c in (b["criteria"] if b else [])}
        for cid in set(ca_by_id) | set(cb_by_id):
            va_c = ca_by_id.get(cid, {}).get("verdict")
            vb_c = cb_by_id.get(cid, {}).get("verdict")
            if va_c != vb_c:
                crit_diffs.append({
                    "criterion_id": cid,
                    "a": va_c,
                    "b": vb_c,
                    "weight": (ca_by_id.get(cid) or cb_by_id.get(cid)).get("weight", 1),
                })

        rows.append({
            "scenario_id": sid,
            "category": category,
            "a_verdict": va,
            "b_verdict": vb,
            "symbol": symbol,
            "classification": classification,
            "a_score": (a or {}).get("weighted_pass", 0),
            "b_score": (b or {}).get("weighted_pass", 0),
            "weighted_total": (a or b or {}).get("weighted_total", 0),
            "criterion_diffs": crit_diffs,
        })
    return rows


def _print_text(rows: list[dict], doc_a: dict, doc_b: dict, color: bool) -> None:
    label_a = doc_a.get("model", "A")
    label_b = doc_b.get("model", "B")
    title_a = f"{label_a} ({doc_a.get('quantization', '?')})"
    title_b = f"{label_b} ({doc_b.get('quantization', '?')})"

    print(f"\nDiff: {title_a}  →  {title_b}\n")

    # Color helpers
    GREEN = "\033[32m"; RED = "\033[31m"; YELLOW = "\033[33m"; DIM = "\033[2m"
    BOLD = "\033[1m"

    # Group by category
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    grand = {"improved": 0, "REGRESSED": 0, "unchanged": 0, "changed": 0,
             "new": 0, "removed": 0}

    for cat in sorted(by_cat):
        cat_rows = by_cat[cat]
        improved = sum(1 for r in cat_rows if r["classification"] == "improved")
        regressed = sum(1 for r in cat_rows if r["classification"] == "REGRESSED")
        print(_color(f"━━ {cat}", BOLD, color),
              f"  ({len(cat_rows)} scenarios,",
              _color(f"+{improved}", GREEN, color) if improved else "",
              _color(f"-{regressed}", RED, color) if regressed else "",
              ")")
        for r in cat_rows:
            sym = r["symbol"]
            klass = r["classification"]
            sym_colored = sym
            if klass == "REGRESSED":
                sym_colored = _color(sym, RED + BOLD, color)
            elif klass == "improved":
                sym_colored = _color(sym, GREEN + BOLD, color)
            elif klass == "changed":
                sym_colored = _color(sym, YELLOW, color)
            elif klass == "unchanged":
                sym_colored = _color(sym, DIM, color)

            verdict_str = f"{r['a_verdict'] or '—':<5} → {r['b_verdict'] or '—':<5}"
            line = f"  {sym_colored}  {r['scenario_id']:<14}  {verdict_str}"
            if klass == "REGRESSED":
                line = _color(line, RED, color)
            print(line)
            for cd in r["criterion_diffs"]:
                detail = f"        {cd['criterion_id']:<28}  {cd['a'] or '—'} → {cd['b'] or '—'}  (w={cd['weight']})"
                print(_color(detail, DIM, color))
            grand[klass] = grand.get(klass, 0) + 1
        print()

    # Summary line
    print(_color("━━ Summary ━━", BOLD, color))
    bits = []
    if grand.get("REGRESSED"):
        bits.append(_color(f"REGRESSED={grand['REGRESSED']}", RED + BOLD, color))
    if grand.get("improved"):
        bits.append(_color(f"improved={grand['improved']}", GREEN, color))
    if grand.get("unchanged"):
        bits.append(_color(f"unchanged={grand['unchanged']}", DIM, color))
    for k in ("changed", "new", "removed"):
        if grand.get(k):
            bits.append(f"{k}={grand[k]}")
    print("  " + "  ".join(bits))

    # Aggregate score comparison
    def pct(doc: dict) -> str:
        s = doc.get("summary", {})
        wp = s.get("weighted_pass")
        wt = s.get("weighted_total")
        if wp is not None and wt:
            return f"{wp:.1f}/{wt:.1f} ({100*wp/wt:.1f}%)"
        return "(no summary)"
    print(f"\n  {label_a}: {pct(doc_a)}")
    print(f"  {label_b}: {pct(doc_b)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a", type=Path, help="Older run (judged.json or run dir)")
    p.add_argument("b", type=Path, help="Newer run (judged.json or run dir)")
    p.add_argument("--category", help="Filter to a single category")
    p.add_argument("--regressions-only", action="store_true",
                   help="Show only scenarios that flipped PASS→FAIL")
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.add_argument("--no-color", action="store_true")
    args = p.parse_args()

    doc_a = _load(args.a)
    doc_b = _load(args.b)
    rows = diff_runs(doc_a, doc_b, args.category, args.regressions_only)

    if args.format == "json":
        json.dump({
            "a": {k: doc_a.get(k) for k in ("model", "quantization", "device", "timestamp")},
            "b": {k: doc_b.get(k) for k in ("model", "quantization", "device", "timestamp")},
            "rows": rows,
        }, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return

    color = sys.stdout.isatty() and not args.no_color
    _print_text(rows, doc_a, doc_b, color)


if __name__ == "__main__":
    main()
