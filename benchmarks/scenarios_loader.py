"""
Shared scenario loader.

Reads benchmarks/scenarios.json (the v4 source of truth) and renders system
prompts using the same _sys() format that the v3 scenarios.py used.

This module is imported by both run_benchmarks.py (cluster) and the porting
script that converts scenarios.py → scenarios.json. The mobile app reads the
same scenarios.json directly (no Python dependency) and reconstructs the
system prompt in TypeScript using identical templates (see
src/eval/buildSystemPrompt.ts in the anchor-app repo).

Schema (v4) — see docs/Benchmarks.md "Scenario Schema" section.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Production system prompt template — must match anchor-app exactly.
# Any change here must be mirrored in src/utils/anchorSystemPrompt.ts and
# src/memory/contextBuilder.ts on the mobile side.
# ──────────────────────────────────────────────────────────────────────────────

APP_BASE_PROMPT = (
    "You are Anchor, a warm and caring AI companion — like a close friend who genuinely listens.\n"
    "Talk naturally. Be curious about the person. Ask follow-up questions. Use their actual words and details back to them.\n"
    "If the conversation has been light and the person suddenly gets serious, drop the casual tone immediately. No jokes, no deflection. Just be present.\n"
    "If someone seems to be in danger or crisis, gently encourage them to reach out to someone they trust or a crisis line.\n"
    "You are an AI. If asked, say so warmly. Never pretend to have lived experiences.\n"
    "Don't lecture."
)

MEMORY_HEADER = "\n".join([
    "============================================================",
    "ABOUT THIS USER (you know this — use it naturally)",
    "============================================================",
    "If the user mentions someone by name, an event, or a coping strategy listed below — reference it.",
    "If they ask for help, suggest ONE strategy from their Helps list by name.",
    "If [Recent sessions] shows a declining mood trend, acknowledge it in your first response — do not open as if meeting them for the first time.",
    'If [Recent sessions] records a health or sleep pattern (poor sleep, fatigue, physical symptoms), connect it when the user describes something similar — e.g. "given how rough your sleep has been, that fogginess tracks".',
    "If [Recent sessions] marks a coping strategy as unhelpful or worsening, do NOT suggest it.",
    "Do not recite this block back verbatim.",
])


def render_system_prompt(seed: dict) -> str:
    """
    Render the production system prompt from a scenario's seed.

    seed shape:
        {
          "profile":       Optional[str],   # rendered as [User] block
          "memory_blocks": Optional[List[str]],  # rendered as [Recent sessions] block, joined with \n
          "biometric":     Optional[str],   # rendered as [Health data — last 7 days] block
        }

    Empty seed (no profile, no memory) returns just APP_BASE_PROMPT — no header.
    """
    profile = seed.get("profile") if seed else None
    memory_blocks = seed.get("memory_blocks") if seed else None
    biometric = seed.get("biometric") if seed else None

    if not profile and not memory_blocks and not biometric:
        return APP_BASE_PROMPT

    blocks: list[str] = []
    if profile:
        blocks.append(f"[User]\n{profile}")
    if memory_blocks:
        # Each memory block is already a formatted line like "[Apr 20] User mentioned …"
        joined = "\n".join(memory_blocks) if isinstance(memory_blocks, list) else str(memory_blocks)
        blocks.append(f"[Recent sessions]\n{joined}")
    if biometric:
        blocks.append(f"[Health data — last 7 days]\n{biometric}")

    return f"{APP_BASE_PROMPT}\n\n{MEMORY_HEADER}\n" + "\n\n".join(blocks)


# ──────────────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_PATH = Path(__file__).parent / "scenarios.json"
_HOLDOUT_PATH = Path(__file__).parent / "scenarios_holdout.py"


def load_scenarios(path: Optional[Path] = None) -> list[dict]:
    """Load scenarios from scenarios.json. Returns the `scenarios` array."""
    p = Path(path) if path else _DEFAULT_PATH
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["scenarios"]


def load_holdout_scenarios() -> list[dict]:
    """
    Load holdout scenarios from benchmarks/scenarios_holdout.py.

    Holdout scenarios (IDs: hd_*) have profiles and phrasings that are
    disjoint from the data-generation pool. Run these only at final release
    ranking with --holdout to guard against benchmark contamination.
    Never include them in routine training-loop evaluations.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "scenarios_holdout", _HOLDOUT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.HOLDOUT_SCENARIOS


def load_metadata(path: Optional[Path] = None) -> dict:
    """Load the top-level metadata block (version, generated_at, etc.)."""
    p = Path(path) if path else _DEFAULT_PATH
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if k != "scenarios"}


def filter_scenarios(
    scenarios: list[dict],
    category: Optional[str] = None,
    scenario_type: Optional[str] = None,
    ids: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
    mobile_safe_only: bool = False,
) -> list[dict]:
    """Filter scenarios by various criteria. All filters AND together."""
    out = scenarios
    if category:
        out = [s for s in out if s["category"] == category]
    if scenario_type:
        out = [s for s in out if s.get("type", "single") == scenario_type]
    if ids:
        ids_set = set(ids)
        out = [s for s in out if s["id"] in ids_set]
    if tags:
        tags_set = set(tags)
        out = [s for s in out if tags_set.issubset(set(s.get("tags", [])))]
    if mobile_safe_only:
        out = [s for s in out if "mobile_safe" in s.get("tags", [])]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Quick verify — `python -m benchmarks.scenarios_loader`
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    try:
        scenarios = load_scenarios()
    except FileNotFoundError:
        print(f"scenarios.json not yet present at {_DEFAULT_PATH}")
        sys.exit(0)

    meta = load_metadata()
    print(f"Schema: {meta.get('schema_version', 'unknown')}  "
          f"generated_at: {meta.get('generated_at', 'unknown')}")
    print(f"Total scenarios: {len(scenarios)}")

    by_cat: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for s in scenarios:
        by_cat[s["category"]] = by_cat.get(s["category"], 0) + 1
        by_type[s.get("type", "single")] = by_type.get(s.get("type", "single"), 0) + 1

    print("\nBy category:")
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat:<26} {n}")
    print("\nBy type:")
    for t, n in sorted(by_type.items()):
        print(f"  {t:<26} {n}")

    # Quick sanity: render the first scenario's system prompt
    if scenarios:
        first = scenarios[0]
        prompt = render_system_prompt(first.get("seed", {}))
        print(f"\nFirst scenario ({first['id']}) system prompt:\n"
              f"  {len(prompt)} chars, {prompt.count(chr(10))} newlines")
