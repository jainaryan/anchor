"""
One-shot port: benchmarks/scenarios.py (v3, Python literals)
                  → benchmarks/scenarios.json (v4, shared schema).

Round-trip validation: for every ported scenario, render the new seed back
through scenarios_loader.render_system_prompt() and compare byte-for-byte
to the original `system` string. If any scenario doesn't round-trip, the
port aborts — we never silently change benchmark semantics.

Usage:
    python -m benchmarks._port_scenarios

This script can be deleted after the port lands; scenarios.json is the new
source of truth.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Import the v3 source-of-truth
sys.path.insert(0, str(Path(__file__).parent.parent))
from benchmarks import scenarios as v3
from benchmarks.scenarios_loader import (
    APP_BASE_PROMPT,
    MEMORY_HEADER,
    render_system_prompt,
)

OUT_PATH = Path(__file__).parent / "scenarios.json"


# ──────────────────────────────────────────────────────────────────────────────
# Seed extraction
# ──────────────────────────────────────────────────────────────────────────────

def _strip_block(text: str, marker: str) -> tuple[str, str]:
    """
    Find `[marker]\\n<content>` in text. Return (content, text_without_block).
    Content runs until the next blank line / next `[xxx]` marker / end.
    """
    needle = f"[{marker}]\n"
    idx = text.find(needle)
    if idx < 0:
        return ("", text)
    start = idx + len(needle)
    # Find end: blank line, next [marker], or end of string
    rest = text[start:]
    end = len(rest)
    for stop in ["\n\n[", "\n\n="]:
        i = rest.find(stop)
        if 0 <= i < end:
            end = i
    content = rest[:end].rstrip("\n")
    new_text = text[:idx] + text[start + end:]
    return (content, new_text)


def extract_seed(system_prompt: str) -> dict:
    """
    Pull profile / memory_blocks / biometric out of a fully-rendered system prompt.
    Empty seed if the prompt is just APP_BASE_PROMPT alone.
    """
    sp = system_prompt.strip()

    # Cheap case: bare base prompt → empty seed
    if sp == APP_BASE_PROMPT.strip():
        return {}

    seed: dict = {}

    # Extract [Health data — last 7 days] first (most specific marker)
    bio, remaining = _strip_block(sp, "Health data — last 7 days")
    if bio:
        seed["biometric"] = bio

    # Extract [Recent sessions]
    memory, remaining = _strip_block(remaining, "Recent sessions")
    if memory:
        # Split memory back into per-session lines
        # Original format: each line starts with "[Mon DD] " or is a contiguous summary
        # For round-trip safety, keep as a single-element list with the raw text
        seed["memory_blocks"] = [line for line in memory.split("\n") if line.strip()]

    # Extract [User]
    profile, _ = _strip_block(remaining, "User")
    if profile:
        seed["profile"] = profile

    return seed


# ──────────────────────────────────────────────────────────────────────────────
# Per-scenario port
# ──────────────────────────────────────────────────────────────────────────────

def port_scenario(s: dict) -> dict:
    """
    Convert one v3 scenario dict → v4 scenario dict.

    v3 fields:
      id, category, type, weight, description, system, turns, judge_criteria
      (dynamic also has: max_turns, user_persona)

    v4 schema (scripted_multiturn / single):
      id, category, type, weight, description, seed, user_turns, judge_criteria, tags

    v4 schema additions for dynamic_multiturn:
      max_turns, min_turns, user_persona, simulator, stop_conditions
    """
    seed = extract_seed(s.get("system", ""))

    out: dict = {
        "id": s["id"],
        "category": s["category"],
        "type": s["type"],          # "single" or "dynamic"
        "weight": s.get("weight", 1),
        "description": s.get("description", ""),
        "seed": seed,
    }

    if s["type"] == "single":
        # v3 single: turns = [{"role": "user", "content": "..."}]
        # v4 single: user_turns = ["..."]
        user_turns = [t["content"] for t in s.get("turns", []) if t["role"] == "user"]
        out["user_turns"] = user_turns

    elif s["type"] == "dynamic":
        # Promote v3 "dynamic" → v4 "dynamic_multiturn"
        out["type"] = "dynamic_multiturn"
        out["max_turns"] = s.get("max_turns", 5)
        out["min_turns"] = s.get("min_turns", 2)
        out["user_persona"] = s.get("user_persona", "")
        # Stable seed per scenario — derive deterministically from id
        out["simulator"] = {
            "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 80,
            "seed": _scenario_simulator_seed(s["id"]),
        }
        # Default stop conditions; can be overridden in v4 scenarios
        out["stop_conditions"] = [
            "user says 'thanks anchor' or 'i think i'm okay'",
            "user goes silent for 2 turns (single word replies)",
            "max_turns reached",
        ]

    # Promote scripted single-turns to scripted_multiturn type for clarity
    # (Phase 2 will rewrite many of these into actual multi-turn flows;
    #  for now, single-turn scripted is the legacy shape.)
    elif s["type"] == "single":
        pass  # keep as "single"

    # Normalize judge criteria — v4 uses explicit id + scope + negative
    v4_criteria = []
    for i, c in enumerate(s.get("judge_criteria", [])):
        v4_criteria.append({
            "id": c.get("id", f"crit_{i+1:02d}"),
            "question": c["question"],
            "pass_if": c.get("pass_if", "YES"),
            "weight": c.get("weight", 1),
            "scope": c.get("scope", "any_turn"),
            "negative": _infer_negative_flag(c["question"]),
        })
    out["judge_criteria"] = v4_criteria

    # Tags — populated heuristically; humans can edit later
    tags: list[str] = []
    if s["category"] in ("CRISIS", "SAFETY_AMBIGUITY"):
        tags.append("safety")
    if s["type"] == "single" and len(out.get("user_turns", [])) <= 1:
        # Short, deterministic scripted scenarios are mobile-safe by default
        tags.append("mobile_safe")
    if s["type"] == "dynamic":
        tags.append("stress_test")
    out["tags"] = tags

    return out


def _scenario_simulator_seed(scenario_id: str) -> int:
    """Deterministic seed per scenario id, stable across runs."""
    h = 0
    for c in scenario_id:
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    # Pick a value that's clearly "seed-like" not random-looking
    return 1000 + (h % 9000)


def _infer_negative_flag(question: str) -> bool:
    """
    Heuristic: criterion is 'negative' if it's checking absence of AI behavior.
    Look for "AI avoids", "AI does not", "AI doesn't" as a phrase, not just the
    bare words (which can appear in flavor text describing user behavior).
    """
    q = question.lower()
    negative_phrases = [
        "ai avoid", "model avoid", "anchor avoid",
        "ai does not", "model does not", "anchor does not",
        "ai doesn't", "model doesn't", "anchor doesn't",
        "ai never", "model never", "anchor never",
        "ai refrain", "model refrain", "anchor refrain",
        "ai not", "model not say", "anchor not",
    ]
    return any(p in q for p in negative_phrases)


# ──────────────────────────────────────────────────────────────────────────────
# Round-trip verification
# ──────────────────────────────────────────────────────────────────────────────

def verify_round_trip(v3_scenario: dict, v4_scenario: dict) -> tuple[bool, str]:
    """
    Render v4 seed → system prompt and compare to v3's `system` field.
    Returns (ok, diagnostic_message).
    """
    expected = v3_scenario.get("system", "")
    actual = render_system_prompt(v4_scenario.get("seed", {}))

    if expected == actual:
        return (True, "")

    # Diff diagnostics
    if len(expected) != len(actual):
        diag = f"length differs (expected={len(expected)}, actual={len(actual)})"
    else:
        # Find first divergence
        diff_at = next((i for i, (a, b) in enumerate(zip(expected, actual)) if a != b),
                       len(expected))
        diag = f"diverges at char {diff_at}: …{expected[max(0,diff_at-20):diff_at+20]!r} vs …{actual[max(0,diff_at-20):diff_at+20]!r}"
    return (False, diag)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    v3_sources = [
        ("CONTEXT_MEMORY", v3.CONTEXT_MEMORY_SCENARIOS),
        ("CONVERSATION_MEMORY", v3.CONVERSATION_MEMORY_SCENARIOS),
        ("CROSS_SESSION_MEMORY", v3.CROSS_SESSION_MEMORY_SCENARIOS),
        ("HELP_MODE", v3.HELP_MODE_SCENARIOS),
        ("CRISIS", v3.CRISIS_SCENARIOS),
        ("NO_HALLUCINATION", v3.NO_HALLUCINATION_SCENARIOS),
        ("BIOMETRIC", v3.BIOMETRIC_SCENARIOS),
        ("FORMAT", v3.FORMAT_SCENARIOS),
        ("COMPANION", v3.COMPANION_SCENARIOS),
    ]

    ported: list[dict] = []
    failures: list[tuple[str, str]] = []

    for cat_name, src_list in v3_sources:
        for s in src_list:
            v4 = port_scenario(s)
            ok, diag = verify_round_trip(s, v4)
            if not ok:
                failures.append((s["id"], diag))
            ported.append(v4)

    # Validate uniqueness
    ids = [s["id"] for s in ported]
    dup_ids = [i for i in set(ids) if ids.count(i) > 1]
    if dup_ids:
        print(f"ERROR: duplicate scenario IDs: {dup_ids}")
        sys.exit(1)

    # Summary
    print(f"Ported {len(ported)} scenarios")
    print(f"Round-trip failures: {len(failures)}")
    for sid, diag in failures[:10]:
        print(f"  ✗ {sid}: {diag}")
    if len(failures) > 10:
        print(f"  ... and {len(failures) - 10} more")

    if failures:
        print("\nABORT: round-trip failures exist. Fix extract_seed() before writing.")
        sys.exit(1)

    # Compose the output document
    doc = {
        "schema_version": "v4.0",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "benchmarks/_port_scenarios.py",
        "source_module": "benchmarks/scenarios.py (v3)",
        "scenarios": ported,
    }

    OUT_PATH.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"\n✓ Wrote {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes)")

    # Category breakdown
    by_cat: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for s in ported:
        by_cat[s["category"]] = by_cat.get(s["category"], 0) + 1
        by_type[s.get("type", "single")] = by_type.get(s.get("type", "single"), 0) + 1
    print("\nBy category:")
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat:<26} {n}")
    print("\nBy type:")
    for t, n in sorted(by_type.items()):
        print(f"  {t:<26} {n}")


if __name__ == "__main__":
    main()
