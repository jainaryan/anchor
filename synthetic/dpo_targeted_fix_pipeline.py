"""
DPO Targeted Fix Pipeline — help_mode + memory_recall

Generates preference pairs for the two eval failure modes identified in
src/eval/EVAL_RESULTS.md:

  help_mode:
    User asks for help while anxious. chosen = names ONE coping strategy from
    their profile. rejected = "you went quiet on me" / generic / probing question.

  memory_recall:
    User mentions a topic/person from a past session. chosen = references the
    specific name/event naturally. rejected = "it's not just about X, is it?" /
    fully generic / "who is that?".

Runs until SIGTERM / wall-time. Appends new pairs to the EXISTING
dpo_train.jsonl / dpo_val.jsonl so the targeted pairs are mixed with the
original 2,500-pair dataset.

Uses the same 8 profile seeds as targeted_fix_pipeline.py, with memory
context injected as a system turn in every generated prompt.
"""

import json
import time
import signal
import random
from pathlib import Path
from utils import TeacherModel, load_prompt, parse_json_robust, calculate_similarity

# ── Config ─────────────────────────────────────────────────────────────────────

TARGET_HELP   = 999999   # unlimited — wall-time controls duration
TARGET_RECALL = 999999
VAL_RATIO      = 0.15
SIMILARITY_THRESHOLD = 0.70

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUTS_DIR  = BASE_DIR / "outputs"
RAW_FILE     = OUTPUTS_DIR / "dpo_targeted_fix_raw.jsonl"

# Append to the existing DPO dataset files (created by dpo_pipeline.py)
TRAIN_FILE   = PROJECT_ROOT / "data" / "dpo_train.jsonl"
VAL_FILE     = PROJECT_ROOT / "data" / "dpo_val.jsonl"

# Partial file for crash recovery (targeted pairs only)
PARTIAL_FILE = PROJECT_ROOT / "data" / "dpo_targeted_fix_partial.jsonl"

CHECKPOINT_EVERY = 25

# ── Profile seeds (same as targeted_fix_pipeline.py) ──────────────────────────

PROFILES = [
    {
        "diagnoses": "GAD",
        "triggers": "work deadlines, crowded places",
        "coping": ["box breathing", "calling Priya", "walking"],
        "support": "Priya (best friend)",
        "recent_session": "Panic attack at work. Called Priya, felt much better.",
        "recent_date": "Apr 18",
    },
    {
        "diagnoses": "anxiety",
        "triggers": "social situations, conflict",
        "coping": ["journaling", "calling Arjun", "deep breathing"],
        "support": "Arjun (brother)",
        "recent_session": "Talked about conflict with coworker. Journaled after, helped a bit.",
        "recent_date": "Apr 19",
    },
    {
        "diagnoses": "MDD, GAD",
        "triggers": "isolation, family stress",
        "coping": ["going for a run", "calling Meera", "listening to music"],
        "support": "Meera (therapist), Dev (partner)",
        "recent_session": "Feeling disconnected from Dev. Low mood most of the week.",
        "recent_date": "Apr 17",
    },
    {
        "diagnoses": "anxiety",
        "triggers": "performance pressure, uncertainty",
        "coping": ["5-4-3-2-1 grounding", "texting Sam", "cold water on face"],
        "support": "Sam (college friend)",
        "recent_session": "Big presentation stress. 5-4-3-2-1 grounding helped stay focused.",
        "recent_date": "Apr 20",
    },
    {
        "diagnoses": "PTSD, anxiety",
        "triggers": "loud arguments, being ignored",
        "coping": ["box breathing", "calling Nadia", "stepping outside"],
        "support": "Nadia (sister)",
        "recent_session": "Parents argued badly. Stepped outside, called Nadia. Felt grounded after.",
        "recent_date": "Apr 16",
    },
    {
        "diagnoses": "depression",
        "triggers": "work stress, loneliness",
        "coping": ["walking", "calling Raj", "cooking something"],
        "support": "Raj (best friend)",
        "recent_session": "Talked about feeling stuck at work. Went for a long walk. Mood lifted slightly.",
        "recent_date": "Apr 21",
    },
    {
        "diagnoses": "social anxiety",
        "triggers": "group settings, public speaking",
        "coping": ["progressive muscle relaxation", "calling Isha", "writing in journal"],
        "support": "Isha (childhood friend)",
        "recent_session": "Pre-event anxiety about work team meeting. PMR helped a lot.",
        "recent_date": "Apr 15",
    },
    {
        "diagnoses": "GAD",
        "triggers": "health worries, financial stress",
        "coping": ["breathing exercises", "texting Vikram", "making tea and sitting quietly"],
        "support": "Vikram (partner)",
        "recent_session": "Spiraled about health symptoms. Breathing exercises + Vikram helped calm it.",
        "recent_date": "Apr 19",
    },
]

# ── Memory recall situations ───────────────────────────────────────────────────

RECALL_SITUATIONS = [
    "User brings up the same person mentioned in a recent session",
    "User mentions an event they were anxious about that was in a past session",
    "User references the coping strategy they tried last session and says it worked or didn't",
    "User says 'that thing I mentioned' or 'you know what I told you' implying prior context",
    "User brings up a person by name who appeared in a recent session",
    "User returns after a few days and mentions the situation from last time got worse",
    "User says they tried the coping strategy and asks what to do when it doesn't work",
    "User mentions a place or event that was in their profile triggers",
    "User references something they talked about last week",
    "User says the person they mentioned last time did something new",
]

# ── Helpers ────────────────────────────────────────────────────────────────────

def build_memory_context(p: dict) -> str:
    return (
        f"[User]\n"
        f"{p['diagnoses']}\n"
        f"Triggers: {p['triggers']}.\n"
        f"Helps: {', '.join(p['coping'])}.\n"
        f"Support: {p['support']}.\n"
        f"[Recent sessions]\n"
        f"[{p['recent_date']}] {p['recent_session']}"
    )

def append_jsonl(data: dict, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def heuristic_check_help(data: dict, profile: dict) -> bool:
    chosen_text  = (data.get("chosen",  [{}])[0].get("content", "") or "").lower()
    rejected_text = (data.get("rejected", [{}])[0].get("content", "") or "").lower()

    if not chosen_text or not rejected_text:
        return False

    # chosen must name at least one coping strategy from the profile
    strategy_found = any(
        any(word in chosen_text for word in strat.lower().split())
        for strat in profile["coping"]
    )
    if not strategy_found:
        return False

    # chosen must NOT contain the hallucination phrase
    bad_phrases = ["you went quiet", "went quiet on me", "been a while", "haven't heard"]
    if any(p in chosen_text for p in bad_phrases):
        return False

    # rejected should be plausible but wrong — not a copy of chosen
    if calculate_similarity(chosen_text, rejected_text) > SIMILARITY_THRESHOLD:
        return False

    # Both must have substance
    if len(chosen_text.split()) < 6 or len(rejected_text.split()) < 6:
        return False

    # Word count parity ≤ 3×
    cw = len(chosen_text.split())
    rw = len(rejected_text.split())
    if max(cw, rw) / max(min(cw, rw), 1) > 3.0:
        return False

    return True

def heuristic_check_recall(data: dict, profile: dict) -> bool:
    chosen_text   = (data.get("chosen",  [{}])[0].get("content", "") or "").lower()
    rejected_text = (data.get("rejected", [{}])[0].get("content", "") or "").lower()

    if not chosen_text or not rejected_text:
        return False

    # chosen must reference at least one detail from the recent session
    recent = profile["recent_session"].lower()
    keywords = [w for w in recent.split() if len(w) > 3 and w.isalpha()]
    support_names = [s.split()[0].lower() for s in profile["support"].split(",")]

    memory_referenced = (
        any(name in chosen_text for name in support_names) or
        any(kw in chosen_text for kw in keywords[:8])
    )
    if not memory_referenced:
        return False

    # chosen must NOT be mind-reading
    if "not just about" in chosen_text or "there's something else" in chosen_text:
        return False

    if calculate_similarity(chosen_text, rejected_text) > SIMILARITY_THRESHOLD:
        return False

    if len(chosen_text.split()) < 6 or len(rejected_text.split()) < 6:
        return False

    cw = len(chosen_text.split())
    rw = len(rejected_text.split())
    if max(cw, rw) / max(min(cw, rw), 1) > 3.0:
        return False

    return True


# ── Generation ─────────────────────────────────────────────────────────────────

def generate_pair(teacher, prompt_template: str, category: str, profile: dict, situation: str) -> dict | None:
    memory_context = build_memory_context(profile)

    prompt = (
        prompt_template
        .replace("{system_prompt}", "You are Anchor, a warm mental-health AI companion.")
        .replace("{memory_context}", memory_context)
        .replace("{category}", category)
        .replace("{situation}", situation)
    )

    response = teacher.generate(prompt, max_new_tokens=1800, temperature=0.80)
    data = parse_json_robust(response, expected_keys=["prompt", "chosen", "rejected"])

    if not data:
        return None

    # Ensure system message in generated prompt contains actual memory context
    for msg in data.get("prompt", []):
        if msg.get("role") == "system" and "{memory_context}" in msg.get("content", ""):
            msg["content"] = msg["content"].replace("{memory_context}", memory_context)

    data["pair_type"] = "targeted_fix"
    data["category"] = category
    append_jsonl(data, RAW_FILE)

    ok = (heuristic_check_help(data, profile) if category == "help_mode"
          else heuristic_check_recall(data, profile))

    if not ok:
        return None

    return {
        "prompt":    data["prompt"],
        "chosen":    data["chosen"],
        "rejected":  data["rejected"],
        "category":  category,
        "pair_type": "targeted_fix",
    }


# ── Merge + save ───────────────────────────────────────────────────────────────

def load_existing_pairs(filepath: Path) -> list:
    pairs = []
    if filepath.exists():
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
    return pairs

def save_merged_split(new_pairs: list):
    """Merge new targeted-fix pairs with existing train/val and re-split."""
    existing_train = load_existing_pairs(TRAIN_FILE)
    existing_val   = load_existing_pairs(VAL_FILE)
    all_existing   = existing_train + existing_val

    # Only add pairs that are truly new (not already in the existing files)
    existing_targeted = {
        json.dumps(p.get("prompt", []), ensure_ascii=False)
        for p in all_existing
        if p.get("pair_type") == "targeted_fix"
    }
    truly_new = [
        p for p in new_pairs
        if json.dumps(p.get("prompt", []), ensure_ascii=False) not in existing_targeted
    ]

    combined = all_existing + truly_new
    random.shuffle(combined)

    val_size = max(50, int(len(combined) * VAL_RATIO))
    val   = combined[:val_size]
    train = combined[val_size:]

    TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for p in train:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(VAL_FILE, "w", encoding="utf-8") as f:
        for p in val:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[Save] {len(train)} train / {len(val)} val "
          f"({len(all_existing)} existing + {len(truly_new)} new targeted-fix pairs)")


# ── Main ───────────────────────────────────────────────────────────────────────

shutdown_requested = False
_new_pairs_ref: list = []

def signal_handler(sig, frame):
    global shutdown_requested
    print("\n[Pipeline] Shutdown signal — saving before exit...")
    if _new_pairs_ref:
        save_merged_split(_new_pairs_ref)
    shutdown_requested = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def main():
    global shutdown_requested

    print("=" * 55)
    print("  MindMate DPO Targeted Fix Pipeline")
    print("  Categories: help_mode + memory_recall")
    print("=" * 55)

    teacher = TeacherModel()
    prompt_template = load_prompt("dpo_targeted_fix_preference.txt")

    help_count   = 0
    recall_count = 0
    attempts     = 0

    # Resume from partial file if present
    new_pairs: list = []
    if PARTIAL_FILE.exists():
        new_pairs = load_existing_pairs(PARTIAL_FILE)
        _new_pairs_ref.extend(new_pairs)
        help_count   = sum(1 for p in new_pairs if p.get("category") == "help_mode")
        recall_count = sum(1 for p in new_pairs if p.get("category") == "memory_recall")
        print(f"[Resume] Loaded {len(new_pairs)} pairs — "
              f"help={help_count} recall={recall_count}")

    while not shutdown_requested and (help_count < TARGET_HELP or recall_count < TARGET_RECALL):
        attempts += 1
        profile = random.choice(PROFILES)

        # Alternate: prioritise whichever category is behind
        if recall_count <= help_count and recall_count < TARGET_RECALL:
            category = "memory_recall"
            situation = random.choice(RECALL_SITUATIONS)
        else:
            category = "help_mode"
            situation = f"user asks for help while anxious about {random.choice(['work', 'this deadline', 'the presentation', 'everything', 'this panic', 'my thoughts'])}"

        print(f"\n[{attempts}] help={help_count} recall={recall_count} | {category}")

        try:
            result = generate_pair(teacher, prompt_template, category, profile, situation)
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(3)
            continue

        if result:
            new_pairs.append(result)
            _new_pairs_ref.append(result)
            append_jsonl(result, PARTIAL_FILE)

            if category == "help_mode":
                help_count += 1
            else:
                recall_count += 1

            print(f"  [Pass] help={help_count} recall={recall_count}")

            if len(new_pairs) % CHECKPOINT_EVERY == 0:
                save_merged_split(new_pairs)
        else:
            print("  [Fail]")

        if attempts % 20 == 0:
            time.sleep(3)

    print(f"\n[Pipeline] Done — help={help_count} recall={recall_count} attempts={attempts}")
    save_merged_split(new_pairs)


if __name__ == "__main__":
    main()
