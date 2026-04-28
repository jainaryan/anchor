"""
Targeted data generation pipeline for two eval failure modes:

Mode 1 — help_mode:
  User explicitly asks "can you help" / "what do i do" during a distress moment.
  System prompt includes a user profile with known coping strategies.
  Model must suggest ONE strategy from the profile by name, not a generic response.

Mode 2 — memory_recall:
  User brings up a topic or person from a past session.
  System prompt includes a recent session summary mentioning that topic.
  Model must connect naturally to the memory (name, event, coping outcome).

Both modes generate dialogues with memory context as a system turn,
teaching the model that when context is present it should use it.
"""

import json
import time
import random
import signal
import sys
from pathlib import Path
from utils import TeacherModel, parse_json_robust, calculate_similarity

TARGET_HELP = 999999  # unlimited — job wall-time controls duration
TARGET_RECALL = 999999
SIMILARITY_THRESHOLD = 0.45

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUT_RAW = OUTPUTS_DIR / "targeted_fix_raw.jsonl"
OUT_TRAIN = BASE_DIR.parent / "data" / "synthetic_train_targeted_fix.jsonl"

shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    print("\n[Pipeline] Shutdown signal. Finishing current dialogue...")
    shutdown_requested = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def append_jsonl(data: dict, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


# ── Profile seeds ──────────────────────────────────────────────────────────────

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

HELP_OPENERS = [
    "i'm spiraling about {topic}. can you help",
    "i don't know what to do about {topic}. help",
    "can you help me calm down, {topic} is getting bad",
    "i need help. {topic} is making me panic",
    "help, i'm really struggling with {topic} right now",
    "things are bad with {topic}. what do i do",
    "i'm losing it over {topic}. can you help me",
    "{topic} is really getting to me. i don't know what to do",
    "i'm not okay. {topic} is just too much right now",
    "i feel like i'm going to spiral. {topic} is everywhere",
]

HELP_TOPICS = [
    "work", "this deadline", "the presentation", "everything at work",
    "the family stuff", "this argument", "the anxiety", "everything",
    "this panic", "the meeting tomorrow", "my thoughts",
]


# ── Prompt templates ───────────────────────────────────────────────────────────

HELP_MODE_PROMPT = """You are generating training data for a mental health AI companion called Anchor.

TASK: Generate a short 2-4 turn dialogue for the HELP MODE failure case.

Context:
- The user has this profile in their system prompt:
{PROFILE_BLOCK}

- The user is asking for help with something anxiety-related.
- The IDEAL assistant response must:
  1. NOT say "you went quiet on me" or imply prior silence
  2. Suggest exactly ONE coping strategy FROM the profile's coping list, by name
  3. Be short (1-2 sentences max)
  4. Feel like a warm friend, not a therapist
  5. Check in OR suggest — not both at once

Generate a JSON object with this exact format:
{
  "conversations": [
    {"role": "system", "content": "<the memory context block>"},
    {"role": "user", "content": "<opening message asking for help>"},
    {"role": "assistant", "content": "<ideal response using ONE named coping strategy>"}
  ]
}

The system content should be formatted like:
[User]
{PROFILE_INLINE}
[Recent sessions]
[{DATE}] {RECENT}

Use realistic, varied language. The user message should feel natural, not scripted.
Output valid JSON only."""

MEMORY_RECALL_PROMPT = """You are generating training data for a mental health AI companion called Anchor.

TASK: Generate a 2-4 turn dialogue for the MEMORY RECALL failure case.

Context:
- The user has this profile and recent session:
{PROFILE_BLOCK}

- The situation: {SITUATION}

- The IDEAL assistant response must:
  1. Reference the specific name, event, or detail from the recent session naturally
  2. NOT ask generic questions like "what's going on?" when the context already explains it
  3. NOT say "it's not just about X, is it?" (mind-reading)
  4. Feel like the assistant genuinely remembers — warm, not clinical
  5. Be short (1-2 sentences)

Generate a JSON object with this exact format:
{
  "conversations": [
    {"role": "system", "content": "<the memory context block>"},
    {"role": "user", "content": "<user message referencing past topic>"},
    {"role": "assistant", "content": "<ideal response that uses the memory naturally>"}
  ]
}

The system content should be formatted like:
[User]
{PROFILE_INLINE}
[Recent sessions]
[{DATE}] {RECENT}

Make the user message sound natural. The memory reference in the assistant response should feel organic, not recited.
Output valid JSON only."""


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_profile_block(p: dict) -> str:
    return (
        f"Diagnoses: {p['diagnoses']}\n"
        f"Triggers: {p['triggers']}\n"
        f"Coping strategies: {', '.join(p['coping'])}\n"
        f"Support people: {p['support']}\n"
        f"Recent session ({p['recent_date']}): {p['recent_session']}"
    )

def build_profile_inline(p: dict) -> str:
    return (
        f"{p['diagnoses']}\n"
        f"Triggers: {p['triggers']}.\n"
        f"Helps: {', '.join(p['coping'])}.\n"
        f"Support: {p['support']}."
    )

def heuristic_check_help(conv: list, profile: dict) -> bool:
    """Validates help mode response uses a named coping strategy."""
    if len(conv) < 3:
        return False
    assistant_turns = [m for m in conv if m["role"] == "assistant"]
    if not assistant_turns:
        return False
    last = assistant_turns[-1]["content"].lower()

    # Must reference at least one coping strategy by name
    strategy_found = any(
        any(word in last for word in strategy.lower().split())
        for strategy in profile["coping"]
    )

    # Must NOT have the hallucination phrase
    bad_phrases = ["you went quiet", "went quiet on me", "been a while", "haven't heard"]
    if any(p in last for p in bad_phrases):
        return False

    # Must NOT be a therapy monologue
    if len(last.split()) > 60:
        return False

    return strategy_found

def heuristic_check_recall(conv: list, profile: dict) -> bool:
    """Validates memory recall response uses a detail from the recent session."""
    if len(conv) < 3:
        return False
    assistant_turns = [m for m in conv if m["role"] == "assistant"]
    if not assistant_turns:
        return False
    last = assistant_turns[-1]["content"].lower()
    recent = profile["recent_session"].lower()

    # Extract key names/words from recent session (words > 3 chars)
    keywords = [w for w in recent.split() if len(w) > 3 and w.isalpha()]
    support_names = [s.split()[0].lower() for s in profile["support"].split(",")]

    # Must reference something from memory (name or key event word)
    memory_referenced = (
        any(name in last for name in support_names) or
        any(kw in last for kw in keywords[:8])
    )

    # Must NOT be mind-reading
    if "not just about" in last or "there's something else" in last:
        return False

    if len(last.split()) > 60:
        return False

    return memory_referenced


# ── Generation functions ───────────────────────────────────────────────────────

def generate_help_example(teacher, profile: dict) -> dict | None:
    topic = random.choice(HELP_TOPICS)
    profile_block = build_profile_block(profile)
    profile_inline = build_profile_inline(profile)

    prompt = (HELP_MODE_PROMPT
              .replace("{PROFILE_BLOCK}", profile_block)
              .replace("{PROFILE_INLINE}", profile_inline)
              .replace("{DATE}", profile["recent_date"])
              .replace("{RECENT}", profile["recent_session"]))

    response = teacher.generate(prompt, max_new_tokens=600, temperature=0.80)
    data = parse_json_robust(response, expected_keys=["conversations"])
    if not data or "conversations" not in data:
        return None

    conv = data["conversations"]
    if not heuristic_check_help(conv, profile):
        return None

    data["meta_mode"] = "help_mode"
    data["meta_profile"] = profile["diagnoses"]
    data["source"] = "targeted_fix"
    return data


def generate_recall_example(teacher, profile: dict) -> dict | None:
    situation = random.choice(RECALL_SITUATIONS)
    profile_block = build_profile_block(profile)
    profile_inline = build_profile_inline(profile)

    prompt = (MEMORY_RECALL_PROMPT
              .replace("{PROFILE_BLOCK}", profile_block)
              .replace("{PROFILE_INLINE}", profile_inline)
              .replace("{DATE}", profile["recent_date"])
              .replace("{RECENT}", profile["recent_session"])
              .replace("{SITUATION}", situation))

    response = teacher.generate(prompt, max_new_tokens=600, temperature=0.80)
    data = parse_json_robust(response, expected_keys=["conversations"])
    if not data or "conversations" not in data:
        return None

    conv = data["conversations"]
    if not heuristic_check_recall(conv, profile):
        return None

    data["meta_mode"] = "memory_recall"
    data["meta_profile"] = profile["diagnoses"]
    data["source"] = "targeted_fix"
    return data


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("[Pipeline] Loading teacher model...")
    teacher = TeacherModel()

    help_count = 0
    recall_count = 0
    attempts = 0

    print(f"[Pipeline] Targets: {TARGET_HELP} help_mode + {TARGET_RECALL} memory_recall")

    while (help_count < TARGET_HELP or recall_count < TARGET_RECALL) and not shutdown_requested:
        attempts += 1
        profile = random.choice(PROFILES)

        # Alternate between modes, prioritising whichever is further behind
        if recall_count < TARGET_RECALL and (help_count >= TARGET_HELP or recall_count <= help_count):
            mode = "recall"
        else:
            mode = "help"

        try:
            if mode == "help" and help_count < TARGET_HELP:
                result = generate_help_example(teacher, profile)
                if result:
                    append_jsonl(result, OUT_RAW)
                    # Strip meta fields for training file
                    train_entry = {"conversations": result["conversations"]}
                    append_jsonl(train_entry, OUT_TRAIN)
                    help_count += 1
                    print(f"[{attempts}] help_mode {help_count}/{TARGET_HELP} | recall {recall_count}/{TARGET_RECALL}")

            elif mode == "recall" and recall_count < TARGET_RECALL:
                result = generate_recall_example(teacher, profile)
                if result:
                    append_jsonl(result, OUT_RAW)
                    train_entry = {"conversations": result["conversations"]}
                    append_jsonl(train_entry, OUT_TRAIN)
                    recall_count += 1
                    print(f"[{attempts}] help_mode {help_count}/{TARGET_HELP} | recall {recall_count}/{TARGET_RECALL}")

        except Exception as e:
            print(f"[{attempts}] Error: {e}")
            time.sleep(2)
            continue

        # Brief pause every 20 attempts to avoid thermal throttle
        if attempts % 20 == 0:
            print(f"[{attempts}] Pausing 3s... help={help_count} recall={recall_count}")
            time.sleep(3)

    print(f"\n[Pipeline] Done. help_mode={help_count}, memory_recall={recall_count}, attempts={attempts}")
    print(f"[Pipeline] Training data saved to: {OUT_TRAIN}")


if __name__ == "__main__":
    main()
