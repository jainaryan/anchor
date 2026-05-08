"""
Conversation + Memory pipeline for MindMate.

ROOT CAUSE BEING FIXED:
  ~60% of training has no memory context; 100% of memory examples are single-turn;
  ALL multi-turn training has 0% memory. The model learned "multi-turn = banter
  without context", overwriting Llama's base instruction-following.

WHAT THIS GENERATES:
  Multi-turn examples (4-6 turns) where:
    - System prompt has both [User] block and [Recent sessions] (production format)
    - User introduces a NEW fact mid-conversation
    - Later turns require referencing BOTH injected memory AND within-conversation facts
    - Casual and emotionally varied turns — not every example is a crisis

TARGET: ~5,000 examples → synthetic_train_conv_memory.jsonl
"""

import json
import time
import random
import signal
import sys
from pathlib import Path
from utils import TeacherModel, parse_json_robust

TARGET = 999999  # wall-time controlled
SIMILARITY_THRESHOLD = 0.3

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUT_RAW = OUTPUTS_DIR / "conv_memory_raw.jsonl"
OUT_TRAIN = BASE_DIR.parent / "data" / "synthetic_train_conv_memory.jsonl"

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


# ─── Profile seeds ─────────────────────────────────────────────────────────────
# Each profile has: diagnoses, triggers, coping, support, recent sessions.
# These get varied slightly before each generation to prevent memorisation.

PROFILES = [
    {
        "name": "Maya",
        "age": "24 F",
        "diagnoses": "GAD",
        "triggers": "work deadlines, crowded places",
        "coping": ["box breathing", "calling Priya", "walking"],
        "support": "Priya (best friend)",
        "sessions": [
            ("Apr 18", "Panic attack at work. Called Priya, felt much better."),
            ("Apr 14", "Anxious about upcoming review. Tried walking, helped a bit."),
        ],
    },
    {
        "name": "Rohan",
        "age": "28 M",
        "diagnoses": "anxiety",
        "triggers": "social situations, conflict with family",
        "coping": ["journaling", "calling Arjun", "deep breathing"],
        "support": "Arjun (brother), Dr Meera (therapist)",
        "sessions": [
            ("Apr 19", "Conflict with mom. Journaled after, helped a bit."),
            ("Apr 12", "Anxious before family dinner. Deep breathing kept it manageable."),
        ],
    },
    {
        "name": "Sana",
        "age": "22 F",
        "diagnoses": "MDD, GAD",
        "triggers": "isolation, family stress",
        "coping": ["going for a run", "calling Meera", "listening to music"],
        "support": "Meera (therapist), Dev (partner)",
        "sessions": [
            ("Apr 17", "Feeling disconnected from Dev. Low mood most of the week."),
            ("Apr 10", "Missed running for a week. Mood noticeably worse."),
        ],
    },
    {
        "name": "Karan",
        "age": "26 M",
        "diagnoses": "anxiety, ADHD",
        "triggers": "performance pressure, uncertainty",
        "coping": ["5-4-3-2-1 grounding", "texting Sam", "cold water on face"],
        "support": "Sam (college friend)",
        "sessions": [
            ("Apr 20", "Big presentation stress. 5-4-3-2-1 grounding helped stay focused."),
            ("Apr 15", "Forgot deadlines twice. Spiralled about being incompetent."),
        ],
    },
    {
        "name": "Aisha",
        "age": "30 F",
        "diagnoses": "PTSD, anxiety",
        "triggers": "loud arguments, being ignored",
        "coping": ["box breathing", "calling Nadia", "stepping outside"],
        "support": "Nadia (sister)",
        "sessions": [
            ("Apr 16", "Parents argued badly. Stepped outside, called Nadia. Felt grounded after."),
            ("Apr 9", "Triggered at work by raised voice. Box breathing helped prevent spiral."),
        ],
    },
    {
        "name": "Vikram",
        "age": "32 M",
        "diagnoses": "depression",
        "triggers": "work stress, loneliness",
        "coping": ["walking", "calling Raj", "cooking something"],
        "support": "Raj (best friend)",
        "sessions": [
            ("Apr 21", "Talked about feeling stuck at work. Went for a long walk. Mood lifted slightly."),
            ("Apr 14", "Canceled plans with Raj. Stayed in all weekend."),
        ],
    },
    {
        "name": "Zara",
        "age": "21 F",
        "diagnoses": "social anxiety",
        "triggers": "group settings, public speaking",
        "coping": ["progressive muscle relaxation", "calling Isha", "writing in journal"],
        "support": "Isha (childhood friend)",
        "sessions": [
            ("Apr 15", "Pre-event anxiety about work team meeting. PMR helped a lot."),
            ("Apr 8", "Avoided a birthday party. Regretted it afterwards."),
        ],
    },
    {
        "name": "Nikhil",
        "age": "27 M",
        "diagnoses": "GAD",
        "triggers": "health worries, financial stress",
        "coping": ["breathing exercises", "texting Vikram", "making tea and sitting quietly"],
        "support": "Vikram (partner)",
        "sessions": [
            ("Apr 19", "Spiraled about health symptoms. Breathing exercises + Vikram helped calm it."),
            ("Apr 12", "Worried about rent. Spiralled for two hours. Texted Vikram, felt better."),
        ],
    },
    {
        "name": "Divya",
        "age": "25 F",
        "diagnoses": "burnout, anxiety",
        "triggers": "overcommitment, perfectionism",
        "coping": ["setting one boundary per day", "calling Tara", "a 10-min walk outside"],
        "support": "Tara (roommate)",
        "sessions": [
            ("Apr 22", "Took on two extra projects. Feeling overwhelmed. Tara helped her say no once."),
            ("Apr 17", "Cried at work. First time. Talked through it with Tara."),
        ],
    },
    {
        "name": "Aryan",
        "age": "23 M",
        "diagnoses": "anxiety",
        "triggers": "academic pressure, future uncertainty",
        "coping": ["making a to-do list", "calling mom", "going to the gym"],
        "support": "mom, Sid (best friend)",
        "sessions": [
            ("Apr 23", "Exam anxiety. Made a study plan, felt more in control."),
            ("Apr 16", "Worried about placement season. Called mom, helped."),
        ],
    },
]

# ─── New-fact seeds ────────────────────────────────────────────────────────────
# Each entry is a fact the user might introduce mid-conversation that the model
# should track and reference later within the same session.

NEW_FACTS = [
    "They just found out their roommate is moving out next month.",
    "They got a small raise at work today but don't feel as happy as they expected.",
    "They've been sleeping only 4-5 hours for the past week.",
    "A friend they haven't spoken to in years reached out today.",
    "They're considering quitting their job but haven't told anyone yet.",
    "They adopted a cat last week and it's been making mornings better.",
    "They had an argument with a close friend two days ago and haven't patched it up.",
    "They just signed up for a 5k run even though they've never run before.",
    "They got a rejection from a job they really wanted.",
    "Their sibling is going through a rough time and they're trying to support them.",
    "They've been avoiding the gym for three weeks.",
    "They booked a solo trip for next month on a whim.",
    "They've started waking up at 6am this week, which is new for them.",
    "They found out their childhood friend is getting married.",
    "They've been eating takeout every day for a week and feeling gross about it.",
    "They just finished a book that hit them really hard.",
    "They've been procrastinating on a big project and deadline is in two days.",
    "They cried on the commute home for no clear reason.",
    "They had a really good therapy session today.",
    "They texted their support person but haven't heard back yet.",
]

# ─── Conversation modes ────────────────────────────────────────────────────────
# Vary the tone so the model doesn't associate memory context with crisis only.

MODES = [
    "casual_check_in",       # Light banter, user seems okay, context used organically
    "venting",               # User venting, no crisis, model listens and recalls context
    "asking_for_help",       # User explicitly asks for a strategy from profile
    "memory_callback",       # User references something from recent sessions
    "mixed_news",            # User has good AND bad news; model balances
    "low_engagement",        # User gives short answers; model keeps things light, uses context to stay relevant
]

# ─── System prompt builder ────────────────────────────────────────────────────

_APP_BASE_PROMPT = (
    "You are Anchor, a warm and caring AI companion — like a close friend who genuinely listens.\n"
    "Talk naturally. Be curious about the person. Ask follow-up questions. Use their actual words and details back to them.\n"
    "If the conversation has been light and the person suddenly gets serious, drop the casual tone immediately. No jokes, no deflection. Just be present.\n"
    "If someone seems to be in danger or crisis, gently encourage them to reach out to someone they trust or a crisis line.\n"
    "You are an AI. If asked, say so warmly. Never pretend to have lived experiences.\n"
    "Don't lecture."
)

_MEMORY_HEADER = "\n".join([
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


def build_system_prompt(profile: dict) -> str:
    p_lines = [
        f"{profile['age']}",
        f"Triggers: {profile['triggers']}.",
        f"Helps: {', '.join(profile['coping'])}.",
        f"Support: {profile['support']}.",
    ]
    if profile.get("diagnoses"):
        p_lines.insert(1, profile["diagnoses"])
    profile_block = "\n".join(p_lines)

    sessions = profile["sessions"]
    session_lines = "\n".join(f"[{date}] {note}" for date, note in sessions)

    return (
        f"{_APP_BASE_PROMPT}\n\n{_MEMORY_HEADER}\n"
        f"[User]\n{profile_block}\n\n"
        f"[Recent sessions]\n{session_lines}"
    )


# ─── Generation prompt ─────────────────────────────────────────────────────────

GENERATION_PROMPT = """You are generating training data for a mental health AI companion called Anchor.

TASK: Generate a realistic {NUM_TURNS}-turn multi-turn conversation.

USER PROFILE (injected into system prompt):
{PROFILE_SUMMARY}

RECENT SESSIONS (injected into system prompt):
{SESSIONS_SUMMARY}

NEW FACT the user introduces during the conversation:
"{NEW_FACT}"

CONVERSATION MODE: {MODE}
{MODE_INSTRUCTION}

CRITICAL RULES FOR THE ASSISTANT:
1. Reference at least one detail from the profile or recent sessions NATURALLY — not as a recitation.
2. After the user mentions the NEW FACT, the assistant should track and use it in a later turn.
3. Keep assistant turns SHORT (1-3 sentences). No monologues or therapy lectures.
4. Do NOT invent diagnoses, names, or events not in the profile. Use only what's given.
5. Do NOT use phrases like "I understand" as openers. Vary how you respond.
6. The LAST assistant turn should feel like a natural conversation pause — not a therapy wrap-up.

Generate a JSON object in this EXACT format:
{{
  "conversations": [
    {{"role": "system", "content": "<system prompt content>"}},
    {{"role": "user", "content": "<turn 1>"}},
    {{"role": "assistant", "content": "<turn 1>"}},
    {{"role": "user", "content": "<turn 2>"}},
    {{"role": "assistant", "content": "<turn 2>"}},
    ...
  ]
}}

The system content must use this EXACT format:
{SYSTEM_PROMPT}

Output valid JSON only. No markdown. No commentary."""

MODE_INSTRUCTIONS = {
    "casual_check_in": (
        "The user opens with something light and unrelated to their triggers. "
        "As the conversation continues, the assistant weaves in context organically when relevant. "
        "Not every turn needs to reference the profile — but at least one should."
    ),
    "venting": (
        "The user is venting about something frustrating. Not a crisis — just annoyed or drained. "
        "The assistant listens, validates, and at some point connects to something from the profile or sessions. "
        "Do NOT immediately jump to coping strategies."
    ),
    "asking_for_help": (
        "The user explicitly asks for help or says 'I don't know what to do'. "
        "The assistant must suggest exactly ONE coping strategy from the profile's Helps list by name. "
        "Not a list. Just one. In a natural, warm way."
    ),
    "memory_callback": (
        "The user references something from their recent sessions (a person, event, or coping outcome). "
        "The assistant recognizes it and responds as if it genuinely remembers. "
        "Do NOT say 'I can see from our last session'. Just respond naturally as a friend would."
    ),
    "mixed_news": (
        "The user has both a positive development and a worry. "
        "The assistant celebrates the good before addressing the concern. "
        "Do not immediately pivot to the negative."
    ),
    "low_engagement": (
        "The user gives short, low-energy answers. They seem okay but quiet. "
        "The assistant stays light, doesn't probe for distress that isn't there. "
        "It uses the profile to ask a relevant, casual question that invites engagement."
    ),
}


# ─── Heuristic validation ──────────────────────────────────────────────────────

def heuristic_check(conv: list[dict], profile: dict, new_fact: str) -> bool:
    """Basic sanity checks — not strict, just filter obvious failures."""
    turns = [m for m in conv if m["role"] != "system"]
    if len(turns) < 4:
        return False

    assistant_turns = [m["content"].lower() for m in conv if m["role"] == "assistant"]
    if not assistant_turns:
        return False

    # Must have at least one turn that's neither too short nor too long
    lengths = [len(t.split()) for t in assistant_turns]
    if all(l < 3 for l in lengths):
        return False
    if any(l > 120 for l in lengths):
        return False

    # System message must be present
    has_system = any(m["role"] == "system" for m in conv)
    if not has_system:
        return False

    # At least one assistant turn should reference something from profile or new_fact
    all_assistant = " ".join(assistant_turns)
    profile_keywords = set()
    for s in profile["coping"] + [profile["support"]]:
        profile_keywords.update(s.lower().split())
    for _, note in profile["sessions"]:
        profile_keywords.update(note.lower().split())
    new_fact_words = set(w for w in new_fact.lower().split() if len(w) > 4)

    has_context_ref = (
        any(kw in all_assistant for kw in profile_keywords if len(kw) > 3)
        or any(w in all_assistant for w in new_fact_words)
    )

    # Filter hallucination phrase
    bad_phrases = ["you went quiet", "been a while since", "haven't heard from you"]
    if any(p in all_assistant for p in bad_phrases):
        return False

    return has_context_ref


# ─── Main generation function ──────────────────────────────────────────────────

def generate_example(teacher, profile: dict) -> dict | None:
    new_fact = random.choice(NEW_FACTS)
    mode = random.choice(MODES)
    num_turns = random.choice([4, 5, 6])

    profile_summary = (
        f"Name: {profile['name']}, {profile['age']}\n"
        f"Diagnoses: {profile['diagnoses']}\n"
        f"Triggers: {profile['triggers']}\n"
        f"Coping: {', '.join(profile['coping'])}\n"
        f"Support: {profile['support']}"
    )
    sessions_summary = "\n".join(f"[{d}] {n}" for d, n in profile["sessions"])
    system_prompt = build_system_prompt(profile)

    prompt = (
        GENERATION_PROMPT
        .replace("{NUM_TURNS}", str(num_turns))
        .replace("{PROFILE_SUMMARY}", profile_summary)
        .replace("{SESSIONS_SUMMARY}", sessions_summary)
        .replace("{NEW_FACT}", new_fact)
        .replace("{MODE}", mode)
        .replace("{MODE_INSTRUCTION}", MODE_INSTRUCTIONS[mode])
        .replace("{SYSTEM_PROMPT}", system_prompt)
    )

    response = teacher.generate(prompt, max_new_tokens=1200, temperature=0.82)
    data = parse_json_robust(response, expected_keys=["conversations"])
    if not data or "conversations" not in data:
        return None

    conv = data["conversations"]
    if not heuristic_check(conv, profile, new_fact):
        return None

    return {
        "conversations": conv,
        "meta": {
            "mode": mode,
            "new_fact": new_fact,
            "profile_name": profile["name"],
            "num_turns": num_turns,
            "source": "conv_memory",
        },
    }


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("[Pipeline] Loading teacher model...")
    teacher = TeacherModel()

    count = 0
    attempts = 0
    skipped = 0

    print(f"[Pipeline] Target: {TARGET} examples (wall-time controlled)")
    print(f"[Pipeline] Output: {OUT_TRAIN}")

    while count < TARGET and not shutdown_requested:
        attempts += 1
        profile = random.choice(PROFILES)

        try:
            result = generate_example(teacher, profile)
            if result:
                append_jsonl(result, OUT_RAW)
                train_entry = {"conversations": result["conversations"]}
                append_jsonl(train_entry, OUT_TRAIN)
                count += 1
                mode = result["meta"]["mode"]
                print(f"[{attempts}] ✓ {count} | mode={mode} | profile={result['meta']['profile_name']}")
            else:
                skipped += 1
                if attempts % 10 == 0:
                    print(f"[{attempts}] skip {skipped} | pass {count}")

        except Exception as e:
            print(f"[{attempts}] Error: {e}")
            time.sleep(2)
            continue

        if attempts % 50 == 0:
            print(f"[{attempts}] Checkpoint: {count} saved, {skipped} skipped, pass_rate={count/(attempts)*100:.1f}%")
            time.sleep(2)

    print(f"\n[Pipeline] Done. examples={count}, attempts={attempts}, skipped={skipped}")
    print(f"[Pipeline] Training data: {OUT_TRAIN}")


if __name__ == "__main__":
    main()
