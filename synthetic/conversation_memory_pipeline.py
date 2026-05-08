"""
Conversation + Memory pipeline for MindMate — teacher-as-Anchor edition.

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

TWO-PHASE GENERATION (teacher-as-Anchor):
  Phase 1 — User simulator:
    Gemma4 with a "simulate a user" system prompt generates all user turns as JSON.
    The mode and new_fact are baked into the user-side instructions.

  Phase 2 — Anchor responder:
    Gemma4 is given the PRODUCTION anchor prompt + injected memory as its actual
    system message. It generates one assistant turn at a time, seeing the full
    conversation history. The teacher is constrained by the exact same prompt
    the student sees at inference time → training-inference distribution aligned.

TARGET: wall-time controlled (72h SLURM job)
"""

import json
import time
import random
import signal
import sys
from pathlib import Path
from utils import TeacherModel, parse_json_robust

TARGET = 999999  # wall-time controlled
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

MODES = [
    "casual_check_in",
    "venting",
    "asking_for_help",
    "memory_callback",
    "mixed_news",
    "low_engagement",
]

# User-side mode instructions — tells the USER simulator how this person is feeling/acting
USER_MODE_INSTRUCTIONS = {
    "casual_check_in": (
        "You're texting casually — bored, procrastinating, or just checking in. "
        "Nothing is wrong, you're just chatting. "
        "Mention the new fact naturally around turn 2-3 as something that's just happening in your life."
    ),
    "venting": (
        "You're mildly frustrated or drained about something (unrelated to the new fact). "
        "Vent about it in the first 1-2 turns. "
        "Bring up the new fact mid-conversation as an aside or additional thing on your mind."
    ),
    "asking_for_help": (
        "You're struggling and want some advice or coping ideas. "
        "In one of the turns, explicitly ask for help or say you don't know what to do. "
        "Mention the new fact as context for why you're struggling."
    ),
    "memory_callback": (
        "You're referencing something that happened recently — an event, a person, or something you tried. "
        "Bring up the new fact as a connected or separate development. "
        "Keep it conversational, not formal."
    ),
    "mixed_news": (
        "The new fact is your main topic — it's mixed (partly good, partly worrying). "
        "You have feelings about both sides of it. Also mention one other thing going on for you."
    ),
    "low_engagement": (
        "You're tired or just not in a talking mood. Give short, low-energy replies. "
        "Mention the new fact briefly at some point — don't elaborate much on it. "
        "You're okay, just quiet."
    ),
}

# ─── System prompt builder (production format) ────────────────────────────────

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
    session_lines = "\n".join(f"[{date}] {note}" for date, note in profile["sessions"])
    return (
        f"{_APP_BASE_PROMPT}\n\n{_MEMORY_HEADER}\n"
        f"[User]\n{profile_block}\n\n"
        f"[Recent sessions]\n{session_lines}"
    )


# ─── Phase 1: User simulator ───────────────────────────────────────────────────

USER_SIM_PROMPT = """\
You are simulating a real person ({AGE_GENDER}, {DIAGNOSES}) texting their AI companion called Anchor.

YOUR NEW FACT (something happening in your life right now): "{NEW_FACT}"

CONVERSATION MODE: {MODE}
{MODE_INSTRUCTION}

Generate EXACTLY {NUM_TURNS} user messages — one per turn in this conversation.
Rules for your messages:
- Casual texting tone: lowercase, contractions, short sentences, occasional typos or filler words
- Turn 1 opens the conversation naturally for the mode (don't open with the new fact unless it's mixed_news)
- Introduce the new fact organically by turn 2 or 3
- Each message 10–60 words. Don't repeat yourself across turns.
- Don't explain the mode explicitly — just write naturally as that person

Output JSON only:
{{"user_turns": ["<turn 1>", "<turn 2>", ..., "<turn {NUM_TURNS}>"]}}"""


def generate_user_turns(teacher, profile: dict, new_fact: str, mode: str, num_turns: int) -> list[str] | None:
    """Phase 1: Gemma4 as user simulator → returns list of user message strings."""
    prompt = (
        USER_SIM_PROMPT
        .replace("{AGE_GENDER}", profile["age"])
        .replace("{DIAGNOSES}", profile.get("diagnoses", "anxiety"))
        .replace("{NEW_FACT}", new_fact)
        .replace("{MODE}", mode)
        .replace("{MODE_INSTRUCTION}", USER_MODE_INSTRUCTIONS[mode])
        .replace("{NUM_TURNS}", str(num_turns))
    )
    response = teacher.generate(prompt, max_new_tokens=600, temperature=0.85)
    data = parse_json_robust(response, expected_keys=["user_turns"])
    if not data or "user_turns" not in data:
        return None
    turns = data["user_turns"]
    if not isinstance(turns, list) or len(turns) < num_turns:
        return None
    return [str(t).strip() for t in turns[:num_turns]]


# ─── Phase 2: Anchor responder ────────────────────────────────────────────────

def generate_anchor_turns(
    teacher, system_prompt: str, user_turns: list[str]
) -> tuple[list[str], list[dict]]:
    """
    Phase 2: Gemma4 acting as Anchor (constrained by production system prompt + memory).
    Returns (list_of_assistant_responses, full_conversation_messages).
    """
    messages = [{"role": "system", "content": system_prompt}]
    assistant_turns = []

    for user_turn in user_turns:
        messages.append({"role": "user", "content": user_turn})
        response = teacher.chat(messages, max_new_tokens=350, temperature=0.82)
        response = response.strip()
        assistant_turns.append(response)
        messages.append({"role": "assistant", "content": response})

    return assistant_turns, messages


# ─── Heuristic validation ──────────────────────────────────────────────────────

def heuristic_check(conv: list[dict], profile: dict, new_fact: str) -> bool:
    """Basic sanity checks — filter obvious generation failures."""
    turns = [m for m in conv if m["role"] != "system"]
    if len(turns) < 4:
        return False

    assistant_turns = [m["content"] for m in conv if m["role"] == "assistant"]
    if not assistant_turns:
        return False

    # Assistant turn lengths
    lengths = [len(t.split()) for t in assistant_turns]
    if all(l < 3 for l in lengths):
        return False
    if any(l > 150 for l in lengths):
        return False

    # System message must be present
    if not any(m["role"] == "system" for m in conv):
        return False

    # At least one assistant turn should reference something from profile or new_fact
    all_assistant = " ".join(assistant_turns).lower()

    profile_keywords = set()
    for s in profile["coping"] + [profile["support"]]:
        profile_keywords.update(s.lower().split())
    for _, note in profile["sessions"]:
        profile_keywords.update(note.lower().split())
    new_fact_words = {w for w in new_fact.lower().split() if len(w) > 4}

    has_context_ref = (
        any(kw in all_assistant for kw in profile_keywords if len(kw) > 3)
        or any(w in all_assistant for w in new_fact_words)
    )
    if not has_context_ref:
        return False

    # Filter known hallucination phrases only
    all_assistant_lower = " ".join(assistant_turns).lower()
    for phrase in ["you went quiet", "been a while since", "haven't heard from you"]:
        if phrase in all_assistant_lower:
            return False

    return True


# ─── Main generation function ──────────────────────────────────────────────────

def generate_example(teacher, profile: dict) -> dict | None:
    new_fact = random.choice(NEW_FACTS)
    mode = random.choice(MODES)
    num_turns = random.choice([4, 5, 6])
    system_prompt = build_system_prompt(profile)

    # ── Phase 1: simulate user turns ─────────────────────────────────────────
    user_turns = generate_user_turns(teacher, profile, new_fact, mode, num_turns)
    if not user_turns:
        return None

    # ── Phase 2: teacher acts as Anchor ──────────────────────────────────────
    _, messages = generate_anchor_turns(teacher, system_prompt, user_turns)

    # messages = [system, user, assistant, user, assistant, ...]
    conv = messages  # already in the right format

    if not heuristic_check(conv, profile, new_fact):
        return None

    return {
        "conversations": conv,
        "meta": {
            "mode": mode,
            "new_fact": new_fact,
            "profile_name": profile["name"],
            "num_turns": num_turns,
            "source": "conv_memory_v2",
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
    print(f"[Pipeline] Mode: teacher-as-Anchor (two-phase generation)")

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
                print(
                    f"[{attempts}] ✓ {count} | "
                    f"mode={result['meta']['mode']} | "
                    f"profile={result['meta']['profile_name']} | "
                    f"turns={result['meta']['num_turns']}"
                )
            else:
                skipped += 1
                if attempts % 10 == 0:
                    print(f"[{attempts}] skip={skipped} pass={count}")

        except Exception as e:
            print(f"[{attempts}] Error: {e}")
            time.sleep(2)
            continue

        if attempts % 50 == 0:
            rate = count / attempts * 100
            print(f"[{attempts}] Checkpoint: {count} saved, {skipped} skipped, pass_rate={rate:.1f}%")
            time.sleep(2)

    print(f"\n[Pipeline] Done. examples={count}, attempts={attempts}, skipped={skipped}")
    print(f"[Pipeline] Training data: {OUT_TRAIN}")


if __name__ == "__main__":
    main()
