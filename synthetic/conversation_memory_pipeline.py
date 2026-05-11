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
import os
import time
import random
import signal
import sys
from pathlib import Path
from utils import TeacherModel, parse_json_robust

TARGET = 999999  # wall-time controlled
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
_label = os.environ.get("OUT_LABEL", "")
_suffix = f"_{_label}" if _label else ""
OUT_RAW = OUTPUTS_DIR / f"conv_memory_raw{_suffix}.jsonl"
OUT_TRAIN = BASE_DIR.parent / "data" / f"synthetic_train_conv_memory{_suffix}.jsonl"

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
    {
        "name": "Priya",
        "age": "29 F",
        "diagnoses": "OCD",
        "triggers": "uncertainty, contamination fears, making decisions",
        "coping": ["ERP exercises", "calling her therapist Dr Anand", "grounding with cold water"],
        "support": "Dr Anand (therapist), Lena (flatmate)",
        "sessions": [
            ("Apr 20", "Spent 40 min checking the stove. ERP helped interrupt it eventually."),
            ("Apr 13", "Good week — intrusive thoughts lower. Felt more present."),
        ],
    },
    {
        "name": "James",
        "age": "35 M",
        "diagnoses": "bipolar II",
        "triggers": "poor sleep, skipping meds, big life changes",
        "coping": ["keeping a sleep log", "texting his psychiatrist", "short walks in the morning"],
        "support": "Dr Patel (psychiatrist), Claire (wife)",
        "sessions": [
            ("Apr 21", "Noticed hypomanic signs — talking fast, less sleep. Texted Dr Patel."),
            ("Apr 15", "Stable week. Sleep consistent. Claire noticed mood was even."),
        ],
    },
    {
        "name": "Sofia",
        "age": "24 F",
        "diagnoses": "panic disorder",
        "triggers": "crowded transport, physical sensations (heart racing), being far from home",
        "coping": ["diaphragmatic breathing", "the DARE method", "texting Rosa"],
        "support": "Rosa (sister)",
        "sessions": [
            ("Apr 22", "Panic attack on the subway. Used DARE, got through it. Shaky afterwards."),
            ("Apr 16", "Rode the bus two stops alone — big win. Texted Rosa right after."),
        ],
    },
    {
        "name": "Marcus",
        "age": "31 M",
        "diagnoses": "depression, chronic pain",
        "triggers": "pain flare-ups, feeling useless, cancelled plans",
        "coping": ["pacing activities", "calling Jay", "heat pad + podcast"],
        "support": "Jay (best friend), pain clinic team",
        "sessions": [
            ("Apr 19", "Bad pain day. Stayed in bed most of it. Jay called unprompted — helped."),
            ("Apr 12", "Managed a short walk despite pain. Mood lifted briefly."),
        ],
    },
    {
        "name": "Leila",
        "age": "26 F",
        "diagnoses": "ADHD",
        "triggers": "open-ended tasks, rejection sensitivity, noisy environments",
        "coping": ["body doubling with a friend", "the Pomodoro timer", "voice memos instead of notes"],
        "support": "her ADHD coach Mia, Tom (boyfriend)",
        "sessions": [
            ("Apr 23", "Missed two deadlines. Shame spiral. Coach Mia helped reframe it."),
            ("Apr 17", "Used body doubling with Tom — finished the report. Felt great."),
        ],
    },
    {
        "name": "Chen",
        "age": "38 M",
        "diagnoses": "grief, adjustment disorder",
        "triggers": "anniversaries, seeing couples, going through dad's things",
        "coping": ["writing letters he doesn't send", "cooking his dad's recipes", "calling uncle Wei"],
        "support": "uncle Wei, grief support group (Thursdays)",
        "sessions": [
            ("Apr 20", "Went through dad's jacket. Cried a lot. Wrote a letter after. Felt a little lighter."),
            ("Apr 14", "First grief group session. Didn't talk much but felt less alone."),
        ],
    },
    {
        "name": "Amara",
        "age": "22 F",
        "diagnoses": "eating disorder recovery (anorexia)",
        "triggers": "diet talk, mirrors, unstructured meal times",
        "coping": ["structured meal plan", "calling her dietitian", "texting the recovery group chat"],
        "support": "dietitian Jess, recovery group chat, mom",
        "sessions": [
            ("Apr 21", "Skipped lunch, didn't tell anyone until now. Felt ashamed. Got back on plan for dinner."),
            ("Apr 15", "Full week on meal plan. Dietitian Jess said good progress."),
        ],
    },
    {
        "name": "Tariq",
        "age": "33 M",
        "diagnoses": "social anxiety, depression",
        "triggers": "networking events, being evaluated, eating alone in public",
        "coping": ["rehearsing conversations", "one social thing per week rule", "calling Bashir"],
        "support": "Bashir (cousin)",
        "sessions": [
            ("Apr 22", "Went to work lunch. Stayed 20 min. Felt like a win."),
            ("Apr 16", "Declined a party. Stayed home. Mood low afterwards."),
        ],
    },
    {
        "name": "Nadia",
        "age": "40 F",
        "diagnoses": "generalized anxiety, insomnia",
        "triggers": "late-night scrolling, work emails after 9pm, worrying about kids",
        "coping": ["phone off at 9pm rule", "progressive muscle relaxation", "talking to her husband Samir"],
        "support": "Samir (husband), Dr Lin (GP)",
        "sessions": [
            ("Apr 20", "Slept 5h. Anxious about eldest's exam results. PMR helped a little."),
            ("Apr 13", "Two nights good sleep after keeping phone out of bedroom. Noticeable difference."),
        ],
    },
    {
        "name": "Eli",
        "age": "27 M",
        "diagnoses": "BPD",
        "triggers": "perceived abandonment, conflict, feeling invisible",
        "coping": ["TIPP skill (ice water)", "texting therapist between sessions", "the check-the-facts worksheet"],
        "support": "therapist Dr Reyes (DBT), Sam (best friend)",
        "sessions": [
            ("Apr 21", "Fight with Sam. Felt like he was pulling away. Used ice water, helped de-escalate."),
            ("Apr 14", "Good DBT session. Practiced opposite action. Felt understood."),
        ],
    },
    {
        "name": "Yuki",
        "age": "29 F",
        "diagnoses": "seasonal depression, anxiety",
        "triggers": "dark mornings, being inside all day, cancelled social plans",
        "coping": ["light therapy lamp at 7am", "daily outside walk even in rain", "calling her friend Hana"],
        "support": "Hana (friend from home), Dr Sato (psychiatrist)",
        "sessions": [
            ("Apr 19", "Skipped the lamp for three days. Mood dipped noticeably. Back on it."),
            ("Apr 12", "Walked every day this week. Energy better. Hana visited."),
        ],
    },
    {
        "name": "Kofi",
        "age": "36 M",
        "diagnoses": "PTSD",
        "triggers": "loud sudden noises, being in confined spaces, being touched unexpectedly",
        "coping": ["grounding 5-4-3-2-1", "telling his partner Ama in the moment", "slow morning routine"],
        "support": "Ama (partner), trauma therapist Dr Osei",
        "sessions": [
            ("Apr 20", "Triggered at the supermarket. Grounding helped. Told Ama after — felt less alone."),
            ("Apr 13", "Good therapy session. Processed one memory without dissociating."),
        ],
    },
    {
        "name": "Isabelle",
        "age": "45 F",
        "diagnoses": "burnout, anxiety",
        "triggers": "back-to-back meetings, feeling like a bad mom, not finishing her to-do list",
        "coping": ["one non-negotiable lunch break", "calling her sister Anne", "5-min journaling at night"],
        "support": "Anne (sister), husband Pierre",
        "sessions": [
            ("Apr 22", "Cried in the car after work. Exhausted. Called Anne. Helped to vent."),
            ("Apr 16", "Took a full lunch break three days in a row. Noticed mood slightly better."),
        ],
    },
    {
        "name": "Dev",
        "age": "24 M",
        "diagnoses": "anxiety, imposter syndrome",
        "triggers": "code reviews, being the most junior person in the room, silence after sharing ideas",
        "coping": ["rubber duck debugging feelings", "texting his mentor Raj", "reframing with 'what would I tell a friend'"],
        "support": "Raj (mentor), his flatmate Kiran",
        "sessions": [
            ("Apr 23", "PR got 12 comments. Spiralled for an hour. Texted Raj. Felt better."),
            ("Apr 17", "Led a standup. Went fine. Still anxious beforehand but proud after."),
        ],
    },
    {
        "name": "Grace",
        "age": "52 F",
        "diagnoses": "depression, empty nest adjustment",
        "triggers": "quiet house, comparing herself to others, doing nothing on weekends",
        "coping": ["volunteering at the library on Saturdays", "calling her daughter Mia", "gardening"],
        "support": "Mia (daughter), book club friends",
        "sessions": [
            ("Apr 20", "Hard Sunday. House too quiet. Gardened for an hour — mood lifted slightly."),
            ("Apr 14", "Book club dinner. Laughed properly for the first time in weeks."),
        ],
    },
]


# ─── Companion profiles (no clinical diagnoses) ──────────────────────────────
# Most Anchor users aren't in therapy — they're regular people who chat, share
# small wins, vent about traffic, ask opinions. These profiles balance the
# dataset so the model doesn't default to therapist-mode for every conversation.

COMPANION_PROFILES = [
    {
        "name": "Alex",
        "age": "25 M",
        "profile_type": "companion",
        "interests": "bouldering, indie music, cooking experiments",
        "triggers": "work deadlines (mild), bad sleep",
        "coping": ["going for a climb", "calling his sister", "long walks with a podcast"],
        "support": "Maya (sister), Theo (roommate)",
        "sessions": [
            ("Apr 22", "Mentioned a new bouldering gym opened nearby. Excited to try it."),
            ("Apr 16", "Talked about getting into a new podcast series."),
        ],
    },
    {
        "name": "Mira",
        "age": "23 F",
        "profile_type": "companion",
        "interests": "thrift shopping, baking, cozy mystery shows",
        "triggers": "loud crowds, work overload",
        "coping": ["baking something simple", "texting her best friend Anika", "a long shower"],
        "support": "Anika (best friend from college)",
        "sessions": [
            ("Apr 21", "Just moved to a new city for work. Slowly settling in."),
            ("Apr 14", "Found a thrift store she likes. Excited about a chair she scored."),
        ],
    },
    {
        "name": "Theo",
        "age": "31 M",
        "profile_type": "companion",
        "interests": "home cooking, weekend hikes, fantasy novels",
        "triggers": "sleep deprivation (new dad), long meetings",
        "coping": ["cooking something elaborate", "a walk around the block", "texting his brother Sam"],
        "support": "Sam (brother), wife Lisa",
        "sessions": [
            ("Apr 22", "Baby slept through the night for the first time. Felt human again."),
            ("Apr 15", "Made fresh pasta with the baby strapped to him. Counted as a win."),
        ],
    },
    {
        "name": "Riley",
        "age": "28 they/them",
        "profile_type": "companion",
        "interests": "running, design podcasts, cafe-hopping",
        "triggers": "client revisions, missed runs",
        "coping": ["morning run", "calling Sasha", "a flat white at their favorite cafe"],
        "support": "Sasha (close friend), Mom",
        "sessions": [
            ("Apr 23", "Signed up for a half marathon. Excited but nervous about training."),
            ("Apr 17", "Landed a freelance project they really wanted."),
        ],
    },
    {
        "name": "Hana",
        "age": "27 F",
        "profile_type": "companion",
        "interests": "K-dramas, board games, learning Spanish on Duolingo",
        "triggers": "thesis stress, comparing herself to peers",
        "coping": ["a K-drama episode", "texting her cohort group chat", "making tea and lighting a candle"],
        "support": "thesis cohort group chat, sister Yuna",
        "sessions": [
            ("Apr 22", "Finished a draft of a thesis chapter. Tired but proud."),
            ("Apr 14", "Got really into a new K-drama. Recommended it to her sister."),
        ],
    },
    {
        "name": "Ben",
        "age": "35 M",
        "profile_type": "companion",
        "interests": "woodworking, smoked BBQ, college football",
        "triggers": "kids fighting, weekend errands piling up",
        "coping": ["a couple hours in the garage workshop", "slow coffee on the porch", "calling his dad"],
        "support": "wife Megan, dad",
        "sessions": [
            ("Apr 22", "Finished a cutting board project. Kids actually liked it."),
            ("Apr 16", "Smoked a brisket for the first time. Came out decent."),
        ],
    },
    {
        "name": "Lin",
        "age": "22 F",
        "profile_type": "companion",
        "interests": "true crime podcasts, film photography, indie games",
        "triggers": "job applications, family pressure",
        "coping": ["a walk with her camera", "texting her group chat", "playing a comfort game"],
        "support": "college group chat (Aanya, Jess, Priya)",
        "sessions": [
            ("Apr 23", "Got a callback for a job interview next week. Cautiously hopeful."),
            ("Apr 17", "Took a roll of film at the park. Loved how the shots came out."),
        ],
    },
    {
        "name": "Carlos",
        "age": "29 M",
        "profile_type": "companion",
        "interests": "running a small food truck, salsa dancing, soccer",
        "triggers": "slow business weeks, supplier delays",
        "coping": ["going for a run", "calling his cousin Rafa", "playing pickup soccer"],
        "support": "Rafa (cousin), girlfriend Sofia",
        "sessions": [
            ("Apr 22", "Sold out at the food truck for the first time. Hyped."),
            ("Apr 15", "Tried a new tamale recipe. Customers loved it."),
        ],
    },
]

PROFILES.extend(COMPANION_PROFILES)


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

# Companion modes — everyday casual chatter, no distress framing.
COMPANION_MODES = [
    "casual_check_in",
    "sharing_win",
    "bored_chatter",
    "opinion_seek",
    "storytelling",
    "small_complaint",
]

# Clinical modes — assume the user is processing something emotionally heavier.
CLINICAL_MODES = [
    "venting",
    "asking_for_help",
    "memory_callback",
    "mixed_news",
    "low_engagement",
]

# Combined reference (used by some logging — order: casual first).
MODES = COMPANION_MODES + CLINICAL_MODES

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
    "sharing_win": (
        "You're sharing something small and good — a tiny win, a fun thing, just something nice. "
        "Tone is light. The new fact is your main reason for messaging, not a heavy topic."
    ),
    "bored_chatter": (
        "You're procrastinating or just bored and want to chat. No big emotions. "
        "Talk about random everyday things — what you're watching, eating, doing. "
        "Bring up the new fact as one casual topic among others. Keep it light."
    ),
    "opinion_seek": (
        "You're asking Anchor for an opinion or recommendation about something light — "
        "what to watch tonight, what to cook, whether to do something fun. "
        "The new fact is part of the context, but the conversation is mostly about deciding."
    ),
    "storytelling": (
        "You're telling Anchor about something that happened today or recently — a story, "
        "an observation, an interaction. The new fact connects to the story but isn't the whole point."
    ),
    "small_complaint": (
        "You're mildly grumbling about something trivial — traffic, slow wifi, an annoying coworker, "
        "weather. Not actually upset, just venting in a lighthearted way. "
        "Bring up the new fact as a side topic."
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
    p_lines = [f"{profile['age']}"]
    if profile.get("diagnoses"):
        p_lines.append(profile["diagnoses"])
    if profile.get("interests"):
        p_lines.append(f"Into: {profile['interests']}.")
    p_lines.append(f"Triggers: {profile['triggers']}.")
    p_lines.append(f"Helps: {', '.join(profile['coping'])}.")
    p_lines.append(f"Support: {profile['support']}.")
    profile_block = "\n".join(p_lines)
    session_lines = "\n".join(f"[{date}] {note}" for date, note in profile["sessions"])
    return (
        f"{_APP_BASE_PROMPT}\n\n{_MEMORY_HEADER}\n"
        f"[User]\n{profile_block}\n\n"
        f"[Recent sessions]\n{session_lines}"
    )


# ─── Phase 1: User simulator ───────────────────────────────────────────────────

USER_SIM_PROMPT = """\
You are simulating a real person ({PERSON_DESCRIPTOR}) texting their AI companion called Anchor.

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
    # Build a descriptor that handles companion profiles (no diagnosis) cleanly.
    parts = [profile["age"]]
    if profile.get("diagnoses"):
        parts.append(profile["diagnoses"])
    if profile.get("interests"):
        parts.append(f"into {profile['interests']}")
    descriptor = ", ".join(parts)

    prompt = (
        USER_SIM_PROMPT
        .replace("{PERSON_DESCRIPTOR}", descriptor)
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

# Memory reference is only required when the mode naturally invites it.
# Default is OPTIONAL — the model should learn that having memory in the system
# prompt doesn't mean it must reference memory in every response. Companion
# profiles never require memory refs regardless of mode.
MEMORY_REQUIRED_MODES = {"memory_callback", "asking_for_help"}


def heuristic_check(conv: list[dict], profile: dict, new_fact: str, mode: str = "") -> bool:
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
    # Memory ref is only required when mode invites it AND profile is clinical.
    # Companion profiles always pass without forced memory refs.
    is_companion = profile.get("profile_type") == "companion"
    if not has_context_ref and mode in MEMORY_REQUIRED_MODES and not is_companion:
        return False

    # Filter known hallucination phrases only
    all_assistant_lower = " ".join(assistant_turns).lower()
    for phrase in ["you went quiet", "been a while since", "haven't heard from you"]:
        if phrase in all_assistant_lower:
            return False

    return True


# ─── Main generation function ──────────────────────────────────────────────────

def pick_mode(profile: dict) -> str:
    """Pick a mode appropriate for the profile type.

    Companion profiles only get companion modes (everyday chatter, no clinical framing).
    Clinical profiles get 65/35 skewed toward companion modes — this matches realistic
    usage where most chat is casual, not therapy-oriented.
    """
    if profile.get("profile_type") == "companion":
        return random.choice(COMPANION_MODES)
    if random.random() < 0.65:
        return random.choice(COMPANION_MODES)
    return random.choice(CLINICAL_MODES)


def generate_example(teacher, profile: dict) -> dict | None:
    new_fact = random.choice(NEW_FACTS)
    mode = pick_mode(profile)
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

    if not heuristic_check(conv, profile, new_fact, mode):
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
