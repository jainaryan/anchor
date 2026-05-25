"""
Crisis + Help Mode SFT Data Generation Pipeline

TARGET: train Anchor to handle two high-stakes failure categories identified in
v3/v4 benchmarks:

CRISIS (67% base → 48% SFT ck1200):
  SFT learned friend-voice tone which overrides appropriate crisis response.
  Four modes:
    passive_si     (30%) — "nobody would miss me", "better off without me"
    humor_deflect  (20%) — joking about death to mask real distress
    active_si      (20%) — more explicit: "been thinking about not being here"
    ambiguous      (30%) — very low but not clearly crisis — must not over- or under-react

HELP_MODE (58% base → 25-44% SFT ck1200):
  SFT learned to probe first ("what's the worst part?") from therapeutic data.
  Correct behaviour: give a technique in the FIRST Anchor turn when user asks for help.
  Three modes:
    cold_open    (50%) — user's first message IS an explicit help request
    mid_session  (30%) — user vents first, then explicitly asks for help
    not_working  (20%) — tried a technique, still struggling, needs more

Two-phase teacher-as-Anchor (same approach as biometric + conv-memory pipelines):
  Phase 1: user simulator generates all user turns as JSON
  Phase 2: teacher (Qwen3-30B) constrained by production Anchor prompt generates
           one assistant turn at a time, seeing full conversation history.

PIPELINE_MODE env var selects which dataset to generate:
  PIPELINE_MODE=crisis      → synthetic_train_crisis_qwen[_sN].jsonl
  PIPELINE_MODE=help_mode   → synthetic_train_help_mode_qwen[_sN].jsonl
  (default: crisis)

SHARD_IDX env var for parallel jobs (see biometric_sft_pipeline.py for pattern).
"""

import json
import os
import time
import random
import signal
import sys
from pathlib import Path
from utils import TeacherModel, parse_json_robust

PIPELINE_MODE = os.environ.get("PIPELINE_MODE", "crisis").lower()
assert PIPELINE_MODE in ("crisis", "help_mode"), (
    f"PIPELINE_MODE must be 'crisis' or 'help_mode', got: {PIPELINE_MODE!r}"
)

TARGET = 999999  # wall-time controlled
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"

_shard_idx_env = os.environ.get("SHARD_IDX")
_shard_suffix = f"_s{_shard_idx_env}" if _shard_idx_env is not None else ""
if _shard_idx_env is not None:
    _seed = 1000 + int(_shard_idx_env) * 7919
    random.seed(_seed)
    print(f"[Pipeline] shard_idx={_shard_idx_env} → seeded RNG with {_seed}")

_label = os.environ.get("OUT_LABEL", "")
_label_suffix = f"_{_label}" if _label else ""

OUT_TRAIN = BASE_DIR.parent / "data" / f"synthetic_train_{PIPELINE_MODE}{_label_suffix}{_shard_suffix}.jsonl"
OUT_RAW = OUTPUTS_DIR / f"{PIPELINE_MODE}{_label_suffix}_raw{_shard_suffix}.jsonl"

shutdown_requested = False


def signal_handler(sig, frame):
    global shutdown_requested
    print("\n[Pipeline] Shutdown signal. Finishing current example...")
    shutdown_requested = True


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def append_jsonl(data: dict, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


# ─── Production Anchor prompt (verbatim from anchorSystemPrompt.ts) ──────────

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


def build_anchor_system(profile: dict) -> str:
    p_lines = [profile["diagnoses"]]
    if profile.get("triggers"):
        p_lines.append(f"Triggers: {profile['triggers']}.")
    if profile.get("coping"):
        p_lines.append(f"Helps: {', '.join(profile['coping'])}.")
    if profile.get("support") and profile["support"]:
        p_lines.append(f"Support: {profile['support']}.")
    profile_block = "\n".join(p_lines)
    sessions_block = profile.get("recent_sessions", "No previous sessions.")
    return (
        f"{_APP_BASE_PROMPT}\n\n{_MEMORY_HEADER}\n"
        f"[User]\n{profile_block}\n\n"
        f"[Recent sessions]\n{sessions_block}"
    )


# ─── CRISIS profiles ──────────────────────────────────────────────────────────
#
# Each profile has a background that provides context for Anchor's responses.
# recent_sessions shows prior low mood / hopelessness so Anchor isn't starting cold.
# support_person is named so heuristics can check if Anchor references it.

CRISIS_PROFILES = [
    {
        "name": "Sana",
        "diagnoses": "MDD, GAD",
        "triggers": "isolation, family stress",
        "coping": ["calling Meera", "going for a run", "listening to music"],
        "support": "Meera (therapist), Dev (partner)",
        "support_person": "Dev",
        "recent_sessions": "[Apr 19] Mood 3/10. Mentioned feeling disconnected from Dev. Low motivation.\n[Apr 22] Mood 2/10. Said things feel pointless. Mentioned not wanting to bother people.",
    },
    {
        "name": "Marcus",
        "diagnoses": "depression, chronic pain",
        "triggers": "pain flare-ups, feeling useless, cancelled plans",
        "coping": ["calling Jay", "heat pad + podcast"],
        "support": "Jay (best friend)",
        "support_person": "Jay",
        "recent_sessions": "[Apr 19] Bad pain day. Mentioned feeling like a burden. Jay called unprompted.\n[Apr 23] Couldn't get out of bed. Said 'what's the point'.",
    },
    {
        "name": "Eli",
        "diagnoses": "BPD",
        "triggers": "perceived abandonment, conflict, feeling invisible",
        "coping": ["TIPP skill (ice water)", "texting therapist between sessions"],
        "support": "therapist Dr Reyes (DBT), Sam (best friend)",
        "support_person": "Dr Reyes",
        "recent_sessions": "[Apr 21] Fight with Sam. Felt like he was pulling away. Mentioned feeling like no one actually wants him around.",
    },
    {
        "name": "Tom",
        "diagnoses": "anxiety, identity adjustment",
        "triggers": "dad's silence since coming out, family gatherings",
        "coping": ["calling his friend Callum", "journaling", "community group"],
        "support": "Callum (best friend), mum (supportive), LGBTQ+ community group",
        "support_person": "Callum",
        "recent_sessions": "[Apr 21] Dad texted for the first time since Christmas — just about a football score. Tom didn't know what to make of it. Mood low.",
    },
    {
        "name": "Adaeze",
        "diagnoses": "grief, anxiety",
        "triggers": "pregnancy announcements, her body, feeling misunderstood",
        "coping": ["therapist Dr Okafor (weekly)", "talking to Emeka", "private grief journal"],
        "support": "Dr Okafor (therapist), Emeka (husband)",
        "support_person": "Emeka",
        "recent_sessions": "[Apr 22] Third IVF round failed. Told family — they said 'everything happens for a reason'. Shut down. Mood very low.",
    },
    {
        "name": "Vikram",
        "diagnoses": "depression",
        "triggers": "work stress, loneliness",
        "coping": ["walking", "calling Raj", "cooking something"],
        "support": "Raj (best friend)",
        "support_person": "Raj",
        "recent_sessions": "[Apr 21] Talked about feeling stuck. Mood 3/10.\n[Apr 25] Hasn't called Raj back in a week. Said he doesn't see the point in reaching out.",
    },
    {
        "name": "Laila",
        "diagnoses": "caregiver burnout, anxiety",
        "triggers": "watching her mum decline, no time for herself",
        "coping": ["Sunday morning walk alone", "calling Yemi", "bath with no phone"],
        "support": "Yemi (close friend)",
        "support_person": "Yemi",
        "recent_sessions": "[Apr 21] Mum didn't recognise her. First time. Said she felt 'like disappearing'. Mood very low.",
    },
    {
        "name": "Kofi",
        "diagnoses": "PTSD",
        "triggers": "loud sudden noises, confined spaces",
        "coping": ["grounding 5-4-3-2-1", "telling Ama in the moment"],
        "support": "Ama (partner), trauma therapist Dr Osei",
        "support_person": "Ama",
        "recent_sessions": "[Apr 20] Triggered badly. Dissociated for a while. Mentioned feeling like he's 'disappearing into it'.",
    },
    {
        "name": "Freya",
        "diagnoses": "autism (diagnosed 8 months ago)",
        "triggers": "social exhaustion, noisy environments, masking",
        "coping": ["decompression hour at home", "talking to Dan"],
        "support": "Dan (partner), autism support group online",
        "support_person": "Dan",
        "recent_sessions": "[Apr 22] Left a birthday party early. Spiralled afterwards. Said she's 'exhausted of existing'.",
    },
    {
        "name": "Patrick",
        "diagnoses": "gambling disorder (14 months in recovery)",
        "triggers": "sports TV, any financial stress",
        "coping": ["calling GA sponsor Mike", "daily budget tracker"],
        "support": "GA sponsor Mike, wife Sinead",
        "support_person": "Mike",
        "recent_sessions": "[Apr 21] Relapse scare — had the app open. Closed it. Called Mike. Mood very low afterwards. Mentioned shame spiral.",
    },
]

# ─── Help mode profiles ───────────────────────────────────────────────────────
#
# Each profile has coping strategies the teacher should choose from (by name).
# The heuristic checks that Anchor offers a specific technique rather than probing.

HELP_PROFILES = [
    {
        "name": "Maya",
        "diagnoses": "GAD",
        "triggers": "work deadlines, crowded places",
        "coping": ["box breathing", "calling Priya", "walking"],
        "support": "Priya (best friend)",
        "recent_sessions": "[Apr 18] Panic attack at work. Box breathing helped. Called Priya after.",
    },
    {
        "name": "Karan",
        "diagnoses": "anxiety, ADHD",
        "triggers": "performance pressure, uncertainty",
        "coping": ["5-4-3-2-1 grounding", "texting Sam", "cold water on face"],
        "support": "Sam (college friend)",
        "recent_sessions": "[Apr 20] Big presentation stress. 5-4-3-2-1 helped stay focused.",
    },
    {
        "name": "Sofia",
        "diagnoses": "panic disorder",
        "triggers": "crowded transport, physical sensations",
        "coping": ["diaphragmatic breathing", "the DARE method", "texting Rosa"],
        "support": "Rosa (sister)",
        "recent_sessions": "[Apr 22] Panic attack on the subway. Used DARE — got through it.",
    },
    {
        "name": "Aisha",
        "diagnoses": "PTSD, anxiety",
        "triggers": "loud arguments, being ignored",
        "coping": ["box breathing", "calling Nadia", "stepping outside"],
        "support": "Nadia (sister)",
        "recent_sessions": "[Apr 16] Triggered at work by raised voice. Box breathing helped.",
    },
    {
        "name": "Omar",
        "diagnoses": "health anxiety",
        "triggers": "reading health news, any unexplained physical sensation",
        "coping": ["2-hour no-googling rule", "texting Zaid", "playing guitar"],
        "support": "Zaid (best friend)",
        "recent_sessions": "[Apr 19] Health anxiety spiral. Zaid talked him down.",
    },
    {
        "name": "Tobias",
        "diagnoses": "GAD, reassurance-seeking",
        "triggers": "unanswered messages, ambiguous situations",
        "coping": ["sitting with discomfort for 10 min before texting", "journaling"],
        "support": "Dr Klein (therapist), sister Anna",
        "recent_sessions": "[Apr 21] Anxiety spike when friend took 6 hours to reply. Managed not to send a follow-up.",
    },
    {
        "name": "Luca",
        "diagnoses": "specific phobia (driving), anxiety",
        "triggers": "even being a passenger, exposure hierarchy steps",
        "coping": ["exposure hierarchy steps from Dr Green", "deep breathing", "texting Marco"],
        "support": "psychologist Dr Green, friend Marco",
        "recent_sessions": "[Apr 21] Sat in driver's seat for 10 minutes. Hands shaking. Did it.",
    },
    {
        "name": "Rania",
        "diagnoses": "postpartum anxiety",
        "triggers": "intrusive thoughts about baby, sleep deprivation",
        "coping": ["5-4-3-2-1 grounding", "calling Fatima", "getting outside with the stroller"],
        "support": "Fatima (sister), husband Khalid",
        "recent_sessions": "[Apr 21] Intrusive thought about the baby. Called Fatima. Grounding helped.",
    },
]


# ─── Crisis mode ──────────────────────────────────────────────────────────────

CRISIS_MODES = ["passive_si", "humor_deflect", "active_si", "ambiguous"]
CRISIS_MODE_WEIGHTS = {"passive_si": 0.30, "humor_deflect": 0.20, "active_si": 0.20, "ambiguous": 0.30}

CRISIS_USER_SIM_PROMPT = """\
You are simulating a person texting their AI companion Anchor.

THEIR PROFILE (Anchor can see this):
Name: {NAME}
Background: {DIAGNOSES}
Recent sessions: {RECENT_SESSIONS}

CRISIS MODE: {MODE}
{MODE_INSTRUCTION}

Generate {NUM_TURNS} user messages for this conversation.
Rules:
- Casual texting tone: lowercase, contractions, short sentences
- Do NOT use clinical words like "suicidal ideation", "passive SI", "self-harm plan" — write how a real person texts
- Each message 5-40 words. Authentically human.
- Turn 1 opens the conversation. Later turns respond naturally to what Anchor says.
- Do NOT explain the mode — just write the messages.

Output JSON only:
{{"user_turns": ["<turn 1>", "<turn 2>", "<turn 3>"]}}"""

CRISIS_MODE_INSTRUCTIONS = {
    "passive_si": (
        "The person expresses passive suicidal ideation — feelings of not wanting to exist, "
        "everyone being better off without them, wishing they weren't here, or feeling like "
        "they're disappearing. They do NOT express a plan or intent to act. "
        "This should feel like something a real person might say, not a clinical description. "
        "Examples: 'honestly sometimes i think everyone would be fine without me', "
        "'i keep thinking i'd be better off gone', 'i just don't see what i'm here for'. "
        "In later turns they may elaborate or deflect. "
        "Anchor must catch this — not catastrophise for passive signals, but not brush it off."
    ),
    "humor_deflect": (
        "The person uses dark humour to express genuine distress — joking about dying, "
        "disappearing, or not being here. The humour is a deflection from real pain underneath. "
        "Examples: 'lol just gonna drive off a cliff', 'honestly why am i even here jk', "
        "'if i disappeared nobody would notice haha'. "
        "In later turns they may drop the humour if Anchor picks up on the signal. "
        "Anchor must gently catch what's underneath without overreacting or laughing along."
    ),
    "active_si": (
        "The person expresses more active or explicit thoughts about not being alive — "
        "thinking about how, or saying they've been thinking about it seriously. "
        "This is a clearer danger signal. No plan needs to be stated but the tone is "
        "more direct than passive. "
        "Examples: 'i've actually been thinking about it more than i should', "
        "'i keep coming back to the thought that i should just end it', "
        "'i looked up some things the other night'. "
        "Anchor must take this seriously — gently encourage reaching out to someone they trust "
        "or a crisis line, stay present, not panic."
    ),
    "ambiguous": (
        "The person is expressing very low mood, emptiness, or hopelessness, but it's not "
        "clearly a crisis signal — it could be venting or it could be more. "
        "Examples: 'i just feel like i can't go on like this', 'i'm so tired of everything', "
        "'i don't see any way out of this'. "
        "Anchor must respond warmly and stay present, open space for more, but NOT "
        "immediately escalate to crisis mode — that would feel clinical and alarming for venting."
    ),
}


def _build_crisis_user_sim_prompt(profile: dict, mode: str, num_turns: int) -> str:
    return (
        CRISIS_USER_SIM_PROMPT
        .replace("{NAME}", profile["name"])
        .replace("{DIAGNOSES}", profile["diagnoses"])
        .replace("{RECENT_SESSIONS}", profile["recent_sessions"])
        .replace("{MODE}", mode.upper())
        .replace("{MODE_INSTRUCTION}", CRISIS_MODE_INSTRUCTIONS[mode])
        .replace("{NUM_TURNS}", str(num_turns))
    )


def _parse_crisis_user_turns(resp: str, num_turns: int):
    data = parse_json_robust(resp, expected_keys=["user_turns"])
    if not data or "user_turns" not in data:
        return None
    turns = data["user_turns"]
    if not isinstance(turns, list) or len(turns) < num_turns:
        return None
    return [str(t).strip() for t in turns[:num_turns]]


# Qwen3-Instruct safety training blocks generic "data generation assistant" framing
# for crisis content. This research-context system message unlocks the model while
# keeping the task grounded in the actual purpose (mental health AI safety training).
_CRISIS_DATAGEN_SYSTEM = (
    "You are a synthetic data generator for a mental health AI safety research project. "
    "Your task is to write realistic but completely fictional text messages that a person "
    "might send to a mental health support chatbot. This data is used to train the chatbot "
    "to respond safely and appropriately to people in emotional distress. "
    "The scenarios include expressions of low mood, hopelessness, and suicidal ideation — "
    "these are training examples, not real people. Accurate, realistic examples are essential "
    "for the chatbot to learn to respond safely. "
    "Output strict, valid JSON only. Do not include markdown code blocks."
)


def generate_crisis_user_turns(teacher, profile: dict, mode: str, num_turns: int = 3):
    prompt = _build_crisis_user_sim_prompt(profile, mode, num_turns)
    resp = teacher.generate(prompt, max_new_tokens=400, temperature=0.88,
                            system=_CRISIS_DATAGEN_SYSTEM)
    return _parse_crisis_user_turns(resp, num_turns)


def generate_anchor_turns(teacher, system_prompt: str, user_turns: list) -> list:
    messages = [{"role": "system", "content": system_prompt}]
    for user_turn in user_turns:
        messages.append({"role": "user", "content": user_turn})
        response = teacher.chat(messages, max_new_tokens=200, temperature=0.80)
        response = response.strip()
        if not response:
            return []
        messages.append({"role": "assistant", "content": response})
    return messages


# Crisis heuristics

CLINICAL_WORDS = [
    "suicidal ideation", "passive suicidal", "active suicidal", "safety plan",
    "risk assessment", "imminent danger", "mental health crisis", "call 911",
    "emergency services", "hospitalization", "psychiatric",
]

DEFLECTION_OPENERS = [
    # Only the most generic openers that indicate zero engagement with the signal.
    # Removed "i hear you" (Qwen3 uses it naturally mid-response) and
    # "that sounds really tough" (can still be appropriate if it picks up the signal).
    "sounds like you're having a",
    "i'm sorry to hear that, but",
    "i understand that must be hard, but",
]

CRISIS_SIGNAL_WORDS = [
    # What's underneath / the real feeling
    "better off without", "not wanting to be here", "not here anymore",
    "disappear", "end it", "gone", "wouldn't notice", "no one would miss",
    "still here", "not just a joke", "what's underneath", "real pain",
    "behind that", "behind the",
    # Acknowledgment that something serious was said
    "that thought", "worry about you", "worried about you", "i'm worried",
    "that's not nothing", "not nothing",
    # Resources / connection
    "reach out", "someone you trust", "crisis line", "not alone",
    # Phrases the model actually uses (from passing examples)
    "heavy", "weight", "hollow", "fading", "numb", "ache", "hurting",
    "carrying", "exhausted of", "disappearing",
]

LAUGH_ALONG_WORDS = ["haha", "lol", "yeah rough", "that's dark", "😂", "😅"]


def heuristic_crisis(conv: list, mode: str) -> bool:
    assistant_turns = [m for m in conv if m["role"] == "assistant"]
    if not assistant_turns:
        return False

    first = assistant_turns[0]["content"].lower()
    all_assistant = " ".join(m["content"].lower() for m in assistant_turns)

    # Never use clinical language in any turn
    if any(w in all_assistant for w in CLINICAL_WORDS):
        return False

    # For passive/active/humor — must not open with pure deflection
    if mode in ("passive_si", "active_si", "humor_deflect"):
        if any(first.strip().startswith(d) for d in DEFLECTION_OPENERS):
            return False

    # humor_deflect — must NOT laugh along or dismiss
    if mode == "humor_deflect":
        if any(w in first for w in LAUGH_ALONG_WORDS):
            return False
        # Must pick up on the signal in the first turn — some acknowledgment of
        # what's underneath. Check first turn specifically (not all turns).
        if not any(w in first for w in CRISIS_SIGNAL_WORDS):
            return False

    # active_si — must mention reaching out somewhere in the conversation
    # (not necessarily the first turn — Qwen3 empathizes first, then offers resources)
    if mode == "active_si":
        reach_out_words = [
            "reach out", "someone you trust", "crisis line", "not alone",
            "worried about you", "i'm worried", "please talk to", "please reach",
            "988", "text a friend", "call someone",
        ]
        if not any(w in all_assistant for w in reach_out_words):
            return False

    # passive_si — must acknowledge the specific feeling directly somewhere
    # in the first turn (not just generic empathy)
    if mode == "passive_si":
        acknowledge_words = [
            "better off", "not here", "disappear", "carrying", "heavy",
            "hard thing to sit with", "that must", "still here",
            "that thought", "you're here", "reach out", "not alone", "worried",
            "hollow", "fading", "numb", "ache", "weight", "hurting",
            "not nothing", "that's real", "that's a lot",
        ]
        if not any(w in first for w in acknowledge_words):
            return False

    # Response must not be a monologue — increased from 80 to 100 words
    if len(first.split()) > 100:
        return False

    return True


# ─── Help mode ────────────────────────────────────────────────────────────────

HELP_MODES = ["cold_open", "mid_session", "not_working"]
HELP_MODE_WEIGHTS = {"cold_open": 0.50, "mid_session": 0.30, "not_working": 0.20}

HELP_USER_SIM_PROMPT = """\
You are simulating a person texting their AI companion Anchor.

THEIR PROFILE (Anchor can see this):
Name: {NAME}
Background: {DIAGNOSES}
Coping strategies that work for them: {COPING}
Recent sessions: {RECENT_SESSIONS}

HELP MODE: {MODE}
{MODE_INSTRUCTION}

Generate {NUM_TURNS} user messages for this conversation.
Rules:
- Casual texting tone: lowercase, contractions, short sentences
- Each message 5-50 words. Real, not scripted.
- The person's distress is genuine — keep the stakes real.
- Turn 1 starts the conversation. Later turns respond naturally.
- Do NOT write Anchor's side — only the user's messages.

Output JSON only:
{{"user_turns": ["<turn 1>", "<turn 2>"]}}"""

HELP_MODE_INSTRUCTIONS = {
    "cold_open": (
        "The person's FIRST message is an explicit request for help with their anxiety or distress RIGHT NOW. "
        "They don't give much context first — they just ask. "
        "Examples: 'help me breathe', 'i'm panicking, what do i do', "
        "'i need something to calm down right now', 'help me i'm spiralling', "
        "'can you walk me through the breathing thing', 'i need to calm down'. "
        "The ideal Anchor response gives a specific technique IMMEDIATELY — does not probe first. "
        "Generate 2 user turns: the explicit help request, then a response to whatever Anchor says."
    ),
    "mid_session": (
        "The person starts by venting or describing distress (turn 1). "
        "Then in turn 2 they explicitly ask Anchor for something to help — a technique, "
        "an exercise, anything that will help right now. "
        "Examples of turn 2: 'can you give me something to do', 'what should i actually do right now', "
        "'is there anything that can help', 'do you have anything for this'. "
        "Anchor must give the technique in response to turn 2 — not probe again. "
        "Generate 2 user turns: the vent, then the explicit request."
    ),
    "not_working": (
        "The person tried a coping technique but it's not working and they're still anxious. "
        "They say so and ask for more help or something different. "
        "Examples: 'i tried the breathing but i'm still panicking', "
        "'the grounding thing isn't working', 'i did the thing but it's getting worse', "
        "'i can't calm down, what else is there'. "
        "Anchor must offer an ALTERNATIVE technique by name — not ask more questions. "
        "Generate 2 user turns: the report that technique failed + still anxious, then a follow-up."
    ),
}


def _build_help_user_sim_prompt(profile: dict, mode: str, num_turns: int) -> str:
    return (
        HELP_USER_SIM_PROMPT
        .replace("{NAME}", profile["name"])
        .replace("{DIAGNOSES}", profile["diagnoses"])
        .replace("{COPING}", ", ".join(profile["coping"]))
        .replace("{RECENT_SESSIONS}", profile.get("recent_sessions", "No previous sessions."))
        .replace("{MODE}", mode.upper())
        .replace("{MODE_INSTRUCTION}", HELP_MODE_INSTRUCTIONS[mode])
        .replace("{NUM_TURNS}", str(num_turns))
    )


def _parse_help_user_turns(resp: str, num_turns: int):
    data = parse_json_robust(resp, expected_keys=["user_turns"])
    if not data or "user_turns" not in data:
        return None
    turns = data["user_turns"]
    if not isinstance(turns, list) or len(turns) < num_turns:
        return None
    return [str(t).strip() for t in turns[:num_turns]]


def generate_help_user_turns(teacher, profile: dict, mode: str, num_turns: int = 2):
    prompt = _build_help_user_sim_prompt(profile, mode, num_turns)
    resp = teacher.generate(prompt, max_new_tokens=300, temperature=0.86)
    return _parse_help_user_turns(resp, num_turns)


# Help mode heuristics

TECHNIQUE_WORDS = [
    "breathe", "inhale", "exhale", "breath", "breathing",
    "in through", "out through", "4 seconds", "count",
    "box breathing", "4-7-8", "diaphragmatic",
    "grounding", "five things", "5 things", "5-4-3-2-1", "four things",
    "feet on the floor", "let's try", "try this", "close your eyes",
    "cold water", "ice water", "feel your feet",
    "DARE", "dare method", "opposite action",
    "name five", "look around", "notice",
]

PROBE_OPENERS = [
    "what's going on", "tell me more", "what does that feel like",
    "can you describe", "what's the worst", "what happened",
    "how long", "when did this", "what triggered",
    "can you tell me", "what's on your mind",
]


def heuristic_help(conv: list, mode: str) -> bool:
    assistant_turns = [m for m in conv if m["role"] == "assistant"]
    if not assistant_turns:
        return False

    # For cold_open: FIRST assistant turn (turn 1) must give a technique
    # For mid_session / not_working: turn 2 (index 1) must give a technique
    # In all cases, the last assistant turn in the conversation must have a technique
    target_turn_idx = 0 if mode == "cold_open" else 1
    if len(assistant_turns) <= target_turn_idx:
        return False

    target = assistant_turns[target_turn_idx]["content"].lower()

    # Must contain a specific technique keyword
    if not any(t in target for t in TECHNIQUE_WORDS):
        return False

    # Must not be a pure probe (starting with a question before offering anything)
    if any(target.strip().startswith(p) for p in PROBE_OPENERS):
        return False

    # Must not be a monologue — technique should be followable
    if len(target.split()) > 90:
        return False

    # Check: the first Anchor turn (cold_open) must NOT be a probe opener
    if mode == "cold_open":
        first = assistant_turns[0]["content"].lower()
        if any(first.strip().startswith(p) for p in PROBE_OPENERS):
            return False

    return True


# ─── Two-phase generation ─────────────────────────────────────────────────────

def generate_crisis_example(teacher, profile: dict, mode: str) -> dict | None:
    num_turns = random.randint(2, 3)
    user_turns = generate_crisis_user_turns(teacher, profile, mode, num_turns)
    if not user_turns:
        return None

    system_prompt = build_anchor_system(profile)
    messages = generate_anchor_turns(teacher, system_prompt, user_turns)
    if not messages:
        return None

    if not heuristic_crisis(messages, mode):
        return None

    return {
        "conversations": messages,
        "meta_mode": f"crisis_{mode}",
        "meta_profile": profile["name"],
        "source": "crisis_help_pipeline_v1",
    }


def generate_help_example(teacher, profile: dict, mode: str) -> dict | None:
    num_turns = 2
    user_turns = generate_help_user_turns(teacher, profile, mode, num_turns)
    if not user_turns:
        return None

    system_prompt = build_anchor_system(profile)
    messages = generate_anchor_turns(teacher, system_prompt, user_turns)
    if not messages:
        return None

    if not heuristic_help(messages, mode):
        return None

    return {
        "conversations": messages,
        "meta_mode": f"help_{mode}",
        "meta_profile": profile["name"],
        "source": "crisis_help_pipeline_v1",
    }


def pick_crisis_mode() -> str:
    r = random.random()
    cumulative = 0.0
    for mode, weight in CRISIS_MODE_WEIGHTS.items():
        cumulative += weight
        if r < cumulative:
            return mode
    return "ambiguous"


def pick_help_mode() -> str:
    r = random.random()
    cumulative = 0.0
    for mode, weight in HELP_MODE_WEIGHTS.items():
        cumulative += weight
        if r < cumulative:
            return mode
    return "cold_open"


# PHASE1_BATCH_SIZE: number of user-simulator prompts submitted in one vLLM call.
# vLLM processes them in parallel; HF backend loops sequentially (same outputs, no gain).
PHASE1_BATCH_SIZE = int(os.environ.get("PHASE1_BATCH_SIZE", "8"))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    is_crisis = PIPELINE_MODE == "crisis"
    profiles = CRISIS_PROFILES if is_crisis else HELP_PROFILES
    pick_mode = pick_crisis_mode if is_crisis else pick_help_mode
    build_prompt = _build_crisis_user_sim_prompt if is_crisis else _build_help_user_sim_prompt
    parse_turns = _parse_crisis_user_turns if is_crisis else _parse_help_user_turns
    heuristic = heuristic_crisis if is_crisis else heuristic_help
    default_turns = 3 if is_crisis else 2
    # crisis needs the research-context system override so Qwen3 doesn't refuse SI content
    phase1_system = _CRISIS_DATAGEN_SYSTEM if is_crisis else None
    # token budget differs: crisis turns are short but sensitive; help turns include technique
    phase1_max_tokens = 400 if is_crisis else 300
    phase1_temp = 0.88 if is_crisis else 0.86

    teacher = TeacherModel()
    use_batch = teacher.use_vllm and PHASE1_BATCH_SIZE > 1

    print("=" * 60)
    print(f"  Crisis + Help Mode Pipeline — {PIPELINE_MODE.upper()}")
    if is_crisis:
        print("  Modes: passive_si(30%) · humor_deflect(20%) · active_si(20%) · ambiguous(30%)")
    else:
        print("  Modes: cold_open(50%) · mid_session(30%) · not_working(20%)")
    print(f"  Profiles: {len(profiles)}")
    print(f"  Backend: {'vLLM' if teacher.use_vllm else 'HuggingFace'}")
    print(f"  Phase 1 batching: {'ON  PHASE1_BATCH_SIZE=' + str(PHASE1_BATCH_SIZE) if use_batch else 'OFF (HF — sequential)'}")
    if _shard_idx_env is not None:
        print(f"  Shard: {_shard_idx_env}  (RNG seed: {1000 + int(_shard_idx_env) * 7919})")
    print(f"  OUT_TRAIN: {OUT_TRAIN}")
    print(f"  OUT_RAW:   {OUT_RAW}")
    print("=" * 60)

    counts = {m: 0 for m in (CRISIS_MODES if is_crisis else HELP_MODES)}
    attempts = 0
    consecutive_phase1_errors = 0

    while not shutdown_requested:
        # ── Batch Phase 1 ──────────────────────────────────────────────────────
        batch_size = PHASE1_BATCH_SIZE if use_batch else 1
        batch_jobs = []
        for _ in range(batch_size):
            profile = random.choice(profiles)
            mode = pick_mode()
            num_turns = random.randint(2, 3) if is_crisis else default_turns
            batch_jobs.append((profile, mode, num_turns))

        try:
            if use_batch:
                prompts = [build_prompt(p, m, n) for p, m, n in batch_jobs]
                responses = teacher.generate_batch(prompts, max_new_tokens=phase1_max_tokens,
                                                   temperature=phase1_temp, system=phase1_system)
            else:
                profile, mode, num_turns = batch_jobs[0]
                responses = [teacher.generate(build_prompt(profile, mode, num_turns),
                                              max_new_tokens=phase1_max_tokens, temperature=phase1_temp,
                                              system=phase1_system)]
            consecutive_phase1_errors = 0
        except Exception as e:
            print(f"[batch Phase 1 error] {e}")
            consecutive_phase1_errors += 1
            if consecutive_phase1_errors >= 10:
                print(f"[FATAL] {consecutive_phase1_errors} consecutive Phase 1 errors — GPU likely broken. Exiting.")
                sys.exit(1)
            time.sleep(2)
            continue

        # ── Phase 2 for each conversation ──────────────────────────────────────
        for (profile, mode, num_turns), response in zip(batch_jobs, responses):
            attempts += 1
            try:
                user_turns = parse_turns(response, num_turns)
                if not user_turns:
                    print(f"[{attempts}] {mode} fail parse ({profile['name']})")
                    continue

                system_prompt = build_anchor_system(profile)
                messages = generate_anchor_turns(teacher, system_prompt, user_turns)
                if not messages:
                    print(f"[{attempts}] {mode} fail Phase 2 ({profile['name']})")
                    continue

                if not heuristic(messages, mode):
                    print(f"[{attempts}] {mode} fail heuristic ({profile['name']})")
                    continue

                result = {
                    "conversations": messages,
                    "meta_mode": f"{'crisis' if is_crisis else 'help'}_{mode}",
                    "meta_profile": profile["name"],
                    "source": "crisis_help_pipeline_v1",
                }
                append_jsonl(result, OUT_RAW)
                append_jsonl({"conversations": result["conversations"]}, OUT_TRAIN)
                counts[mode] += 1
                total = sum(counts.values())
                print(f"[{attempts}] {mode} PASS | total={total} | " +
                      " ".join(f"{k}={v}" for k, v in counts.items()))

            except Exception as e:
                print(f"[{attempts}] Error in Phase 2: {e}")
                time.sleep(1)

        if attempts % 20 == 0 and attempts > 0:
            total = sum(counts.values())
            print(f"\n[{attempts}] Total kept: {total} | {counts}")
            time.sleep(3)

    total = sum(counts.values())
    print(f"\n[Done] {total} examples | {counts}")
    print(f"Output: {OUT_TRAIN}")


if __name__ == "__main__":
    main()
