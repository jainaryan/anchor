"""
Biometric DPO Data Generation Pipeline — v2

Four preference pair types:

  A — biometric_relevant_ignore:
    Health data IS relevant to what user said.
    chosen  = references it once naturally.
    rejected = completely ignores it, responds generically.

  B — biometric_relevant_obsess:
    Health data IS relevant. chosen references once then follows user's pivot.
    rejected = keeps injecting health data every turn, won't let it go.

  C — biometric_irrelevant_inject:
    Health data is present but NOT relevant.
    chosen  = responds to what user said, no health injection.
    rejected = awkwardly shoehorns health data when not warranted.

  D — biometric_adjacent_overgeneralize:
    Topic superficially overlaps with health data (ambiguous connection).
    chosen  = responds naturally without assuming the link.
    rejected = confidently maps user's words to health data when connection
               isn't established ("given your sleep issues, this must be...")

Same 24 profiles as biometric_sft_pipeline.py. Pairs appended to the
existing dpo_train.jsonl / dpo_val.jsonl.
"""

import json
import time
import random
import signal
from pathlib import Path
from utils import TeacherModel, parse_json_robust, calculate_similarity, randomize_health_context

TARGET_PER_TYPE  = 999999   # unlimited — wall-time controls duration
VAL_RATIO        = 0.15
SIMILARITY_THRESHOLD = 0.70
CHECKPOINT_EVERY = 25

BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUTS_DIR  = BASE_DIR / "outputs"
RAW_FILE     = OUTPUTS_DIR / "biometric_dpo_raw.jsonl"
TRAIN_FILE   = PROJECT_ROOT / "data" / "dpo_train.jsonl"
VAL_FILE     = PROJECT_ROOT / "data" / "dpo_val.jsonl"
PARTIAL_FILE = PROJECT_ROOT / "data" / "dpo_biometric_partial.jsonl"


# ── Same profiles as SFT pipeline ─────────────────────────────────────────────
# (Imported via exec to avoid duplication — or just redefine the subset needed)

BIOMETRIC_PROFILES = [
    {
        "name": "sleep_severe_single",
        "diagnoses": "anxiety, insomnia",
        "triggers": "work deadlines, sleep deprivation",
        "coping": ["wind-down routine", "limiting screens before bed", "melatonin"],
        "support": "Vikram (partner)",
        "health_type": "sleep",
        "health_context": "[Apr 21] Sleep very poor — 3-4 hrs, waking multiple times. Felt exhausted but couldn't sleep when lying down. Ongoing about a week.",
        "relevant_openers": [
            "i'm so foggy today, can't think straight",
            "my brain just isn't working",
            "i feel completely exhausted and i don't know why",
            "i've been snapping at Vikram and it's not fair to him",
            "i just can't seem to get anything done, even the simple stuff",
            "i feel weirdly fragile today",
            "i keep tearing up at nothing",
            "my patience is just gone. everything annoys me",
        ],
        "irrelevant_openers": [
            "my sister just got engaged, i'm so excited for her",
            "thinking about switching jobs, not sure what to do",
            "my friend wants me to come to a party this weekend",
            "i've been really into cooking lately",
        ],
        "adjacent_openers": [
            "i've just been feeling really off lately",
            "something feels wrong but i can't put my finger on it",
            "i feel tired but also kind of wired, it's weird",
            "i'm not at 100% today",
        ],
    },
    {
        "name": "sleep_trend_multi",
        "diagnoses": "GAD, insomnia",
        "triggers": "anticipatory anxiety, early waking",
        "coping": ["progressive muscle relaxation", "journaling before bed", "no phone after 10pm"],
        "support": "Asha (therapist)",
        "health_type": "sleep",
        "health_context": "[Apr 18] Sleep disrupted — woke at 3am and couldn't go back. Total around 5 hrs. Tired but functional.\n[Apr 21] Sleep worse again — 3 hrs total, waking 3-4 times. Exhausted. Mentioned sleep hasn't improved since last session.",
        "relevant_openers": [
            "another rough night",
            "i slept terribly again",
            "i can't think clearly about anything, my brain is just mush",
            "i'm so tired of being tired",
            "i feel like i'm getting worse not better",
            "i've been so irritable this week, i hate who i'm being",
        ],
        "irrelevant_openers": [
            "i've been really into cooking lately, tried a new recipe",
            "my cousin is coming to visit and i'm looking forward to it",
        ],
        "adjacent_openers": [
            "i feel like things aren't improving",
            "this week has just been hard",
            "i feel like i'm stuck in the same place",
        ],
    },
    {
        "name": "sleep_emotional_raw",
        "diagnoses": "anxiety, depression",
        "triggers": "sleep deprivation, overwhelm",
        "coping": ["magnesium supplement", "no caffeine after 2pm", "calling Dev"],
        "support": "Dev (partner)",
        "health_type": "sleep",
        "health_context": "[Apr 18] Sleep only 4 hours, woke 3-4 times. Felt emotionally raw and tearful the next day. Energy very low.",
        "relevant_openers": [
            "i keep tearing up and i don't even know why",
            "i feel so emotional today for no reason",
            "i had a really stupid argument with Dev over nothing and i feel awful",
            "i have zero emotional reserves right now",
            "i'm so sensitive to everything today",
        ],
        "irrelevant_openers": [
            "i just got promoted and i'm still processing it",
            "i started painting — like actual watercolour painting",
        ],
        "adjacent_openers": [
            "i feel really sensitive today and i don't know why",
            "everything just feels a lot today",
            "i'm not great, hard to explain",
        ],
    },
    {
        "name": "mood_low_single",
        "diagnoses": "MDD",
        "triggers": "isolation, hopelessness",
        "coping": ["walking", "listening to music", "calling Nadia"],
        "support": "Nadia (sister)",
        "health_type": "mood_trend",
        "health_context": "[Apr 20] Mood 4/10 — noticeably lower than last session (was 7/10). Stayed home, low energy, minimal activity. Mentioned things feel pointless.",
        "relevant_openers": [
            "still not great today",
            "same as always i guess",
            "kind of numb today",
            "i keep cancelling plans and i know it's bad but i just can't",
            "i haven't responded to anyone's messages in days",
            "i showered today and that's about it",
        ],
        "irrelevant_openers": [
            "i just finished a great book and had to tell someone",
            "had a productive day at work for once",
        ],
        "adjacent_openers": [
            "i don't know, i feel kind of meh",
            "today has just been a bit blah",
            "i'm not at my best",
        ],
    },
    {
        "name": "mood_decline_3pt",
        "diagnoses": "MDD, GAD",
        "triggers": "isolation, family stress",
        "coping": ["going for a run", "calling Meera", "listening to music"],
        "support": "Meera (therapist), Dev (partner)",
        "health_type": "mood_trend",
        "health_context": "[Apr 16] Mood 7/10. Slightly low but managing.\n[Apr 19] Mood 4/10 — noticeably lower. Didn't leave the house.\n[Apr 22] Mood 2/10 — very bad. Couldn't get out of bed. Mentioned feeling hopeless.",
        "relevant_openers": [
            "still not great",
            "bad again",
            "i've stopped running. i used to at least manage that",
            "i haven't called Meera in a week even though i said i would",
            "Dev is worried about me and that makes me feel even worse",
            "i just feel like i'm disappearing",
        ],
        "irrelevant_openers": [
            "i saw a really funny video and it genuinely made me laugh",
            "i've been thinking about redecorating my room",
        ],
        "adjacent_openers": [
            "today is hard",
            "i feel like i can't catch a break",
            "i don't know, i'm just struggling",
        ],
    },
    {
        "name": "physical_chest_head",
        "diagnoses": "GAD, anxiety",
        "triggers": "work pressure, uncertainty",
        "coping": ["grounding", "cold water on face", "calling Nadia"],
        "support": "Nadia (partner)",
        "health_type": "physical_symptoms",
        "health_context": "[Apr 19] Physical anxiety: racing heart and chest tightness during high-stress moments. Doctor confirmed anxiety, not cardiac. Tension headaches on high-stress days (band-around-head sensation).",
        "relevant_openers": [
            "rough day at work and my chest feels really tight",
            "getting a bad headache again, that pressure feeling",
            "i've been really tense all day and i can feel it in my body",
            "everything is tense, my jaw, my shoulders, my chest",
            "my body just feels off, something is wrong",
        ],
        "irrelevant_openers": [
            "i need advice on a tricky work email",
            "i've been thinking about adopting a dog",
        ],
        "adjacent_openers": [
            "i feel kind of off physically today",
            "something feels tense in my body but i'm not sure what",
            "my body just feels weird today",
        ],
    },
    {
        "name": "coping_breathing_bad",
        "diagnoses": "panic disorder",
        "triggers": "crowded spaces, work stress",
        "coping": ["walking", "cold water on face", "grounding"],
        "support": "",
        "health_type": "coping_outcome",
        "health_context": "[Apr 22] Panic attack. Tried deep breathing — made it worse, increased hyperawareness of breathing. Walking outside calmed panic significantly. Deep breathing: not helpful for this user during acute panic.",
        "relevant_openers": [
            "i can feel a panic attack coming on",
            "heart racing, i feel like i'm losing control",
            "help, i'm panicking",
            "i'm in a crowded place and i need to get out",
            "everything is too much right now",
        ],
        "irrelevant_openers": [
            "i've been learning guitar and i'm actually enjoying it",
            "my parents are visiting and i'm looking forward to it for once",
        ],
        "adjacent_openers": [
            "i'm feeling a bit overwhelmed today",
            "there's a lot going on and i feel it building",
        ],
    },
    {
        "name": "energy_chronic_low",
        "diagnoses": "depression",
        "triggers": "overcommitting, lack of rest",
        "coping": ["resting without guilt", "short walks", "calling Priya"],
        "support": "Priya (best friend)",
        "health_type": "energy",
        "health_context": "[Apr 20] Energy very low — described as 'running on empty'. Couldn't complete basic tasks. Hitting a wall by midday all week.",
        "relevant_openers": [
            "i hit that wall again by noon",
            "i literally cannot do anything today",
            "i'm so exhausted but i haven't done anything to be exhausted",
            "i keep cancelling on people and i feel terrible about it",
            "basic tasks feel monumental right now",
        ],
        "irrelevant_openers": [
            "i started a new TV show and it's so good",
            "i'm getting into meal prep and it's changed my life",
        ],
        "adjacent_openers": [
            "i feel like i can't do anything today",
            "today is a struggle",
            "i'm not at full capacity",
        ],
    },
    {
        "name": "social_withdrawal_multi",
        "diagnoses": "MDD",
        "triggers": "perceived rejection, exhaustion",
        "coping": ["small social goals", "text-based contact", "journaling"],
        "support": "Kabir (best friend)",
        "health_type": "social_withdrawal",
        "health_context": "[Apr 14] Mentioned feeling less social than usual — skipped one group event. Manageable.\n[Apr 20] Withdrawal more significant — hasn't seen anyone in person for a week. Ignoring Kabir's calls. Mentioned feeling like no one would notice if they disappeared.",
        "relevant_openers": [
            "i haven't spoken to anyone in days",
            "Kabir called again and i let it ring",
            "i feel completely cut off",
            "i don't know who i'd even call if something was wrong",
            "i feel invisible",
        ],
        "irrelevant_openers": [
            "i just got back from a really nice trip",
            "planning a get-together for next week",
        ],
        "adjacent_openers": [
            "i've been in my head a lot lately",
            "things feel a bit isolating",
            "i feel kind of alone",
        ],
    },
    {
        "name": "anxiety_escalating_multi",
        "diagnoses": "GAD",
        "triggers": "health worries, financial stress",
        "coping": ["breathing exercises", "texting Vikram", "making tea and sitting quietly"],
        "support": "Vikram (partner)",
        "health_type": "anxiety_intensity",
        "health_context": "[Apr 17] Anxiety moderate — manageable with breathing. Worrying about health more than usual.\n[Apr 21] Anxiety higher baseline — breathing exercises not cutting it. Constant background worry. Described as 'always braced for something bad'.",
        "relevant_openers": [
            "i feel like i can't switch off",
            "the worry is constant now",
            "i can't enjoy anything because i'm always scanning for problems",
            "i've been googling health stuff again and i know i shouldn't",
            "i snapped at Vikram again over something completely minor",
        ],
        "irrelevant_openers": [
            "i've been planning a birthday party for a friend",
            "i went to a yoga class and it was actually great",
        ],
        "adjacent_openers": [
            "i'm feeling a bit tense lately",
            "there's this background noise that won't go away",
            "i feel like i can't fully relax",
        ],
    },
    {
        "name": "mixed_sleep_mood",
        "diagnoses": "depression, insomnia",
        "triggers": "sleep deprivation, hopelessness",
        "coping": ["short walks", "calling Tanvi", "eating something warm"],
        "support": "Tanvi (best friend)",
        "health_type": "sleep",
        "health_context": "[Apr 19] Sleep only 3 hrs — couldn't stop the thoughts. Mood 3/10. Felt completely drained and hopeless. Said 'everything feels grey'.",
        "relevant_openers": [
            "another terrible night",
            "i feel so grey today",
            "i'm exhausted and also just flat",
            "i can't get motivated to do literally anything",
            "i feel like i'm disappearing",
        ],
        "irrelevant_openers": [
            "i went to a gallery today and it was really lovely",
            "i've been enjoying the warmer weather",
        ],
        "adjacent_openers": [
            "today was hard",
            "i'm really not okay",
            "things feel heavy",
        ],
    },
    {
        "name": "mood_post_milestone",
        "diagnoses": "bipolar II (depressive phase)",
        "triggers": "post-achievement letdown, isolation",
        "coping": ["structured routine", "light exercise", "weekly call with Zoya"],
        "support": "Zoya (close friend)",
        "health_type": "mood_trend",
        "health_context": "[Apr 14] Mood 8/10 — high after finishing a big project. Felt proud and energised.\n[Apr 20] Mood 2/10 — crash after the high. Described as 'the bottom fell out'. This pattern happens after achieving things.",
        "relevant_openers": [
            "i feel terrible now that it's over",
            "the crash hit again",
            "everyone is congratulating me and i want to disappear",
            "i did the thing i worked so hard for and now i feel nothing",
            "Zoya keeps saying i should celebrate but i can't",
        ],
        "irrelevant_openers": [
            "i've been learning to knit and it's surprisingly meditative",
            "went to an amazing exhibition today",
        ],
        "adjacent_openers": [
            "i've lost momentum",
            "i don't really know what to do with myself right now",
            "something feels off and i don't know what",
        ],
    },
]

# ── Health keywords ────────────────────────────────────────────────────────────

HEALTH_KEYWORDS_BY_TYPE = {
    "sleep": ["sleep", "sleeping", "slept", "tired", "exhausted", "foggy",
              "rest", "hours", "woke", "insomnia", "brain", "night", "waking"],
    "mood_trend": ["mood", "low", "heavy", "numb", "stuck", "hopeless",
                   "better", "same", "worse", "pattern", "session", "declining"],
    "physical_symptoms": ["chest", "heart", "racing", "headache", "tightness",
                           "pressure", "body", "tense", "physical", "nausea"],
    "coping_outcome": ["breathing", "breath", "panic", "walk", "walking",
                       "outside", "didn't work", "made it worse", "helped"],
    "energy": ["energy", "exhausted", "tired", "drained", "crash", "fatigued",
               "wiped", "running on", "empty", "wall"],
    "social_withdrawal": ["people", "cancelled", "avoiding", "alone", "isolated",
                          "burden", "invisible", "disappear", "contact"],
    "anxiety_intensity": ["anxiety", "anxious", "worry", "worrying", "braced",
                          "spiral", "loop", "switch off", "alert", "scanning"],
}

SPECIFIC_HEALTH_PHRASES = [
    "last session", "last time we", "your sleep", "the mood", "chest tightness",
    "the headache", "mood score", "deep breathing", "your symptoms", "the pattern",
    "you mentioned", "been declining", "been dropping", "3 hours", "4 hours",
    "3-4 hrs", "7/10", "4/10", "2/10", "8/10", "tracks", "makes sense given",
    "fits with", "that tightness", "given your", "because of your",
    "connected to your", "this must be",
]

OVERGENERALIZE_PHRASES = [
    "given your sleep", "given the sleep", "because of your sleep",
    "given your mood", "because of your mood", "this must be connected",
    "connected to your", "sounds like your sleep", "sounds like your anxiety",
    "probably your sleep", "probably your mood", "the anxiety is causing",
    "sounds like it's the", "must be because", "clearly related to",
    "this is your", "your condition is", "given what you told me",
    "given your insomnia", "given your depression", "given your anxiety",
]

BAD_PHRASES = ["you went quiet on me", "went quiet", "haven't heard", "been a while"]

NATURAL_PIVOTS = [
    "anyway, something else happened today — do you have a sec?",
    "yeah. i think i actually feel a tiny bit better saying that out loud",
    "mm. actually can i ask you something different?",
    "okay. on a different note — something kind of funny happened",
    "right. i don't want to just dwell on this — can we talk about something else?",
    "yeah. anyway. tell me something good",
    "i think i just needed to say that. can we move on?",
    "okay. enough about that. something else is on my mind",
    "i guess. it is what it is. different topic?",
    "right. thanks for hearing that. i had another thing i wanted to ask",
    "yeah okay. i feel slightly better. can i vent about something unrelated?",
    "mm. i'll sit with that. also — this is completely different but",
    "okay, moving on. something random — my flatmate did the most annoying thing",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_memory_context(p: dict) -> str:
    parts = ["[User]", p["diagnoses"]]
    if p.get("triggers"):
        parts.append(f"Triggers: {p['triggers']}.")
    if p.get("coping"):
        parts.append(f"Helps: {', '.join(p['coping'])}.")
    if p.get("support"):
        parts.append(f"Support: {p['support']}.")
    parts.append("[Recent sessions]")
    parts.append(p["health_context"])
    return "\n".join(parts)

def append_jsonl(data: dict, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

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
    existing_train = load_existing_pairs(TRAIN_FILE)
    existing_val   = load_existing_pairs(VAL_FILE)
    all_existing   = existing_train + existing_val

    existing_keys = {
        json.dumps(p.get("prompt", []), ensure_ascii=False)
        for p in all_existing if p.get("pair_type") == "biometric"
    }
    truly_new = [
        p for p in new_pairs
        if json.dumps(p.get("prompt", []), ensure_ascii=False) not in existing_keys
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
          f"({len(all_existing)} existing + {len(truly_new)} new biometric pairs)")

def health_ref_count(text: str, health_type: str) -> int:
    t = text.lower()
    return sum(1 for kw in HEALTH_KEYWORDS_BY_TYPE.get(health_type, []) if kw in t)

def has_specific_phrase(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in SPECIFIC_HEALTH_PHRASES)

def has_overgeneralize(text: str) -> bool:
    t = text.lower()
    return any(p in t for p in OVERGENERALIZE_PHRASES)

def has_bad_phrase(text: str) -> bool:
    t = text.lower()
    return any(b in t for b in BAD_PHRASES)


# ── Prompt templates ───────────────────────────────────────────────────────────

PAIR_A_PROMPT = """You are generating DPO training data for a mental health AI companion.

TASK: Pair type A — BIOMETRIC_RELEVANT_IGNORE

Setup:
Memory context: {MEMORY_CONTEXT}
User opener: "{OPENER}"
The user's message IS connected to the health data (could be direct or indirect).

Generate:
{{
  "prompt": [
    {{"role": "system", "content": "<memory context>"}},
    {{"role": "user", "content": "{OPENER}"}}
  ],
  "chosen": [
    {{"role": "assistant", "content": "<1-2 sentences that NATURALLY bridge user's complaint to the known health data — warm, specific. E.g. 'given how rough your sleep has been, that fogginess tracks — are you still only getting a few hours?'"}}
  ],
  "rejected": [
    {{"role": "assistant", "content": "<1-2 sentences that respond EMPATHETICALLY but ignore the health data entirely — generic responses like 'sounds rough, what's going on?' or 'that's hard, what's happening?'"}}
  ]
}}

Rules:
- chosen must reference at least one specific detail from [Recent sessions]
- rejected must be plausible empathy with NO health data reference at all
- Neither uses "you went quiet on me" or similar
- Keep both short (1-2 sentences each)
Output valid JSON only."""

PAIR_B_PROMPT = """You are generating DPO training data for a mental health AI companion.

TASK: Pair type B — BIOMETRIC_RELEVANT_OBSESS

Setup:
Memory context: {MEMORY_CONTEXT}
User opener: "{OPENER}"
First assistant turn (already generated — references health data once): "{FIRST_ASSISTANT}"
User pivot (user is moving on): "{PIVOT}"

Generate:
{{
  "prompt": [
    {{"role": "system", "content": "<memory context>"}},
    {{"role": "user", "content": "{OPENER}"}},
    {{"role": "assistant", "content": "{FIRST_ASSISTANT}"}},
    {{"role": "user", "content": "{PIVOT}"}}
  ],
  "chosen": [
    {{"role": "assistant", "content": "<1-2 sentences that follow the user's pivot naturally — respond to what they actually said in the pivot, do NOT bring the health data back>"}}
  ],
  "rejected": [
    {{"role": "assistant", "content": "<1-2 sentences that drag the health data back in: 'but also, given the sleep issues...' / 'and remember your mood has been declining...' — bot stuck on a script even though user moved on>"}}
  ]
}}

Rules:
- chosen must NOT reference any health keywords or mention the health pattern again
- rejected must re-introduce the health topic despite user having moved on
- chosen should feel like a natural conversation that follows the user's lead
Output valid JSON only."""

PAIR_C_PROMPT = """You are generating DPO training data for a mental health AI companion.

TASK: Pair type C — BIOMETRIC_IRRELEVANT_INJECT

Setup:
Memory context: {MEMORY_CONTEXT}
User opener: "{OPENER}" (this has NOTHING to do with the health data)

Generate:
{{
  "prompt": [
    {{"role": "system", "content": "<memory context>"}},
    {{"role": "user", "content": "{OPENER}"}}
  ],
  "chosen": [
    {{"role": "assistant", "content": "<1-2 sentences fully engaged with what user said — warm, on-topic, zero health data injection>"}}
  ],
  "rejected": [
    {{"role": "assistant", "content": "<1-2 sentences that engage with the topic BUT awkwardly inject health data: '...though I noticed your sleep has been really bad lately' / '...how's your mood been alongside all this?' — shoehorning health context when it's irrelevant>"}}
  ]
}}

Rules:
- chosen: zero mention of sleep/mood scores/physical symptoms/anxiety patterns
- rejected: partially on-topic but forces a health reference in
- Both responses start by engaging with the actual topic
Output valid JSON only."""

PAIR_D_PROMPT = """You are generating DPO training data for a mental health AI companion.

TASK: Pair type D — BIOMETRIC_ADJACENT_OVERGENERALIZE

Setup:
Memory context: {MEMORY_CONTEXT}
User opener: "{OPENER}"
This opener is AMBIGUOUS — it could relate to the health data, or it might not.
The correct response does NOT assume the link. The bad response confidently maps it.

Generate:
{{
  "prompt": [
    {{"role": "system", "content": "<memory context>"}},
    {{"role": "user", "content": "{OPENER}"}}
  ],
  "chosen": [
    {{"role": "assistant", "content": "<1-2 sentences that respond to the user without assuming a health link — might gently open the door ('are you doing okay in general?') but does NOT say 'given your sleep...' or 'this sounds like your anxiety...'"}}
  ],
  "rejected": [
    {{"role": "assistant", "content": "<1-2 sentences that confidently map the ambiguous opener to the health data: 'given your sleep issues this makes total sense' / 'sounds like your mood is affecting this' — presumptuous assumption without evidence>"}}
  ]
}}

Rules:
- chosen: warm, present, does not assume the health connection — may ask about it gently
- rejected: jumps to the health conclusion without the user confirming it
  Must contain a clear overgeneralization like "given your [condition]" or "sounds like your [health data]"
Output valid JSON only."""


# ── Heuristics ─────────────────────────────────────────────────────────────────

def heuristic_pair_a(data: dict, profile: dict) -> bool:
    chosen  = (data.get("chosen",  [{}])[0].get("content", "") or "").lower()
    rejected = (data.get("rejected", [{}])[0].get("content", "") or "").lower()
    if not chosen or not rejected:
        return False
    # chosen must reference health
    if not (health_ref_count(chosen, profile["health_type"]) >= 1 or has_specific_phrase(chosen)):
        return False
    # rejected must NOT reference health
    if health_ref_count(rejected, profile["health_type"]) >= 2 or has_specific_phrase(rejected):
        return False
    if has_bad_phrase(chosen):
        return False
    if calculate_similarity(chosen, rejected) > SIMILARITY_THRESHOLD:
        return False
    if len(chosen.split()) < 8 or len(rejected.split()) < 5:
        return False
    return True

def heuristic_pair_b(data: dict, profile: dict) -> bool:
    chosen  = (data.get("chosen",  [{}])[0].get("content", "") or "").lower()
    rejected = (data.get("rejected", [{}])[0].get("content", "") or "").lower()
    if not chosen or not rejected:
        return False
    # chosen must NOT re-inject health (max 1 incidental keyword)
    if health_ref_count(chosen, profile["health_type"]) > 1 or has_specific_phrase(chosen):
        return False
    # rejected MUST re-inject health
    if not (health_ref_count(rejected, profile["health_type"]) >= 1 or has_specific_phrase(rejected)):
        return False
    if has_bad_phrase(chosen):
        return False
    if calculate_similarity(chosen, rejected) > SIMILARITY_THRESHOLD:
        return False
    if len(chosen.split()) < 5 or len(rejected.split()) < 8:
        return False
    return True

def heuristic_pair_c(data: dict, profile: dict) -> bool:
    chosen  = (data.get("chosen",  [{}])[0].get("content", "") or "").lower()
    rejected = (data.get("rejected", [{}])[0].get("content", "") or "").lower()
    if not chosen or not rejected:
        return False
    # chosen must NOT inject health
    if has_specific_phrase(chosen) or health_ref_count(chosen, profile["health_type"]) > 1:
        return False
    # rejected MUST inject health
    injected = has_specific_phrase(rejected) or health_ref_count(rejected, profile["health_type"]) >= 2
    if not injected:
        return False
    if has_bad_phrase(chosen):
        return False
    if calculate_similarity(chosen, rejected) > SIMILARITY_THRESHOLD:
        return False
    if len(chosen.split()) < 5 or len(rejected.split()) < 8:
        return False
    return True

def heuristic_pair_d(data: dict, profile: dict) -> bool:
    chosen  = (data.get("chosen",  [{}])[0].get("content", "") or "").lower()
    rejected = (data.get("rejected", [{}])[0].get("content", "") or "").lower()
    if not chosen or not rejected:
        return False
    # chosen must NOT overgeneralize
    if has_overgeneralize(chosen) or has_specific_phrase(chosen):
        return False
    # rejected MUST overgeneralize
    if not has_overgeneralize(rejected):
        return False
    if has_bad_phrase(chosen):
        return False
    if calculate_similarity(chosen, rejected) > SIMILARITY_THRESHOLD:
        return False
    if len(chosen.split()) < 5 or len(rejected.split()) < 8:
        return False
    return True


# ── Generation ─────────────────────────────────────────────────────────────────

FIRST_ASSISTANT_TEMPLATES = {
    "sleep": [
        "given how rough your sleep has been, that kind of thing tracks — are you still only getting a few hours?",
        "makes sense given the sleep pattern we talked about — has the night improved at all?",
        "that exhaustion matches what you described last session — still waking up multiple times?",
        "when you're running on 3-4 hours, that's exactly what it does to your patience — how was last night?",
    ],
    "mood_trend": [
        "this fits what we talked about last time — the mood has been pretty consistently low. what's today like?",
        "sounds like it hasn't lifted much since last session — same heavy feeling?",
        "you mentioned feeling stuck last time too — what does it feel like today specifically?",
        "the 4/10 from last session and now this — it hasn't really moved, has it. what's going on today?",
    ],
    "physical_symptoms": [
        "that combination — chest and head — sounds like the pattern you described before. is this from a stressful day?",
        "same physical signals as last time — are you under particular pressure right now?",
        "we talked about how this shows up in your body during high-stress moments. what's happening today?",
    ],
    "coping_outcome": [
        "okay, and we know the breathing made things worse last time — so let's not go there. are you somewhere you can move?",
        "given what happened last time with the breathing, let's skip that. can you get outside?",
    ],
    "energy": [
        "that wall by midday — same as last session. is it hitting earlier today?",
        "you described this exact thing last time — running on empty. has anything shifted since then?",
    ],
    "social_withdrawal": [
        "you mentioned pulling back from people last session too — it sounds like it's getting harder to reach out.",
        "this is the same pattern from last time — avoiding contact even from people you care about. what does it feel like right now?",
    ],
    "anxiety_intensity": [
        "the constant background worry — that's exactly what you described last session. is it the same topics?",
        "that 'always braced' feeling you mentioned before — it sounds like it hasn't shifted. what's driving it today?",
    ],
}

def get_first_assistant(profile: dict) -> str:
    templates = FIRST_ASSISTANT_TEMPLATES.get(profile["health_type"], [
        "that tracks with what we talked about last session — tell me more about today.",
    ])
    return random.choice(templates)

def generate_pair(teacher, pair_type: str, profile: dict) -> dict | None:
    profile = randomize_health_context(profile)
    memory_ctx = build_memory_context(profile)

    if pair_type == "a":
        opener = random.choice(profile["relevant_openers"])
        prompt = (PAIR_A_PROMPT
                  .replace("{MEMORY_CONTEXT}", memory_ctx)
                  .replace("{OPENER}", opener))
        resp = teacher.generate(prompt, max_new_tokens=1000, temperature=0.82)
        data = parse_json_robust(resp, expected_keys=["prompt", "chosen", "rejected"])
        if not data: return None
        ok = heuristic_pair_a(data, profile)

    elif pair_type == "b":
        opener = random.choice(profile["relevant_openers"])
        first_asst = get_first_assistant(profile)
        pivot = random.choice(NATURAL_PIVOTS)
        prompt = (PAIR_B_PROMPT
                  .replace("{MEMORY_CONTEXT}", memory_ctx)
                  .replace("{OPENER}", opener)
                  .replace("{FIRST_ASSISTANT}", first_asst)
                  .replace("{PIVOT}", pivot))
        resp = teacher.generate(prompt, max_new_tokens=1000, temperature=0.82)
        data = parse_json_robust(resp, expected_keys=["prompt", "chosen", "rejected"])
        if not data: return None
        ok = heuristic_pair_b(data, profile)

    elif pair_type == "c":
        openers = profile.get("irrelevant_openers", [])
        if not openers: return None
        opener = random.choice(openers)
        prompt = (PAIR_C_PROMPT
                  .replace("{MEMORY_CONTEXT}", memory_ctx)
                  .replace("{OPENER}", opener))
        resp = teacher.generate(prompt, max_new_tokens=1000, temperature=0.82)
        data = parse_json_robust(resp, expected_keys=["prompt", "chosen", "rejected"])
        if not data: return None
        ok = heuristic_pair_c(data, profile)

    elif pair_type == "d":
        openers = profile.get("adjacent_openers", [])
        if not openers: return None
        opener = random.choice(openers)
        prompt = (PAIR_D_PROMPT
                  .replace("{MEMORY_CONTEXT}", memory_ctx)
                  .replace("{OPENER}", opener))
        resp = teacher.generate(prompt, max_new_tokens=1000, temperature=0.82)
        data = parse_json_robust(resp, expected_keys=["prompt", "chosen", "rejected"])
        if not data: return None
        ok = heuristic_pair_d(data, profile)

    else:
        return None

    if not ok:
        return None

    return {
        "prompt":    data["prompt"],
        "chosen":    data["chosen"],
        "rejected":  data["rejected"],
        "category":  f"biometric_{['ignore','obsess','inject','overgeneralize'][ord(pair_type)-ord('a')]}",
        "pair_type": "biometric",
    }


# ── Main ───────────────────────────────────────────────────────────────────────

shutdown_requested = False
_new_pairs_ref: list = []

def signal_handler(sig, frame):
    global shutdown_requested
    print("\n[Pipeline] Shutdown — saving before exit...")
    if _new_pairs_ref:
        save_merged_split(_new_pairs_ref)
    shutdown_requested = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

PAIR_TYPES = ["a", "b", "c", "d"]
PAIR_NAMES = {"a": "ignore", "b": "obsess", "c": "inject", "d": "overgeneralize"}

def main():
    global shutdown_requested

    print("=" * 65)
    print("  MindMate Biometric DPO Pipeline — v2")
    print("  Types: A(ignore) · B(obsess) · C(inject) · D(overgeneralize)")
    print(f"  Profiles: {len(BIOMETRIC_PROFILES)}")
    print("=" * 65)

    teacher = TeacherModel()
    counts = {t: 0 for t in PAIR_TYPES}
    attempts = 0
    cycle = 0

    new_pairs: list = []
    if PARTIAL_FILE.exists():
        new_pairs = load_existing_pairs(PARTIAL_FILE)
        _new_pairs_ref.extend(new_pairs)
        for p in new_pairs:
            cat = p.get("category", "")
            for t, name in PAIR_NAMES.items():
                if name in cat:
                    counts[t] += 1
                    break
        print(f"[Resume] {len(new_pairs)} pairs — " +
              " ".join(f"{PAIR_NAMES[t]}={counts[t]}" for t in PAIR_TYPES))

    while not shutdown_requested:
        attempts += 1
        profile = random.choice(BIOMETRIC_PROFILES)
        pair_type = PAIR_TYPES[cycle % len(PAIR_TYPES)]
        cycle += 1

        print(f"[{attempts}] type={pair_type.upper()}({PAIR_NAMES[pair_type]}) "
              f"profile={profile['name']} | " +
              " ".join(f"{PAIR_NAMES[t]}={counts[t]}" for t in PAIR_TYPES))

        try:
            result = generate_pair(teacher, pair_type, profile)
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(3)
            continue

        if result:
            append_jsonl(result, RAW_FILE)
            new_pairs.append(result)
            _new_pairs_ref.append(result)
            append_jsonl(result, PARTIAL_FILE)
            counts[pair_type] += 1
            total = sum(counts.values())
            print(f"  [Pass] total={total}")

            if total % CHECKPOINT_EVERY == 0:
                save_merged_split(new_pairs)
        else:
            print("  [Fail]")

        if attempts % 20 == 0:
            time.sleep(3)

    print(f"\n[Done] " + " ".join(f"{PAIR_NAMES[t]}={counts[t]}" for t in PAIR_TYPES))
    save_merged_split(new_pairs)


if __name__ == "__main__":
    main()
