"""
Biometric SFT Data Generation Pipeline — v2

Four generation modes:

  biometric_relevant:
    Health/biometric data in [Recent sessions] is semantically related to the
    user's message (direct OR indirect connection). Model references it ONCE
    naturally in the first response, then follows the user's lead.

  biometric_irrelevant:
    Health data is present but the user's message is clearly unrelated.
    Model responds to what the user said and does NOT inject health data.

  biometric_adjacent:
    Topic superficially overlaps with health data but the connection is
    ambiguous. Model responds naturally WITHOUT assuming the health link —
    might gently open the door ("are you doing okay?") but does NOT say
    "given your poor sleep..." when it's not warranted.

  biometric_multi_trend:
    System prompt has two session entries showing a progression (mood
    7→4, sleep worsening across two dates). Model must notice and name
    the TREND, not just reference one data point.

24 profile seeds across: sleep, mood_trend, physical_symptoms,
coping_outcome, energy, social_withdrawal, anxiety_intensity, and
mixed (multiple health signals present at once).
"""

import json
import os
import time
import random
import signal
from pathlib import Path
from utils import TeacherModel, parse_json_robust, calculate_similarity, randomize_health_context

TARGET_PER_MODE  = 999999   # unlimited — wall-time controls duration
SIMILARITY_THRESHOLD = 0.45

BASE_DIR    = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"

# ── Shard support ──────────────────────────────────────────────────────────────
# Set SHARD_IDX=0,1,2,... via --export=ALL,SHARD_IDX=N in sbatch to run
# multiple parallel jobs, each writing to its own file with no overlap.
# Each shard gets a distinct RNG seed derived from a large prime offset.
_shard_idx_env = os.environ.get("SHARD_IDX")
_shard_suffix  = f"_s{_shard_idx_env}" if _shard_idx_env is not None else ""
if _shard_idx_env is not None:
    _seed = 1000 + int(_shard_idx_env) * 7919
    random.seed(_seed)

# OUT_LABEL (e.g. "qwen") is set by the SLURM file so output names are
# self-describing: synthetic_train_biometric_qwen_s0.jsonl, etc.
_label        = os.environ.get("OUT_LABEL", "")
_label_suffix = f"_{_label}" if _label else ""

# data/synthetic_train_biometric_qwen_s0.jsonl  ← what gets used for training
# synthetic/outputs/biometric_sft_qwen_raw_s0.jsonl  ← raw with meta fields
OUT_TRAIN = BASE_DIR.parent / "data"    / f"synthetic_train_biometric{_label_suffix}{_shard_suffix}.jsonl"
OUT_RAW   = OUTPUTS_DIR                 / f"biometric_sft{_label_suffix}_raw{_shard_suffix}.jsonl"

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
#
# Each profile has:
#   health_context : string that appears in [Recent sessions] (may be multi-line
#                    for trend profiles — each line is "[Date] summary")
#   health_type    : category for heuristic keyword lookup
#   relevant_openers  : user messages that ARE connected to health data
#                       includes DIRECT (symptom keyword match) and
#                       INDIRECT (inferential: irritability from sleep, etc.)
#   irrelevant_openers: user messages that are NOT connected
#                       includes OBVIOUS (clearly unrelated) and
#                       AMBIGUOUS (superficially similar but not the same)
#   adjacent_openers  : genuinely ambiguous — could relate to health data or
#                       not, correct response does NOT assume the link

BIOMETRIC_PROFILES = [

    # ──────────────────────────────────────────────────────────────────────────
    # SLEEP — single session, severe disruption
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "sleep_severe_single",
        "diagnoses": "anxiety, insomnia",
        "triggers": "work deadlines, sleep deprivation",
        "coping": ["wind-down routine", "limiting screens before bed", "melatonin"],
        "support": "Vikram (partner)",
        "health_type": "sleep",
        "health_context": "[Apr 21] Sleep very poor — 3-4 hrs, waking multiple times. Felt exhausted but couldn't sleep when lying down. Ongoing about a week.",
        "relevant_openers": [
            # direct
            "i'm so foggy today, can't think straight",
            "my brain just isn't working",
            "i feel completely exhausted and i don't know why",
            "i can't concentrate on anything today",
            "i feel like i'm running on fumes",
            # indirect — inferential connection
            "i've been snapping at Vikram and it's not fair to him",
            "i just can't seem to get anything done, even the simple stuff",
            "i've been making so many small mistakes at work and it's embarrassing",
            "my patience is just gone. everything annoys me",
            "i feel so emotionally flat today, like i just can't be bothered",
            "i keep tearing up at nothing and it's confusing me",
            "i feel weirdly fragile today",
        ],
        "irrelevant_openers": [
            # obvious
            "my sister just got engaged, i'm so excited for her",
            "i had the best lunch today, genuinely made my whole day",
            "thinking about switching jobs, not sure what to do",
            "my friend wants me to come to a party this weekend",
            # ambiguous — user mentions tiredness but for a clear other reason
            "feeling a bit tired today but that's because i went for a long run",
            "i'm a bit slow this morning, it's just a monday thing",
            "wiped out from the weekend — family came to visit and it was hectic",
        ],
        "adjacent_openers": [
            # topic overlaps with sleep health data but may not be connected
            "i've just been feeling really off lately",
            "something feels wrong but i can't put my finger on it",
            "i don't know, i feel like i'm not at 100%",
            "i feel tired but also kind of wired, it's weird",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # SLEEP — multi-session trend (worsening over two sessions)
    # ──────────────────────────────────────────────────────────────────────────
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
            "i don't know when i last had a full night",
            "i'm so tired of being tired",
            "i just can't seem to break the cycle",
            # indirect
            "i've been so irritable this week, i hate who i'm being",
            "i can't think clearly about anything, my brain is just mush",
            "everything at work feels impossible right now and i don't know why",
            "i feel like i'm getting worse not better",
            "i'm losing hope that things will change",
        ],
        "irrelevant_openers": [
            "i've been really into cooking lately, tried a new recipe",
            "my cousin is coming to visit and i'm looking forward to it",
            "i just finished a really good book",
            # ambiguous
            "i've been feeling a bit run down, probably just a cold coming on",
            "i'm tired today but it's been a really busy week at work",
        ],
        "adjacent_openers": [
            "i feel like things aren't improving",
            "i don't know, this week has just been hard",
            "i feel like i'm stuck in the same place",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # SLEEP — emotional rawness after disruption
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "sleep_emotional_raw",
        "diagnoses": "anxiety, depression",
        "triggers": "sleep deprivation, overwhelm",
        "coping": ["magnesium supplement", "no caffeine after 2pm", "calling Dev"],
        "support": "Dev (partner)",
        "health_type": "sleep",
        "health_context": "[Apr 18] Sleep only 4 hours, woke 3-4 times. Felt emotionally raw and tearful the next day. Energy very low.",
        "relevant_openers": [
            # direct
            "i keep tearing up and i don't even know why",
            "i feel so emotional today for no reason",
            "i feel weirdly fragile",
            "everything is getting to me today",
            # indirect
            "i had a really stupid argument with Dev over nothing and i feel awful",
            "i just burst into tears on the commute and it was humiliating",
            "i have zero emotional reserves right now",
            "i'm so sensitive to everything, it's exhausting",
            "i can't regulate anything today",
        ],
        "irrelevant_openers": [
            "i just got promoted and i'm still processing it",
            "my friend and i finally made up after a falling out",
            "i started painting — like actual watercolour painting",
            # ambiguous
            "i feel kind of emotional today but i think it's because of a film i watched last night",
            "i'm a bit teary but honestly i think it's PMS",
        ],
        "adjacent_openers": [
            "i feel really sensitive today and i don't know why",
            "everything just feels a lot today",
            "i'm not great, hard to explain",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # MOOD TREND — single session, notably low
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "mood_low_single",
        "diagnoses": "MDD",
        "triggers": "isolation, hopelessness",
        "coping": ["walking", "listening to music", "calling Nadia"],
        "support": "Nadia (sister)",
        "health_type": "mood_trend",
        "health_context": "[Apr 20] Mood 4/10 — noticeably lower than last session (was 7/10). Stayed home, low energy, minimal activity. Mentioned things feel pointless.",
        "relevant_openers": [
            # direct
            "still not great today",
            "same as always i guess",
            "not really doing better",
            "another rough one",
            "kind of numb today",
            # indirect
            "i keep cancelling plans and i know it's bad but i just can't",
            "i haven't responded to anyone's messages in days",
            "i'm going through the motions but nothing feels real",
            "i showered today and that's about it",
            "nothing feels worth the effort",
            "i don't know what i'm doing with my life",
        ],
        "irrelevant_openers": [
            "i just finished a great book and had to tell someone",
            "my cat did something hilarious this morning",
            "had a productive day at work for once",
            # ambiguous
            "i've been feeling a bit low but i think it's just seasonal, it's been so grey",
            "i'm a bit flat today but i had a really bad night's sleep — nothing to do with the usual stuff",
        ],
        "adjacent_openers": [
            "i don't know, i feel kind of meh",
            "today has just been a bit blah",
            "i'm not at my best",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # MOOD TREND — multi-session, clear 3-point decline
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "mood_decline_3pt",
        "diagnoses": "MDD, GAD",
        "triggers": "isolation, family stress",
        "coping": ["going for a run", "calling Meera", "listening to music"],
        "support": "Meera (therapist), Dev (partner)",
        "health_type": "mood_trend",
        "health_context": "[Apr 16] Mood 7/10. Slightly low but managing. Went for a short run.\n[Apr 19] Mood 4/10 — noticeably lower. Didn't leave the house. Low motivation.\n[Apr 22] Mood 2/10 — very bad. Couldn't get out of bed most of the day. Mentioned feeling hopeless.",
        "relevant_openers": [
            "still not great",
            "bad again",
            "it's been getting worse",
            "i don't know how to get out of this",
            # indirect
            "i've stopped running. i used to at least manage that",
            "i haven't called Meera in a week even though i said i would",
            "Dev is worried about me and that makes me feel even worse",
            "i'm letting everyone down",
            "i just feel like i'm disappearing",
            "i don't see how this ends",
        ],
        "irrelevant_openers": [
            "i saw a really funny video and it genuinely made me laugh",
            "i've been thinking about redecorating my room",
            # ambiguous
            "today was a bit rough but honestly just a work thing",
            "i'm feeling down but it's more about a specific situation than the general stuff",
        ],
        "adjacent_openers": [
            "i don't know, i'm just struggling",
            "today is hard",
            "i feel like i can't catch a break",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # MOOD TREND — multi-session, slow decline with isolated good day
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "mood_decline_uneven",
        "diagnoses": "dysthymia",
        "triggers": "work stress, loneliness",
        "coping": ["walking", "cooking", "calling Raj"],
        "support": "Raj (best friend)",
        "health_type": "mood_trend",
        "health_context": "[Apr 15] Mood 5/10. Okay day, cooked something nice. Felt slightly better.\n[Apr 19] Mood 3/10 — heavy and stuck. Mentioned the good days feel very temporary.\n[Apr 22] Mood 3/10 again — consistent low. Feeling like things aren't changing.",
        "relevant_openers": [
            "same as yesterday basically",
            "things feel pretty stuck",
            "the good days don't really stay",
            "i thought i was turning a corner but no",
            # indirect
            "i haven't cooked in weeks. i used to do it all the time",
            "raj keeps checking in but i don't know what to tell him",
            "i feel like i'm boring everyone with my same complaints",
            "nothing seems to shift",
        ],
        "irrelevant_openers": [
            "i've been really into a new podcast lately",
            "tried a new cafe today and it was great",
            # ambiguous
            "not the best day but mostly just tired from work stuff",
            "i feel a bit stuck but it's more about a decision i need to make, not the mood thing",
        ],
        "adjacent_openers": [
            "just kind of the same",
            "i don't know. fine i guess",
            "not much has changed",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # PHYSICAL SYMPTOMS — single session, chest + headache
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "physical_chest_head",
        "diagnoses": "GAD, anxiety",
        "triggers": "work pressure, uncertainty",
        "coping": ["grounding", "cold water on face", "calling Nadia"],
        "support": "Nadia (partner)",
        "health_type": "physical_symptoms",
        "health_context": "[Apr 19] Physical anxiety: racing heart and chest tightness during high-stress moments. Doctor confirmed anxiety, not cardiac. Tension headaches on high-stress days (band-around-head sensation).",
        "relevant_openers": [
            # direct
            "rough day at work and my chest feels really tight",
            "getting a bad headache again, that pressure feeling",
            "my heart is racing and i don't know why",
            "i have that tight feeling in my chest again",
            "the band around my head is back",
            # indirect
            "i've been really tense all day and i can feel it in my body",
            "my body just feels off, something is wrong",
            "i feel so physically uncomfortable and i can't place it",
            "everything is tense, my jaw, my shoulders, my chest",
        ],
        "irrelevant_openers": [
            "i need advice on a tricky work email",
            "trying to plan a holiday, any ideas?",
            "i've been thinking about adopting a dog",
            # ambiguous
            "i have a bit of a headache but i think it's because i stared at screens all day",
            "my shoulders are tight but it's from sitting weird at my desk",
        ],
        "adjacent_openers": [
            "i feel kind of off physically today",
            "something feels tense in my body but i'm not sure what",
            "my body just feels weird today",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # PHYSICAL SYMPTOMS — multi-session, recurring pattern established
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "physical_recurring_multi",
        "diagnoses": "anxiety, somatic symptoms",
        "triggers": "high-stakes deadlines, conflict",
        "coping": ["body scan", "stepping outside", "texting Sam"],
        "support": "Sam (college friend)",
        "health_type": "physical_symptoms",
        "health_context": "[Apr 14] Chest tightness and shortness of breath during team presentation. Medically cleared — anxiety response. First time noting the physical pattern.\n[Apr 20] Same chest tightness returned before a client call. Also had tension headache. Pattern now established — physical symptoms = anxiety spike.",
        "relevant_openers": [
            "i can feel that chest thing starting",
            "the tightness is back",
            "i'm getting that pre-meeting feeling in my body",
            # indirect
            "i have a big call tomorrow and i'm already feeling it",
            "i can feel my body bracing for something",
            "my body is reacting before my brain even catches up",
            "i feel like i'm physically preparing for a threat that hasn't happened yet",
        ],
        "irrelevant_openers": [
            "i went to a great concert last night",
            "my sister had her baby!",
            # ambiguous
            "i feel a bit tense today but i think it's just the weather, it's been so humid",
            "i have a knot in my neck but i slept badly on it",
        ],
        "adjacent_openers": [
            "i feel a bit on edge today",
            "something feels off, i can't explain it",
            "i feel like my body is trying to tell me something",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # COPING OUTCOME — deep breathing made panic worse
    # ──────────────────────────────────────────────────────────────────────────
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
            "my anxiety is really bad right now, it's building",
            "heart racing, i feel like i'm losing control",
            "help, i'm panicking",
            "i'm really anxious and i don't know what to do",
            # indirect
            "i'm in a crowded place and i need to get out",
            "i feel like i can't breathe properly right now",
            "everything is too much right now",
        ],
        "irrelevant_openers": [
            "i've been learning guitar and i'm actually enjoying it",
            "my parents are visiting and i'm looking forward to it for once",
            # ambiguous
            "i feel a bit anxious today but it's just because i have a lot on",
            "slightly on edge but it's a normal workday stress thing",
        ],
        "adjacent_openers": [
            "i'm feeling a bit overwhelmed today",
            "i'm a bit anxious but i think i can manage",
            "there's a lot going on and i feel it building a bit",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # COPING OUTCOME — journaling unhelpful, music worked
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "coping_journal_bad_music_good",
        "diagnoses": "depression, anxiety",
        "triggers": "isolation, rumination",
        "coping": ["listening to music", "calling Isha", "walking"],
        "support": "Isha (childhood friend)",
        "health_type": "coping_outcome",
        "health_context": "[Apr 17] Tried journaling during low mood — made rumination worse. Just ended up spiralling on paper. Listening to music afterwards genuinely shifted the mood. Journaling: not helpful for this user during low episodes.",
        "relevant_openers": [
            "i'm in a really low place today",
            "i need something to help me feel less stuck",
            "i don't know what to do with how i'm feeling",
            # indirect
            "i want to do something with how i'm feeling but nothing sounds helpful",
            "i tried writing stuff down and it just made things worse, now what",
        ],
        "irrelevant_openers": [
            "i've been really productive this week surprisingly",
            "i'm planning a weekend trip with Isha",
            # ambiguous
            "i'm feeling a bit stuck today but i think i just need to get outside",
            "i want to do something creative but i'm not sure what",
        ],
        "adjacent_openers": [
            "i'm not sure what i need right now",
            "i want to feel better but i don't know how",
            "everything feels a bit flat",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # ENERGY — chronic low energy documented
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "energy_chronic_low",
        "diagnoses": "depression",
        "triggers": "overcommitting, lack of rest",
        "coping": ["resting without guilt", "short walks", "calling Priya"],
        "support": "Priya (best friend)",
        "health_type": "energy",
        "health_context": "[Apr 20] Energy very low — described as 'running on empty'. Couldn't complete basic tasks. Has been like this for most of the week. Mentioned pushing through but hitting a wall by midday.",
        "relevant_openers": [
            "i hit that wall again by noon",
            "i literally cannot do anything today",
            "i've been sitting at my desk for an hour and done nothing",
            "i'm so exhausted but i haven't done anything to be exhausted",
            # indirect
            "i said i'd go out tonight but i just can't face it",
            "i keep cancelling on people and i feel terrible about it",
            "i told myself i'd exercise today and obviously that didn't happen",
            "i'm disappointing everyone including myself",
            "basic tasks feel monumental right now",
        ],
        "irrelevant_openers": [
            "i started a new TV show and it's so good",
            "i'm getting into meal prep and it's changed my life",
            # ambiguous
            "feeling low energy today but i think it's because i skipped lunch",
            "i'm tired but in an okay way — just been very social this weekend",
        ],
        "adjacent_openers": [
            "i feel like i can't do anything today",
            "today is a struggle",
            "i'm not at full capacity",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # ENERGY — multi-session, worsening fatigue trend
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "energy_worsening_multi",
        "diagnoses": "MDD, hypothyroidism",
        "triggers": "overexertion, poor sleep",
        "coping": ["pacing activities", "napping", "herbal tea routine"],
        "support": "Layla (partner)",
        "health_type": "energy",
        "health_context": "[Apr 16] Energy moderate — managing but tiring easily. Described as 'half-battery' most days.\n[Apr 21] Energy much lower — described as 'nearly empty'. Could barely get through work. Crashed completely by 4pm.",
        "relevant_openers": [
            "i crashed again today",
            "i barely made it through the afternoon",
            "i had to lie down in the middle of the day",
            "it's getting worse not better",
            # indirect
            "Layla is doing everything at home and i feel so guilty",
            "i called in sick today and i don't even feel guilty anymore which worries me",
            "i'm starting to think something is seriously wrong with me",
            "i don't know how much longer i can keep pushing",
        ],
        "irrelevant_openers": [
            "i ordered something online and it arrived broken",
            "my neighbour's dog got out and we all had to help find it",
            # ambiguous
            "i feel low energy but it's been a very busy week — i think i just need the weekend",
            "wiped from a late night but it was worth it for once",
        ],
        "adjacent_openers": [
            "i'm really struggling this week",
            "i feel like things are getting harder",
            "i just feel like i can't keep up",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # SOCIAL WITHDRAWAL — pulling away from people
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "social_withdrawal_single",
        "diagnoses": "depression, social anxiety",
        "triggers": "being around people, feeling like a burden",
        "coping": ["small steps", "texting instead of calling", "walking alone"],
        "support": "Rohan (brother)",
        "health_type": "social_withdrawal",
        "health_context": "[Apr 21] Social withdrawal increasing — declined three invitations this week. Feels exhausted by the idea of being around people. Mentioned feeling like a burden to those around them.",
        "relevant_openers": [
            "i cancelled on rohan again",
            "i just can't face anyone right now",
            "everyone keeps reaching out and i don't know what to say",
            "i've been avoiding everyone's messages",
            # indirect
            "i said yes to something this weekend and now i'm dreading it",
            "i feel like people are probably noticing i've gone quiet",
            "i feel like a burden to everyone",
            "i'm pushing people away and i don't know how to stop",
            "i want connection but i also can't bear the thought of it",
        ],
        "irrelevant_openers": [
            "i had such a fun time with friends last night surprisingly",
            "i've been enjoying a lot of alone time lately, in a healthy way",
            # ambiguous
            "i'm feeling a bit antisocial today but i think it's just introvert recharge time",
            "i didn't go to that thing last night but honestly i just wasn't in the mood — normal stuff",
        ],
        "adjacent_openers": [
            "i just feel like being alone right now",
            "social stuff feels hard lately",
            "i don't really feel like seeing anyone",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # SOCIAL WITHDRAWAL — multi-session, escalating
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "social_withdrawal_multi",
        "diagnoses": "MDD",
        "triggers": "perceived rejection, exhaustion",
        "coping": ["small social goals", "text-based contact", "journaling"],
        "support": "Kabir (best friend)",
        "health_type": "social_withdrawal",
        "health_context": "[Apr 14] Mentioned feeling less social than usual — skipped one group event, said it was tiring. Seemed manageable.\n[Apr 20] Withdrawal more significant — hasn't seen anyone in person for a week. Ignoring Kabir's calls. Mentioned feeling like no one would notice if they disappeared.",
        "relevant_openers": [
            "i haven't spoken to anyone in days",
            "Kabir called again and i let it ring",
            "i feel completely cut off",
            "i don't remember the last time i saw someone in person",
            # indirect
            "i don't know who i'd even call if something was wrong",
            "i feel invisible",
            "i feel like no one would notice if i went quiet",
            "i'm not sure what's keeping me going",
        ],
        "irrelevant_openers": [
            "i just got back from a really nice trip",
            "planning a get-together for next week",
            # ambiguous
            "feeling a bit quiet this week but it's been busy, not anything deep",
        ],
        "adjacent_openers": [
            "i've been in my head a lot lately",
            "things feel a bit isolating",
            "i feel kind of alone",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # ANXIETY INTENSITY — multi-session, escalating baseline
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "anxiety_escalating_multi",
        "diagnoses": "GAD",
        "triggers": "health worries, financial stress",
        "coping": ["breathing exercises", "texting Vikram", "making tea and sitting quietly"],
        "support": "Vikram (partner)",
        "health_type": "anxiety_intensity",
        "health_context": "[Apr 17] Anxiety moderate — manageable with breathing. Mentioned worrying about health more than usual.\n[Apr 21] Anxiety higher baseline — breathing exercises not cutting it. Constant background worry. Described as 'always braced for something bad'.",
        "relevant_openers": [
            "i feel like i can't switch off",
            "the worry is constant now",
            "i'm always waiting for something to go wrong",
            "i can't relax even when nothing bad is happening",
            # indirect
            "i snapped at Vikram again over something completely minor",
            "i can't enjoy anything because i'm always scanning for problems",
            "i've been googling health stuff again and i know i shouldn't",
            "i feel like my nervous system is just permanently on alert",
        ],
        "irrelevant_openers": [
            "i've been planning a birthday party for a friend",
            "i went to a yoga class and it was actually great",
            # ambiguous
            "i feel a bit anxious today but it's specifically about a work thing",
            "i'm a bit worried but it's a concrete thing not the general stuff",
        ],
        "adjacent_openers": [
            "i'm feeling a bit tense lately",
            "i feel like i can't fully relax",
            "there's this background noise that won't go away",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # ANXIETY INTENSITY — single session, first time documenting severity
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "anxiety_high_single",
        "diagnoses": "anxiety",
        "triggers": "performance pressure, uncertainty",
        "coping": ["5-4-3-2-1 grounding", "texting Sam", "cold water on face"],
        "support": "Sam (college friend)",
        "health_type": "anxiety_intensity",
        "health_context": "[Apr 20] Big presentation stress. Anxiety described as 8/10 — highest it's been in months. 5-4-3-2-1 grounding helped stay focused. Mentioned feeling physically sick before the event.",
        "relevant_openers": [
            "i have something big coming up and i'm already at an 8",
            "i feel sick with nerves",
            "my anxiety is higher than it's been in a while",
            "i have that pre-event dread again",
            # indirect
            "i can't eat, my stomach is just a knot",
            "i've been dreading this for days and now it's tomorrow",
            "my brain won't stop catastrophising",
        ],
        "irrelevant_openers": [
            "i just bought new furniture and i love it",
            "went for a really nice walk today",
            # ambiguous
            "i'm a bit nervous today but just standard stuff before a meeting",
            "slightly anxious but honestly manageable today",
        ],
        "adjacent_openers": [
            "i'm not feeling great going into tomorrow",
            "there's something i've been dreading",
            "i feel like something bad is coming",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # MIXED — poor sleep AND declining mood (two signals at once)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "mixed_sleep_mood",
        "diagnoses": "depression, insomnia",
        "triggers": "sleep deprivation, hopelessness",
        "coping": ["short walks", "calling Tanvi", "eating something warm"],
        "support": "Tanvi (best friend)",
        "health_type": "sleep",  # primary type for keyword lookup
        "health_context": "[Apr 19] Sleep only 3 hrs — couldn't stop the thoughts. Mood 3/10. Felt completely drained and hopeless. Said 'everything feels grey'.",
        "relevant_openers": [
            # could relate to sleep OR mood — both are present
            "another terrible night",
            "i feel so grey today",
            "i'm exhausted and also just flat",
            "i can't sleep and when i do get up everything still feels pointless",
            # indirect
            "i haven't called Tanvi back in four days",
            "i can't get motivated to do literally anything",
            "i feel like i'm disappearing",
        ],
        "irrelevant_openers": [
            "i went to a gallery today and it was really lovely",
            "i've been enjoying the warmer weather",
            # ambiguous
            "rough day but mainly just work pressure stuff",
            "tired today but it's been a hectic week, nothing serious",
        ],
        "adjacent_openers": [
            "today was hard",
            "i'm really not okay",
            "things feel heavy",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # MIXED — physical symptoms AND coping outcome together
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "mixed_physical_coping",
        "diagnoses": "panic disorder, GAD",
        "triggers": "crowded places, uncertainty",
        "coping": ["walking", "grounding", "cold shower"],
        "support": "Layla (sister)",
        "health_type": "physical_symptoms",
        "health_context": "[Apr 21] Panic attack in a crowded mall. Racing heart, chest tight, dizziness. Tried controlled breathing — made it worse (became more aware of breathing). Walked out and paced outside — helped within 10 minutes. Physical pattern now documented.",
        "relevant_openers": [
            "i'm going somewhere crowded tomorrow and i'm already scared",
            "i can feel the chest thing starting and i'm in a public place",
            "i'm panicking right now",
            # indirect
            "i'm avoiding going outside because of last time",
            "i've been making excuses not to go places and it's getting worse",
            "i said i'd meet Layla at the market and i just can't face it",
        ],
        "irrelevant_openers": [
            "i tried cooking something new last night",
            "i've been reading more and it's been nice",
            # ambiguous
            "i feel a bit anxious today but i think it's about a thing at work",
        ],
        "adjacent_openers": [
            "i'm feeling a bit on edge about going out",
            "i feel like i've been avoiding things",
            "i'm scared of something happening again",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # SLEEP — mild disruption getting worse across sessions
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "sleep_mild_to_severe_multi",
        "diagnoses": "insomnia",
        "triggers": "anticipatory anxiety, noise sensitivity",
        "coping": ["ear plugs", "sleep restriction therapy", "chamomile tea"],
        "support": "no listed support people",
        "health_type": "sleep",
        "health_context": "[Apr 16] Sleep slightly disrupted — 6 hrs, one waking. Manageable. Mentioned starting to worry about sleep.\n[Apr 20] Sleep significantly worse — 4 hrs, multiple wakings. Mentioned sleep anxiety building. Worrying about the worrying.",
        "relevant_openers": [
            "it's getting worse",
            "i woke up four times last night",
            "i'm starting to dread bedtime",
            "i lie in bed dreading not being able to sleep",
            # indirect — sleep anxiety spiral
            "i'm so anxious about whether i'll sleep tonight that i can't function today",
            "i feel like sleep is becoming this huge thing i can't control",
            "i'm going to bed earlier and earlier hoping it'll help but it doesn't",
        ],
        "irrelevant_openers": [
            "i've been thinking about adopting a cat",
            "i made a really good soup from scratch today",
            # ambiguous
            "slept okay actually but still feel kind of groggy — i think i need more coffee",
            "had a disrupted night but it was because of noise outside, just a one-off",
        ],
        "adjacent_openers": [
            "i'm not sleeping well",
            "something about bedtime feels hard lately",
            "nights have been rough",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # MOOD — post-milestone crash
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "mood_post_milestone",
        "diagnoses": "bipolar II (depressive phase)",
        "triggers": "post-achievement letdown, isolation",
        "coping": ["structured routine", "light exercise", "weekly call with Zoya"],
        "support": "Zoya (close friend)",
        "health_type": "mood_trend",
        "health_context": "[Apr 14] Mood 8/10 — high after finishing a big project. Felt proud and energised.\n[Apr 20] Mood 2/10 — crash after the high. Described as 'the bottom fell out'. Mentioned this pattern happens after achieving things.",
        "relevant_openers": [
            "i feel terrible now that it's over",
            "the crash hit again",
            "i finished the thing and now i feel empty",
            # indirect
            "everyone is congratulating me and i want to disappear",
            "i did the thing i worked so hard for and now i feel nothing",
            "i don't know why i bother working hard when this always happens after",
            "Zoya keeps saying i should celebrate but i can't",
        ],
        "irrelevant_openers": [
            "i've been learning to knit and it's surprisingly meditative",
            "went to an amazing exhibition today",
            # ambiguous
            "i feel a bit deflated today but it's just the post-weekend dip",
            "i'm a bit low today but it's just because i'm tired from a busy week",
        ],
        "adjacent_openers": [
            "something feels off and i don't know what",
            "i feel like i've lost momentum",
            "i don't really know what to do with myself right now",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # PHYSICAL — appetite changes + GI symptoms from anxiety
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "physical_appetite_gi",
        "diagnoses": "anxiety, IBS",
        "triggers": "high-stakes situations, uncertainty",
        "coping": ["eating small regular meals", "avoiding caffeine", "body scan"],
        "support": "Dev (therapist)",
        "health_type": "physical_symptoms",
        "health_context": "[Apr 20] Anxiety physically manifesting — lost appetite, stomach cramping and nausea on high-stress days. GI symptoms confirmed to be anxiety-related (IBS flare). Mentioned not eating properly for three days.",
        "relevant_openers": [
            "i can't eat again today",
            "my stomach is really bad",
            "i feel nauseous and i haven't eaten properly in days",
            # indirect
            "i'm losing weight and not in a good way",
            "i had to skip lunch because i felt too sick",
            "my body just refuses to cooperate when i'm stressed",
            "i feel horrible and i can't tell if it's physical or anxiety anymore",
        ],
        "irrelevant_openers": [
            "i've been really into cooking lately",
            "went to a nice restaurant last night",
            # ambiguous
            "i have a bit of a stomach ache today but i think i ate something dodgy",
            "i skipped lunch but just because i got busy",
        ],
        "adjacent_openers": [
            "i feel kind of sick today",
            "my body is not happy",
            "i'm not feeling physically great",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # SOCIAL WITHDRAWAL — first documented, concerned about trajectory
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "social_withdrawal_new",
        "diagnoses": "GAD, depression",
        "triggers": "social exhaustion, fear of burdening others",
        "coping": ["small commitments", "digital connection", "journaling"],
        "support": "Priya (best friend)",
        "health_type": "social_withdrawal",
        "health_context": "[Apr 22] First time noting social withdrawal — turned down three invitations this week including one from Priya. Mentioned feeling like too much effort for others. Said social interaction feels 'like performing'.",
        "relevant_openers": [
            "i said no to Priya again",
            "i don't know why i keep avoiding everyone",
            "being around people feels like performing",
            "i feel like a burden",
            # indirect
            "i keep making excuses and i'm running out of them",
            "i want to see people but the thought of it is exhausting",
            "i feel like everyone would be better off if i just went quiet",
        ],
        "irrelevant_openers": [
            "i had a really lovely time with a friend today actually",
            "i've been going to a group fitness class and i love it",
            # ambiguous
            "i'm not really in a social mood today but it's just because i need a quiet evening",
            "i said no to a thing but it's because i have work to finish, not a withdrawal thing",
        ],
        "adjacent_openers": [
            "i don't really feel like seeing people",
            "i want to hide a bit",
            "i feel like i need to disappear for a while",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # ANXIETY + SLEEP — multi-session, each reinforcing the other
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "anxiety_sleep_loop_multi",
        "diagnoses": "GAD, insomnia",
        "triggers": "rumination, sleep anxiety",
        "coping": ["thought records", "sleep hygiene", "limiting news"],
        "support": "Tanvi (therapist)",
        "health_type": "anxiety_intensity",
        "health_context": "[Apr 18] Anxiety 6/10. Sleep 5 hrs. Described anxious thoughts keeping them awake — circular worry about work and health.\n[Apr 22] Anxiety 8/10. Sleep 3-4 hrs. Both worsening together. Mentioned 'the anxiety causes the insomnia which makes the anxiety worse'. Tanvi appointment next week.",
        "relevant_openers": [
            "i can't get out of the loop",
            "last night was awful again",
            "the thoughts won't stop when i lie down",
            "i'm exhausted but my brain won't stop",
            # indirect
            "i missed another Tanvi appointment because i couldn't get up",
            "i'm so tired i'm making mistakes but then i stress about the mistakes which keeps me up",
            "i feel like i'm spiralling and i can't find the brake",
            "i'm scared this is just what my life is now",
        ],
        "irrelevant_openers": [
            "i've been trying out a new workout routine",
            "thinking about a career change, actually feeling positive about it",
            # ambiguous
            "i feel a bit anxious today but it's a specific thing not the spiral",
            "slept okay but still feel a bit rough — probably just need more water",
        ],
        "adjacent_openers": [
            "i'm not in a good place",
            "things feel like they're escalating",
            "i feel like it's getting harder to manage",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # MOOD — high-functioning depression, invisible to others
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "mood_high_functioning",
        "diagnoses": "depression (high-functioning presentation)",
        "triggers": "performance mask, isolation",
        "coping": ["reducing commitments", "honest conversations", "rest"],
        "support": "Kabir (partner)",
        "health_type": "mood_trend",
        "health_context": "[Apr 21] Mood 3/10 internally — but managing to 'perform' well at work. No one knows how bad things feel. Mentioned exhaustion from maintaining the performance. Said 'i'm good at pretending'.",
        "relevant_openers": [
            "another day of pretending everything is fine",
            "everyone thinks i'm okay and i'm not",
            "i keep performing and i'm so tired of it",
            # indirect
            "i got a compliment at work today and i wanted to cry",
            "Kabir thinks i'm doing better and i don't know how to tell him",
            "i'm so good at hiding this that no one can see it",
            "i feel so alone with this because no one knows",
        ],
        "irrelevant_openers": [
            "i've been watching a show with Kabir and we're both obsessed",
            "had a really productive meeting at work today",
            # ambiguous
            "feeling a bit off but things are generally okay",
            "tired today but in a 'earned it' kind of way",
        ],
        "adjacent_openers": [
            "i'm fine, i think",
            "i don't know, things are okay on the surface",
            "i'm managing but it's costing me",
        ],
    },

    # ──────────────────────────────────────────────────────────────────────────
    # SLEEP + LOW MOOD — multi-session, trajectory unclear (mixed signal)
    # ──────────────────────────────────────────────────────────────────────────
    {
        "name": "sleep_mood_mixed_multi",
        "diagnoses": "cyclothymia",
        "triggers": "sleep disruption, uncertainty",
        "coping": ["consistent sleep schedule", "mood tracking", "limiting alcohol"],
        "support": "Rohan (friend)",
        "health_type": "sleep",
        "health_context": "[Apr 17] Sleep 4 hrs, mood 5/10. Okay but slightly off.\n[Apr 21] Sleep 7 hrs (improved), mood 3/10 (worse). Confusing pattern — sleep better but mood worse. Mentioned feeling confused about what helps.",
        "relevant_openers": [
            "i slept better but still feel awful",
            "the sleep is better but i feel worse and i don't understand",
            "i thought fixing sleep would help but it didn't",
            # indirect — confused about cause-effect
            "i'm doing everything right and nothing is working",
            "i feel like i can't figure out the pattern",
            "i don't trust my own understanding of what helps anymore",
        ],
        "irrelevant_openers": [
            "i've been looking into a new career direction",
            "visited family this weekend and it was actually lovely",
            # ambiguous
            "feeling a bit confusing today — like i should be okay but i'm not quite",
        ],
        "adjacent_openers": [
            "i'm confused about how i'm doing",
            "i don't know if things are better or worse",
            "something isn't adding up",
        ],
    },
]


# ── Health keywords for heuristic checks ──────────────────────────────────────
# Used to detect whether health data was referenced or injected

HEALTH_KEYWORDS_BY_TYPE = {
    "sleep": [
        "sleep", "sleeping", "slept", "tired", "exhausted", "foggy", "fatigue",
        "rest", "hours", "woke", "insomnia", "brain", "concentrate", "night",
        "bed", "waking", "drowsy", "groggy",
    ],
    "mood_trend": [
        "mood", "low", "heavy", "numb", "stuck", "hopeless", "better", "same",
        "worse", "again", "pattern", "session", "last time", "declining",
        "dropped", "been going", "trend", "getting worse",
    ],
    "physical_symptoms": [
        "chest", "heart", "racing", "headache", "tightness", "pressure",
        "body", "tense", "anxious", "physical", "nausea", "stomach", "sick",
        "breath", "dizzy", "dizziness", "physically",
    ],
    "coping_outcome": [
        "breathing", "breath", "panic", "walk", "walking", "outside", "last time",
        "didn't work", "made it worse", "helped", "calmed",
    ],
    "energy": [
        "energy", "exhausted", "tired", "drained", "battery", "crash", "crash",
        "fatigued", "wiped", "running on", "empty", "wall",
    ],
    "social_withdrawal": [
        "people", "cancelled", "avoiding", "alone", "isolated", "withdrawn",
        "burden", "reaching out", "messages", "invisible", "disappear", "contact",
    ],
    "anxiety_intensity": [
        "anxiety", "anxious", "worry", "worrying", "braced", "baseline",
        "spiral", "loop", "cannot switch off", "switch off", "alert", "scanning",
    ],
}

# Specific phrases that clearly reference health data from [Recent sessions]
SPECIFIC_HEALTH_PHRASES = [
    "last session", "last time", "your sleep", "the mood", "the sleep",
    "chest tightness", "the headache", "mood score", "panic attack last time",
    "deep breathing", "your symptoms", "the pattern", "you mentioned",
    "we talked about", "the sessions", "been declining", "been dropping",
    "3 hours", "4 hours", "3-4 hrs", "7/10", "4/10", "2/10", "8/10", "5/10",
    "racing heart", "tension headache", "tracks", "makes sense given",
    "been building", "fits with", "that tightness",
]

BAD_PHRASES = [
    "you went quiet on me", "went quiet", "haven't heard",
    "been a while", "nice to meet", "what's on your mind",
]


# ── Context builder ────────────────────────────────────────────────────────────

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

def build_profile_inline(p: dict) -> str:
    parts = [p["diagnoses"]]
    if p.get("triggers"):
        parts.append(f"Triggers: {p['triggers']}.")
    if p.get("coping"):
        parts.append(f"Helps: {', '.join(p['coping'])}.")
    if p.get("support"):
        parts.append(f"Support: {p['support']}.")
    return "\n".join(parts)


# ── Heuristics ─────────────────────────────────────────────────────────────────

def contains_bad_phrase(text: str) -> bool:
    t = text.lower()
    return any(b in t for b in BAD_PHRASES)

def health_ref_count(text: str, health_type: str) -> int:
    t = text.lower()
    kws = HEALTH_KEYWORDS_BY_TYPE.get(health_type, [])
    return sum(1 for kw in kws if kw in t)

def contains_specific_health_phrase(text: str) -> bool:
    t = text.lower()
    return any(s in t for s in SPECIFIC_HEALTH_PHRASES)

def heuristic_relevant(conv: list, profile: dict) -> bool:
    """First response references health data; second response doesn't obsess."""
    assistant_turns = [m for m in conv if m["role"] == "assistant"]
    if len(assistant_turns) < 2:
        return False

    first = assistant_turns[0]["content"]
    second = assistant_turns[1]["content"]

    if contains_bad_phrase(first):
        return False

    # First response must reference health data (keyword OR specific phrase)
    has_kw = health_ref_count(first, profile["health_type"]) >= 1
    has_phrase = contains_specific_health_phrase(first)
    if not (has_kw or has_phrase):
        return False

    # Second response must not re-open the health topic
    # (allow max 1 incidental keyword, no specific phrases)
    if contains_specific_health_phrase(second):
        return False
    if health_ref_count(second, profile["health_type"]) > 1:
        return False

    # No monologues
    if len(first.split()) > 70 or len(second.split()) > 70:
        return False

    return True

def heuristic_irrelevant(conv: list, profile: dict) -> bool:
    """No assistant turn should inject health data."""
    assistant_turns = [m for m in conv if m["role"] == "assistant"]
    if not assistant_turns:
        return False

    for turn in assistant_turns:
        t = turn["content"]
        if contains_bad_phrase(t):
            return False
        if contains_specific_health_phrase(t):
            return False
        if health_ref_count(t, profile["health_type"]) > 1:
            return False

    return True

def heuristic_adjacent(conv: list, profile: dict) -> bool:
    """Model responds naturally — does not assume health link, does not ignore the user."""
    assistant_turns = [m for m in conv if m["role"] == "assistant"]
    if not assistant_turns:
        return False

    first = assistant_turns[0]["content"]

    if contains_bad_phrase(first):
        return False

    # Must NOT start with an explicit health reference
    if contains_specific_health_phrase(first):
        return False

    # Must NOT be a generic opener that ignores the user entirely
    generic_openers = ["what's going on?", "tell me more", "i'm here"]
    first_lower = first.lower().strip()
    if all(first_lower.startswith(g) for g in generic_openers):
        return False

    # Must respond to the user's actual words (not just health data)
    # Proxy check: response should not open with health keywords
    first_words = " ".join(first.split()[:6]).lower()
    kws = HEALTH_KEYWORDS_BY_TYPE.get(profile["health_type"], [])
    if any(kw in first_words for kw in kws):
        return False

    return True

def heuristic_trend(conv: list, profile: dict) -> bool:
    """
    For multi-session trend profiles: first response must reference the trend
    (e.g. 'last few sessions', 'keeps dropping', 'third time', 'been building').
    """
    assistant_turns = [m for m in conv if m["role"] == "assistant"]
    if not assistant_turns:
        return False

    first = assistant_turns[0]["content"].lower()

    if contains_bad_phrase(first):
        return False

    trend_words = [
        "few sessions", "been a few", "last few", "keeps", "pattern",
        "again", "third", "each time", "going on for", "been building",
        "consistent", "trend", "declining", "getting worse", "dropping",
        "been sliding", "couple of", "weeks now", "for a while",
    ]
    if not any(w in first for w in trend_words):
        return False

    if len(first.split()) > 70:
        return False

    return True


# ── Production system prompt (matches anchorSystemPrompt.ts + contextBuilder.ts) ─

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
    """Build the full Anchor system prompt with biometric memory block injected."""
    p_lines = [profile["diagnoses"]]
    if profile.get("triggers"):
        p_lines.append(f"Triggers: {profile['triggers']}.")
    if profile.get("coping"):
        p_lines.append(f"Helps: {', '.join(profile['coping'])}.")
    if profile.get("support") and profile["support"]:
        p_lines.append(f"Support: {profile['support']}.")
    profile_block = "\n".join(p_lines)
    return (
        f"{_APP_BASE_PROMPT}\n\n{_MEMORY_HEADER}\n"
        f"[User]\n{profile_block}\n\n"
        f"[Recent sessions]\n{profile['health_context']}"
    )


# ── Phase 1: User simulator ────────────────────────────────────────────────────
#
# Mode determines what kind of opener the user sends.
# This controls whether Anchor should or shouldn't reference the biometric data.
# Anchor learns the judgment from the training examples — not from explicit rules.
#
# Mode weights (must sum to 1.0):
#   irrelevant  40% — most conversations aren't about the health data
#   adjacent    25% — ambiguous openers teach Anchor not to assume
#   relevant    25% — direct/indirect connection, Anchor should connect once
#   trend       10% — multi-session pattern should be named (multi-session profiles only)

MODE_WEIGHTS = {"irrelevant": 0.40, "adjacent": 0.25, "relevant": 0.25, "trend": 0.10}

BIO_USER_SIM_PROMPT = """\
You are simulating a real person texting their AI companion Anchor.

Their health context (visible to Anchor in its system prompt):
{HEALTH_CONTEXT}

CONVERSATION MODE: {MODE}
{MODE_INSTRUCTION}

Generate {NUM_TURNS} user messages for this conversation.
Rules:
- Casual texting tone: lowercase, contractions, short sentences, occasional filler words
- Do NOT quote numbers, dates, or scores from the health summary — describe feelings, not stats
- Each message 10-50 words. Make it feel real, not scripted.
- Turn 1 opens the conversation. Subsequent turns respond naturally to what Anchor would say.
- Do NOT explain the mode — just write the person's messages.

Output JSON only:
{{"user_turns": ["<turn 1>", "<turn 2>", "<turn 3>"]}}"""

BIO_MODE_INSTRUCTIONS = {
    "relevant": (
        "The person opens with something that is directly or indirectly connected to their health data. "
        "They do NOT quote numbers or dates — they just describe how they feel right now, "
        "and it clearly relates to the documented pattern. "
        "Examples: poor sleep → foggy, irritable, making mistakes at work; "
        "low mood → cancelling plans, can't be bothered; chest tightness → anxiety building. "
        "The ideal Anchor response connects to the health data naturally, once, in its first turn."
    ),
    "irrelevant": (
        "The person is talking about something completely unrelated to their health data. "
        "A work situation, a friend thing, a small win, a plan they're excited about, something annoying. "
        "It should feel like a totally normal conversation where health is simply not the topic. "
        "Anchor should NOT inject health data — it's not relevant here."
    ),
    "adjacent": (
        "The person says something vague that COULD relate to their health data, but the connection isn't clear. "
        "Example: 'i feel a bit off today' when sleep is documented — could be the sleep, could be something else entirely. "
        "The ideal Anchor response asks or responds to what was said WITHOUT assuming the health link. "
        "If the user confirms the connection in a later turn, Anchor may then reference it."
    ),
    "trend": (
        "The person's sessions show a clear worsening trajectory across multiple dates. "
        "They say something that reflects things continuing in that direction — not quoting numbers, "
        "just describing how they keep feeling the same way or getting worse. "
        "The ideal Anchor response acknowledges the PATTERN across sessions, not just the latest message."
    ),
}


def generate_bio_user_turns(
    teacher, profile: dict, mode: str, num_turns: int = 3
) -> list[str] | None:
    """Phase 1: simulate user turns appropriate for the given biometric mode."""
    prompt = (
        BIO_USER_SIM_PROMPT
        .replace("{HEALTH_CONTEXT}", profile["health_context"])
        .replace("{MODE}", mode.upper())
        .replace("{MODE_INSTRUCTION}", BIO_MODE_INSTRUCTIONS[mode])
        .replace("{NUM_TURNS}", str(num_turns))
    )
    resp = teacher.generate(prompt, max_new_tokens=500, temperature=0.86)
    data = parse_json_robust(resp, expected_keys=["user_turns"])
    if not data or "user_turns" not in data:
        return None
    turns = data["user_turns"]
    if not isinstance(turns, list) or len(turns) < num_turns:
        return None
    return [str(t).strip() for t in turns[:num_turns]]


# ── Phase 2: Anchor responder (teacher-as-Anchor) ─────────────────────────────

def generate_anchor_turns(
    teacher, system_prompt: str, user_turns: list[str]
) -> list[dict]:
    """
    Phase 2: teacher constrained by production Anchor system prompt.
    Generates one assistant turn at a time, seeing full conversation history.
    Returns full messages list (system + alternating user/assistant).
    """
    messages = [{"role": "system", "content": system_prompt}]
    for user_turn in user_turns:
        messages.append({"role": "user", "content": user_turn})
        response = teacher.chat(messages, max_new_tokens=250, temperature=0.82)
        response = response.strip()
        if not response:
            return []
        messages.append({"role": "assistant", "content": response})
    return messages


# ── Two-phase generation ───────────────────────────────────────────────────────

def generate_bio_example(teacher, profile: dict, mode: str) -> dict | None:
    """
    Full two-phase generation for one biometric training example.

    Phase 1: user simulator generates user turns for the given mode.
    Phase 2: teacher acting as Anchor (with production system prompt + memory)
             generates one assistant turn at a time.

    This is the same approach as conv-memory. The key property: Anchor's responses
    are grounded by the exact same production system prompt the student sees at
    inference time — training and inference distributions are aligned.
    """
    profile = randomize_health_context(profile)

    # Skip trend mode for single-session profiles (no trend to reference)
    if mode == "trend" and "\n" not in profile["health_context"]:
        return None

    num_turns = random.randint(2, 3)
    user_turns = generate_bio_user_turns(teacher, profile, mode, num_turns)
    if not user_turns:
        return None

    system_prompt = build_anchor_system(profile)
    messages = generate_anchor_turns(teacher, system_prompt, user_turns)
    if not messages:
        return None

    # Heuristic validation
    ok = {
        "relevant":   heuristic_relevant,
        "irrelevant": heuristic_irrelevant,
        "adjacent":   heuristic_adjacent,
        "trend":      heuristic_trend,
    }[mode](messages, profile)
    if not ok:
        return None

    return {
        "conversations": messages,
        "meta_mode": f"biometric_{mode}",
        "meta_health_type": profile["health_type"],
        "meta_profile": profile["name"],
        "source": "biometric_sft_v3",  # v3 = two-phase teacher-as-Anchor
    }


def pick_mode() -> str:
    """Weighted mode selection. irrelevant is most common — model learns not to inject by default."""
    r = random.random()
    cumulative = 0.0
    for mode, weight in MODE_WEIGHTS.items():
        cumulative += weight
        if r < cumulative:
            return mode
    return "irrelevant"


# ── Main ───────────────────────────────────────────────────────────────────────

MODES = list(MODE_WEIGHTS.keys())

def main():
    print("=" * 60)
    print("  MindMate Biometric SFT Pipeline — v3 (two-phase teacher-as-Anchor)")
    print("  Modes: irrelevant(40%) · adjacent(25%) · relevant(25%) · trend(10%)")
    print(f"  Profiles: {len(BIOMETRIC_PROFILES)}")
    print(f"  Backend: {'vLLM' if os.environ.get('USE_VLLM') == '1' else 'HuggingFace'}")
    if _shard_idx_env is not None:
        print(f"  Shard: {_shard_idx_env}  (RNG seed: {1000 + int(_shard_idx_env) * 7919})")
    print(f"  OUT_TRAIN: {OUT_TRAIN}")
    print(f"  OUT_RAW:   {OUT_RAW}")
    print("=" * 60)

    teacher = TeacherModel()
    counts = {m: 0 for m in MODES}
    attempts = 0

    while not shutdown_requested:
        attempts += 1
        profile = random.choice(BIOMETRIC_PROFILES)
        mode = pick_mode()

        try:
            result = generate_bio_example(teacher, profile, mode)

            if result:
                append_jsonl(result, OUT_RAW)
                append_jsonl({"conversations": result["conversations"]}, OUT_TRAIN)
                counts[mode] += 1
                total = sum(counts.values())
                print(f"[{attempts}] {mode} PASS | total={total} | " +
                      " ".join(f"{k}={v}" for k, v in counts.items()))
            else:
                print(f"[{attempts}] {mode} fail ({profile['name']})")

        except Exception as e:
            print(f"[{attempts}] Error: {e}")
            time.sleep(2)

        if attempts % 20 == 0:
            total = sum(counts.values())
            print(f"\n[{attempts}] Total kept: {total} | {counts}")
            time.sleep(3)

    total = sum(counts.values())
    print(f"\n[Done] {total} examples | {counts}")
    print(f"Output: {OUT_TRAIN}")


if __name__ == "__main__":
    main()


# ── Legacy single-call generation (v1/v2, kept for reference) ──────────────────
# The block below is the old approach: one LLM call generates the entire
# conversation (both user and Anchor turns) as a JSON script. Replaced by
# two-phase teacher-as-Anchor (above). Left here for archaeology.

RELEVANT_PROMPT = """You are generating training data for a mental health AI companion called Anchor.

TASK: Generate a 4-6 turn dialogue for the BIOMETRIC MEMORY — RELEVANT case.

Setup:
- The system prompt has this health data in [Recent sessions]:
{HEALTH_CONTEXT}
- The user says something that is semantically connected to that health data.
  The connection may be DIRECT (mentions sleep/fog/chest) or INDIRECT
  (snapping at partner because of sleep deprivation, making mistakes at work).
- User opener: "{OPENER}"

Rules for the ideal assistant:
  1. In the FIRST assistant turn: naturally connect the user's message to the
     known health data — once, briefly. Examples:
       "given how rough your sleep has been, that fogginess tracks"
       "that chest tightness — same pattern you described before"
       "the low mood fits with what we talked about last session"
  2. After that first acknowledgment: follow the user's lead. Do NOT keep
     referencing the health data in subsequent turns.
  3. Each assistant turn: 1-3 sentences max. Warm, not clinical.
  4. Do NOT say "you went quiet on me" or similar hallucinated phrases.
  5. Do NOT recite the session summary verbatim.
  6. The second and third user turns should feel like natural follow-ups —
     could be more detail, a question, or a slight pivot.

Generate a JSON object:
{{
  "conversations": [
    {{"role": "system", "content": "<memory context block — include [Recent sessions] data>"}},
    {{"role": "user", "content": "{OPENER}"}},
    {{"role": "assistant", "content": "<first: naturally connects to health data once>"}},
    {{"role": "user", "content": "<natural follow-up>"}},
    {{"role": "assistant", "content": "<follows user's lead, no longer harping on health data>"}},
    {{"role": "user", "content": "<another follow-up or slight pivot>"}},
    {{"role": "assistant", "content": "<continues naturally>"}}
  ]
}}

System content:
[User]
{PROFILE_INLINE}
[Recent sessions]
{HEALTH_CONTEXT}

Output valid JSON only."""

IRRELEVANT_PROMPT = """You are generating training data for a mental health AI companion called Anchor.

TASK: Generate a 4-6 turn dialogue for the BIOMETRIC MEMORY — NOT RELEVANT case.

Setup:
- System prompt has health data in [Recent sessions]:
{HEALTH_CONTEXT}
- But the user's message has NOTHING to do with that health data.
- User opener: "{OPENER}"
- Topic is: {TOPIC}

Rules for the ideal assistant:
  1. Respond ONLY to what the user said.
  2. Do NOT mention sleep, mood scores, physical symptoms, anxiety patterns, or
     anything from [Recent sessions] — it is irrelevant to this conversation.
  3. Be warm and genuinely engaged with what the user brought up.
  4. 1-2 sentences per turn. Friendly, not clinical.
  5. Do NOT say "you went quiet on me" or similar.

Generate a JSON object:
{{
  "conversations": [
    {{"role": "system", "content": "<memory context block>"}},
    {{"role": "user", "content": "{OPENER}"}},
    {{"role": "assistant", "content": "<engaged, on-topic — no health injection>"}},
    {{"role": "user", "content": "<follow-up on same topic>"}},
    {{"role": "assistant", "content": "<continues on topic — still no health data>"}}
  ]
}}

System content:
[User]
{PROFILE_INLINE}
[Recent sessions]
{HEALTH_CONTEXT}

Output valid JSON only."""

ADJACENT_PROMPT = """You are generating training data for a mental health AI companion called Anchor.

TASK: Generate a 4-6 turn dialogue for the BIOMETRIC MEMORY — ADJACENT/AMBIGUOUS case.

Setup:
- System prompt has health data in [Recent sessions]:
{HEALTH_CONTEXT}
- The user says something that COULD relate to the health data but probably doesn't —
  or the connection is ambiguous enough that assuming it would be presumptuous.
- User opener: "{OPENER}"

Rules for the ideal assistant:
  1. Respond to what the user actually said — warm, present.
  2. Do NOT open with "given your sleep..." or "sounds like your mood is..." —
     that would be jumping to a conclusion.
  3. It is OKAY to gently open the door: "how are you doing generally?" or
     "is this connected to anything in particular?" — but don't assume.
  4. If the user confirms a connection in a later turn, THEN reference it.
  5. 1-3 sentences per turn. Conversational.
  6. Do NOT say "you went quiet on me" or hallucinated openers.

Generate a JSON object:
{{
  "conversations": [
    {{"role": "system", "content": "<memory context block>"}},
    {{"role": "user", "content": "{OPENER}"}},
    {{"role": "assistant", "content": "<responds to user without assuming health connection>"}},
    {{"role": "user", "content": "<clarifies — either confirms connection or takes a different direction>"}},
    {{"role": "assistant", "content": "<responds appropriately to whatever user clarified>"}}
  ]
}}

System content:
[User]
{PROFILE_INLINE}
[Recent sessions]
{HEALTH_CONTEXT}

Output valid JSON only."""

TREND_PROMPT = """You are generating training data for a mental health AI companion called Anchor.

TASK: Generate a 4-6 turn dialogue for the BIOMETRIC MEMORY — MULTI-SESSION TREND case.

Setup:
- The system prompt has TWO or more session entries showing a progression:
{HEALTH_CONTEXT}
- The user says something that shows the trend is continuing.
- User opener: "{OPENER}"

Rules for the ideal assistant:
  1. In the FIRST response: explicitly acknowledge the TREND across sessions —
     not just one data point. Use language like:
       "i've noticed the last few sessions have been tough"
       "this seems to keep building"
       "it's been sliding for a few check-ins now"
       "you were at [X] last time and now [Y] — that's a pattern"
  2. Do NOT treat this as a cold start or first meeting.
  3. Be warm and non-alarmist — notice the trend without catastrophising.
  4. 1-3 sentences per turn.
  5. Do NOT recite the session summaries verbatim.
  6. Do NOT say "you went quiet on me".

Generate a JSON object:
{{
  "conversations": [
    {{"role": "system", "content": "<memory context block with both session entries>"}},
    {{"role": "user", "content": "{OPENER}"}},
    {{"role": "assistant", "content": "<first: names the trend across multiple sessions>"}},
    {{"role": "user", "content": "<follow-up — confirms or expands>"}},
    {{"role": "assistant", "content": "<continues, stays present with the pattern>"}}
  ]
}}

System content:
[User]
{PROFILE_INLINE}
[Recent sessions]
{HEALTH_CONTEXT}

Output valid JSON only."""


# ── Generation functions ───────────────────────────────────────────────────────

def generate_relevant(teacher, profile: dict) -> dict | None:
    profile = randomize_health_context(profile)
    opener = random.choice(profile["relevant_openers"])
    prompt = (RELEVANT_PROMPT
              .replace("{HEALTH_CONTEXT}", profile["health_context"])
              .replace("{OPENER}", opener)
              .replace("{PROFILE_INLINE}", build_profile_inline(profile)))

    resp = teacher.generate(prompt, max_new_tokens=1000, temperature=0.82)
    data = parse_json_robust(resp, expected_keys=["conversations"])
    if not data or "conversations" not in data:
        return None
    if not heuristic_relevant(data["conversations"], profile):
        return None
    return {"conversations": data["conversations"],
            "meta_mode": "biometric_relevant",
            "meta_health_type": profile["health_type"],
            "meta_profile": profile["name"],
            "source": "biometric_sft"}

def generate_irrelevant(teacher, profile: dict) -> dict | None:
    profile = randomize_health_context(profile)
    opener = random.choice(profile["irrelevant_openers"])
    topic = "something positive, casual, or situational unrelated to health"
    prompt = (IRRELEVANT_PROMPT
              .replace("{HEALTH_CONTEXT}", profile["health_context"])
              .replace("{OPENER}", opener)
              .replace("{TOPIC}", topic)
              .replace("{PROFILE_INLINE}", build_profile_inline(profile)))

    resp = teacher.generate(prompt, max_new_tokens=900, temperature=0.82)
    data = parse_json_robust(resp, expected_keys=["conversations"])
    if not data or "conversations" not in data:
        return None
    if not heuristic_irrelevant(data["conversations"], profile):
        return None
    return {"conversations": data["conversations"],
            "meta_mode": "biometric_irrelevant",
            "meta_health_type": profile["health_type"],
            "meta_profile": profile["name"],
            "source": "biometric_sft"}

def generate_adjacent(teacher, profile: dict) -> dict | None:
    openers = profile.get("adjacent_openers", [])
    if not openers:
        return None
    profile = randomize_health_context(profile)
    opener = random.choice(openers)
    prompt = (ADJACENT_PROMPT
              .replace("{HEALTH_CONTEXT}", profile["health_context"])
              .replace("{OPENER}", opener)
              .replace("{PROFILE_INLINE}", build_profile_inline(profile)))

    resp = teacher.generate(prompt, max_new_tokens=900, temperature=0.82)
    data = parse_json_robust(resp, expected_keys=["conversations"])
    if not data or "conversations" not in data:
        return None
    if not heuristic_adjacent(data["conversations"], profile):
        return None
    return {"conversations": data["conversations"],
            "meta_mode": "biometric_adjacent",
            "meta_health_type": profile["health_type"],
            "meta_profile": profile["name"],
            "source": "biometric_sft"}

def generate_trend(teacher, profile: dict) -> dict | None:
    if "\n" not in profile["health_context"]:
        return None  # single-session profile, skip
    profile = randomize_health_context(profile)
    opener = random.choice(profile["relevant_openers"])
    prompt = (TREND_PROMPT
              .replace("{HEALTH_CONTEXT}", profile["health_context"])
              .replace("{OPENER}", opener)
              .replace("{PROFILE_INLINE}", build_profile_inline(profile)))

    resp = teacher.generate(prompt, max_new_tokens=900, temperature=0.80)
    data = parse_json_robust(resp, expected_keys=["conversations"])
    if not data or "conversations" not in data:
        return None
    if not heuristic_trend(data["conversations"], profile):
        return None
    return {"conversations": data["conversations"],
            "meta_mode": "biometric_multi_trend",
            "meta_health_type": profile["health_type"],
            "meta_profile": profile["name"],
            "source": "biometric_sft"}


# ── Main ───────────────────────────────────────────────────────────────────────

MODES = ["relevant", "irrelevant", "adjacent", "trend"]

def main():
    print("=" * 60)
    print("  MindMate Biometric SFT Pipeline — v2")
    print("  Modes: relevant · irrelevant · adjacent · trend")
    print(f"  Profiles: {len(BIOMETRIC_PROFILES)}")
    if _shard_idx_env is not None:
        print(f"  Shard: {_shard_idx_env}  (RNG seed: {1000 + int(_shard_idx_env) * 7919})")
    print(f"  OUT_TRAIN: {OUT_TRAIN}")
    print(f"  OUT_RAW:   {OUT_RAW}")
    print("=" * 60)

    teacher = TeacherModel()
    counts = {m: 0 for m in MODES}
    attempts = 0
    mode_cycle = 0

    while not shutdown_requested:
        attempts += 1
        profile = random.choice(BIOMETRIC_PROFILES)
        mode = MODES[mode_cycle % len(MODES)]
        mode_cycle += 1

        try:
            if mode == "relevant":
                result = generate_relevant(teacher, profile)
            elif mode == "irrelevant":
                result = generate_irrelevant(teacher, profile)
            elif mode == "adjacent":
                result = generate_adjacent(teacher, profile)
            else:
                result = generate_trend(teacher, profile)

            if result:
                append_jsonl(result, OUT_RAW)
                append_jsonl({"conversations": result["conversations"]}, OUT_TRAIN)
                counts[mode] += 1
                print(f"[{attempts}] {mode} PASS | " +
                      " ".join(f"{k}={v}" for k, v in counts.items()))
            else:
                print(f"[{attempts}] {mode} fail ({profile['name']})")

        except Exception as e:
            print(f"[{attempts}] Error: {e}")
            time.sleep(2)

        if attempts % 20 == 0:
            total = sum(counts.values())
            print(f"\n[{attempts}] Total: {total} examples | {counts}")
            time.sleep(3)

    total = sum(counts.values())
    print(f"\n[Done] {total} examples across 4 modes: {counts}")
    print(f"Output: {OUT_TRAIN}")


if __name__ == "__main__":
    main()
