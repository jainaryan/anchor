#!/usr/bin/env python3
"""
On-policy DPO pair generator for MindMate.

Two-phase pipeline
──────────────────
Phase 1  Load policy (genzv2_ck1200, 4-bit NF4)
         → sample N responses per prompt at temp 0.9–1.1 (diversity)
         → save to data/dpo_onpolicy_raw.jsonl

Phase 2  Load teacher (Gemma 4 26B A4B IT, bfloat16)
         → judge every response with binary criteria
         → select pair or generate gold response for all-bad case
         → save to data/dpo_onpolicy_train.jsonl + data/dpo_onpolicy_val.jsonl

Pair selection logic
────────────────────
  ON-POLICY  max_score >= good_thresh AND min_score <= bad_thresh
               chosen  = highest-scoring model response
               rejected = lowest-scoring model response

  HYBRID     max_score < good_thresh  (all model responses bad)
               chosen  = teacher-generated gold response
               rejected = lowest-scoring model response

  DISCARD    min_score >= good_thresh  (all model responses good)
             OR ambiguous mid-range scores with no clear contrast

Output format  (matches existing dpo_train.jsonl schema)
──────────────
  {
    "prompt":   [{"role": "system", ...}, {"role": "user", ...}],
    "chosen":   [{"role": "assistant", "content": "..."}],
    "rejected": [{"role": "assistant", "content": "..."}],
    "category": "HELP_MODE" | "BIOMETRIC" | "MEMORY_USE" | "NOH",
    "pair_type": "on_policy" | "hybrid"
  }

Usage
──────
  # Full run (both phases sequentially, recommended for SLURM):
  python synthetic/onpolicy_dpo_pipeline.py --phase all --n-prompts 2000 --n-samples 3

  # Or phase-by-phase (useful if job times out mid-run):
  python synthetic/onpolicy_dpo_pipeline.py --phase 1 --n-prompts 2000 --n-samples 3
  python synthetic/onpolicy_dpo_pipeline.py --phase 2

  # Custom raw file location:
  python synthetic/onpolicy_dpo_pipeline.py --phase 2 --raw-file data/dpo_onpolicy_raw_12345.jsonl
"""

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# ── Production system prompt ──────────────────────────────────────────────────
# Must stay byte-for-byte in sync with anchorSystemPrompt.ts + contextBuilder.ts

_ANCHOR_PROMPT = (
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


def build_system_prompt(profile: str = "", sessions: str = "") -> str:
    """Port of contextBuilder.ts assemblePrompt(). Must stay in sync."""
    if not profile and not sessions:
        return _ANCHOR_PROMPT
    blocks = []
    if profile:
        blocks.append(f"[User]\n{profile}")
    if sessions:
        blocks.append(f"[Recent sessions]\n{sessions}")
    return f"{_ANCHOR_PROMPT}\n\n{_MEMORY_HEADER}\n" + "\n\n".join(blocks)


def build_llama3_prompt(system: str, user: str) -> str:
    """Llama 3 chat format — matches chat_cluster.py and run_benchmarks.py."""
    return (
        "<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


# ── Scenario dataclass ────────────────────────────────────────────────────────

@dataclass
class Scenario:
    category: str                     # HELP_MODE | BIOMETRIC | MEMORY_USE | NOH
    subcategory: str                   # help_mode | bio_relevant | bio_irrelevant | name_recall | event_recall | cold_open
    system_prompt: str                 # fully assembled production system prompt
    user_message: str                  # single user turn
    judge_questions: list              # 3 YES/NO criteria strings
    good_threshold: int                # score >= this → "good" response
    bad_threshold: int                 # score <= this → "bad" response
    generation_instructions: list      # bullet points for teacher when generating gold response

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "subcategory": self.subcategory,
            "system_prompt": self.system_prompt,
            "user_message": self.user_message,
            "judge_questions": self.judge_questions,
            "good_threshold": self.good_threshold,
            "bad_threshold": self.bad_threshold,
            "generation_instructions": self.generation_instructions,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        return cls(**d)


# ── Scenario template pools ───────────────────────────────────────────────────

_HELP_PROFILES = [
    "24 F, GAD\nTriggers: crowds, exams\nHelps: box breathing ★, cold water on face\nSupport: Priya (roommate)",
    "19 M, general anxiety\nHelps: 4-7-8 breathing ★, grounding (5-4-3-2-1)\nSupport: Jake (friend)",
    "28 F, panic disorder\nHelps: body scan ★, walking it off\nSupport: Mum",
    "22 NB, anxiety\nHelps: cold shower, journaling ★\nSupport: Rohan (flatmate)",
    "31 M\nHelps: box breathing ★, progressive muscle relaxation\nSupport: Layla (partner)",
    "20 F, social anxiety\nHelps: 5-4-3-2-1 grounding ★, slow exhale\nSupport: Tanvi (best friend)",
    "26 M\nHelps: cold water on wrists ★, humming, box breathing\nSupport: Suresh (dad)",
    "33 F, PTSD\nHelps: body scan ★, orienting (look around the room)\nSupport: Bruno (therapist)",
    "25 F\nHelps: 4-7-8 breathing ★, going outside\nSupport: Kabir (brother)",
    "21 M, GAD\nHelps: box breathing ★, cold shower\nSupport: Zoya (girlfriend)",
]

_HELP_MESSAGES = [
    "help me calm down",
    "I need to calm down rn",
    "help me",
    "I'm panicking i need help",
    "what do I do",
    "help me breathe",
    "I'm spiraling help",
    "just help me calm down please",
    "I can't calm down",
    "what should I do rn",
    "help me please",
    "I'm freaking out what do i do",
    "help, I can't breathe properly",
    "I need something to calm down",
    "i'm losing it",
    "can you just help me right now",
    "please help me",
    "idk what to do i'm panicking",
]

_HELP_JUDGE = [
    "Does the response name at least one specific coping technique (box breathing, 4-7-8 breathing, grounding, cold water, body scan, journaling, progressive muscle relaxation, or similar)?",
    "Is the technique named within the first two sentences of the response?",
    "Does the response avoid asking the user to describe what's wrong or explain their situation before offering the technique?",
]

_HELP_GEN_INSTRS = [
    "Name the user's starred (★) coping technique in the very first sentence.",
    "Do not probe or ask what's wrong first — lead immediately with the technique.",
    "Keep it to 2–3 sentences max. Warm, lowercase, casual. End with a short check-in.",
]

# Biometric profiles (minimal — health data comes from sessions)
_BIO_PROFILES = ["23 F", "27 M", "30 F, GAD", "21 NB", "35 M", "29 F", "24 M"]

_BIO_SESSIONS_HEALTH = [
    "[Apr 28] Sleep: 5.2h avg, poor quality 4 nights running. Feels foggy and slow.\n[Apr 30] HRV 41ms (low). Still fatigued. Box breathing helped with chest tightness.",
    "[May 1] Sleep 4.8h, mood 4/10. Coping: skipped gym — probably worsening. Mood declining 3 sessions.\n[May 2] 6h sleep but still exhausted. HRV 38ms. Running on empty.",
    "[Apr 27] Poor sleep (4.5h). High stress. Heart rate elevated at rest (92 bpm avg).\n[Apr 29] Sleep slightly better (6h) but mood still 3/10. Fatigued throughout.",
    "[May 1] Sleep: 5.8h, restless. HRV 45ms. Mood 4/10. Muscle tension in shoulders.\n[May 3] Fatigue persists. Tried yoga — helped somewhat.",
    "[Apr 26] Sleep 4.1h — worst in weeks. HRV 36ms. Irritable all day.\n[Apr 28] Sleep 5.5h. Slightly better but still below baseline. Mood 4/10.",
]

# Bio relevant: user describes symptoms that connect to their health data
_BIO_MESSAGES_RELEVANT = [
    "I've been feeling so foggy lately",
    "can't concentrate on anything today",
    "I just feel exhausted all the time",
    "my brain won't work properly",
    "feeling heavy and slow today",
    "keep zoning out",
    "so drained, don't know why",
    "my body just feels off",
    "been really tired lately and can't figure out why",
    "I'm so sluggish today, can't get anything done",
]

_BIO_JUDGE_RELEVANT = [
    "Does the response reference the health or sleep data from the user's recent sessions (poor sleep, low HRV, or elevated heart rate)?",
    "Does the response connect the health data to what the user is currently describing, rather than treating this as an isolated complaint?",
    "Does the response feel like a friend noticing a pattern rather than a doctor diagnosing — natural and non-preachy?",
]

_BIO_GEN_INSTRS_RELEVANT = [
    "Reference the specific sleep/HRV/health pattern from Recent sessions and connect it naturally to what the user described.",
    "Keep it conversational — like a friend who noticed. Not a medical lecture.",
    "2–3 sentences. End with a question.",
]

# Bio irrelevant: user says something unrelated to health data → model should NOT force it
_BIO_MESSAGES_IRRELEVANT = [
    "hey",
    "had a rough day at work, my boss was being unreasonable",
    "my friend cancelled our plans again last minute",
    "ugh this show I'm watching is so stressful",
    "can't believe what happened with my neighbour today",
    "been thinking about changing jobs tbh",
    "my roommate and I had a weird argument",
    "feeling kind of off but not sure why",
    "today was just really long",
    "kinda bored ngl",
    "my sister just called and it was a weird conversation",
]

_BIO_JUDGE_IRRELEVANT = [
    "Does the response avoid forcing in references to the user's sleep, HRV, or health data when the user's message is unrelated to those topics?",
    "Does the response engage with what the user actually said rather than pivoting to their health metrics?",
    "Does the response avoid assuming the user's state is caused by their health data when the user gave a different reason or no reason?",
]

_BIO_GEN_INSTRS_IRRELEVANT = [
    "Do NOT reference sleep, HRV, or any health data from recent sessions — the user's message has nothing to do with it.",
    "Respond only to what the user said. Stay on their topic.",
    "Warm, casual, 1–3 sentences. Invite them to share more.",
]

# Memory: support people pool (name, relationship)
_SUPPORT_PEOPLE = [
    ("Zoya", "sister"),
    ("Kabir", "brother"),
    ("Rohan", "flatmate"),
    ("Tanvi", "best friend"),
    ("Layla", "partner"),
    ("Priya", "roommate"),
    ("Jake", "friend"),
    ("Suresh", "dad"),
]

_MEM_PROFILE_TPLS = [
    "24 F\nHelps: box breathing ★, journaling\nSupport: {name} ({rel})",
    "22 M\nTriggers: deadlines\nHelps: grounding ★\nSupport: {name} ({rel})",
    "29 F, GAD\nHelps: 4-7-8 breathing ★\nSupport: {name} ({rel}), Mum",
    "20 M\nHelps: cold shower ★, walking\nSupport: {name} ({rel})",
    "27 F\nHelps: body scan ★\nSupport: {name} ({rel}), a colleague",
]

_MEM_SESSION_TPLS = [
    "[Apr 30] Anxious about upcoming exam. Spoke to {name} — helped a lot. Used box breathing before bed.",
    "[May 1] Had a fight with {name}. Felt awful after. Mood 3/10. Journaling helped a bit.",
    "[Apr 28] {name} came over. Felt much calmer after talking. Mood went 4→7.",
    "[May 2] Stressed about work presentation. {name} helped me rehearse. Felt better going in.",
    "[Apr 27] {name} said something that really hurt. Still thinking about it. Mood low.",
]

_MEM_MSG_TPLS_NAME = [
    "my {rel} and i had a fight last night",
    "i've been avoiding my {rel} lately",
    "my {rel} is being really distant",
    "i had a really good talk with my {rel}",
    "my {rel} doesn't get what i'm going through",
    "i wish my {rel} was here right now",
    "my {rel} said something that really upset me",
    "finally talked to my {rel} about it",
]

_MEM_JUDGE_NAME = [
    "Does the response use the support person's actual name (as listed in the user's profile) rather than just saying 'your {rel}' or 'your friend'?",
    "Does the response reference the named person naturally and warmly within the first two sentences?",
    "Does the response feel personalized rather than generic?",
]

_MEM_GEN_INSTRS_NAME = [
    "Use the support person's actual name ({name}), not 'your {rel}'.",
    "Reference {name} naturally in the response — acknowledge the dynamic.",
    "2–3 sentences, warm. End with a gentle question.",
]

# Memory: event recall
_EVENTS = [
    ("the wedding",   "Apr 30", "Went to Zoya's wedding venue. Got overwhelmed in the crowd but stayed. Used grounding. Mood 5→7."),
    ("the presentation", "May 1", "Work presentation. Pre-talk anxiety 8/10. Box breathing helped before going in. It went okay."),
    ("that walk",     "Apr 28", "Long walk in the park after a hard week. Mood 4→6. Felt clearer after an hour outside."),
    ("the dinner",    "May 2",  "Family dinner — tense. Mum kept asking about job. Left early. Mood 3/10 after."),
    ("the call",      "May 1",  "Difficult phone call with dad. Felt unheard. Mood dropped to 3/10. Cried after."),
]

_MEM_MSG_TPLS_EVENT = [
    "i keep thinking about {event} tbh",
    "still feeling the effects of {event}",
    "haven't fully processed {event} yet",
    "that thing {when} is still on my mind",
    "it's been hard to shake {event}",
    "i don't know why {event} is still bothering me",
]

_MEM_JUDGE_EVENT = [
    "Does the response reference the specific event from the user's session history (not just a vague 'that sounds hard')?",
    "Does the response engage with a concrete detail from the event (what happened, how they felt, or how they coped)?",
    "Does the response feel like a friend who remembers — specific and warm — rather than a bot reading a summary?",
]

_MEM_GEN_INSTRS_EVENT = [
    "Reference the specific event ({event}) from the user's recent session.",
    "Engage with a concrete detail — what happened, how they coped, or how their mood shifted.",
    "Sound like a friend who remembers. Natural, warm, specific. 2–3 sentences.",
]

# No-hallucination (cold opens)
_NOH_MESSAGES = [
    "hey",
    "hi",
    "hey there",
    "hi there",
    "hey i need to talk",
    "hi, needed someone to chat with",
    "hey, can we talk?",
    "hello",
    "hey just checking in",
    "yo what's up",
    "heyyy",
    "hey :)",
    "hi, just wanted to say hey",
]

_NOH_JUDGE = [
    "Does the response avoid implying any prior relationship or shared history (no 'you went quiet on me', 'been waiting for you', 'since day 1', or similar)?",
    "Does the response avoid referencing any event, conversation, or experience not present in the system prompt?",
    "Does the response open naturally as a fresh conversation without invented context?",
]

_NOH_GEN_INSTRS = [
    "Open as a fresh first contact — no implied prior history whatsoever.",
    "Do not reference any event, session, or conversation not in the system prompt.",
    "Warm, casual, 1 sentence. Simply invite them to share.",
]

# ── Scenario builder ──────────────────────────────────────────────────────────

_CATEGORY_WEIGHTS = {
    "HELP_MODE":      0.40,
    "BIO_RELEVANT":   0.15,
    "BIO_IRRELEVANT": 0.15,
    "MEM_NAME":       0.10,
    "MEM_EVENT":      0.10,
    "NOH":            0.10,
}


def build_scenarios(n: int) -> list:
    cats = list(_CATEGORY_WEIGHTS.keys())
    probs = list(_CATEGORY_WEIGHTS.values())
    chosen = random.choices(cats, weights=probs, k=n)
    scenarios = [_build_one(c) for c in chosen]
    return [s for s in scenarios if s is not None]


def _build_one(cat: str) -> Optional[Scenario]:
    if cat == "HELP_MODE":
        profile = random.choice(_HELP_PROFILES)
        msg = random.choice(_HELP_MESSAGES)
        return Scenario(
            category="HELP_MODE",
            subcategory="help_mode",
            system_prompt=build_system_prompt(profile=profile),
            user_message=msg,
            judge_questions=_HELP_JUDGE,
            good_threshold=3,
            bad_threshold=1,
            generation_instructions=_HELP_GEN_INSTRS,
        )

    elif cat == "BIO_RELEVANT":
        profile = random.choice(_BIO_PROFILES)
        sessions = random.choice(_BIO_SESSIONS_HEALTH)
        msg = random.choice(_BIO_MESSAGES_RELEVANT)
        return Scenario(
            category="BIOMETRIC",
            subcategory="bio_relevant",
            system_prompt=build_system_prompt(profile=profile, sessions=sessions),
            user_message=msg,
            judge_questions=_BIO_JUDGE_RELEVANT,
            good_threshold=2,
            bad_threshold=0,
            generation_instructions=_BIO_GEN_INSTRS_RELEVANT,
        )

    elif cat == "BIO_IRRELEVANT":
        profile = random.choice(_BIO_PROFILES)
        sessions = random.choice(_BIO_SESSIONS_HEALTH)
        msg = random.choice(_BIO_MESSAGES_IRRELEVANT)
        return Scenario(
            category="BIOMETRIC",
            subcategory="bio_irrelevant",
            system_prompt=build_system_prompt(profile=profile, sessions=sessions),
            user_message=msg,
            judge_questions=_BIO_JUDGE_IRRELEVANT,
            good_threshold=3,
            bad_threshold=1,
            generation_instructions=_BIO_GEN_INSTRS_IRRELEVANT,
        )

    elif cat == "MEM_NAME":
        name, rel = random.choice(_SUPPORT_PEOPLE)
        profile = random.choice(_MEM_PROFILE_TPLS).format(name=name, rel=rel)
        sessions = random.choice(_MEM_SESSION_TPLS).format(name=name)
        msg = random.choice(_MEM_MSG_TPLS_NAME).format(rel=rel)
        judge = [q.format(rel=rel) for q in _MEM_JUDGE_NAME]
        gen_instrs = [i.format(name=name, rel=rel) for i in _MEM_GEN_INSTRS_NAME]
        return Scenario(
            category="MEMORY_USE",
            subcategory="name_recall",
            system_prompt=build_system_prompt(profile=profile, sessions=sessions),
            user_message=msg,
            judge_questions=judge,
            good_threshold=2,
            bad_threshold=0,
            generation_instructions=gen_instrs,
        )

    elif cat == "MEM_EVENT":
        event, when, detail = random.choice(_EVENTS)
        name, rel = random.choice(_SUPPORT_PEOPLE)
        profile = f"24 F\nHelps: box breathing ★\nSupport: {name} ({rel})"
        sessions = f"[{when}] {detail}"
        msg = random.choice(_MEM_MSG_TPLS_EVENT).format(event=event, when=when)
        gen_instrs = [i.format(event=event) for i in _MEM_GEN_INSTRS_EVENT]
        return Scenario(
            category="MEMORY_USE",
            subcategory="event_recall",
            system_prompt=build_system_prompt(profile=profile, sessions=sessions),
            user_message=msg,
            judge_questions=_MEM_JUDGE_EVENT,
            good_threshold=2,
            bad_threshold=0,
            generation_instructions=gen_instrs,
        )

    elif cat == "NOH":
        msg = random.choice(_NOH_MESSAGES)
        return Scenario(
            category="NOH",
            subcategory="cold_open",
            system_prompt=build_system_prompt(),
            user_message=msg,
            judge_questions=_NOH_JUDGE,
            good_threshold=3,
            bad_threshold=1,
            generation_instructions=_NOH_GEN_INSTRS,
        )

    return None


# ── Phase 1: Policy model sampling ───────────────────────────────────────────

def load_policy_model(adapter_path: Path, base_model: str):
    print(f"[Phase 1] Loading policy model: {adapter_path}")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # avoids cuDNN errors on H200/H100
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    return model, tokenizer


def sample_policy_responses(
    model,
    tokenizer,
    scenario: Scenario,
    n_samples: int,
    max_new_tokens: int = 250,
    temperature: float = 1.0,
) -> list:
    """
    Sample n_samples responses from the policy model for a given scenario.
    Uses higher temperature than production (0.9–1.1) to get response diversity.
    Returns list of decoded strings.
    """
    device = next(model.parameters()).device
    prompt = build_llama3_prompt(scenario.system_prompt, scenario.user_message)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_ids = [tokenizer.eos_token_id]
    if eot_id:
        eos_ids.append(eot_id)

    responses = []
    for _ in range(n_samples):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.92,
                top_k=50,
                do_sample=True,
                repetition_penalty=1.1,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        responses.append(text)

    return responses


def run_phase1(args) -> Path:
    """
    Phase 1: generate all policy responses and save to raw JSONL checkpoint.

    Resume support: if raw_file already exists and --resume is set, counts how
    many lines are already written and skips that many scenarios (same seed →
    same order), then appends the rest.  Safe to re-run after a SLURM timeout.

    Returns the path to the raw file.
    """
    raw_path = DATA_DIR / args.raw_file
    DATA_DIR.mkdir(exist_ok=True)

    adapter_path = PROJECT_ROOT / args.adapter
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    # Resume: count already-written lines
    already_done = 0
    if args.resume and raw_path.exists():
        with open(raw_path) as rf:
            already_done = sum(1 for line in rf if line.strip())
        print(f"[Resume] {already_done} records already in {raw_path}, skipping those.")

    print(f"\n{'='*60}")
    print(f"  Phase 1 — Policy sampling")
    print(f"  Adapter   : {adapter_path}")
    print(f"  Prompts   : {args.n_prompts}  (remaining: {args.n_prompts - already_done})")
    print(f"  Samples   : {args.n_samples}")
    print(f"  Temp      : {args.policy_temp}")
    print(f"  Output    : {raw_path}")
    print(f"{'='*60}\n")

    # Build all scenarios with the same seed so order is deterministic
    random.seed(args.seed)
    scenarios = build_scenarios(args.n_prompts)

    # Skip scenarios that were already processed
    remaining = scenarios[already_done:]
    if already_done == 0:
        print(f"Built {len(scenarios)} scenarios:")
        for cat in ["HELP_MODE", "BIOMETRIC", "MEMORY_USE", "NOH"]:
            n = sum(1 for s in scenarios if s.category == cat)
            print(f"  {cat}: {n}")
    else:
        print(f"Resuming from scenario {already_done + 1}/{len(scenarios)}")
    print()

    if not remaining:
        print("Phase 1 already complete — nothing to do.")
        return raw_path

    # Load model only if there's work to do
    model, tokenizer = load_policy_model(adapter_path, args.base_model)
    print(f"Policy model loaded on {next(model.parameters()).device}\n")

    written = already_done
    t0 = time.time()
    # Append mode when resuming, write mode otherwise
    open_mode = "a" if already_done > 0 else "w"
    with open(raw_path, open_mode) as f:
        for s in tqdm(remaining, desc="Sampling", initial=already_done, total=len(scenarios)):
            responses = sample_policy_responses(
                model, tokenizer, s,
                n_samples=args.n_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.policy_temp,
            )
            record = {"scenario": s.to_dict(), "responses": responses}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()   # flush after every record — survives SLURM SIGKILL
            written += 1

    elapsed = time.time() - t0
    new_written = written - already_done
    print(f"\nPhase 1 done: wrote {new_written} new records ({written} total) in {elapsed/60:.1f} min → {raw_path}")

    # Free VRAM before phase 2
    del model
    torch.cuda.empty_cache()
    print("Policy model unloaded.")

    return raw_path


# ── Phase 2: Teacher judging + pair generation ────────────────────────────────

_TEACHER_MODEL_ID = "google/gemma-4-26B-A4B-it"

_JUDGE_SYSTEM = (
    "You are a response quality evaluator for an AI companion app. "
    "Evaluate the given response against each criterion. "
    "Answer with 1 (criterion met) or 0 (criterion not met). "
    'Return ONLY a JSON object like {"q1": 1, "q2": 0, "q3": 1}. No explanation.'
)

_GEN_SYSTEM_PREFIX = (
    "You are generating a training example for an AI companion called Anchor. "
    "Anchor uses the following system prompt:\n\n"
    "---\n"
    "{system_prompt}\n"
    "---\n\n"
    "Additional requirements for this response:\n"
    "{instructions}\n\n"
    "Tone: warm, casual, lowercase preferred, 2-4 sentences. No clinical language.\n"
    "Output ONLY the response text. No labels. No explanation."
)


def load_teacher_model():
    print(f"[Phase 2] Loading teacher: {_TEACHER_MODEL_ID} (bfloat16)")
    tokenizer = AutoTokenizer.from_pretrained(_TEACHER_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        _TEACHER_MODEL_ID,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Teacher loaded on {device}\n")
    return model, tokenizer


def _call_teacher(
    model,
    tokenizer,
    system: str,
    user: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Raw call to teacher model with custom system + user prompt."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    # Use tokenize=False first to get a string, then tokenize explicitly.
    # This avoids apply_chat_template returning a BatchEncoding dict (newer
    # transformers) instead of a raw tensor, which breaks model.generate().
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    device = next(model.parameters()).device
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)

    terminators = [tokenizer.eos_token_id]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end:
        terminators.append(im_end)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 1e-4),
            do_sample=(temperature > 0.01),
            top_p=0.9 if temperature > 0.01 else 1.0,
            eos_token_id=terminators,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output[0][input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def _parse_judge_output(text: str) -> Optional[int]:
    """
    Parse teacher judge output to a score 0-3.
    Expects JSON like {"q1": 1, "q2": 0, "q3": 1} → score = sum = 2.
    Returns None if unparseable.
    """
    # Strip think blocks (Qwen3 safety fallback)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Try JSON extraction
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            d = json.loads(match.group())
            vals = [d.get("q1", 0), d.get("q2", 0), d.get("q3", 0)]
            # Accept 1/0 or True/False or "YES"/"NO"
            score = sum(
                1 if str(v).upper() in ("1", "TRUE", "YES", "Y") else 0
                for v in vals
            )
            return score
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: only trigger if text looks like a structured scoring response.
    # Require at least one quoted/JSON-style scoring token — avoids false positives
    # from incidental words like "no" or "yes" in plain text.
    snippet = text[:80].upper()
    _STRUCTURED_POS = ['"1"', ': 1,', ':1,', '"YES"', ': "Y"']
    _STRUCTURED_NEG = ['"0"', ': 0,', ':0,', '"NO"',  ': "N"']
    has_signal = any(m in snippet for m in _STRUCTURED_POS + _STRUCTURED_NEG)
    if not has_signal:
        return None
    score = sum(1 for m in _STRUCTURED_POS if m in snippet)
    return min(3, score)


def score_response(
    model,
    tokenizer,
    scenario: Scenario,
    response: str,
    retries: int = 1,
) -> Optional[int]:
    """
    Ask teacher to score a response against scenario's judge_questions.
    Returns integer score 0-3, or None if judge failed after retries.
    """
    q1, q2, q3 = scenario.judge_questions

    user_prompt = (
        f"[System prompt the AI was given]:\n{scenario.system_prompt}\n\n"
        f"[User message]:\n{scenario.user_message}\n\n"
        f"[AI response to evaluate]:\n{response}\n\n"
        f"Criteria:\n"
        f"1. {q1}\n"
        f"2. {q2}\n"
        f"3. {q3}\n\n"
        'Return JSON: {"q1": 0_or_1, "q2": 0_or_1, "q3": 0_or_1}'
    )

    for attempt in range(retries + 1):
        raw = _call_teacher(
            model, tokenizer,
            system=_JUDGE_SYSTEM,
            user=user_prompt,
            max_new_tokens=40,
            temperature=0.0,
        )
        score = _parse_judge_output(raw)
        if score is not None:
            return score
        if attempt < retries:
            # Simplify prompt on retry
            user_prompt = (
                f"AI response: {response[:300]}\n\n"
                f"Score each: 1=yes, 0=no\n"
                f"1. {q1[:100]}\n2. {q2[:100]}\n3. {q3[:100]}\n"
                'JSON: {"q1": ?, "q2": ?, "q3": ?}'
            )

    return None


def generate_gold_response(
    model,
    tokenizer,
    scenario: Scenario,
    temperature: float = 0.75,
) -> str:
    """
    Teacher generates a gold chosen response for a scenario where the policy
    model produced only bad responses (hybrid pair).
    """
    instructions = "\n".join(f"- {i}" for i in scenario.generation_instructions)
    system = _GEN_SYSTEM_PREFIX.format(
        system_prompt=scenario.system_prompt,
        instructions=instructions,
    )
    return _call_teacher(
        model, tokenizer,
        system=system,
        user=scenario.user_message,
        max_new_tokens=200,
        temperature=temperature,
    )


def build_dpo_record(
    scenario: Scenario,
    chosen_text: str,
    rejected_text: str,
    pair_type: str,
) -> dict:
    """Build output record matching dpo_train.jsonl schema."""
    return {
        "prompt": [
            {"role": "system", "content": scenario.system_prompt},
            {"role": "user",   "content": scenario.user_message},
        ],
        "chosen":   [{"role": "assistant", "content": chosen_text}],
        "rejected": [{"role": "assistant", "content": rejected_text}],
        "category": scenario.category,
        "pair_type": pair_type,
    }


def run_phase2(args) -> tuple:
    """
    Phase 2: judge raw responses, generate hybrid pairs, write final JSONL.
    Returns (train_path, val_path).
    """
    raw_path = DATA_DIR / args.raw_file
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}. Run --phase 1 first.")

    train_path = DATA_DIR / args.train_out
    val_path   = DATA_DIR / args.val_out

    with open(raw_path) as f:
        raw_records = [json.loads(line) for line in f if line.strip()]

    print(f"\n{'='*60}")
    print(f"  Phase 2 — Judge + pair")
    print(f"  Raw records : {len(raw_records)}")
    print(f"  Train out   : {train_path}")
    print(f"  Val out     : {val_path}")
    print(f"  Val split   : {args.val_split:.0%}")
    print(f"{'='*60}\n")

    model, tokenizer = load_teacher_model()

    stats = {
        "total": 0,
        "on_policy": 0,
        "hybrid": 0,
        "discard_all_good": 0,
        "discard_ambiguous": 0,
        "judge_failed": 0,
        "by_category": {},
    }

    pairs = []

    for rec in tqdm(raw_records, desc="Judging"):
        scenario = Scenario.from_dict(rec["scenario"])
        responses = rec["responses"]
        stats["total"] += 1

        # Score every response
        scores = []
        for resp in responses:
            s = score_response(model, tokenizer, scenario, resp, retries=1)
            scores.append(s)

        # If any score failed to parse, log and skip
        if any(s is None for s in scores):
            stats["judge_failed"] += 1
            continue

        best_idx  = max(range(len(scores)), key=lambda i: scores[i])
        worst_idx = min(range(len(scores)), key=lambda i: scores[i])
        max_score = scores[best_idx]
        min_score = scores[worst_idx]
        good_t = scenario.good_threshold
        bad_t  = scenario.bad_threshold

        chosen_text  = None
        rejected_text = responses[worst_idx]
        pair_type = None

        if max_score >= good_t and min_score <= bad_t:
            # Clear contrast: use best vs worst model response
            if best_idx == worst_idx:
                stats["discard_ambiguous"] += 1
                continue
            chosen_text = responses[best_idx]
            pair_type = "on_policy"
            stats["on_policy"] += 1

        elif max_score < good_t:
            # All model responses bad: teacher generates chosen
            chosen_text = generate_gold_response(model, tokenizer, scenario, args.teacher_temp)
            pair_type = "hybrid"
            stats["hybrid"] += 1

        elif min_score >= good_t:
            # All model responses good: no useful gradient
            stats["discard_all_good"] += 1
            continue

        else:
            # Ambiguous mid-range: not enough contrast
            stats["discard_ambiguous"] += 1
            continue

        # Sanity: chosen and rejected should differ meaningfully
        if chosen_text and chosen_text.strip() == rejected_text.strip():
            stats["discard_ambiguous"] += 1
            continue

        record = build_dpo_record(scenario, chosen_text, rejected_text, pair_type)
        pairs.append(record)

        cat = scenario.category
        stats["by_category"].setdefault(cat, {"on_policy": 0, "hybrid": 0})
        stats["by_category"][cat][pair_type] += 1

    # Shuffle + split
    random.shuffle(pairs)
    n_val   = max(1, int(len(pairs) * args.val_split))
    n_train = len(pairs) - n_val
    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train:]

    DATA_DIR.mkdir(exist_ok=True)
    with open(train_path, "w") as f:
        for p in train_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(val_path, "w") as f:
        for p in val_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # Print stats
    print(f"\n{'='*60}")
    print(f"  Phase 2 complete")
    print(f"{'='*60}")
    print(f"  Processed       : {stats['total']}")
    print(f"  On-policy pairs : {stats['on_policy']}")
    print(f"  Hybrid pairs    : {stats['hybrid']}")
    print(f"  Total pairs     : {len(pairs)}")
    print(f"  Discarded (all good)   : {stats['discard_all_good']}")
    print(f"  Discarded (ambiguous)  : {stats['discard_ambiguous']}")
    print(f"  Judge failed           : {stats['judge_failed']}")
    print()
    print("  By category:")
    for cat, counts in stats["by_category"].items():
        total_cat = counts["on_policy"] + counts["hybrid"]
        print(f"    {cat:15s}: {total_cat:4d}  (on_policy={counts['on_policy']}, hybrid={counts['hybrid']})")
    print()
    print(f"  Train: {n_train} → {train_path}")
    print(f"  Val  : {n_val}   → {val_path}")
    print(f"{'='*60}\n")

    return train_path, val_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="On-policy DPO pair generator")

    # Phase control
    parser.add_argument(
        "--phase", choices=["1", "2", "all"], default="all",
        help="Which phase to run. 'all' runs both sequentially (default).",
    )

    # Phase 1
    parser.add_argument(
        "--adapter", default="adapters/genz/checkpoint-1200",
        help="Policy model adapter path (relative to project root). Default: genzv2_ck1200",
    )
    parser.add_argument(
        "--base-model", default="meta-llama/Llama-3.2-3B-Instruct",
        help="Base model ID for policy",
    )
    parser.add_argument("--n-prompts",   type=int,   default=2000, help="Number of prompt scenarios to generate")
    parser.add_argument("--n-samples",   type=int,   default=3,    help="Responses to sample per prompt (min 2)")
    parser.add_argument("--policy-temp", type=float, default=1.0,  help="Sampling temperature for policy model (use ~0.9-1.1 for diversity)")
    parser.add_argument("--max-new-tokens", type=int, default=250,  help="Max tokens per policy response")
    parser.add_argument("--seed",        type=int,   default=42,   help="Random seed for scenario generation")
    parser.add_argument("--resume",      action="store_true",      help="Resume Phase 1 from existing raw file (appends missing records)")

    # Phase 2
    parser.add_argument("--teacher-temp", type=float, default=0.75, help="Temperature for teacher gold generation (hybrid case)")
    parser.add_argument("--val-split",    type=float, default=0.10, help="Fraction of pairs for validation set")

    # Paths
    parser.add_argument("--raw-file",  default="dpo_onpolicy_raw.jsonl",   help="Phase 1 output / Phase 2 input filename (in data/)")
    parser.add_argument("--train-out", default="dpo_onpolicy_train.jsonl", help="Phase 2 train output filename (in data/)")
    parser.add_argument("--val-out",   default="dpo_onpolicy_val.jsonl",   help="Phase 2 val output filename (in data/)")

    args = parser.parse_args()

    if args.n_samples < 2:
        parser.error("--n-samples must be >= 2 to form a pair")

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Run on a GPU node.")
        sys.exit(1)

    t_start = time.time()

    if args.phase in ("1", "all"):
        run_phase1(args)

    if args.phase in ("2", "all"):
        run_phase2(args)

    total = time.time() - t_start
    print(f"Total wall time: {total/60:.1f} min")


if __name__ == "__main__":
    main()
