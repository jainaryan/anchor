"""
MindMate DPO Preference Pair Pipeline
Generates (prompt, chosen, rejected) triplets for DPO training.

50% single-mode pairs (one focused failure mode per pair)
50% mixed-mode pairs (casual → emotional → panic arc)

Uses Qwen3-72B in 4-bit on H100/H200 (set TEACHER_MODEL=72b).

Runs until TARGET_COUNT is reached or SLURM kills the job.
"""

import json
import time
import signal
import random
from pathlib import Path
from utils import (
    TeacherModel, load_prompt, parse_json_robust,
    calculate_similarity
)

# Configuration
TARGET_COUNT = 2500
VAL_RATIO = 0.15
SIMILARITY_THRESHOLD = 0.70  # reject pairs that are too similar

# Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
RAW_FILE = OUTPUTS_DIR / "dpo_pairs_raw.jsonl"
TRAIN_FILE = PROJECT_ROOT / "data" / "dpo_train.jsonl"
VAL_FILE = PROJECT_ROOT / "data" / "dpo_val.jsonl"
PARTIAL_FILE = PROJECT_ROOT / "data" / "dpo_pairs_partial.jsonl"  # incremental saves

CHECKPOINT_EVERY = 25  # save train/val split every N accepted pairs

SYSTEM_PROMPT = (PROJECT_ROOT / "system_prompt.txt").read_text(encoding="utf-8").strip()

# Graceful shutdown
shutdown_requested = False
_all_pairs_ref = []  # holds reference for signal handler to save on SIGTERM

def _checkpoint_save():
    if _all_pairs_ref:
        save_train_val_split(_all_pairs_ref)
        print(f"[Checkpoint] Saved {len(_all_pairs_ref)} pairs.")

def signal_handler(sig, frame):
    global shutdown_requested
    print("\n[Pipeline] Shutdown signal received — saving current pairs before exit...")
    _checkpoint_save()
    shutdown_requested = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def append_jsonl(data: dict, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


# Single-mode situations
SINGLE_MODE_SITUATIONS = [
    {"category": "casual_sad", "situation": "User texts casually saying they feel sad/off, not in crisis — just a low-key bleh day"},
    {"category": "casual_sad", "situation": "User says they've been feeling empty lately but brushes it off as 'probably nothing'"},
    {"category": "casual_sad", "situation": "User mentions they've been unmotivated and tired, in a very casual tone"},
    {"category": "casual_sad", "situation": "User says 'idk i just feel kinda down today' mid casual conversation"},
    {"category": "casual_sad", "situation": "User admits they've been crying but doesn't want to make a big deal of it"},
    {"category": "transition", "situation": "Casual banter about a movie, then user mentions it reminded them of a person they lost"},
    {"category": "transition", "situation": "Joking about being broke, then user reveals they got fired last week"},
    {"category": "transition", "situation": "Chatting about weekend plans, then user says they've been avoiding people lately"},
    {"category": "transition", "situation": "Talking about food, then user admits they haven't been eating properly because they're depressed"},
    {"category": "transition", "situation": "Banter about a game, then user mentions the friend they used to play it with passed away"},
    {"category": "panic_mode", "situation": "User has an exam in 15 minutes and is spiralling"},
    {"category": "panic_mode", "situation": "User needs to give a presentation in 10 minutes and their mind went blank"},
    {"category": "panic_mode", "situation": "User is having a panic attack right now and doesn't know what to do"},
    {"category": "panic_mode", "situation": "User has a difficult phone call in 5 minutes they've been dreading"},
    {"category": "system_compliance", "situation": "User keeps saying 'yeah' and 'idk' — conversation has stalled for 3 turns"},
    {"category": "system_compliance", "situation": "User expressed sadness; model has already validated once — user sends another short reply"},
    {"category": "system_compliance", "situation": "User says they feel better now; model should exit emotional support mode"},
    {"category": "hallucination_guard", "situation": "User mentioned losing a pet; model response invents details about the pet's name and personality"},
    {"category": "hallucination_guard", "situation": "User said they're stressed; model references a relationship the user never mentioned"},
    {"category": "hallucination_guard", "situation": "User mentioned a bad day; model invents context about their job or family situation"},
]

# Mixed-mode situations
MIXED_MODE_SITUATIONS = [
    {"category": "mixed_mode", "situation": "Starts casual (procrastinating, chatting), user reveals they've been struggling emotionally, then mentions a deadline in 20 minutes"},
    {"category": "mixed_mode", "situation": "Starts with banter about a show, user opens up about feeling lonely since moving cities, then says they have a job interview in 10 minutes"},
    {"category": "mixed_mode", "situation": "Casual talk about food, user admits they haven't been eating properly due to grief, then panics about a family call happening right now"},
    {"category": "mixed_mode", "situation": "Joking about being tired, user reveals they haven't slept in days due to anxiety, then says they have an exam starting now"},
    {"category": "mixed_mode", "situation": "Chatting about weekend, user mentions missing their ex badly, then realizes they have to see them in 15 minutes"},
    {"category": "mixed_mode", "situation": "Casual venting about work, user opens up about feeling like a failure, then a urgent work message arrives and they panic"},
    {"category": "mixed_mode", "situation": "Talking about music, user reveals the song reminds them of someone who passed, then says they have to give a eulogy speech in 20 minutes"},
    {"category": "mixed_mode", "situation": "Banter about school stress, user admits they've been having thoughts of dropping out, then realizes their parents are calling right now"},
    {"category": "mixed_mode", "situation": "Chatting casually, user mentions they've been feeling invisible lately, then says they have a therapy appointment in 10 minutes and feels unprepared"},
    {"category": "mixed_mode", "situation": "Talking about random stuff, user opens up about a fight with their best friend, then realizes they have to meet that friend in 15 minutes"},
]


def heuristic_check(data: dict) -> bool:
    prompt = data.get("prompt", [])
    chosen = data.get("chosen", [])
    rejected = data.get("rejected", [])

    if not prompt or not chosen or not rejected:
        return False

    chosen_text = chosen[0].get("content", "") if chosen else ""
    rejected_text = rejected[0].get("content", "") if rejected else ""

    if not chosen_text or not rejected_text:
        return False

    # Must be different enough
    if calculate_similarity(chosen_text, rejected_text) > SIMILARITY_THRESHOLD:
        return False

    # Both must have substance
    if len(chosen_text.split()) < 8 or len(rejected_text.split()) < 8:
        return False

    # Length parity — rejected shouldn't be way longer/shorter than chosen
    chosen_words = len(chosen_text.split())
    rejected_words = len(rejected_text.split())
    if max(chosen_words, rejected_words) / max(min(chosen_words, rejected_words), 1) > 2.5:
        return False

    # Chosen should NOT contain unsolicited grounding for casual_sad/transition
    category = data.get("category", "")
    grounding_phrases = ["4-7-8", "inhale for", "breathe in", "grounding exercise",
                         "name 5 things", "box breathing", "body scan"]
    if category in ("casual_sad", "transition"):
        if any(p in chosen_text.lower() for p in grounding_phrases):
            return False

    # Prompt must have at least one user message
    user_turns = [m for m in prompt if m.get("role") == "user"]
    if not user_turns:
        return False

    # Mixed mode needs longer prompt
    if category == "mixed_mode" and len(prompt) < 7:
        return False

    return True


def get_next_situation(attempt: int) -> dict:
    # 50% mixed, 50% single
    if attempt % 2 == 0:
        return random.choice(MIXED_MODE_SITUATIONS)
    else:
        return random.choice(SINGLE_MODE_SITUATIONS)


def generate_pair(teacher, gen_prompt_template: str, situation: dict) -> dict | None:
    pair_type = "mixed_mode" if situation["category"] == "mixed_mode" else "single_mode"

    # Use .replace() not .format() — the template contains literal JSON braces
    # which conflict with Python's str.format() placeholder syntax.
    prompt = (gen_prompt_template
        .replace("{system_prompt}", SYSTEM_PROMPT)
        .replace("{pair_type}", pair_type)
        .replace("{category}", situation["category"])
        .replace("{situation}", situation["situation"])
    )

    response = teacher.generate(prompt, max_new_tokens=1800, temperature=0.78)
    data = parse_json_robust(response, expected_keys=["prompt", "chosen", "rejected"])

    if data:
        data["pair_type"] = pair_type
        data["meta_situation"] = situation
        append_jsonl(data, RAW_FILE)

    if not data or "prompt" not in data or "chosen" not in data or "rejected" not in data:
        return None

    # Inject real system prompt if teacher left a placeholder
    for msg in data.get("prompt", []):
        if msg.get("role") == "system":
            msg["content"] = SYSTEM_PROMPT

    if heuristic_check(data):
        # Strip reason before saving to training file
        clean = {
            "prompt": data["prompt"],
            "chosen": data["chosen"],
            "rejected": data["rejected"],
            "category": data.get("category", situation["category"]),
            "pair_type": pair_type,
        }
        return clean

    return None


def save_train_val_split(all_pairs: list):
    random.shuffle(all_pairs)
    val_size = max(50, int(len(all_pairs) * VAL_RATIO))
    val = all_pairs[:val_size]
    train = all_pairs[val_size:]

    TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for p in train:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(VAL_FILE, "w", encoding="utf-8") as f:
        for p in val:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Saved: {len(train)} train / {len(val)} val")
    print(f"  Train: {TRAIN_FILE}")
    print(f"  Val:   {VAL_FILE}")


def main():
    print("=========================================")
    print("   MindMate DPO Preference Pipeline      ")
    print("=========================================")
    print(f"Target: {TARGET_COUNT} pairs (50% mixed / 50% single)\n")

    start_time = time.time()
    teacher = TeacherModel()
    gen_prompt_template = load_prompt("dpo_preference.txt")

    all_pairs = []
    _all_pairs_ref.clear()
    attempts = 0

    # Resume from partial file if it exists
    if PARTIAL_FILE.exists():
        with open(PARTIAL_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_pairs.append(json.loads(line))
        _all_pairs_ref.extend(all_pairs)
        print(f"[Resume] Loaded {len(all_pairs)} pairs from previous partial save.")

    while not shutdown_requested and len(all_pairs) < TARGET_COUNT:
        try:
            attempts += 1
            elapsed_hrs = (time.time() - start_time) / 3600
            situation = get_next_situation(attempts)

            print(f"\n[Attempt {attempts}] ({elapsed_hrs:.1f}h | Saved: {len(all_pairs)}/{TARGET_COUNT})")
            print(f"  Type: {situation['category']}")
            print(f"  Situation: {situation['situation'][:70]}...")
            print(f"  Generating...", end="", flush=True)

            result = generate_pair(teacher, gen_prompt_template, situation)

            if result:
                all_pairs.append(result)
                _all_pairs_ref.append(result)
                # Append immediately to partial file — survives hard kills
                append_jsonl(result, PARTIAL_FILE)
                print(f" [Pass] Total: {len(all_pairs)}")
                # Periodic checkpoint: rewrite train/val split every N pairs
                if len(all_pairs) % CHECKPOINT_EVERY == 0:
                    save_train_val_split(all_pairs)
                    print(f"[Checkpoint] {len(all_pairs)} pairs saved to train/val files.")
            else:
                print(f" [Fail]")

        except Exception as e:
            print(f"\n[ERROR] {type(e).__name__}: {e}")
            print("[Recovery] Continuing...")
            time.sleep(5)
            continue

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Pipeline finished after {elapsed/3600:.1f} hours")
    print(f"Total attempts: {attempts}")
    print(f"Total pairs saved: {len(all_pairs)}")

    save_train_val_split(all_pairs)
    print(f"{'='*50}")
    print("Pipeline finished!")


if __name__ == "__main__":
    main()
