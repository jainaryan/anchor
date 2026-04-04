"""
MindMate Transition Dialogue Data Pipeline
Generates conversations that start casual and shift to emotional support,
teaching the model when and how to change tone naturally.
Runs until target count is reached or SLURM kills the job.
"""
import json
import time
import signal
import sys
from pathlib import Path
from utils import (
    TeacherModel, load_prompt, parse_json_robust,
    calculate_similarity
)

# Configuration
TARGET_COUNT = 10000         # Run for 2 days, grab as many as possible
SIMILARITY_THRESHOLD = 0.45

# Paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
RAW_FILE = OUTPUTS_DIR / "transition_dialogues_raw.jsonl"
TRAIN_FILE = BASE_DIR.parent / "data" / "synthetic_train_transition.jsonl"

# Graceful shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    print("\n[Pipeline] Shutdown signal received. Finishing current dialogue...")
    shutdown_requested = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def append_jsonl(data: dict, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


# Situations: casual opener + an emotional undercurrent that surfaces mid-convo
SITUATIONS = [
    {"situation": "user is procrastinating and chatting, then mentions they haven't spoken to their best friend in months after a fallout"},
    {"situation": "user is bored at home, then admits they've been avoiding going out because they feel anxious around people lately"},
    {"situation": "user just got back from a long day, then lets slip they cried in the bathroom at work"},
    {"situation": "user is watching a movie, then says it reminded them of their ex and they miss them"},
    {"situation": "user is talking about food, then mentions they haven't been eating properly because they've been really low"},
    {"situation": "user is chatting about a show, then reveals their dad is sick and they're trying not to think about it"},
    {"situation": "user is killing time before sleep, then admits they've been having nightmares and not sleeping well"},
    {"situation": "user is on a break at school, then says they've been feeling really invisible lately"},
    {"situation": "user is excited about a new game, then mentions they used to play it with someone who passed away"},
    {"situation": "user is complaining about a minor inconvenience, then suddenly says everything feels heavy lately"},
    {"situation": "user is joking around, then admits they've been really lonely since moving to a new city"},
    {"situation": "user is chatting casually, then says they don't know why they're crying right now"},
    {"situation": "user is talking about being tired, then reveals they've been having thoughts of just disappearing"},
    {"situation": "user is discussing weekend plans, then mentions they got rejected from something they really wanted"},
    {"situation": "user is casually venting about something small, then opens up about feeling like a failure"},
    {"situation": "user is chatting before bed, then says they're scared of tomorrow because of an important event"},
    {"situation": "user is talking about music, then says a song reminds them of when they were happier"},
    {"situation": "user is asking for food recommendations, then mentions they've been stress eating and feeling bad about it"},
    {"situation": "user is chatting about nothing, then says they feel like no one would notice if they just vanished"},
    {"situation": "user is on a walk and checking in, then admits they've been going on walks just to avoid being home"},
    {"situation": "user is talking about a friend group, then says they feel like they don't belong anywhere"},
    {"situation": "user is mildly excited about something, then suddenly gets quiet and says never mind, it doesn't matter"},
    {"situation": "user is procrastinating, then says they're scared they'll fail out and disappoint everyone"},
    {"situation": "user is chatting about random things, then mentions their parents are fighting a lot and it's affecting them"},
    {"situation": "user is casually talking, then reveals they just found out some bad news about their health"},
]


def heuristic_check(conv: list) -> bool:
    """Validates that the conversation has a real casual-to-serious shift."""
    if not conv or len(conv) < 8:  # at least 4 turns
        return False

    user_tokens = 0
    assistant_tokens = 0
    full_text = ""
    for msg in conv:
        tokens = len(msg["content"].split())
        if msg["role"] == "assistant":
            assistant_tokens += tokens
        else:
            user_tokens += tokens
        full_text += msg["content"] + " "

    total = user_tokens + assistant_tokens
    if total == 0:
        return False
    if (assistant_tokens / total) < 0.25:
        return False

    # Must NOT be all therapy-speak from the start
    therapy_heavy = ["grounding", "breathe in", "what do you see", "breathe out",
                     "breathing exercise", "let's do a quick"]
    if any(bad in full_text.lower() for bad in therapy_heavy):
        return False

    # Must NOT be all jokes with no shift
    forbidden_endings = ["lol", "haha", "😂", "lmao"]
    last_assistant = ""
    for msg in reversed(conv):
        if msg["role"] == "assistant":
            last_assistant = msg["content"].lower()
            break
    if any(last_assistant.endswith(e) for e in forbidden_endings):
        return False

    return True


def get_next_situation(attempt: int) -> dict:
    import random
    idx = attempt % len(SITUATIONS)
    if random.random() > 0.5:
        idx = random.randint(0, len(SITUATIONS) - 1)
    return SITUATIONS[idx]


def generate_transition_dialogue(teacher, gen_prompt, situation: dict) -> dict | None:
    prompt = gen_prompt.replace("{SITUATION_JSON}", json.dumps(situation))
    response = teacher.generate(prompt, max_new_tokens=1500, temperature=0.82)
    data = parse_json_robust(response, expected_keys=["conversations"])

    if data:
        append_jsonl(data, RAW_FILE)

    if not data or "conversations" not in data:
        return None

    conv = data["conversations"]
    if heuristic_check(conv):
        data["meta_situation"] = situation
        data["source"] = "transition"
        return data
    return None


def main():
    print("=========================================")
    print("   MindMate Transition Data Pipeline     ")
    print("=========================================")
    print(f"Target: {TARGET_COUNT} dialogues (2-day run)\n")

    start_time = time.time()
    teacher = TeacherModel()
    gen_prompt = load_prompt("transition_dialogue.txt")

    total_dialogues = 0
    attempts = 0

    while not shutdown_requested and total_dialogues < TARGET_COUNT:
        try:
            attempts += 1
            elapsed = time.time() - start_time
            elapsed_hrs = elapsed / 3600

            situation = get_next_situation(attempts)

            print(f"\n[Attempt {attempts}] ({elapsed_hrs:.1f}h | Saved: {total_dialogues}/{TARGET_COUNT})")
            print(f"  Situation: {situation['situation'][:70]}...")
            print(f"  Generating...", end="", flush=True)

            result = generate_transition_dialogue(teacher, gen_prompt, situation)

            if result:
                append_jsonl(result, TRAIN_FILE)
                total_dialogues += 1
                print(f" [Pass] Total: {total_dialogues}")
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
    print(f"Total dialogues saved: {total_dialogues}")
    print(f"Output: {TRAIN_FILE}")
    print(f"{'='*50}")
    print("Pipeline finished!")


if __name__ == "__main__":
    main()
