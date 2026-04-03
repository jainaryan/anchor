"""
MindMate Casual Conversation Data Pipeline
Generates casual, non-distress conversations to balance the training distribution.
Runs until target count is reached or Slurm kills the job.
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
TARGET_COUNT = 5000          # Stop after this many saved dialogues
SIMILARITY_THRESHOLD = 0.45

# Paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
RAW_FILE = OUTPUTS_DIR / "casual_dialogues_raw.jsonl"
TRAIN_FILE = BASE_DIR.parent / "data" / "synthetic_train_casual.jsonl"

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


SITUATIONS = [
    {"situation": "user is procrastinating on studying or work and just wants to chat"},
    {"situation": "user is bored at home with nothing to do"},
    {"situation": "user just got back from a long day and wants to decompress by chatting"},
    {"situation": "user is waiting for something (food delivery, a friend, a bus) and killing time"},
    {"situation": "user is watching TV or a movie and wants to talk about it"},
    {"situation": "user just woke up and is slowly getting going"},
    {"situation": "user is on a break at work/school and checking in"},
    {"situation": "user is mildly annoyed about something small and trivial (not serious)"},
    {"situation": "user is excited about something low-stakes like a new game, show, or food"},
    {"situation": "user is curious about a random topic and wants to chat about it"},
    {"situation": "user had a decent day, nothing special, just checking in"},
    {"situation": "user is trying to decide what to eat and asks for thoughts"},
    {"situation": "user is slightly tired but not distressed, just chatting before bed"},
    {"situation": "user is restless and wants distraction, not emotional support"},
    {"situation": "user wants to vent lightly about a minor inconvenience, not a crisis"},
    {"situation": "user is in a good mood and just wants to hang"},
    {"situation": "user is curious what MindMate thinks about something mundane"},
    {"situation": "user is taking a walk and checking in mid-way"},
    {"situation": "user just finished a task and feels mildly satisfied"},
    {"situation": "user is a bit lonely but not sad, just wants some company"},
]


def heuristic_check(conv: list) -> bool:
    if not conv or len(conv) < 6:  # at least 3 turns
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
    # For casual convos, assistant ratio can be lower (short replies are fine)
    if (assistant_tokens / total) < 0.25:
        return False
    forbidden = ["as an AI", "language model", "sin ", "repent", "god's plan",
                 "breathe in", "breathe out", "grounding", "what do you see",
                 "that sounds heavy", "you're not alone in this"]
    if any(bad in full_text.lower() for bad in forbidden):
        return False
    return True


def get_next_situation(used_situations: list) -> dict:
    """Cycle through situations with some repetition allowed."""
    import random
    # Weight toward less-used situations
    idx = len(used_situations) % len(SITUATIONS)
    # Add some randomness
    if random.random() > 0.6:
        idx = random.randint(0, len(SITUATIONS) - 1)
    return SITUATIONS[idx]


def generate_casual_dialogue(teacher, gen_prompt, situation: dict) -> dict | None:
    prompt = gen_prompt.replace("{SITUATION_JSON}", json.dumps(situation))
    response = teacher.generate(prompt, max_new_tokens=1200, temperature=0.85)
    data = parse_json_robust(response, expected_keys=["conversations"])

    if data:
        append_jsonl(data, RAW_FILE)

    if not data or "conversations" not in data:
        return None

    conv = data["conversations"]
    if heuristic_check(conv):
        data["meta_situation"] = situation
        data["source"] = "casual"
        return data
    return None


def main():
    print("=========================================")
    print("   MindMate Casual Data Pipeline         ")
    print("=========================================")
    print(f"Target: {TARGET_COUNT} dialogues\n")

    start_time = time.time()
    teacher = TeacherModel()
    gen_prompt = load_prompt("casual_dialogue.txt")

    used_situations = []
    total_dialogues = 0
    attempts = 0

    while not shutdown_requested and total_dialogues < TARGET_COUNT:
        try:
            attempts += 1
            elapsed = time.time() - start_time
            elapsed_hrs = elapsed / 3600

            situation = get_next_situation(used_situations)
            used_situations.append(situation)

            print(f"\n[Attempt {attempts}] ({elapsed_hrs:.1f}h | Saved: {total_dialogues}/{TARGET_COUNT})")
            print(f"  Situation: {situation['situation'][:70]}...")
            print(f"  Generating...", end="", flush=True)

            result = generate_casual_dialogue(teacher, gen_prompt, situation)

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


if __name__ == "__main__":
    main()
