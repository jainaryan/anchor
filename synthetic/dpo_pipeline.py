"""
MindMate DPO Preference Pair Pipeline
Generates (prompt, chosen, rejected) triplets for DPO training.

Category-quota weighted sampler — targets a fixed distribution per category.
Uses Qwen3-30B-A3B-Instruct-2507 teacher model.

Runs until TARGET_COUNT is reached or SLURM kills the job (resumes from partial).
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

# ── Configuration ─────────────────────────────────────────────────────────────
TARGET_COUNT = 8000
VAL_RATIO = 0.15
SIMILARITY_THRESHOLD = 0.70  # reject pairs that are too similar
CHECKPOINT_EVERY = 25        # save train/val split every N accepted pairs

# Category target distribution (must sum to 1.0)
CATEGORY_TARGETS = {
    "casual_sad":         0.20,
    "transition":         0.17,
    "mixed_mode":         0.23,
    "panic_mode":         0.18,
    "system_compliance":  0.09,
    "hallucination_guard":0.13,
}

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
OUTPUTS_DIR  = BASE_DIR / "outputs"
RAW_FILE     = OUTPUTS_DIR / "dpo_pairs_raw_v2.jsonl"
TRAIN_FILE   = PROJECT_ROOT / "data" / "dpo_train_v2.jsonl"
VAL_FILE     = PROJECT_ROOT / "data" / "dpo_val_v2.jsonl"
PARTIAL_FILE = PROJECT_ROOT / "data" / "dpo_pairs_partial_v2.jsonl"

SYSTEM_PROMPT = (PROJECT_ROOT / "system_prompt.txt").read_text(encoding="utf-8").strip()

# ── Graceful shutdown ─────────────────────────────────────────────────────────
shutdown_requested = False
_all_pairs_ref = []

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


# ── Helpers ───────────────────────────────────────────────────────────────────
def append_jsonl(data: dict, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


# ── Situations ────────────────────────────────────────────────────────────────
SINGLE_MODE_SITUATIONS = [
    # casual_sad (12 situations)
    {"category": "casual_sad", "situation": "User texts casually saying they feel sad/off, not in crisis — just a low-key bleh day"},
    {"category": "casual_sad", "situation": "User says they've been feeling empty lately but brushes it off as 'probably nothing'"},
    {"category": "casual_sad", "situation": "User mentions they've been unmotivated and tired, in a very casual tone"},
    {"category": "casual_sad", "situation": "User says 'idk i just feel kinda down today' mid casual conversation"},
    {"category": "casual_sad", "situation": "User admits they've been crying but doesn't want to make a big deal of it"},
    {"category": "casual_sad", "situation": "User says 'ngl been feeling super lonely lately' then quickly changes subject"},
    {"category": "casual_sad", "situation": "User mentions they've been staying in bed all day and laughs it off with 'classic me'"},
    {"category": "casual_sad", "situation": "User says 'i don't know why but i just feel really heavy today'"},
    {"category": "casual_sad", "situation": "User admits they've been avoiding friends and can't explain why, tone is casual not crisis"},
    {"category": "casual_sad", "situation": "User says they cried watching a random video and seems embarrassed about it"},
    {"category": "casual_sad", "situation": "User mentions they've been feeling disconnected from everything lately, like they're just going through the motions"},
    {"category": "casual_sad", "situation": "User says 'everything's fine i guess i'm just being dramatic' after mentioning feeling off"},

    # transition (12 situations)
    {"category": "transition", "situation": "Casual banter about a movie, then user mentions it reminded them of a person they lost"},
    {"category": "transition", "situation": "Joking about being broke, then user reveals they got fired last week"},
    {"category": "transition", "situation": "Chatting about weekend plans, then user says they've been avoiding people lately"},
    {"category": "transition", "situation": "Talking about food, then user admits they haven't been eating properly because they're depressed"},
    {"category": "transition", "situation": "Banter about a game, then user mentions the friend they used to play it with passed away"},
    {"category": "transition", "situation": "Joking about school stress, then user drops 'nah but for real I've been really depressed lately'"},
    {"category": "transition", "situation": "Two turns of banter about the weekend, user suddenly says 'I think my parents are getting divorced'"},
    {"category": "transition", "situation": "Laughing about a meme, user goes quiet then says 'sorry i'm just not really okay today'"},
    {"category": "transition", "situation": "Talking about a TV show, user says 'the main character reminds me of my dad who passed last year'"},
    {"category": "transition", "situation": "Casual chat about music, user admits the song they were talking about was playing when they got some really bad news"},
    {"category": "transition", "situation": "Joking about being tired, user says 'honestly haven't been sleeping because of anxiety' in a serious tone"},
    {"category": "transition", "situation": "Banter about social media, user says 'yeah i deleted everything because seeing people happy was making me feel worse'"},

    # panic_mode (10 situations)
    {"category": "panic_mode", "situation": "User has an exam in 15 minutes and is spiralling"},
    {"category": "panic_mode", "situation": "User needs to give a presentation in 10 minutes and their mind went blank"},
    {"category": "panic_mode", "situation": "User is having a panic attack right now and doesn't know what to do"},
    {"category": "panic_mode", "situation": "User has a difficult phone call in 5 minutes they've been dreading"},
    {"category": "panic_mode", "situation": "User has to send a difficult email in the next few minutes or miss the deadline"},
    {"category": "panic_mode", "situation": "User is sitting outside a job interview room right now and completely frozen"},
    {"category": "panic_mode", "situation": "User has to walk into a family confrontation happening in the next room right now"},
    {"category": "panic_mode", "situation": "User is about to break up with someone in 10 minutes and is shaking"},
    {"category": "panic_mode", "situation": "User's heart is racing, they feel dizzy, they don't know if it's a panic attack or something physical"},
    {"category": "panic_mode", "situation": "User has 5 minutes before a doctor's appointment they've been scared of and is about to bolt"},

    # system_compliance (8 situations)
    {"category": "system_compliance", "situation": "User keeps saying 'yeah' and 'idk' — conversation has stalled for 3 turns"},
    {"category": "system_compliance", "situation": "User expressed sadness; model has already validated once — user sends another short reply"},
    {"category": "system_compliance", "situation": "User says they feel better now; model should exit emotional support mode"},
    {"category": "system_compliance", "situation": "User gave two 'mhm' replies in a row — model keeps probing deeper instead of switching strategy"},
    {"category": "system_compliance", "situation": "User has been in distress for several turns; model keeps repeating 'I'm here for you' instead of evolving"},
    {"category": "system_compliance", "situation": "User said 'I don't want to talk about it' — model should respect this and shift to quiet presence"},
    {"category": "system_compliance", "situation": "Model asked a question; user gave a one-word answer; model immediately fires two more questions"},
    {"category": "system_compliance", "situation": "User says 'nevermind it's fine' after opening up — model should gently acknowledge, not push"},

    # hallucination_guard (10 situations)
    {"category": "hallucination_guard", "situation": "User said 'I've been really stressed lately' — model invents a specific cause like a breakup or job pressure they never mentioned"},
    {"category": "hallucination_guard", "situation": "User mentioned losing a pet — model invents the pet's name, breed, or how long they had it"},
    {"category": "hallucination_guard", "situation": "User said they're stressed about work — model references a specific boss, colleague, or project the user never mentioned"},
    {"category": "hallucination_guard", "situation": "User mentioned a bad day — model invents context about their family situation or living arrangements"},
    {"category": "hallucination_guard", "situation": "User said they feel lonely — model references a specific breakup or falling out with a friend the user never disclosed"},
    {"category": "hallucination_guard", "situation": "User said 'things have been hard lately' — model responds as if it knows it's about money or health when nothing was specified"},
    {"category": "hallucination_guard", "situation": "User mentioned missing someone — model invents whether it's a death, a move, or a relationship ending"},
    {"category": "hallucination_guard", "situation": "User said 'I had a rough night' — model invents a specific reason like insomnia or a nightmare or a fight"},
    {"category": "hallucination_guard", "situation": "User expressed feeling stuck in life — model references specific details about their career path or age that they never mentioned"},
    {"category": "hallucination_guard", "situation": "User mentioned feeling disconnected from people — model invents that they recently moved cities or changed schools"},
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
    {"category": "mixed_mode", "situation": "Joking about hating mornings, user admits they've been avoiding getting up because they feel hopeless, then says they have to be somewhere in 20 minutes"},
    {"category": "mixed_mode", "situation": "Casual chat about a hobby, user reveals they stopped doing it after a close friend drifted away, then gets a text from that friend and panics"},
]


# ── Quota-weighted sampler ────────────────────────────────────────────────────
def get_next_situation(category_counts: dict, total: int) -> dict:
    """
    Sample the next situation weighted by how far each category is below its target.
    Ensures the final distribution tracks CATEGORY_TARGETS throughout the run.
    """
    deficits = {}
    for cat, target_pct in CATEGORY_TARGETS.items():
        current_pct = category_counts.get(cat, 0) / max(total, 1)
        deficits[cat] = max(0.0, target_pct - current_pct)

    total_deficit = sum(deficits.values())

    if total_deficit == 0:
        # All categories on target — sample uniformly
        cat = random.choice(list(CATEGORY_TARGETS.keys()))
    else:
        r = random.random() * total_deficit
        cumulative = 0.0
        cat = list(CATEGORY_TARGETS.keys())[-1]  # fallback
        for c, d in deficits.items():
            cumulative += d
            if r <= cumulative:
                cat = c
                break

    if cat == "mixed_mode":
        return random.choice(MIXED_MODE_SITUATIONS)
    pool = [s for s in SINGLE_MODE_SITUATIONS if s["category"] == cat]
    return random.choice(pool) if pool else random.choice(SINGLE_MODE_SITUATIONS)


# ── Heuristics ────────────────────────────────────────────────────────────────
def heuristic_check(data: dict) -> bool:
    prompt   = data.get("prompt", [])
    chosen   = data.get("chosen", [])
    rejected = data.get("rejected", [])

    if not prompt or not chosen or not rejected:
        return False

    chosen_text   = chosen[0].get("content", "") if chosen else ""
    rejected_text = rejected[0].get("content", "") if rejected else ""

    if not chosen_text or not rejected_text:
        return False

    # Must be different enough
    if calculate_similarity(chosen_text, rejected_text) > SIMILARITY_THRESHOLD:
        return False

    # Both must have substance
    chosen_words   = len(chosen_text.split())
    rejected_words = len(rejected_text.split())
    if chosen_words < 8 or rejected_words < 8:
        return False

    # Tighter length ratio cap (was 2.5x)
    if max(chosen_words, rejected_words) / max(min(chosen_words, rejected_words), 1) > 2.0:
        return False

    # For non-panic categories: chosen must not be extremely short relative to rejected
    # This breaks the spurious "shorter = better" signal
    category = data.get("category", "")
    if category not in ("panic_mode",) and chosen_words < rejected_words * 0.6:
        return False

    # Chosen should NOT contain unsolicited grounding for casual_sad/transition
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


# ── Pair generation ───────────────────────────────────────────────────────────
def generate_pair(teacher, gen_prompt_template: str, situation: dict) -> dict | None:
    pair_type = "mixed_mode" if situation["category"] == "mixed_mode" else "single_mode"

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
        clean = {
            "prompt":   data["prompt"],
            "chosen":   data["chosen"],
            "rejected": data["rejected"],
            "category": data.get("category", situation["category"]),
            "pair_type": pair_type,
        }
        return clean

    return None


# ── Train/val split ───────────────────────────────────────────────────────────
def save_train_val_split(all_pairs: list):
    random.shuffle(all_pairs)
    val_size = max(50, int(len(all_pairs) * VAL_RATIO))
    val   = all_pairs[:val_size]
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=========================================")
    print("   MindMate DPO Preference Pipeline v2   ")
    print("=========================================")
    print(f"Target: {TARGET_COUNT} pairs")
    print(f"Category targets: {CATEGORY_TARGETS}\n")

    start_time = time.time()
    teacher = TeacherModel()
    gen_prompt_template = load_prompt("dpo_preference.txt")

    all_pairs = []
    _all_pairs_ref.clear()
    attempts = 0

    # Per-category counters for quota sampler
    category_counts: dict[str, int] = {cat: 0 for cat in CATEGORY_TARGETS}

    # Resume from partial file if it exists
    if PARTIAL_FILE.exists():
        with open(PARTIAL_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pair = json.loads(line)
                    all_pairs.append(pair)
                    cat = pair.get("category", "")
                    if cat in category_counts:
                        category_counts[cat] += 1
        _all_pairs_ref.extend(all_pairs)
        print(f"[Resume] Loaded {len(all_pairs)} pairs from previous partial save.")
        print(f"[Resume] Category counts: {category_counts}")

    while not shutdown_requested and len(all_pairs) < TARGET_COUNT:
        try:
            attempts += 1
            elapsed_hrs = (time.time() - start_time) / 3600
            situation = get_next_situation(category_counts, len(all_pairs))

            # Print current distribution every 50 attempts
            if attempts % 50 == 1 and len(all_pairs) > 0:
                print(f"\n[Distribution @ {len(all_pairs)} pairs]")
                for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
                    pct = 100 * count / len(all_pairs)
                    target_pct = 100 * CATEGORY_TARGETS.get(cat, 0)
                    print(f"  {cat:25s}: {count:4d} ({pct:.0f}% / target {target_pct:.0f}%)")

            print(f"\n[Attempt {attempts}] ({elapsed_hrs:.1f}h | Saved: {len(all_pairs)}/{TARGET_COUNT})")
            print(f"  Type: {situation['category']}")
            print(f"  Situation: {situation['situation'][:70]}...")
            print(f"  Generating...", end="", flush=True)

            result = generate_pair(teacher, gen_prompt_template, situation)

            if result:
                all_pairs.append(result)
                _all_pairs_ref.append(result)
                cat = result.get("category", "")
                if cat in category_counts:
                    category_counts[cat] += 1
                append_jsonl(result, PARTIAL_FILE)
                print(f" [Pass] Total: {len(all_pairs)}")
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
    print(f"Acceptance rate: {len(all_pairs)/max(attempts,1)*100:.1f}%")
    print(f"Total pairs saved: {len(all_pairs)}")
    print(f"Final distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(len(all_pairs), 1)
        print(f"  {cat:25s}: {count} ({pct:.1f}%)")

    save_train_val_split(all_pairs)
    print(f"{'='*50}")
    print("Pipeline finished!")


if __name__ == "__main__":
    main()
