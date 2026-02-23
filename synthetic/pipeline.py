"""
MindMate Continuous Synthetic Data Pipeline
Runs indefinitely, generating scenarios and dialogues until Slurm kills the job.
Saves incrementally after each dialogue so no work is lost.
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
DIALOGUES_PER_SCENARIO = 2
SIMILARITY_THRESHOLD = 0.45
# Paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
SCENARIOS_FILE = OUTPUTS_DIR / "scenarios.jsonl"
RAW_FILE = OUTPUTS_DIR / "dialogues_raw.jsonl"
TRAIN_FILE = BASE_DIR.parent / "data" / "synthetic_train.jsonl"
# Graceful shutdown flag
shutdown_requested = False
def signal_handler(sig, frame):
    global shutdown_requested
    print("\n[Pipeline] Shutdown signal received. Finishing current dialogue...")
    shutdown_requested = True
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
def append_jsonl(data: dict, filepath: Path):
    """Append a single JSON entry to a JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
def heuristic_check(conv: list) -> bool:
    """Returns True if dialogue passes basic quality checks."""
    if not conv or len(conv) < 4:
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
    if (assistant_tokens / total) < 0.35:
        return False
    forbidden = ["as an AI", "language model", "sin ", "repent", "god's plan"]
    if any(bad in full_text.lower() for bad in forbidden):
        return False
    return True
def generate_one_scenario(teacher, scenario_prompt_template, valid_scenarios):
    """Generate a single unique scenario, checking similarity against previous ones."""
    expected = ["core_emotion", "context", "intensity", "personality"]
    max_attempts = 5
    for _ in range(max_attempts):
        recent_str = "\n".join(
            [f"- {s['context']}" for s in valid_scenarios[-5:]]
        ) if valid_scenarios else "None yet."
        prompt = scenario_prompt_template.replace("{RECENT_CONTEXTS}", recent_str)
        response = teacher.generate(prompt, max_new_tokens=800, temperature=0.9)
        data = parse_json_robust(response, expected_keys=expected)
        if not data:
            continue
        # Similarity check
        core_emo = data.get("core_emotion", "unknown")
        ctx = data.get("context", "unknown")
        comp = data.get("complicating_factor", "none")
        current_text = f"{core_emo} {ctx} {comp}"
        is_too_similar = False
        for prev in valid_scenarios[-50:]:  # Only check last 50 for speed
            prev_text = f"{prev.get('core_emotion', '')} {prev.get('context', '')} {prev.get('complicating_factor', '')}"
            if calculate_similarity(current_text, prev_text) > SIMILARITY_THRESHOLD:
                is_too_similar = True
                break
        if not is_too_similar:
            return data
    return None
def generate_dialogues_for_scenario(teacher, scenario, gen_prompt, critique_prompt):
    """Generate DIALOGUES_PER_SCENARIO dialogues for a given scenario."""
    results = []
    for d_idx in range(DIALOGUES_PER_SCENARIO):
        print(f"  Dialogue {d_idx + 1}: Generating...", end="", flush=True)
        raw_prompt = gen_prompt.replace("{SCENARIO_JSON}", json.dumps(scenario))
        response = teacher.generate(raw_prompt)
        data = parse_json_robust(response, expected_keys=["conversations"])
        # Save raw attempt
        if data:
            append_jsonl(data, RAW_FILE)
        if not data or "conversations" not in data:
            print(" [Fail: No JSON]")
            continue
        conv = data["conversations"]
        if heuristic_check(conv):
            print(" [Pass]")
            data["meta_scenario"] = scenario
            data["source"] = "original"
            results.append(data)
        else:
            # Critique / Repair
            print(" [Repairing...", end="", flush=True)
            conv_text = json.dumps({"conversations": conv}, indent=2)
            rewrite_req = f"Review this therapy conversation:\n\n{conv_text}\n\n{critique_prompt}"
            rewrite_resp = teacher.generate(rewrite_req)
            rewritten = parse_json_robust(rewrite_resp, expected_keys=["conversations"])
            if rewritten:
                rewritten["meta_scenario"] = scenario
                rewritten["source"] = "rewrite_attempt"
                append_jsonl(rewritten, RAW_FILE)
            if rewritten and "conversations" in rewritten and heuristic_check(rewritten["conversations"]):
                print(" Repaired]")
                rewritten["source"] = "repaired"
                results.append(rewritten)
            else:
                print(" Failed]")
    return results
def main():
    print("=========================================")
    print("   MindMate Continuous Data Pipeline     ")
    print("=========================================")
    print("Running until Slurm time limit is reached.\n")
    start_time = time.time()
    # Load model once
    teacher = TeacherModel()
    scenario_prompt = load_prompt("scenario.txt")
    gen_prompt = load_prompt("friend_dialogue.txt")
    critique_prompt = load_prompt("critique.txt")
    # Track all generated scenarios for diversity
    all_scenarios = []
    total_dialogues = 0
    scenario_count = 0
    while not shutdown_requested:
        scenario_count += 1
        elapsed = time.time() - start_time
        elapsed_hrs = elapsed / 3600
        print(f"\n{'='*50}")
        print(f"[Scenario {scenario_count}] (Elapsed: {elapsed_hrs:.1f}h | Dialogues so far: {total_dialogues})")
        print(f"{'='*50}")
        # 1. Generate a unique scenario
        scenario = generate_one_scenario(teacher, scenario_prompt, all_scenarios)
        if not scenario:
            print("[Skip] Could not generate a unique scenario after 5 attempts.")
            continue
        all_scenarios.append(scenario)
        append_jsonl(scenario, SCENARIOS_FILE)
        print(f"[Scenario] {scenario.get('core_emotion', '?')} - {scenario.get('context', '?')[:60]}...")
        # 2. Generate dialogues for this scenario
        dialogues = generate_dialogues_for_scenario(
            teacher, scenario, gen_prompt, critique_prompt
        )
        # 3. Save each dialogue immediately
        for d in dialogues:
            append_jsonl(d, TRAIN_FILE)
            total_dialogues += 1
        print(f"[Progress] +{len(dialogues)} dialogues | Total: {total_dialogues}")
        if shutdown_requested:
            break
    # Final summary
    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Pipeline stopped after {elapsed/3600:.1f} hours")
    print(f"Total scenarios: {scenario_count}")
    print(f"Total dialogues saved: {total_dialogues}")
    print(f"{'='*50}")
if __name__ == "__main__":
    main()