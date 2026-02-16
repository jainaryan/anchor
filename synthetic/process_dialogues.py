from utils import TeacherModel, load_prompt, save_jsonl, parse_json_robust
import json
from pathlib import Path

# Configuration
INPUT_FILE = "scenarios.jsonl"
RAW_FILE = "dialogues_raw.jsonl"
OUTPUT_FILE = "../../data/synthetic_train.jsonl"
DIALOGUES_PER_SCENARIO = 2
PROMPT_FILE = "friend_dialogue.txt"

def heuristic_check(conv: list) -> bool:
    """Returns True if dialogue passes basic quality checks."""
    if not conv or len(conv) < 4:
        return False # production threshold
        
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
    if total == 0: return False
    
    # Assistant Token Ratio (> 35%)
    if (assistant_tokens / total) < 0.35:
        return False
        
    # Forbidden Phrases & Topics
    forbidden = ["as an AI", "language model", "sin ", "repent", "god's plan"]
    if any(bad in full_text.lower() for bad in forbidden):
        return False
        
    return True

def main():
    print(f"--- Starting Integrated Dialogue Processing ({PROMPT_FILE}) ---")
    teacher = TeacherModel()
    gen_prompt = load_prompt(PROMPT_FILE)
    critique_prompt = load_prompt("critique.txt")
    
    scenario_path = Path(__file__).resolve().parent / "outputs" / INPUT_FILE
    if not scenario_path.exists():
        print(f"[Error] No scenarios found at {scenario_path}")
        return

    scenarios = []
    with open(scenario_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): scenarios.append(json.loads(line))
            
    final_data = []
    base_path = Path(__file__).resolve().parent / "outputs"
    raw_path = base_path / RAW_FILE
    
    # Ensure raw file exists/cleared if starting fresh? 
    # For now, append mode is safer for long runs.
    
    for s_idx, scenario in enumerate(scenarios):
        print(f"\n[Scenario {s_idx + 1}/{len(scenarios)}] {scenario['core_emotion']}")
        
        for d_idx in range(DIALOGUES_PER_SCENARIO):
            print(f"  Dialogue {d_idx + 1}: Generating...", end="", flush=True)
            
            # 1. Generate Raw
            raw_prompt = gen_prompt.replace("{SCENARIO_JSON}", json.dumps(scenario))
            response = teacher.generate(raw_prompt)
            data = parse_json_robust(response, expected_keys=["conversations"])
            
            # --- SAVE RAW ALWAYS ---
            if data:
                save_jsonl([data], str(RAW_FILE)) # utils.save_jsonl handles relative path
            
            if not data or "conversations" not in data:
                print(" [Fail: No JSON]")
                continue
                
            conv = data["conversations"]
            
            # 2. Heuristic Filter
            if heuristic_check(conv):
                print(" [Pass: Heuristics]")
                data["meta_scenario"] = scenario
                data["source"] = "original"
                final_data.append(data)
            else:
                # 3. Critique / Rewrite Loop (REPAIR)
                print(" [Fail: Heuristics] -> Repairing...", end="", flush=True)
                conv_text = json.dumps({"conversations": conv}, indent=2)
                rewrite_req = f"Review this therapy conversation:\n\n{conv_text}\n\n{critique_prompt}"
                
                rewrite_resp = teacher.generate(rewrite_req)
                rewritten_data = parse_json_robust(rewrite_resp, expected_keys=["conversations"])
                
                # --- SAVE REWRITTEN TO RAW ALSO (for debug) ---
                if rewritten_data:
                    rewritten_data["meta_scenario"] = scenario
                    rewritten_data["source"] = "rewrite_attempt"
                    save_jsonl([rewritten_data], str(RAW_FILE))
                
                if rewritten_data and "conversations" in rewritten_data:
                    if heuristic_check(rewritten_data["conversations"]):
                         print(" [Success: Repaired]")
                         rewritten_data["source"] = "repaired"
                         final_data.append(rewritten_data)
                    else:
                         print(" [Fail: Repair still below threshold]")
                else:
                    print(" [Fail: Repair Parse Error]")

    # Save to final data dir
    output_path = Path(__file__).resolve().parent / "outputs" / OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving {len(final_data)} high-quality dialogues to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in final_data:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    main()
