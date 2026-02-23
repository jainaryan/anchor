from utils import TeacherModel, load_prompt, save_jsonl, parse_json_robust, calculate_similarity
import tqdm
import json
import re

# Configuration
NUM_SCENARIOS =  # 2 mock scenarios
OUTPUT_FILE = "scenarios.jsonl"
SIMILARITY_THRESHOLD = 0.45 

def main():
    print("--- Starting Open-Ended Scenario Generation ---")
    teacher = TeacherModel()
    prompt_template = load_prompt("scenario.txt")
    
    valid_scenarios = []
    expected = ["core_emotion", "context", "intensity", "personality"]
    
    # Track recent contexts to inform the model
    recent_contexts = []
    
    pbar = tqdm.tqdm(total=NUM_SCENARIOS)
    while len(valid_scenarios) < NUM_SCENARIOS:
        # Prepare iterative diversity list
        recent_str = "\n".join([f"- {s['context']}" for s in valid_scenarios[-5:]]) if valid_scenarios else "None yet."
        prompt = prompt_template.replace("{RECENT_CONTEXTS}", recent_str)
        
        response = teacher.generate(prompt, max_new_tokens=800, temperature=0.9)
        data = parse_json_robust(response, expected_keys=expected)
        
        if data:
            # 1. Similarity Check
            core_emo = data.get('core_emotion', 'unknown')
            ctx = data.get('context', 'unknown')
            comp = data.get('complicating_factor', 'none')
            
            current_text = f"{core_emo} {ctx} {comp}"
            is_too_similar = False
            
            for prev in valid_scenarios:
                prev_text = f"{prev.get('core_emotion', '')} {prev.get('context', '')} {prev.get('complicating_factor', '')}"
                sim = calculate_similarity(current_text, prev_text)
                if sim > SIMILARITY_THRESHOLD:
                    print(f"\n[Discarded] Too similar (Score: {sim:.2f}) to existing scenario.")
                    is_too_similar = True
                    break
            
            if not is_too_similar:
                print(f"\n[Success] Scenario {len(valid_scenarios)+1}: {data['core_emotion']} - {data['context'][:50]}...")
                valid_scenarios.append(data)
                pbar.update(1)
        else:
            print("\n[Fail] Failed to parse scenario JSON.")

    pbar.close()
    print(f"\nTotal valid & unique: {len(valid_scenarios)}")
    save_jsonl(valid_scenarios, OUTPUT_FILE)
    print(f"Saved to synthetic/outputs/{OUTPUT_FILE}")

if __name__ == "__main__":
    main()
