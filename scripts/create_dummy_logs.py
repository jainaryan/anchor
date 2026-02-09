import json
import random
from datetime import datetime, timedelta
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "web" / "chat_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Sample data
scenarios = [
    {"text": "I am having a panic attack, I can't breathe", "mood": "Panic"},
    {"text": "I feel so anxious about my exam tomorrow", "mood": "Anxious"},
    {"text": "I am feeling really sad and lonely today", "mood": "Sad"},
    {"text": "I am okay, just a bit tired", "mood": "Neutral"},
    {"text": "I feel meaningful and calm today", "mood": "Calm"},
    {"text": "I am so happy, I passed my test!", "mood": "Happy"},
    {"text": "Everything is closing in on me", "mood": "Panic"},
    {"text": "I feel good, thanks for asking", "mood": "Happy"},
    {"text": "Just chilling", "mood": "Neutral"},
    {"text": "My heart is racing and I am scared", "mood": "Panic"},
]

def create_logs():
    print(f"Generating logs in {LOG_DIR}...")
    
    # Generate 10 logs over the last 10 days
    for i in range(10):
        days_ago = 10 - i
        timestamp = (datetime.now() - timedelta(days=days_ago)).isoformat()
        
        scenario = scenarios[i % len(scenarios)]
        
        user_msg = scenario["text"]
        
        session_id = f"session_dummy_{i}_{int(random.random()*1000)}"
        
        data = {
            "id": session_id,
            "timestamp": timestamp,
            "messages": [
                {"role": "system", "content": "System prompt..."},
                {"role": "assistant", "content": "Hi, how are you?"},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": "I hear you."}
            ]
        }
        
        file_path = LOG_DIR / f"{session_id}.json"
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
            
        print(f"Created {file_path.name} ({scenario['mood']})")

if __name__ == "__main__":
    create_logs()
