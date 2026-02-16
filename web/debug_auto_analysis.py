
import requests
import time
import json

BASE_URL = "http://127.0.0.1:8001"


def ensure_profile():
    profile_path = "web/user_profile.json"
    try:
        with open(profile_path, 'r') as f:
            pass
    except FileNotFoundError:
        print("Creating dummy profile...")
        dummy_data = {
            "static_profile": {"name": "TestUser", "key_facts": []},
            "emotional_state": {"current_mood": "Neutral", "last_updated": ""},
            "recurring_patterns": [],
            "session_history": []
        }
        with open(profile_path, 'w') as f:
            json.dump(dummy_data, f)

def test_auto_analysis():
    ensure_profile()
    print("Testing /end_session endpoint...")
    
    # 1. Create a fake session via /chat to generate a log file
    session_id = f"test_auto_{int(time.time())}"
    print(f"Creating session: {session_id}")
    
    # Send user message
    res = requests.post(f"{BASE_URL}/chat", json={
        "messages": [{"role": "user", "content": "I feel anxious about my exams."}],
        "sessionId": session_id
    })
    print(f"Chat Response: {res.status_code}")
    
    if res.status_code != 200:
        print("Failed to create chat.")
        return

    # 2. Trigger end_session
    print(f"Triggering end_session for {session_id}...")
    res = requests.post(f"{BASE_URL}/end_session", json={"session_id": session_id})
    print(f"End Session Response: {res.status_code} - {res.text}")
    
    if res.status_code == 200:
        print("SUCCESS: Endpoint is reachable and processing started.")
    else:
        print("FAILURE: Endpoint returned error.")

if __name__ == "__main__":
    test_auto_analysis()
