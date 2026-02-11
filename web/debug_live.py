import requests
import sys

BASE_URL = "http://localhost:8001"

def debug_live_server():
    print("Debugging Live Server...")
    
    # 1. Check Initial Status
    try:
        res = requests.get(f"{BASE_URL}/onboarding_status")
        print(f"Initial Status: {res.json()}")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    # 2. Trigger Reset
    print("Triggering Reset...")
    res = requests.post(f"{BASE_URL}/reset_memory")
    print(f"Reset Response: {res.json()}")

    # 3. Check Status Again
    res = requests.get(f"{BASE_URL}/onboarding_status")
    print(f"Post-Reset Status: {res.json()}")
    
    if res.json().get('onboarding_required'):
        print("SUCCESS: Onboarding IS required.")
    else:
        print("FAILURE: Onboarding is NOT required (Profile still exists?).")

    # 4. Trigger Chat (Simulate Step 0)
    print("Triggering Step 0 Chat...")
    res = requests.post(f"{BASE_URL}/chat", json={"messages": [], "sessionId": "debug_session"})
    print(f"Chat Response: {res.json()}")

if __name__ == "__main__":
    debug_live_server()
