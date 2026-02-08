import threading
import json
import re
import shutil
import os
from pathlib import Path
from datetime import datetime
from mlx_lm import generate

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = PROJECT_ROOT / "web" / "user_profile.json"

# Lock for thread-safe file I/O


# Lock for thread-safe file I/O
profile_lock = threading.RLock()

import time

# Flag to track background analysis status
analysis_active = False

def wait_for_analysis(timeout=30):
    """Blocks until analysis is complete or timeout reached."""
    global analysis_active
    if not analysis_active:
        return
        
    print(f"[MemoryEngine] Waiting for analysis to complete (timeout={timeout}s)...")
    start = time.time()
    while analysis_active:
        time.sleep(0.5)
        if time.time() - start > timeout:
            print("[MemoryEngine] Wait timed out.")
            break

def get_profile_context():
    """Returns a formatted string of the user profile for the system prompt."""
    # Wait for any pending analysis to ensure we have the latest summary
    wait_for_analysis()
    
    profile = load_profile()
    
    name = profile['static_profile'].get('name', 'User')
    role = profile['static_profile'].get('role', 'Unknown')
    additional = profile['static_profile'].get('additional_notes', 'None')
    style = profile.get('preferences', {}).get('interaction_style', 'Mixed')
    
    mood = profile['emotional_state'].get('current_mood', 'Unknown')
    
    patterns = profile.get('recurring_patterns', [])
    # Sort by count desc and take top 5
    top_patterns = sorted(patterns, key=lambda x: x.get('count', 0), reverse=True)[:5]
    pattern_str = ", ".join([p['topic'] for p in top_patterns]) if top_patterns else "None detected yet"
    
    context = f"""
[USER PROFILE]
Name: {name}
Role: {role}
Interaction Style: {style}
Additional Notes: {additional}
Current Mood: {mood}
Recurring Themes: {pattern_str}
"""
    return context.strip()

def init_profile():
    """Ensures user_profile.json exists with the correct schema."""
    with profile_lock:
        if not PROFILE_PATH.exists():
            initial_data = {
                "static_profile": {"name": "User", "key_facts": []},
                "emotional_state": {"current_mood": "Neutral", "last_updated": ""},
                "recurring_patterns": [],
                "session_history": []
            }
            save_profile_atomic(initial_data)

def save_profile_atomic(data):
    """Saves data to user_profile.json atomically to prevent corruption."""
    temp_path = PROFILE_PATH.with_suffix(".tmp")
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    shutil.move(temp_path, PROFILE_PATH)

def profile_exists():
    """Checks if user_profile.json exists."""
    return PROFILE_PATH.exists()

def create_profile(data):
    """Creates a new profile with the given data (from onboarding)."""
    with profile_lock:
        profile = {
            "static_profile": {
                "name": data.get("name", "User"),
                "role": data.get("role", ""),
                "additional_notes": data.get("additional_notes", ""),
                "key_facts": []
            },
            "emotional_state": {"current_mood": "Neutral", "last_updated": ""},
            "preferences": {
                "interaction_style": data.get("interaction_style", "Mixed")
            },
            "recurring_patterns": [],
            "session_history": []
        }
        save_profile_atomic(profile)

def load_profile():
    """Loads the user profile. Returns None if not found."""
    with profile_lock:
        if not PROFILE_PATH.exists():
            return None # Do not auto-init, let app handle it
        with open(PROFILE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)

def reset_data(create_new=True):
    """Hard reset: Clears user profile and all chat logs."""
    with profile_lock:
        # 1. Reset Profile
        if PROFILE_PATH.exists():
            os.remove(PROFILE_PATH)
        
        if create_new:
            init_profile() # Re-create fresh if requested
        
        # 2. Clear Chat Logs
        log_dir = PROJECT_ROOT / "web" / "chat_logs"
        if log_dir.exists():
            shutil.rmtree(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)


def process_post_session(chat_history, model, tokenizer):
    """
    Starts the background processing thread.
    chat_history: List of {"role":Str, "content":Str}
    """
    global analysis_active
    analysis_active = True
    
    thread = threading.Thread(
        target=_background_worker,
        args=(chat_history, model, tokenizer),
        daemon=True
    )
    thread.start()

def _background_worker(chat_history, model, tokenizer):
    """
    Refines the chat history, generates a summary using the LLM,
    parses the output, detects patterns, and updates the profile.
    """
    global analysis_active
    try:
        print("[MemoryEngine] Starting background analysis...")
        
        # 1. Construct Prompt
        # Filter out system messages for the summary to focus on interaction
        conversation_text = ""
        for msg in chat_history:
            role = msg.get("role", "unknown")
            if role == "system":
                continue
            content = msg.get("content", "")
            conversation_text += f"{role.upper()}: {content}\n"
        
        prompt = f"""
You are an expert psychologist AI. Analyze the following conversation deeply.

CONVERSATION:
{conversation_text}

INSTRUCTIONS:
1. Think step-by-step about the user's emotional state, potential risks, and key topics discussed.
2. Output your analysis in the required format.

FORMAT:
[ANALYSIS]
... (your chain of thought) ...

[FINAL_UPDATE]
MOOD: (1-2 words describing the user's current mood)
TOPICS: (Comma-separated list of main topics)
RISK_LEVEL: (Low/Medium/High)
SUMMARY: (Concise 1-3 sentences summary of the session)
"""
        
        # 2. Generate with LLM
        # We use a lower max_tokens since we just need a summary
        # Note: 'generate' is blocking, so this thread will wait.
        response_text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=512,
            verbose=True
        )
        
        if isinstance(response_text, dict) and "text" in response_text:
            response_text = response_text["text"]  # Handle MLX return format if it changes

        # DEBUG: Save raw response
        try:
            with open(PROJECT_ROOT / "web" / "last_llm_response.txt", "w", encoding="utf-8") as f:
                f.write(response_text)
        except Exception as e:
            print(f"[MemoryEngine] Failed to save debug log: {e}")

        # 3. Parse Output
        parsed_data = _parse_llm_output(response_text)
        
        if not parsed_data:
            print("[MemoryEngine] Failed to parse LLM output.")
            return

        # 4. Pattern Detection & Update
        _update_user_profile(parsed_data)
        print("[MemoryEngine] Background analysis complete.")
        
    except Exception as e:
        print(f"[MemoryEngine] Error in background worker: {e}")
        
    finally:
        analysis_active = False

def _parse_llm_output(raw_text):
    """
    Extracts MOOD, TOPICS, RISK_LEVEL, SUMMARY using Regex.
    Returns a dict or None if parsing fails.
    """
    try:
        # We look for the [FINAL_UPDATE] section
        if "[FINAL_UPDATE]" in raw_text:
            final_section = raw_text.split("[FINAL_UPDATE]")[1]
        else:
            final_section = raw_text # Fallback: try parsing the whole text if tag missing
            
        data = {}
        
        # Regex (case insensitive, allowing whitespace)
        mood_match = re.search(r"MOOD:\s*(.+)", final_section, re.IGNORECASE)
        topics_match = re.search(r"TOPICS:\s*(.+)", final_section, re.IGNORECASE)
        risk_match = re.search(r"RISK_LEVEL:\s*(.+)", final_section, re.IGNORECASE)
        summary_match = re.search(r"SUMMARY:\s*(.+)", final_section, re.IGNORECASE | re.DOTALL)
        
        if mood_match:
            data['mood'] = mood_match.group(1).strip()
        if topics_match:
            # Split by comma and clean
            raw_topics = topics_match.group(1).strip()
            data['topics'] = [t.strip() for t in raw_topics.split(',') if t.strip()]
        if risk_match:
            data['risk_level'] = risk_match.group(1).strip()
        if summary_match:
            data['summary'] = summary_match.group(1).strip()
            
        return data
        
    except Exception as e:
        print(f"[MemoryEngine] Regex parsing error: {e}")
        return None

def _update_user_profile(parsed_data):
    """
    Updates the session history, emotional state, and recurring patterns.
    """
    with profile_lock:
        profile = load_profile()
        
        # 1. Append Session History
        new_session = {
            "date": datetime.now().isoformat(),
            "summary": parsed_data.get('summary', 'No summary available.'),
            "mood": parsed_data.get('mood', 'Unknown'),
            "topics": parsed_data.get('topics', []),
            "risk_level": parsed_data.get('risk_level', 'Low')
        }
        profile['session_history'].append(new_session)
        
        # 2. Update Emotional State
        if parsed_data.get('mood'):
            profile['emotional_state']['current_mood'] = parsed_data['mood']
            profile['emotional_state']['last_updated'] = new_session['date']
            
        # 3. Pattern Detection (Running logic in Python)
        # Look at last 10 sessions including this one
        recent_sessions = profile['session_history'][-10:]
        
        # Count topics
        topic_counts = {}
        for session in recent_sessions:
            for topic in session.get('topics', []):
                topic_lower = topic.lower()
                topic_counts[topic_lower] = topic_counts.get(topic_lower, 0) + 1
        
        # Update recurring patterns
        # Existing patterns list
        existing_patterns = {p['topic'].lower(): p for p in profile['recurring_patterns']}
        
        for topic, count in topic_counts.items():
            if count >= 3:
                if topic in existing_patterns:
                    existing_patterns[topic]['count'] = count # Update count
                else:
                    # Add new pattern
                    profile['recurring_patterns'].append({
                        "topic": topic,
                        "count": count,
                        "first_detected": datetime.now().isoformat()
                    })
        
        # Save atomically
        save_profile_atomic(profile)
