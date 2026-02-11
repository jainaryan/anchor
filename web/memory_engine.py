import threading
import json
import re
import shutil
import os
import subprocess
from pathlib import Path
from datetime import datetime
from mlx_lm import generate

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILE_PATH = PROJECT_ROOT / "web" / "user_profile.json"

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
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        
        # Atomic replacement
        os.replace(temp_path, PROFILE_PATH)
        # print(f"[DEBUG] Profile saved to {PROFILE_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to save profile atomically: {e}")
        # Attempt minimal recovery or logging
        if temp_path.exists():
            try:
                os.replace(temp_path, PROFILE_PATH)
            except:
                pass

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


from model_service import service

def process_post_session(session_id):
    """
    Queues a background analysis task via ModelService.
    """
    global analysis_active
    analysis_active = True
    
    try:
        SESSION_FILE = PROJECT_ROOT / "web" / "chat_logs" / f"{session_id}.json"

        # 1. Load Session Data
        try:
            with open(SESSION_FILE, 'r') as f:
                data = json.load(f)
                messages = data.get('messages', [])
        except Exception as e:
            print(f"[MemoryEngine] Failed to load session: {e}")
            analysis_active = False # Reset here since we return early
            return

        # 2. Filter system prompts
        conversation_text = ""
        for msg in messages:
            role = msg.get("role", "unknown")
            if role == "system":
                continue
            content = msg.get("content", "")
            conversation_text += f"{role.upper()}: {content}\n"
    except Exception as e:
        print(f"[MemoryEngine] Error in process_post_session setup: {e}")
        analysis_active = False
        return

    # 3. Construct Messages for Chat Template
    system_instruction = "You are an expert psychologist AI. Your task is to analyze conversations deeply and extract emotional insights."
    
    user_content = f"""
ANALYZE THIS CONVERSATION:
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
SUMMARY: (Concise 1-3 sentences stating ONLY what the user shared about their life/situation. Do NOT include advice, recommendations, or future plans.)
"""

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content}
    ]
    
    print(f"[MemoryEngine] queueing analysis for {session_id}...")

    # Apply Template
    tokenizer = service.get_tokenizer()
    if tokenizer:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback if tokenizer not ready (shouldn't happen if service started)
        prompt = f"{system_instruction}\n\n{user_content}"

    # 4. Define Callback 
    def on_analysis_complete(response_text):
        try:
            # Parse Output
            parsed_data = _parse_llm_output(response_text)
            
            if not parsed_data:
                print(f"[MemoryEngine] Failed to parse LLM output. Raw: {response_text}")
                return

            # Pattern Detection & Update
            _update_user_profile(parsed_data)
            print(f"[MemoryEngine] Analysis complete for {session_id}.")
            
        except Exception as e:
            print(f"[MemoryEngine] Error in analysis callback: {e}")
        finally:
            global analysis_active
            analysis_active = False
            
    # 5. Submit to Service
    service.submit_analysis(prompt, on_analysis_complete)

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
        print(f"[MemoryEngine] Appended new session. Total sessions: {len(profile['session_history'])}")
        
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
                # Handle string or list of strings
                if isinstance(topic, str):
                    t_list = [t.strip() for t in topic.split(',')]
                else:
                    t_list = [str(t).strip()]
                
                for t in t_list:
                    topic_lower = t.lower()
                    topic_counts[topic_lower] = topic_counts.get(topic_lower, 0) + 1
        
        # Update recurring patterns
        new_patterns = []
        for topic, count in topic_counts.items():
            if count >= 2: # Lower threshold for testing
                new_patterns.append({
                    "topic": topic,
                    "count": count,
                    "first_detected": datetime.now().isoformat() 
                })
        
        profile['recurring_patterns'] = new_patterns
        print(f"[MemoryEngine] Updated patterns: {new_patterns}")
        
        # Save atomically
        save_profile_atomic(profile)
        print("[MemoryEngine] Profile saved safely.")

def sync_session_history():
    """
    Scans chat_logs directory and adds missing sessions to user_profile.
    Uses a heuristic for mood if not already analyzed.
    """
    with profile_lock:
        profile = load_profile()
        if not profile: return # Should exist by now
        
        log_dir = PROJECT_ROOT / "web" / "chat_logs"
        if not log_dir.exists(): return
        
        # Get existing IDs
        existing_ids = set()
        for s in profile.get('session_history', []):
            if 'id' in s:
                existing_ids.add(s['id'])
                
        # Scan logs
        changes_made = False
        for log_file in log_dir.glob("*.json"):
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                    
                sess_id = data.get('id', log_file.stem)
                if sess_id in existing_ids:
                    continue
                    
                # New Session Found - Backfill
                timestamp = data.get('timestamp', datetime.now().isoformat())
                messages = data.get('messages', [])
                
                # Heuristic Analysis
                user_text = " ".join([m['content'] for m in messages if m['role'] == 'user']).lower()
                
                mood = "Neutral"
                risk = "Low"
                if "panic" in user_text or "dying" in user_text or "scared" in user_text:
                    mood = "Panic"
                    risk = "High"
                elif "anxious" in user_text or "worried" in user_text:
                    mood = "Anxious"
                elif "sad" in user_text or "tired" in user_text:
                    mood = "Sad"
                elif "thank" in user_text or "good" in user_text or "happy" in user_text:
                    mood = "Happy"
                elif "calm" in user_text or "better" in user_text:
                    mood = "Calm"
                    
                summary = "Legacy session (auto-synced)."
                if messages:
                    first_user = next((m['content'] for m in messages if m['role'] == 'user'), "No user input")
                    summary = first_user[:50] + "..."
                    
                new_entry = {
                    "id": sess_id,
                    "date": timestamp,
                    "summary": summary,
                    "mood": mood,
                    "topics": ["legacy"],
                    "risk_level": risk
                }
                
                profile['session_history'].append(new_entry)
                existing_ids.add(sess_id)
                changes_made = True
                print(f"[MemoryEngine] Backfilled session {sess_id} with mood {mood}")
                
            except Exception as e:
                print(f"Error syncing log {log_file}: {e}")
                
        if changes_made:
            # Sort by date
            profile['session_history'].sort(key=lambda x: x['date'])
            
            # Update current mood from latest session
            if profile['session_history']:
                last_session = profile['session_history'][-1]
                if last_session.get('mood'):
                     profile['emotional_state']['current_mood'] = last_session['mood']
                     profile['emotional_state']['last_updated'] = last_session['date']
                     print(f"[MemoryEngine] Updated current mood to {last_session['mood']} from backfill.")

            save_profile_atomic(profile)
