from mlx_lm import load, generate
from pathlib import Path
from flask import Flask, request, jsonify, Response, stream_with_context
import json
import shutil
import memory_engine

# Define paths (consistent with inference/chat_mindmate.py)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "mlx_base_llama32_3b"
ADAPTER_DIR = PROJECT_ROOT / "adapters" / "mindmate_llama32_3b_qlora_nl10_3072_lr3e5"
BEST_CHECKPOINT = ADAPTER_DIR / "0001000_adapters.safetensors"
ACTIVE_ADAPTER_FILE = ADAPTER_DIR / "adapters.safetensors"
# Note: User metadata indicates system prompt might be in inference folder
PROMPT_PATH = PROJECT_ROOT / "inference" / "system_prompt.txt"
if not PROMPT_PATH.exists():
    PROMPT_PATH = PROJECT_ROOT / "system_prompt.txt" # Fallback

app = Flask(__name__, static_url_path='', static_folder='.')

from model_service import service

import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Hume AI Credentials
HUME_API_KEY = os.getenv("HUME_API_KEY")
HUME_SECRET_KEY = os.getenv("HUME_SECRET_KEY")

# Global variables for model (handled by service)
system_prompt = ""

# Model Configuration
USE_ADAPTER = False # Set to False to use base model, True for finetuned

def ensure_best_adapter_active():
    if not ADAPTER_DIR.exists():
        print(f"Error: Adapter directory not found: {ADAPTER_DIR}")
        return
    if not BEST_CHECKPOINT.exists():
         print(f"Warning: Expected checkpoint not found: {BEST_CHECKPOINT}")
         return
    shutil.copy2(BEST_CHECKPOINT, ACTIVE_ADAPTER_FILE)
    print(f"[info] set active adapter to: {BEST_CHECKPOINT.name}")

def load_system_prompt():
    try:
        with PROMPT_PATH.open("r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"[warn] system prompt not found at {PROMPT_PATH}")
        return "You are MindMate, an empathetic AI companion."

def init_model():
    global system_prompt
    print("Initializing model service...")
    
    adapter_path_arg = None
    if USE_ADAPTER:
        print("[INFO] Adapter enabled. Loading fine-tuned weights.")
        ensure_best_adapter_active()
        adapter_path_arg = str(ADAPTER_DIR)
    else:
        print("[INFO] Adapter disabled. Loading base model.")
    
    # Initialize Service (Background Thread)
    service.start(str(MODEL_PATH), adapter_path_arg)
    
    system_prompt = load_system_prompt()
    print("Model service initialized.")

@app.route('/')
def index():
    return app.send_static_file('index.html')

# Onboarding State: {session_id: {step: 0, data: {}}}
onboarding_state = {}

@app.route('/onboarding_status', methods=['GET'])
def get_onboarding_status():
    return jsonify({"onboarding_required": not memory_engine.profile_exists()})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    messages = data.get('messages', [])
    session_id = data.get('sessionId')

    # 1. Global User Input Extraction (Safe)
    user_input = ""
    if messages and messages[-1]["role"] == "user":
        user_input = messages[-1]["content"]

    # 2. Real-Time Keyword Detection (Runs for BOTH Onboarding and Normal Flow)
    trigger_tool = None
    if user_input:
        user_input_lower = user_input.lower()
        TRIGGER_PHRASES = [
            "panic attack", "can't breathe", "trouble breathing", 
            "heart is racing", "feels like i'm dying", "suffocating", 
            "chest is tight", "hyperventilating", "help me calm down", 
            "freaking out", "everything is closing in", "so overwhelmed"
        ]
        
        for phrase in TRIGGER_PHRASES:
            if phrase in user_input_lower:
                trigger_tool = "breathing"
                print(f"[Trigger] Detected distress phrase: '{phrase}' -> Tools: Breathing")
                break
    
    # Check    # Onboarding Flow
    if not memory_engine.profile_exists():
        if session_id not in onboarding_state:
            onboarding_state[session_id] = {"step": 0, "data": {}}
            
        state = onboarding_state[session_id]
        step = state["step"]
        # user_input is already defined above
        
        response_text = ""
        
        # Helper for Extraction
        def extract_info(text, field_desc):
            if not text: return ""
            try:
                tokenizer = service.get_tokenizer()
                messages = [
                    {"role": "user", "content": f"Extract the user's {field_desc} from the text below. Return ONLY the extracted value. No other text.\nInput: {text}"}
                ]
                extraction_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                print(f"[DEBUG] Extracting {field_desc} from: {text}")
                
                # Use Service
                # generate_chat now returns a generator, so we must consume it
                response_generator = service.generate_chat(extraction_prompt, max_tokens=20)
                response = "".join([token for token in response_generator])
                
                print(f"[DEBUG] Extraction Result: {response}")
                
                # Robust cleaning
                cleaned = response.strip()
                # Remove common artifacts if they appear (just in case)
                for tag in ["[/INST]", "<|eot_id|>", "<|start_header_id|>", "assistant"]:
                    if tag in cleaned:
                        cleaned = cleaned.replace(tag, "")
                cleaned = cleaned.split('\n')[0].strip()
                
                return cleaned
            except Exception as e:
                print(f"Extraction failed: {e}")
                return text # Fallback to raw

        # Step 0: Welcome -> Ask Name
        if step == 0:
            response_text = "Hello, I am Anchor. I'm here to support you. To get started, what should I call you?"
            state["step"] = 1
            
        # Step 1: Capture Name -> Ask Role
        elif step == 1:
            clean_name = extract_info(user_input, "Name (e.g. Aryan)")
            state["data"]["name"] = clean_name
            response_text = f"Nice to meet you, {clean_name}. To help me understand your context, are you currently studying, working, or doing something else?"
            state["step"] = 2
            
        # Step 2: Capture Role -> Ask Role -> Ask Style
        elif step == 2:
            clean_role = extract_info(user_input, "Role/Occupation (e.g. Student, Engineer)")
            state["data"]["role"] = clean_role
            response_text = "Understood. Last quick question: When we talk, do you prefer me to just listen/vent, or do you want practical solutions?"
            state["step"] = 3
            
        # Step 3: Capture Style -> Finish
        elif step == 3:
            clean_style = extract_info(user_input, "Interaction Style (e.g. Listener, Solver)")
            state["data"]["interaction_style"] = clean_style
            state["data"]["additional_notes"] = "" 
            
            # Create Profile
            memory_engine.create_profile(state["data"])
            
            # Cleanup
            del onboarding_state[session_id]
            
            # Force reload of system prompt for next turn
            global system_prompt
            system_prompt = load_system_prompt()

            response_text = "Got it. I've set up your profile. I'm listening—what's on your mind today?"

        # Return static response (bypass LLM)
        # Update session logs manually since we bypass normal flow
        final_response = {"role": "assistant", "content": response_text}
        if trigger_tool:
             final_response["trigger_tool"] = trigger_tool
             
        return jsonify(final_response)

    # Normal Flow
    # Prepend system prompt if not present
    if not messages or messages[0]['role'] != 'system':
        # Dynamic System Prompt
        base_prompt = system_prompt
        # If system_prompt is empty (e.g. cold start just finished), reload it
        if not base_prompt: 
             base_prompt = load_system_prompt()
             system_prompt = base_prompt
             
        # Refresh context to get latest mood/patterns
        memory_context = memory_engine.get_profile_context()
        
        # FIX: The instruction "[USER PROFILE]" is in the prompt text, so simple inclusion check fails.
        # We need to check if the *content* is there. We assume the content always starts with the header on a new line.
        if "\n[USER PROFILE]" in base_prompt:
             full_system_prompt = base_prompt
        else:
             full_system_prompt = f"{base_prompt}\n\n{memory_context}"
        
        messages.insert(0, {"role": "system", "content": full_system_prompt})

    # Prepare for Generation
    gen_messages = list(messages)
    
    # Check for New Session (only system prompt exists)
    if len(gen_messages) == 1 and gen_messages[0]['role'] == 'system':
        # Inject hidden instruction for greeting
        gen_messages.append({
            "role": "user", 
            "content": "I'm back. Welcome me to this new session based on my profile and history. Keep it warm but brief."
        })

    # (Previous detection block removed, relying on top-level detection)

    tokenizer = service.get_tokenizer()
    prompt = tokenizer.apply_chat_template(gen_messages, tokenize=False, add_generation_prompt=True)
    
    # Streaming response
    @stream_with_context
    def generate_stream():
        final_reply = ""
        
        try:
            # Iterate over the generator from ModelService
            for token in service.generate_chat(prompt, max_tokens=1024):
                final_reply += token
                
                # Yield chunk as JSON line
                chunk_data = {"role": "assistant", "content": token}
                yield json.dumps(chunk_data) + "\n"
            
            # Post-generation: check triggers and save session
            
            # Clean up response if it echoes prompt
            if final_reply.startswith(prompt):
                 final_reply = final_reply[len(prompt):]
            final_reply = final_reply.strip()

            # Trigger Tool Check (on full text)
            trigger_tool_found = None
            # Re-check trigger phrases on final response? 
            # Actually, user input trigger checking was done before generation.
            # If we want to send the trigger tool signal, we should do it at the start or end.
            # Let's send it at the end for now, or as a separate event.
            
            if trigger_tool:
                yield json.dumps({"trigger_tool": trigger_tool}) + "\n"

            # SAVE SESSION
            if session_id:
                try:
                    # Append AI response to messages
                    gen_messages.append({"role": "assistant", "content": final_reply})
                    # Note: We append to gen_messages which was local, but we need to save the full session
                    # The original 'messages' list is what we want to save + new reply
                    messages.append({"role": "assistant", "content": final_reply})
                    
                    # Ensure log directory exists
                    log_dir = PROJECT_ROOT / "web" / "chat_logs"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    
                    session_file = log_dir / f"{session_id}.json"
                    
                    # Save to file
                    session_data = {
                        "id": session_id,
                        "timestamp": import_time_now_iso(),
                        "messages": messages
                    }
                    with open(session_file, 'w') as f:
                        json.dump(session_data, f, indent=2)
                except Exception as e:
                    print(f"Error saving session: {e}")

        except Exception as e:
            print(f"Generation error: {e}")
            yield json.dumps({"error": str(e)}) + "\n"

    return Response(generate_stream(), mimetype='application/x-ndjson')

def import_time_now_iso():
    from datetime import datetime
    return datetime.now().isoformat()

@app.route('/sessions', methods=['GET'])
def get_sessions():
    log_dir = PROJECT_ROOT / "web" / "chat_logs"
    print(f"[DEBUG] Fetching sessions from {log_dir}")
    if not log_dir.exists():
        print("[DEBUG] Log dir does not exist")
        return jsonify([])
    
    sessions = []
    for f in log_dir.glob("*.json"):
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                # Create a simple summary
                first_msg = "New Chat"
                if len(data.get("messages", [])) > 1:
                     # Find first user message
                     for m in data["messages"]:
                         if m["role"] == "user":
                             first_msg = m["content"][:30] + "..."
                             break
                
                sessions.append({
                    "id": data.get("id", f.stem),
                    "timestamp": data.get("timestamp", ""),
                    "summary": first_msg
                })
        except:
            continue
            
    # Sort by timestamp desc
    sessions.sort(key=lambda x: x['timestamp'], reverse=True)
    return jsonify(sessions)

@app.route('/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    log_dir = PROJECT_ROOT / "web" / "chat_logs"
    session_file = log_dir / f"{session_id}.json"
    
    if not session_file.exists():
        return jsonify({"error": "Session not found"}), 404
        
    with open(session_file, 'r') as f:
        data = json.load(f)
        
    return jsonify(data)

@app.route('/end_session', methods=['POST'])
def end_session():
    data = request.json
    session_id = data.get('session_id')
    print(f"[DEBUG] /end_session called for {session_id}")
    if not session_id:
        return jsonify({"error": "Missing session_id"}), 400
    
    # Load session logs
    log_dir = PROJECT_ROOT / "web" / "chat_logs"
    session_file = log_dir / f"{session_id}.json"
    
    if not session_file.exists():
        return jsonify({"error": "Session file not found"}), 404
        
    try:
        with open(session_file, 'r') as f:
            session_data = json.load(f)
            messages = session_data.get('messages', [])
            
        # Trigger background processing
        if service._model: # Basic check if service is running
            memory_engine.process_post_session(session_id)
            return jsonify({"status": "processing_started"})
        else:
             return jsonify({"error": "Model service not initialized"}), 500
             
    except Exception as e:
        print(f"Error in end_session: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset_memory', methods=['POST'])
def reset_memory():
    try:
        memory_engine.reset_data(create_new=False)
        global system_prompt
        # Reset prompt to empty so next chat triggers context rebuild/onboarding check
        system_prompt = "" 
        return jsonify({"status": "memory_reset"})
    except Exception as e:
        print(f"Error resetting memory: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/profile', methods=['GET'])
def get_profile():
    """Returns the full user profile including session history."""
    profile = memory_engine.load_profile()
    if not profile:
        return jsonify({"error": "Profile not found"}), 404
    return jsonify(profile)

@app.route('/tts', methods=['POST'])
def tts():
    import base64 # Import locally to avoid messing up top-level diffs
    
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({"error": "Missing text"}), 400
        
    if not HUME_API_KEY:
        print("[Error] HUME_API_KEY not set")
        return jsonify({"error": "TTS not configured"}), 500

    try:
        # Hume TTS API (Standard)
        # Docs: https://dev.hume.ai/reference/tts-synthesize-json
        url = "https://api.hume.ai/v0/tts"
        headers = {
            "X-Hume-Api-Key": HUME_API_KEY,
            "Content-Type": "application/json"
        }
        
        # Construct payload with utterances list
        payload = {
            "utterances": [
                {
                    "text": text,
                }
            ]
        }
        
        # Request
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            print(f"[TTS Error] {response.status_code} - {response.text}")
            return jsonify({"error": "TTS generation failed"}), response.status_code

        # Parse JSON to get audio
        try:
            res_data = response.json()
            if "generations" not in res_data or not res_data["generations"]:
                return jsonify({"error": "No audio generated"}), 500
                
            b64_audio = res_data["generations"][0]["audio"]
            audio_bytes = base64.b64decode(b64_audio)
            
            return Response(audio_bytes, mimetype="audio/mpeg")
            
        except Exception as e:
            print(f"[TTS Parse Error] {e}")
            return jsonify({"error": "Failed to parse TTS response"}), 500

    except Exception as e:
        print(f"[TTS Exception] {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    init_model()
    # Sync legacy logs to profile
    memory_engine.sync_session_history()
    
    # Allow port configuration or default to 8001
    import sys
    port = 8001
    if len(sys.argv) > 1 and sys.argv[1] == '--port':
        port = int(sys.argv[2])
    app.run(port=port, debug=False, use_reloader=False)
