import sys
from pathlib import Path
from flask import Flask, request, jsonify, Response, stream_with_context
import json
import shutil
from mlx_lm import load, generate
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

# Global variables for model
model = None
tokenizer = None
system_prompt = ""

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
    global model, tokenizer, system_prompt
    print("Initializing model...")
    ensure_best_adapter_active()
    print(f"Loading model from {MODEL_PATH}")
    print(f"Loading adapter from {ADAPTER_DIR}")
    model, tokenizer = load(str(MODEL_PATH), adapter_path=str(ADAPTER_DIR))
    system_prompt = load_system_prompt()
    print("Model initialized.")

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
    
    # Check    # Onboarding Flow
    if not memory_engine.profile_exists():
        if session_id not in onboarding_state:
            onboarding_state[session_id] = {"step": 0, "data": {}}
            
        state = onboarding_state[session_id]
        step = state["step"]
        user_input = messages[-1]["content"] if messages else ""
        
        response_text = ""
        
        # Helper for Extraction
        def extract_info(text, field_desc):
            if not text: return ""
            try:
                extraction_prompt = f"""[INST] Extract the user's {field_desc} from the text below. Return ONLY the extracted value. No other text.
Input: {text}
Extracted {field_desc}: [/INST]"""
                response = generate(model, tokenizer, prompt=extraction_prompt, max_tokens=20, verbose=False)
                return response.strip().split('\n')[0].strip() # Take first line
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
            
        # Step 2: Capture Role -> Ask Style
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
            system_prompt = memory_engine.get_profile_context()

            response_text = "Got it. I've set up your profile. I'm listening—what's on your mind today?"

        # Return static response (bypass LLM)
        # Update session logs manually since we bypass normal flow
        return jsonify({"role": "assistant", "content": response_text})

    # Normal Flow
    # Prepend system prompt if not present
    if not messages or messages[0]['role'] != 'system':
        # Dynamic System Prompt
        base_prompt = system_prompt
        # If system_prompt is empty (e.g. cold start just finished), reload it
        if not base_prompt or "Unknown" in base_prompt: 
             base_prompt = memory_engine.get_profile_context()
             
        # Refresh context to get latest mood/patterns
        memory_context = memory_engine.get_profile_context()
        
        if "[USER PROFILE]" in base_prompt:
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

    prompt = tokenizer.apply_chat_template(gen_messages, tokenize=False, add_generation_prompt=True)
    
    # Streaming response
    def generate_stream():
        response_text = ""
        
        output = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=1024,
            verbose=False
        )
        text = output
        
        reply = text
        if isinstance(text, dict) and "text" in text:
            reply = text["text"]
        
        if reply.startswith(prompt):
             reply = reply[len(prompt):]
        
        # Clean up response
        final_reply = reply.strip()
        
        # SAVE SESSION
        if session_id:
            try:
                # Append AI response to messages
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

        yield json.dumps({"role": "assistant", "content": final_reply})

    return Response(generate_stream(), mimetype='application/json')

def import_time_now_iso():
    from datetime import datetime
    return datetime.now().isoformat()

@app.route('/sessions', methods=['GET'])
def get_sessions():
    log_dir = PROJECT_ROOT / "web" / "chat_logs"
    if not log_dir.exists():
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
        if model and tokenizer:
            memory_engine.process_post_session(messages, model, tokenizer)
            return jsonify({"status": "processing_started"})
        else:
             return jsonify({"error": "Model not initialized"}), 500
             
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

if __name__ == '__main__':
    init_model()
    # Allow port configuration or default to 8001
    import sys
    port = 8001
    if len(sys.argv) > 1 and sys.argv[1] == '--port':
        port = int(sys.argv[2])
    app.run(port=port, debug=False, use_reloader=False)
