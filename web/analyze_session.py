
import sys
import json
import argparse
from pathlib import Path
from mlx_lm import load, generate

def analyze(session_file, model_path, adapter_path):
    # Load session data
    try:
        with open(session_file, 'r') as f:
            data = json.load(f)
            messages = data.get('messages', [])
    except Exception as e:
        print(json.dumps({"error": f"Failed to load session: {e}"}))
        return

    # Filter system prompts
    conversation_text = ""
    for msg in messages:
        role = msg.get("role", "unknown")
        if role == "system":
            continue
        content = msg.get("content", "")
        conversation_text += f"{role.upper()}: {content}\n"

    # Construct Prompt
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

    # Load Model
    # print(json.dumps({"status": "loading_model"})) # Optional debug
    try:
        model, tokenizer = load(model_path, adapter_path=adapter_path)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load model: {e}"}))
        return

    # Generate
    try:
        response_text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=512,
            verbose=False
        )
        
        # Output raw text wrapped in JSON for safety
        print(json.dumps({"result": response_text}))
        
    except Exception as e:
        print(json.dumps({"error": f"Generation failed: {e}"}))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_file", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", required=True)
    args = parser.parse_args()
    
    analyze(args.session_file, args.model_path, args.adapter_path)
