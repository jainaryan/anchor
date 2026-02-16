
prompt_text = """
MEMORY & CONTEXT
You have access to the user's profile, current mood, and recent recurring themes (provided below in [USER PROFILE]).
You MUST use this context to personalize your responses.
"""

memory_context = """
[USER PROFILE]
Name: Aryan
Role: Developer
"""

# New Logic Simulation
print(f"Prompt has '\\n[USER PROFILE]': {'\n[USER PROFILE]' in prompt_text}")

if "\n[USER PROFILE]" in prompt_text:
    full_prompt = prompt_text
else:
    full_prompt = f"{prompt_text}\n\n{memory_context}"

print("-" * 20)
print("FINAL PROMPT:")
print(full_prompt)
print("-" * 20)

if memory_context not in full_prompt:
    print("FAIL: Memory context was NOT injected.")
else:
    print("SUCCESS: Memory context was injected correctly.")
