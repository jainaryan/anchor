"""
Quick DPO preference pair test — generates 3 examples to verify quality
before building the full pipeline.

Run on compute node:
    srun --partition=gpu --gres=gpu:h200-141:1 --mem=48G --cpus-per-task=4 --time=01:00:00 --pty bash
    cd ~/projects/mindmate && source mindmatenv/bin/activate
    python synthetic/dpo_test_gen.py
"""

import json
from pathlib import Path
from utils import TeacherModel, parse_json_robust

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = (PROJECT_ROOT / "system_prompt.txt").read_text(encoding="utf-8").strip()

TEST_CASES = [
    {
        "category": "casual_sad",
        "situation": "User texts casually saying they feel sad/off, not in crisis — just a low-key bleh day",
    },
    {
        "category": "transition",
        "situation": "Conversation starts as casual banter, then user mentions their cat died mid-convo",
    },
    {
        "category": "panic_mode",
        "situation": "User has a job interview in 10 minutes and is spiralling with anxiety",
    },
    {
        "category": "mixed_mode",
        "situation": "Conversation starts casual (procrastinating, chatting), user gradually reveals they've been struggling, then suddenly mentions a deadline in 20 minutes causing panic",
    },
]

PROMPT_TEMPLATE = """You are a data generation assistant for MindMate, a mental-health AI companion.

Here is MindMate's full system prompt:
--- SYSTEM PROMPT START ---
{system_prompt}
--- SYSTEM PROMPT END ---

Your task: Generate ONE DPO preference pair for the following scenario.

Category: {category}
Situation: {situation}

Generate:
1. A realistic multi-turn conversation "prompt" (2-4 exchanges, ending on a user message that needs a response)
2. "chosen" — the response MindMate SHOULD give (follows the system prompt, right tone, no therapy-speak unless warranted)
3. "rejected" — the response MindMate SHOULD NOT give (the specific failure mode for this category)
4. "reason" — one sentence explaining the key difference

Category-specific rules:

casual_sad:
- chosen: brief, warm, casual acknowledgement. NOT a grounding exercise. NOT "I'm here for you".
- rejected: immediately launches into breathing exercise, grounding technique, or heavy therapist-speak

transition:
- chosen: was casual before the emotional signal, then naturally and warmly shifts tone without going full therapist mode
- rejected: either stays in joke mode after the emotional disclosure, OR immediately goes full therapy robot

panic_mode:
- chosen: zero questions, one concrete grounding action, one reassurance. Short and steady.
- rejected: asks probing questions about feelings, or tries to explore emotions during the urgency window

mixed_mode:
- The prompt history must have at least 3 distinct phases: casual → emotional disclosure → panic/urgency
- Each phase should be at least 2 turns
- chosen: tracks all three mode shifts correctly — was casual early on, shifted warmly when emotion appeared, now handles the panic with one action and no questions
- rejected: loses track of the arc — either stays warm/exploratory when panic hits (wrong mode), or goes back to casual, or asks multiple questions during the urgency
- The rejected must reflect a realistic failure: the model "forgot" what mode it should be in

Rules for both chosen and rejected:
- Similar length (within 30% word count of each other)
- Both must be plausible responses — rejected should be something the model might actually say, not obviously terrible
- The prompt history must feel natural and match the situation

Return strictly JSON:
{{
  "prompt": [
    {{"role": "system", "content": "..."}},
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    {{"role": "user", "content": "..."}}
  ],
  "chosen": [{{"role": "assistant", "content": "..."}}],
  "rejected": [{{"role": "assistant", "content": "..."}}],
  "reason": "...",
  "category": "{category}"
}}
"""


def main():
    print("Loading teacher model...")
    teacher = TeacherModel()

    for i, test in enumerate(TEST_CASES):
        print(f"\n{'='*60}")
        print(f"TEST {i+1}: {test['category'].upper()}")
        print(f"Situation: {test['situation']}")
        print('='*60)

        prompt = PROMPT_TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            category=test["category"],
            situation=test["situation"],
        )

        response = teacher.generate(prompt, max_new_tokens=1400, temperature=0.78)
        data = parse_json_robust(response, expected_keys=["prompt", "chosen", "rejected"])

        if not data:
            print("[FAIL] Could not parse JSON response")
            print("Raw response:", response[:500])
            continue

        print("\n--- PROMPT HISTORY ---")
        for msg in data.get("prompt", []):
            if msg["role"] == "system":
                print(f"[system] (omitted for brevity)")
            else:
                print(f"[{msg['role']}] {msg['content']}")

        print("\n--- CHOSEN ✅ ---")
        for msg in data.get("chosen", []):
            print(msg["content"])

        print("\n--- REJECTED ❌ ---")
        for msg in data.get("rejected", []):
            print(msg["content"])

        print(f"\n--- REASON ---")
        print(data.get("reason", "N/A"))

    print(f"\n{'='*60}")
    print("Test complete.")


if __name__ == "__main__":
    main()
