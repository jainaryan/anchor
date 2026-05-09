---
tags: [anchor, system-prompt]
---

# System Prompt

← [[Home]]

---

## Overview

Two system prompt files exist in anchor-app. They serve different purposes and must not be confused.

| File | Used by | Content |
|---|---|---|
| `src/utils/anchorSystemPrompt.ts` | **Production chat** (`ChatScreen.tsx`) | "You are Anchor..." 6-line prompt |
| `src/constants/mindmatePrompt.ts` | **Eval runner only** (`EvalRunner.ts`, `MultiTurnRunner.ts`) | ~150-line structured prompt (old name) |

`anchorSystemPrompt.ts` is what real users see. `mindmatePrompt.ts` is only in the in-app eval screen.

---

## Production System Prompt (`anchorSystemPrompt.ts`)

The 6-line "You are Anchor..." prompt. This is the base that gets memory injected on top.

**Benchmark fidelity (verified 2026-05-03):** `benchmarks/scenarios.py` uses `_APP_BASE_PROMPT` which is a byte-for-byte match of `anchorSystemPrompt.ts`. The `_sys()` / `_MEMORY_HEADER` format matches `contextBuilder.ts assemblePrompt()` exactly. All benchmark scores reflect real production conditions.

---

## Memory Injection (contextBuilder.ts)

`ChatScreen.tsx` passes `getAnchorSystemPrompt()` to `useChatSession` → `prepareCompletion()` calls `buildEnhancedSystemPrompt(basePrompt, userMessage)` → injects tiers from DB.

**What the model sees:**
```
You are Anchor... [6-line base prompt]

ABOUT THIS USER
[User]
Name: ...
Age: ...
Diagnoses: ...
Triggers: ...
Coping strategies: ...
Risk flags: ...

[Recent sessions]
Session 1: ...
Session 2: ...
Session 3: ...
```

### Memory tiers

| Tier | Content | Always included? |
|---|---|---|
| Tier 1 | User profile card (demographics, diagnoses, triggers, coping, support people, risk flags) | ✅ |
| Tier 2 | Recent 3 sessions — `summary` text only | ✅ |
| Tier 3 | Keyword-matched past sessions (deduped from Tier 2) | Only if relevant |

Only the `summary` field of `EpisodicMemoryData` is rendered — structured fields (moodStart, heartRate, etc.) are invisible to the LLM.

### Directives in the memory header (added 2026-04-22)

Five directives added to make the model use context proactively:
1. Reference names/events/strategies the user has mentioned
2. Suggest ONE coping strategy by name when asked for help
3. If Recent sessions shows declining mood trend — acknowledge in first response
4. If Recent sessions records health/sleep pattern — connect it when user describes something similar
5. If a coping strategy is marked unhelpful — do NOT suggest it

---

## Training Format (after c3acdc9)

Since commit c3acdc9 (2026-05-09), all 42,038 training examples use the production format:

```json
{"conversations": [
  {
    "role": "system",
    "content": "You are Anchor...\n\nABOUT THIS USER\n[User]\n...\n[Recent sessions]\n..."
  },
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

**Before c3acdc9 (the bug):**
- `targeted_fix` + `biometric`: had `[User]\n...\n[Recent sessions]\n...` only — missing preamble + header
- All other files: no system message at all

This format mismatch was the root cause of base-beats-SFT on all memory categories.

---

## Benchmark Prompt Architecture (`benchmarks/scenarios.py`)

```python
_APP_BASE_PROMPT = "You are Anchor..."  # byte-for-byte match of anchorSystemPrompt.ts

def _sys(user_block, session_block):
    return _APP_BASE_PROMPT + _MEMORY_HEADER + user_block + session_block

# _MEMORY_HEADER = "\n\nABOUT THIS USER\n"
# Matches contextBuilder.ts assemblePrompt() exactly
```

---

## See also

- [[Production]] — how system prompt is used in the app
- [[Data]] — training format normalization
- [[Bug Log]] — format mismatch root cause (May 9)
