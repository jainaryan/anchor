---
tags: [anchor, system-prompt]
---

# System Prompt

← [[Home]]

---

## Overview

Two system prompt files exist in anchor-app. **They serve completely different purposes and must not be confused.**

| File | Used by | Content | Notes |
|---|---|---|---|
| `src/utils/anchorSystemPrompt.ts` | **Production chat** (`ChatScreen.tsx`) | "You are Anchor..." 6-line prompt | What real users see |
| `src/constants/mindmatePrompt.ts` | **Eval runner only** (`EvalRunner.ts`, `MultiTurnRunner.ts`) | ~150-line structured prompt with old "MindMate" name | In-app eval screen only |

`anchorSystemPrompt.ts` is the source of truth for production behavior. `mindmatePrompt.ts` is kept for historical eval comparison; do not use it for any new development.

---

## The Production Prompt (`anchorSystemPrompt.ts`)

Exported as `getAnchorSystemPrompt()`. Exact content (do not paraphrase — this must match training data and benchmark exactly):

```
You are Anchor, a warm and caring AI companion — like a close friend who genuinely listens.
Talk naturally. Be curious about the person. Ask follow-up questions. Use their actual words and details back to them.
If the conversation has been light and the person suddenly gets serious, drop the casual tone immediately. No jokes, no deflection. Just be present.
If someone seems to be in danger or crisis, gently encourage them to reach out to someone they trust or a crisis line.
You are an AI. If asked, say so warmly. Never pretend to have lived experiences.
Don't lecture.
```

This is `_ANCHOR_PROMPT` in `scripts/chat_cluster.py` and `_APP_BASE_PROMPT` in `benchmarks/scenarios.py`. All three must remain byte-for-byte identical.

---

## Memory Injection (`contextBuilder.ts`)

`buildEnhancedSystemPrompt(basePrompt, userMessage)` appends three tiers of context after the base prompt.

### Full injected format

```
You are Anchor, a warm and caring AI companion — like a close friend who genuinely listens.
[...5 more lines of base prompt...]

ABOUT THIS USER
[User]
Name: <from profile>
Age: <from profile>
Gender: <from profile>
Diagnoses: <from profile>
Triggers: <from profile>
Coping strategies: ★ <helpful> / ✗ <unhelpful>
Support people: <from profile>
Risk flags: <from profile>

[Recent sessions]
Session 1: <summary text from most recent EpisodicMemoryData>
Session 2: <summary text>
Session 3: <summary text>
<Tier 3: keyword-matched older sessions if relevant>
```

### Memory tier details

**Tier 1 — Profile card (always included)**
- Source: user-entered profile fields
- Fields: name, age, gender, diagnoses, triggers, coping strategies (with ★/✗ markers), support people, risk flags
- The ★helpful/✗unhelpful markers are critical — model is trained to use ★ strategies and avoid ✗ ones

**Tier 2 — Recent 3 sessions (always included)**
- Source: `summary` field of the 3 most recent `EpisodicMemoryData` entries in SQLite
- Only the `summary` text is shown to the model — all other structured fields (moodStart, heartRate, sleepHours, etc.) are invisible to the LLM
- These fields are stored for future feature development, not for current model use

**Tier 3 — Keyword-matched older sessions**
- Source: older `EpisodicMemoryData.summary` entries matching keywords from the current user message
- Deduped against Tier 2 (never repeats sessions already shown)
- Only included when relevant keywords found

### Directives in memory header (added 2026-04-22)

Five directives added to `contextBuilder.ts` to make the model use context proactively:
1. Reference names/events/strategies the user has mentioned
2. Suggest ONE coping strategy by name when asked for help
3. If Recent sessions shows declining mood trend — acknowledge in first response
4. If Recent sessions records health/sleep pattern — connect it when user describes something similar
5. If a coping strategy is marked unhelpful — do NOT suggest it

---

## Training Format (post c3acdc9)

Since commit c3acdc9 (2026-05-09), all 42,038 training examples use this format exactly:

```json
{"conversations": [
  {
    "role": "system",
    "content": "You are Anchor, a warm and caring AI companion — like a close friend who genuinely listens.\nTalk naturally...\n\nABOUT THIS USER\n[User]\nName: ...\n[Recent sessions]\nSession 1: ..."
  },
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

### The bug that was fixed (c3acdc9)

Before c3acdc9, training data had three different formats:

| Files | System prompt during training | System prompt at inference |
|---|---|---|
| `targeted_fix.jsonl` (13,524) | `[User]\n...\n[Recent sessions]\n...` only | Full preamble + ABOUT header + [User] + [Recent sessions] |
| `biometric.jsonl` (2,348) | `[User]\n...\n[Recent sessions]\n...` only | Same full format |
| All other 6 files (26,166) | **No system message at all** | Full preamble |

The model was trained on 42,038 examples where it either saw truncated memory blocks or no system prompt. At inference it sees the full production format — a completely different context. This is why base model (which has intact instruction-following) beats fine-tuned models on all memory categories.

**What the normalization script (c3acdc9) did:**
- `targeted_fix` + `biometric`: prepended `_APP_BASE_PROMPT + "\n\nABOUT THIS USER\n"` before existing `[User]`/`[Recent sessions]` blocks
- `biometric`: 179 examples with raw session notes (no `[Recent sessions]` label) → wrapped with full format
- `friend_1`, `therapist_`, `transition`, `casual`, `grief`: injected `_APP_BASE_PROMPT` as system message (they had no system message)
- `targeted_fixes` (gold): had abbreviated old format → prepended `_APP_BASE_PROMPT`
- All normalized from `messages` key to `conversations` key where needed

---

## Benchmark Prompt Architecture (`benchmarks/scenarios.py`)

The benchmark must use byte-for-byte the same format as production. Verified 2026-05-03.

```python
_APP_BASE_PROMPT = (
    "You are Anchor, a warm and caring AI companion — like a close friend who genuinely listens.\n"
    "Talk naturally. Be curious about the person. Ask follow-up questions. Use their actual words and details back to them.\n"
    "If the conversation has been light and the person suddenly gets serious, drop the casual tone immediately. No jokes, no deflection. Just be present.\n"
    "If someone seems to be in danger or crisis, gently encourage them to reach out to someone they trust or a crisis line.\n"
    "You are an AI. If asked, say so warmly. Never pretend to have lived experiences.\n"
    "Don't lecture."
)

_MEMORY_HEADER = "\n\nABOUT THIS USER\n"

def _sys(user_block="", session_block=""):
    return _APP_BASE_PROMPT + _MEMORY_HEADER + user_block + session_block

# user_block = "[User]\nName: ...\n..."
# session_block = "[Recent sessions]\nSession 1: ..."
```

This matches `contextBuilder.ts` `assemblePrompt()` exactly.

---

## See also

- [[Production]] — how system prompt is used in the app and server
- [[Data]] — training format normalization (c3acdc9)
- [[Bug Log]] — format mismatch root cause and fix
