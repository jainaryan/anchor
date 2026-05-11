---
tags: [anchor, app, android]
---

# Anchor App

← [[Home]]

---

## Overview

React Native Android app. Forked from PocketPal AI (open-source GGUF runner). Runs any GGUF model on-device via llama.cpp (JNI bridge). Extended with: episodic memory system, profile setup, panic screen, diary/reflection, and an in-app eval harness.

| | |
|---|---|
| **Repo** | github.com/tanmaykay/Anchor |
| **Branch** | `aryan_branch` |
| **Local path** | `~/projects/anchor-app/` |
| **Package ID** | `com.pocketpalai` — intentionally kept from upstream (changing breaks installs) |
| **React Native** | 0.82.1 |
| **DB** | WatermelonDB 0.28 (SQLite) |
| **Inference** | llama.rn (llama.cpp JNI bindings) |
| **Default model** | `jainaryan/mindmate-gguf/mindmate_genzv2_ck1200_q4_k_m.gguf` |

---

## Chat Session Lifecycle

```
User opens app
        ↓
ModelStore: load GGUF → initLlama() → LlamaContext
        ↓
ChatScreen mounts
  → getAnchorSystemPrompt(mode)           # base 6-line Anchor prompt
  → buildEnhancedSystemPrompt(base, msg)  # injects memory tiers + biometric
  → fullPrompt stored in useChatSession ref (fixed for entire session)
        ↓
User sends message
  → detectPanic() → PanicRiskLevel ('none' | 'watch' | 'urgent')
       'urgent' → onPanicDetected() + return (message blocked)
       'watch'  → onPanicDetected() called, message still sends
       'none'   → normal flow
  → prepareCompletion() builds OpenAI-compatible messages array
  → truncateHistory() caps at MAX_HISTORY_TURNS (8 turns = 16 messages)
  → modelStore.activeModel.completion(params) → token stream
        ↓
Completion finishes
  → scheduleSessionExtraction()
       ⚠️ sessionId + messages snapshot captured NOW (not at timer fire)
       Debounce: 5-min timer
  → runExtraction(sessionId, messagesSnapshot, moodStart, moodEnd)
    → extractTopics / extractPeople / extractUserFacts / extractCopingWithOutcomes / detectMilestone
    → memoryRepository.saveMemory(EpisodicMemoryData)
    → memoryRepository.addKnownName / addUserFact
```

---

## Memory System

### `src/memory/contextBuilder.ts` — System Prompt Assembly

`buildEnhancedSystemPrompt(basePrompt, currentUserMessage, overrideContext?)` → `BuiltContext`

Returns `{tier1, tier2, tier3, biometric, fullPrompt}`.

**Production mode** (no `overrideContext`):
```typescript
const [profile, recentMemories, biometricSummaries] = await Promise.all([
  memoryRepository.getOrCreateProfile(),
  memoryRepository.getRecentMemories(3),   // always 3 most recent
  getDailySummaries(7),                    // last 7 days of health data
]);
const keywords = extractKeywords(currentUserMessage);  // filters stop words
const matchedMemories = await memoryRepository.searchByKeywords(keywords);
// Tier 3: matched memories deduped against Tier 2 (recentIds excluded)
```

**Eval mode** (`overrideContext` provided): uses inline data, no DB access. Used by EvalRunner.

**Per-tier character budgets** (enforced before assembly, ~4 chars/token):

| Tier | Budget | ≈ tokens |
|---|---|---|
| Tier 1 (`[User]`) | 2,400 chars | ~600 |
| Tier 2 (`[Recent sessions]`) | 3,200 chars | ~800 |
| Tier 3 (`[Relevant past]`) | 1,600 chars | ~400 |
| Biometric | 1,600 chars | ~400 |

Each tier is truncated at the nearest line boundary within its budget (`[…]` appended). Chat history is truncated separately via `truncateHistory()`.

### Memory Tiers

| Tier | Label in prompt | Content | Always shown? |
|---|---|---|---|
| 1 | `[User]` | Profile card — gender, age, location, diagnoses, therapy, risk flags, triggers, coping strategies (Helps), support people, user facts, communication prefs | ✅ Yes |
| 2 | `[Recent sessions]` | 3 most recent session summaries, formatted as `[Mon DD] <summary>` | ✅ Yes |
| 3 | `[Relevant past]` | Keyword-matched older session summaries, deduped from Tier 2 | Only if keywords match |
| Biometric | `[Health data — last 7 days]` | Daily: steps, sleep (h, quality), resting HR, HRV, SpO2, calories, exercise minutes, recovery score (Whoop), strain (Whoop) | Only if health data available |

**Critical:** Only `EpisodicMemory.summary` text is rendered to the model. Fields like `moodStart`, `moodEnd`, `heartRateAvg`, `sleepHours` etc. are stored in DB for future use but are **invisible to the LLM**.

### `assemblePrompt()` — Exact Format

```
You are Anchor, a warm and caring AI companion — like a close friend who genuinely listens.
[...5 more lines of base prompt...]

============================================================
ABOUT THIS USER (you know this — use it naturally)
============================================================
If the user mentions someone by name, an event, or a coping strategy listed below — reference it.
If they ask for help, suggest ONE strategy from their Helps list by name.
If [Recent sessions] shows a declining mood trend, acknowledge it in your first response...
[...8 more directive lines about health data, unhelpful strategies, etc.]
Do not recite this block back verbatim.

[User]
<tier1 profile card>

[Recent sessions]
[May 10] <session summary>
[May 8] <session summary>
[May 5] <session summary>

[Relevant past]      ← only if tier3 non-empty
[Apr 22] <matched session>

[Health data — last 7 days]    ← only if biometric data available
2026-05-09 | steps 4,231 | sleep 6h15m (fair) | rhr 62 bpm | hrv 44 ms
```

### `src/memory/sessionExtractor.ts` — Post-Session Extraction

`runExtraction(sessionId, messages, moodStart?, moodEnd?)` runs after every session.

Extracts from message text:
- **Topics** — matched against `TOPIC_SEEDS` (anxiety, sleep, work, family, relationship, etc.)
- **People mentioned** — from `knownNames` in profile + `"my [relationship] [Name]"` regex
- **User facts** — `"my [noun] [Name]"` pattern (e.g. "my dog Mochi", "my therapist Priya")
- **Coping used + outcome** — matched against `COPING_STEMS` (breathing, walking, journaling, etc.), outcome inferred from sentiment words in next 3 messages
- **Milestones** — keyword matching (death, breakup, engaged, new job, etc.) with exclusion list ("killing it", "dying of boredom")

**Casing invariant:** `getUserMessages()` preserves original message casing. `extractTopics()` and `detectMilestone()` lowercase internally. The intro regex for new-person detection (`"my [relationship] [Name]"`) matches `[A-Za-z]` and Title-cases the result — fixes a prior bug where the `[A-Z]` pattern never matched lowercased text.

Saves to DB via `memoryRepository.saveMemory()`. Also updates `knownNames` and `userFacts` on the profile.

**Coping outcome detection:**
- `partially_helpful` — "a little", "kind of", "somewhat"
- `unhelpful` — "didn't help", "still feel", "worse", "nothing works"
- `helpful` — "helped", "better", "calmer", "worked", "relieved"
- `unknown` — none of the above signals found

---

## Database (`@nozbe/watermelondb`)

SQLite via WatermelonDB. All models in `src/database/models/`.

### `EpisodicMemory` — `episodic_memories` table

| Column | Type | Notes |
|---|---|---|
| `session_id` | text | Unique per chat session |
| `date` | text | ISO string |
| `mood_start` | number | 1–10, optional |
| `mood_end` | number | 1–10, optional |
| `summary` | text | **Only field shown to LLM** |
| `topics_json` | text | JSON array of topic strings |
| `people_mentioned_json` | text | JSON array of names |
| `coping_used_json` | text | JSON array of canonical strategy names |
| `coping_outcome_json` | text | JSON map of strategy → outcome |
| `is_milestone` | boolean | |
| `milestone_type` | text | e.g. `"negative:death"`, `"positive:new job"` |

### `UserProfile` — `user_profiles` table

Stores the profile card (Tier 1). Key fields: `name`, `age`, `gender`, `location`, `diagnoses_json`, `triggers_json`, `coping_strategies_json`, `support_people_json`, `risk_flags_json`, `known_names_json`, `user_facts_json`, `communication_prefs_json`, `therapy_status_json`.

### Other models

| Model | Table | Purpose |
|---|---|---|
| `ChatSession` | `chat_sessions` | Chat session metadata |
| `Message` | `messages` | Individual chat messages |
| `BiometricRecord` | `biometric_records` | Health data (future use) |
| `GlobalSetting` | `global_settings` | App-wide settings |

---

## Screen Inventory

| Screen | Path | Purpose |
|---|---|---|
| `ChatScreen` | `src/screens/ChatScreen/ChatScreen.tsx` | Main chat. Wires model + memory + diary flow + panic detection |
| `ProfileSetupScreen` | `src/screens/ProfileSetupScreen/` | 4-step profile wizard: About you / What helps / Your people / Boundaries |
| `ModelsScreen` | `src/screens/ModelsScreen/` | GGUF model browser, download from HuggingFace, local model management |
| `MemoryViewScreen` | `src/screens/MemoryViewScreen/MemoryViewScreen.tsx` | Inspect episodic memory entries stored in DB |
| `DiaryScreen` | `src/screens/DiaryScreen/` | View past session reflections |
| `DiaryEditorScreen` | `src/screens/DiaryEditorScreen/` | Edit/view a single diary entry |
| `PanicScreen` | `src/screens/PanicScreen.tsx` | Crisis screen — grounding techniques, crisis line numbers |
| `EvalScreen` | `src/screens/EvalScreen/` | In-app eval harness UI |
| `BenchmarkScreen` | `src/screens/BenchmarkScreen/` | Model benchmarking UI |
| `SettingsScreen` | `src/screens/SettingsScreen/` | App settings |
| `AboutScreen` | `src/screens/AboutScreen/` | Attribution — keeps PocketPal upstream links intentionally |
| `DevToolsScreen` | `src/screens/DevToolsScreen/` | Dev/debug tools |

---

## Key Source Files

### Anchor-specific (not upstream PocketPal)

| File | Purpose |
|---|---|
| `src/utils/anchorSystemPrompt.ts` | **Production system prompt.** `getAnchorSystemPrompt(mode)` returns 6-line BASE_PROMPT. `AnchorConversationMode = 'reflect' \| 'calm' \| 'focus'` (mode currently unused — all return same prompt). |
| `src/memory/contextBuilder.ts` | Builds full system prompt with memory. `buildEnhancedSystemPrompt()`, `truncateHistory()`. |
| `src/memory/sessionExtractor.ts` | Post-session extraction. `runExtraction()`. |
| `src/repositories/MemoryRepository.ts` | DB access: `getOrCreateProfile()`, `getRecentMemories(n)`, `searchByKeywords(keywords)`, `saveMemory()`, `addKnownName()`, `addUserFact()`. `searchByKeywords` scores against topics + peopleMentioned + summary text, with recency bonus (memories < 14 days old score +0–1.0 on top of keyword hits). |
| `src/screens/ChatScreen/ChatScreen.tsx` | Chat orchestration. Imports `getAnchorSystemPrompt`, calls `buildEnhancedSystemPrompt` via `useChatSession`. |
| `src/screens/PanicScreen.tsx` | Crisis screen. |
| `src/screens/ProfileSetupScreen/` | Profile setup wizard. Uses `loadProfile`/`saveProfile` from `src/utils/profileStorage.ts`. |
| `src/store/ModelStore.ts` | `lastUsedModelId` default: `jainaryan/mindmate-gguf/mindmate_genzv2_ck1200_q4_k_m.gguf` |

### Do not confuse with production

| File | What it actually is |
|---|---|
| `src/constants/mindmatePrompt.ts` | **~150-line structured prompt. Eval runner only** (`EvalRunner.ts`, `MultiTurnRunner.ts`). Not in production chat. Has old "MindMate" name. |
| `src/store/FeedbackStore.ts` | Uses AsyncStorage key `@pocketpal_ai/app_feedback_id` — **do not rename**, breaks existing users. |
| `src/screens/AboutScreen/` | PocketPal upstream URLs — **intentional attribution, do not remove**. |

### Chat internals

| File | Purpose |
|---|---|
| `src/hooks/useChatSession.ts` | Core chat hook. Calls `buildEnhancedSystemPrompt`, `prepareCompletion`, `runExtraction`. |
| `src/utils/systemPromptResolver.ts` | Resolves which system prompt to use (Anchor vs custom). |
| `src/utils/panicDetection.ts` | `detectPanic(msg)` → `PanicRiskLevel`. `'urgent'` = explicit self-harm intent (blocks message, hard redirect). `'watch'` = general distress phrasing (notifies caller, message still sends). `'none'` = safe. See tiered pattern lists in the file. |
| `src/utils/chat.ts` | `convertToChatMessages`, `removeThinkingParts`, `getHFDefaultSettings`. |
| `src/utils/profileStorage.ts` | `loadProfile()` / `saveProfile()` — AsyncStorage-backed profile persistence. |
| `src/utils/profileExtractor.ts` | Extracts profile suggestions from conversation (e.g. if user mentions age). |
| `src/utils/diarySummary.ts` | `generateReflectionSummary()` — generates diary summary from session. |

---

## Eval Harness

### Overview

29 scenarios in `src/eval/fixtures/scenarios.ts`:
- 14 single-turn
- 10 multi-turn
- 5 biometric

`EvalRunner.ts` runs single-turn scenarios. `MultiTurnRunner.ts` runs multi-turn.

**As of 2026-05-10:** `EvalRunner.ts` now defaults to the **production prompt** (`getAnchorSystemPrompt('reflect')`). To use the legacy 150-line structured prompt for historical comparison, pass it as the third arg: `runScenario(scenario, onStatus, MINDMATE_SYSTEM_PROMPT)`.

This means new eval runs are directly comparable to what users experience. Old runs (pre 2026-05-10) used `MINDMATE_SYSTEM_PROMPT` — note this when comparing historical scores.

Results: `src/eval/EVAL_RESULTS.md` — includes last run (2026-04-23, Pixel 8a) plus a warning block documenting all post-run fixes.

Held-out eval names (not in training data): **Zoya, Kabir, Tanvi, Layla, Rohan**.

### Running eval

Open app → navigate to Eval screen → select model → run. Results shown in-app and written to `EVAL_RESULTS.md`.

---

## Conversation Modes

`AnchorConversationMode = 'reflect' | 'calm' | 'focus'`

Defined in `src/utils/anchorSystemPrompt.ts`. All three currently return the same `BASE_PROMPT` — mode switching is implemented in the UI but the prompt differentiation is not yet built. Planned future feature.

---

## Building & Running

```bash
cd ~/projects/anchor-app

# Install dependencies
yarn install

# iOS (not primary — Android is the target)
cd ios && pod install && cd ..
yarn ios

# Android
yarn android

# TypeScript check
yarn tsc --noEmit

# Tests
yarn test

# Lint
yarn lint
```

### Commit style (enforced by commitlint + husky)

Subjects must be lowercase:
```bash
# ✅ Correct
git commit -m "feat: add biometric memory injection"
git commit -m "fix: context builder crash on empty profile"

# ❌ Rejected by hook
git commit -m "Feat: Add Biometric Memory Injection"
```

---

## On-device Performance (Pixel 8a, genzv2_ck1200 Q4_K_M)

| Metric | Value |
|---|---|
| Generation TPS | ~5.4–6.0 |
| TTFT cold (1,100+ token prompt) | 60–66s (~18 tok/s prefill) |
| TTFT KV-cached | ~6s |
| Peak heap | ~3.12–3.17 GB |
| Available RAM after model load | ~800–875 MB |
| Thermal (single session) | LIGHT |
| Thermal (45–60 min sustained) | MODERATE |

Model copy on first load: app copies GGUF to `DocumentDirectoryPath/models/local/` (1.9GB, shows full-screen spinner).

---

## Health Data Integration

`src/services/health/HealthDataService.ts` — `getDailySummaries(days)` → `DailySummary[]`

Reads from device health APIs (Android Health Connect / iOS HealthKit). Returns per-day summaries with: steps, sleepDurationMin, sleepQuality, restingHeartRate, hrv, spo2, activeCalories, exerciseMinutes, recoveryScore (Whoop), strainScore (Whoop).

Biometric context is injected into the system prompt as `[Health data — last 7 days]` (Tier 4). The contextBuilder includes specific directives about how to interpret poor sleep, low HRV, low recovery scores, and high strain.

---

## See also

- [[System Prompt]] — exact prompt text, memory injection format, training alignment
- [[Production]] — deploy details, model paths, on-device perf
- [[Models]] — which GGUF to use and why
