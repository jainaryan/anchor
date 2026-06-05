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
| **Package ID** | `com.anchor.app` — changed from `com.pocketpalai` on 2026-05-25 to prevent Play Store auto-updates from overwriting custom builds |
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
| `HomeScreen` | `src/screens/HomeScreen/` | Landing screen — animated orb, greeting, 3 action tiles (Write/Chat/Breathe). Hidden from sidebar, app opens here |
| `ChatScreen` | `src/screens/ChatScreen/ChatScreen.tsx` | Main chat. Wires model + memory + diary flow + panic detection |
| `ProfileSetupScreen` | `src/screens/ProfileSetupScreen/` | 4-step profile wizard: About you / What helps / Your people / Boundaries |
| `ModelsScreen` | `src/screens/ModelsScreen/` | GGUF model browser, download from HuggingFace, local model management |
| `MemoryViewScreen` | `src/screens/MemoryViewScreen/MemoryViewScreen.tsx` | Inspect episodic memory entries stored in DB |
| `DiaryScreen` | `src/screens/DiaryScreen/` | Diary entry list — search, per-entry preview, FAB for new entry, hamburger to open drawer |
| `DiaryEditorScreen` | `src/screens/DiaryEditorScreen/` | Edit/write a diary entry — text only, no mood picker |
| `PanicScreen` | `src/screens/PanicScreen.tsx` | Crisis screen — grounding techniques, crisis line numbers |
| `SettingsScreen` | `src/screens/SettingsScreen/` | App settings — Appearance (8 chat themes), Model, Notifications, Dev Settings |
| `DevSettingsScreen` | `src/screens/DevSettingsScreen/` | Dev-only — links to Benchmark, Dev Tools, Eval Harness screens (hidden from main sidebar) |
| `EvalScreen` | `src/screens/EvalScreen/` | In-app eval harness UI (accessible via Dev Settings only) |
| `BenchmarkScreen` | `src/screens/BenchmarkScreen/` | Model benchmarking UI (accessible via Dev Settings only) |
| `AboutScreen` | `src/screens/AboutScreen/` | Attribution — keeps PocketPal upstream links intentionally |
| `DevToolsScreen` | `src/screens/DevToolsScreen/` | Dev/debug tools (accessible via Dev Settings only) |

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

### v4 mobile eval infrastructure (in progress, 2026-05-15)

The mobile eval is being rewritten to share schema + judge with the cluster benchmark:

| Component | Status | File |
|---|---|---|
| Structured logger — JSONL + human + transcripts + failures, schema-shared with cluster | ✅ | `src/eval/logger.ts` |
| Per-turn telemetry — TTFT, gen TPS, memory, thermal, with regression thresholds | ✅ | `src/eval/telemetry.ts` |
| Panic detection unit tests (61 cases, 100% coverage on `panicDetection.ts`) | ✅ | `src/utils/__tests__/panicDetection.test.ts` |
| EvalRunner rewrite — multi-turn, no on-device judge, structured output | ⬜ | `src/eval/EvalRunner.ts` |
| Memory persistence integration test | ⬜ | `src/eval/integration/memoryPersistence.test.ts` |
| Shared scenarios.json (build-time copy from mindmate repo) | ⬜ | `src/eval/fixtures/scenarios.json` |
| `EVAL_RESULTS.md` becomes auto-generated from judged JSON | ⬜ | scripts |

**Per-turn telemetry thresholds** (`PERF_THRESHOLDS` in `src/eval/telemetry.ts`):
- min_gen_tps_per_turn: 4.0 (Pixel 8a baseline ~5.5)
- max_ttft_cold_ms: 90,000 / max_ttft_cached_ms: 12,000
- max_native_memory_mb: 3500 / min_available_ram_mb: 500
- max_gen_tps_drop_pct: 25 (cross-turn — flags thermal throttle)
- max_thermal_state: MODERATE (SEVERE/CRITICAL → warn)
- max_native_memory_growth_mb: 200 (cross-turn — memory leak signal)

Breaches surface as `perf_threshold_breach` warning events in `trace.log` — they don't fail the run.

The mobile logger writes to `${RNFS.ExternalDirectoryPath}/eval_runs/<run_id>/` so the entire run directory is `adb pull`-able. After eval finishes:

```bash
adb pull /sdcard/Android/data/com.pocketpalai/files/eval_runs/<run_id>/ local/
rsync -az local/<run_id>/ nus-student-cluster:~/projects/mindmate/results_pending/
ssh nus-student-cluster "sbatch --export=ALL,MOBILE_RESULTS=results_pending/<run_id>/raw.json \\
                                 ~/projects/mindmate/benchmarks/judge_mobile.slurm"
```

---

### ⚠️ Current eval results are stale — re-run needed

Last run: **2026-04-22**, model: **genzv2_ck1600**, before all targeted fix data and before c3acdc9.

**What changed since the last run that affects results:**
- c3acdc9 (2026-05-09): training format fix — should recover cross-session memory failures
- 8 anti-hallucination gold examples: should fix "you went quiet on me there" (3 failures in last run)
- 30 name_resolution + 18 session_recall examples: should recover `memory_recall_wedding`
- 15 profile_coping examples: should recover `coping_from_profile`
- Best model is now genzv2_ck1200, not genzv2_ck1600

**What to expect on re-run (genzv2_ck1200 on Pixel 8a):**
- `memory_recall_wedding`, `coping_from_profile`, `help_mode_direct_opener` → likely improved
- Hallucination scenarios (`empty_profile_first_use`, `cross_session_mood_trend`) → likely improved
- `cross_session_coping_outcome_followup` → likely still failing (cluster CROSS_SESSION_MEMORY = 0% for ck1200)
- Crisis scenarios → watch carefully; cluster shows CRISIS degraded to 48% for ck1200 vs base 67%

**Action before genzv5:** Re-run in-app eval on Pixel 8a with genzv2_ck1200 Q4_K_M. These results establish the on-device baseline genzv5 must beat. See `src/eval/EVAL_RESULTS.md`.

---

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

## UI Revamp (2026-05-25)

### Session 6 changes (2026-05-27)

- **Release APK build** — switched from debug (Metro-dependent) to release build. JS bundle baked into APK via `npx react-native bundle --dev false`. Old `com.mindmate` package uninstalled from device — was causing old app to load on restart. Release APK installed permanently; works without Metro.
- **App icon color swap** — previous session had icon colors inverted (white anchor on teal). Fixed by swapping `ic_launcher_background` color (`#3E706C` → `#FFFFFF`) and `ic_launcher_foreground.xml` fill (`#FFFFFF` → `#3E706C`). Note: adaptive icon XML (`mipmap-anydpi-v26/`) overrides PNG mipmaps on Android 8+ — PNG changes alone don't affect the visible icon.
- **App icon centering** — anchor was 29px above center in the 1024px viewport. Fixed by wrapping paths in `<group android:translateY="29">` in `ic_launcher_foreground.xml`.
- **`Anchor-Icon.png` recentered** — replaced flat-scale render with crop-to-bbox + center-with-equal-padding approach using PIL. Anchor is now visually centered in the 512×512 image. Color: teal (`#3E706C`) on white.

### Session 5 changes (2026-05-26)

- **Screen transitions** — `withFadeTransition` HOC added to `App.tsx`. Wraps every Drawer screen in a `Reanimated.View`; uses `useFocusEffect` to fire a 230ms fade + 6px upward translate on every navigation (not just first mount, because `useFocusEffect` re-triggers on each focus). Modal-style screens (DiaryEditor, Panic, ProfileSetup) get 14px translate for a more pronounced slide-up. Bug note: original implementation reset opacity to 0 in the cleanup function, which snapped the incoming screen invisible mid-drawer-slide — fixed by resetting at the START of the focus callback instead (no cleanup reset). `drawerType: 'back'` also reverted after it conflicted with the opacity animation.
- **ChatEmptyPlaceholder: remove "Download Anchor" button** — replaced `Button` with a `TouchableOpacity` wrapping the whole card. Title/description text now uses `palette.ink` / `palette.inkSoft` (was `theme.colors.onSurface` — invisible on dark themes). Description for the no-models state simplified to direct user to sidebar.
- **Chat header logo** — `ChatHeaderTitle` logo size `20×20` → `32×32` with `borderRadius: 8`.
- **App icon PNGs regenerated** — all 10 mipmap PNGs (mdpi→xxxhdpi, `ic_launcher` + `ic_launcher_round`) regenerated using `cairosvg` from `anchor-logo.svg`: white anchor (`#FFFFFF`) on teal background (`#3E706C`). Previous icons had anchor in near-identical teal (invisible). Generated at 72% fill ratio so anchor sits within safe zone.
- **`Anchor-Icon.png` replaced** — was 587×425 RGBA with grey anchor on transparent background (broke on dark themes). Replaced with 512×512 white-anchor-on-teal to match app icon and look correct on all chat themes.

### Session 4 changes (2026-05-26)

- **Chat text visibility fix (all themes)** — `TextMessage.tsx` `textColor` was hardcoded `#1F2933` (dark), invisible on midnight/dark themes. Now uses `palette.aiFg` (AI messages) / `palette.userFg` (user messages) from `uiStore.activeChatPalette`. `TextMessage/styles.ts` link-preview colors updated the same way.
- **ChatInput palette-aware colors** — `textColor`, `secondaryTextColor`, `accentColor` were all hardcoded. Now use `palette.ink`, `palette.inkSoft`, `palette.sendBgBottom` respectively.
- **ChatView composer background** — `inputBackgroundColor` was `theme.dark ? surface : '#FFFFFF'` (ignored chat theme). Now uses `palette.composerBg`.
- **ChatScreen session tools + empty state** — `TEXT_PRIMARY/SECONDARY/MUTED` constants removed. `styles` converted from static `StyleSheet.create` to `createStyles(palette)` function called inside the observer component. All text colors now use `palette.ink` / `palette.inkSoft`. UI cards (mood strip, toolbar, pills) adapt opacity for dark themes.
- **SettingsScreen theme swatch label** — selected state used hardcoded `#1F2933`; now uses `theme.palette.ink` (each theme's own contrast color).

### Session 3 changes (2026-05-25)

- **Home screen: 3D orb** — breathing orb replaced with react-native-svg `RadialGradient` sphere. Five render layers: base diffuse (light source top-left 30%/26%), bottom-right shadow pass, soft highlight bloom, sharp specular ellipse, tiny bright specular core. Palette tokens `avatarOrbTop/Mid/Bottom` drive all layers per theme.
- **Sidebar visual redesign** — `SidebarContent` now uses `LinearGradient` background (`#FBFDFC → #F2F7F6 → #E5EFEC`). New header: 36×36 SVG teal orb (radial gradient + white stroke anchor icon, drop shadow) + "Anchor" serif italic. All nav items replaced with custom `DrawerNavItem` (34×34 rounded-square icon wrap, hint subtitle, 3px teal active left bar). Privacy footer: lock icon + "Everything stays on this device. / No cloud, no account, no analytics."
- **ChatInput placeholder** — changed to `"Tell Anchor anything…"`.
- **Adaptive app icon** — added `mipmap-anydpi-v26/ic_launcher.xml` + `ic_launcher_round.xml` (adaptive icon XML), `drawable/ic_launcher_foreground.xml` (vector drawable, white anchor on transparent, 108dp canvas), `values/colors.xml` (`ic_launcher_background: #3E706C`). Android 8+ launchers now apply correct shape masking (circle/squircle/etc).
- **Package ID changed** — `applicationId` changed from `com.pocketpalai` → `com.anchor.app` (`namespace` stays `com.pocketpal` for Codegen compatibility). Prevents Play Store auto-updates from overwriting dev builds. Requires uninstalling old app before first install: `adb uninstall com.pocketpalai`.

### Session 2 changes (2026-05-24)

- **Home screen** — new `HomeScreen` with animated breathing orb, per-theme greetings, 3 action tiles. Registered as first Drawer.Screen but hidden from sidebar; app opens here.
- **8 chat themes** — `uiStore.activeChatThemeKey` drives `activeChatPalette` (harbor/midnight/paper/sunset/lagoon/amethyst/sage/mono). Palette tokens used everywhere: `ink`, `inkSoft`, `composerBg`, `composerBorder`, `avatarOrbTop/Mid/Ring`, `blob1Color`, `blob2Color`.
- **SettingsScreen restructure** — Appearance section with theme picker grid; Dev Settings moved to `DevSettingsScreen` (separate screen); Benchmark/Dev Tools/Eval Harness accessible only from Dev Settings.
- **Sidebar cleanup** — Home added as first nav item; Benchmark/Dev Tools/Eval removed from sidebar; `AnchorLogoIcon` replaces `BenchmarkIcon`.
- **Diary: mood removed** — mood picker removed from `DiaryEditorScreen`. `DiaryEntry.mood` still stored in DB (defaults to `'calm'` on new entries, preserved on edit) to avoid breaking existing storage.
- **Diary: FAB + hamburger** — FAB for new entry (teal pill, bottom-right). Hamburger opens navigation drawer.
- **Diary: warm paper card design** — cards use `rgba(251,246,233,0.88)` background with ink/inkSoft colors.
- **App icon** — all Android mipmap PNGs (`ic_launcher` + `ic_launcher_round`) replaced with the real Anchor logo (white anchor on teal `#3E706C` background), generated at mdpi/hdpi/xhdpi/xxhdpi/xxxhdpi.
- **iOS-only module shims** — `appleAuthShim`, `healthKitShim`, `healthConnectShim` platform shims prevent Metro from crashing on Android; `crypto-js` replaced with native WebCrypto (`crypto.subtle`).

---

## Legacy UX Plan (session 6)

Three UX updates are now part of the near-term app roadmap.

### 1) Calmer, simpler chat home

**Goal:** First-time users should see only mood check-in, input, and safe empty-state.

**Change:**
- Move secondary controls behind a subtle `Session tools` drawer:
  - conversation mode pills (`reflect/calm/focus`)
  - thinking toggle
  - profile suggestion banner/actions
- Keep the default chat surface low-noise (no dense control strip above messages).

**Primary files:**
- `src/screens/ChatScreen/ChatScreen.tsx`
- `src/components/ChatView/ChatView.tsx` (if drawer trigger/control lives there)

**Acceptance criteria:**
- On empty or first-use sessions, users see only core chat UI and safety copy.
- Advanced controls are discoverable but collapsed by default.
- No regression in mode switching, thinking toggle persistence, or profile suggestion save/dismiss.

### 2) Passive diary auto-capture + editable draft

**Goal:** Remove "save" friction mid-chat while preserving edit control.

**Change:**
- Replace manual mid-conversation "Save to diary" behavior with passive auto-capture:
  - at session-end extraction window, auto-generate reflection draft
  - show lightweight `Review & Save` card/snackbar
  - tap opens `DiaryEditor` prefilled with draft for quick edits
- Keep explicit save affordance only in the review card, not as persistent toolbar action.

**Primary files:**
- `src/screens/ChatScreen/ChatScreen.tsx`
- `src/hooks/useChatSession.ts`
- `src/utils/diarySummary.ts`
- `src/screens/DiaryEditorScreen/DiaryEditorScreen.tsx`

**Acceptance criteria:**
- Users do not need to manually press save during live conversation.
- Draft appears reliably after session-end debounce.
- Users can edit before final save, with no duplicate draft entries.

### 3) Progressive profile onboarding

**Goal:** Increase completion by splitting setup into fast essentials + optional depth.

**Change:**
- Restructure `ProfileSetupScreen` into:
  - `2-minute essentials` (age/gender/location, coping basics, one support person)
  - `optional deep profile` (diagnoses detail, therapy metadata, boundaries, avoid topics)
- Keep completion momentum with clear "Finish essentials" and "Add more later" paths.

**Primary files:**
- `src/screens/ProfileSetupScreen/ProfileSetupScreen.tsx`
- `src/screens/MemoryViewScreen/MemoryViewScreen.tsx` (entry points to optional deep-edit)

**Acceptance criteria:**
- User can complete core onboarding quickly with fewer required interactions.
- Optional deep profile remains accessible after onboarding.
- No data loss across essentials/deep sections.

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

## GPU / CPU Inference Defaults

**Default since 2026-06-05: CPU-only (`n_gpu_layers = 0`). GPU is opt-in.**

### Background
The GPU path (OpenCL via Adreno) is only enabled when a phone passes three checks: `hasAdreno && hasI8mm && hasDotProd` ([`deviceCapabilities.ts`](../src/utils/deviceCapabilities.ts)). Phones that fail (e.g. Pixel with Mali GPU) are silently downgraded to CPU. All development and benchmarking has been done on the Pixel 8a, which always ran CPU — so the GPU/OpenCL path with `n_gpu_layers=99` was completely untested and caused freezes/OOM on Snapdragon devices.

### Fix
Three defaults changed in [`contextInitParamsVersions.ts`](../src/utils/contextInitParamsVersions.ts):
- `createContextInitParams`: `n_gpu_layers ?? 99` → `0`
- `createDefaultContextInitParams`: `n_gpu_layers: 99` → `0`
- Migration 1.0→2.0: when GPU was previously enabled and n_gpu_layers was unset, no longer upgrades it to 99

GPU is still configurable via Settings — users on Snapdragon phones can manually enable it to experiment.

### GPU support matrix (Android)

| GPU family | `checkGpuSupport()` | Result |
|---|---|---|
| Adreno (Qualcomm/Snapdragon) | ✅ passes (if i8mm + dotprod present) | GPU path available in settings |
| Mali (Pixel/Tensor, Exynos, MediaTek) | ❌ fails `hasAdreno` | Forced to CPU |

All perf benchmarks below are CPU-path numbers (Pixel 8a). No validated GPU-path numbers exist yet.

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

## Play Store Publishing (2026-06-05)

De-PocketPal pass + self-review for Google Play submission. Two scoping decisions: **full removal** of inherited PocketPal features that don't fit a focused local mental-health companion, and **document (not rename)** deep internal identifiers (renaming breaks existing-user data and native codegen).

### Features removed (full removal)
- **PalsHub** persona marketplace — utils (`exportPal`/`importPals` + `imageUtils.ts`), screens, fixtures, mocks, tests.
- **Local-server mode** + **remote-model / OpenAI-compatible API** — `src/api/openai.ts`, `src/api/sseParser.ts`, `OpenAICompletionEngine` (only `LocalCompletionEngine` is instantiated now, ModelStore.ts:1605). `remoteModels` getter stubbed to `[]`; residual `ModelOrigin.REMOTE` guards left as inert defensive no-ops (never reachable — no creation path remains). Safe to delete in a later cleanup.
- **Auth** — Supabase, Google Sign-In, Apple Authentication deps dropped from package.json; jest mappers/mocks removed.
- Verified: no new `tsc` errors (only the pre-existing 7-error health-shim baseline). Jest delta confirms **zero new test failures** — clean HEAD = 58 failed suites / 13 failed tests; after removal = 50 failed suites / 12 failed tests (253 fewer total tests, all from deleted-feature suites). Residual failures are pre-existing: `react-native-health` not in package.json (~50 suites fail to load) + `activeChatPalette` missing from the `uiStore` test mock. Fixed one stale test (`contextInitParamsVersions` expected `n_gpu_layers` 99 → updated to 0, matching the GPU-opt-in default).

### On-device verification (Pixel 8a / akita, 2026-06-05)
`yarn install` → `./gradlew assembleDebug` → **BUILD SUCCESSFUL** (removing the 3 native auth deps did not break the native build). Installed + launched on the Pixel: home screen renders with **Anchor** branding (anchor icon + name), Diary and "Talk it out" Chat entry points work, navigation into Chat is clean, no FATAL exceptions. Chat correctly shows "Model not loaded" on a fresh install (no GGUF downloaded yet). The `palette.ink` code path that fails in the unit-test mock works fine on real hardware (the real `uiStore` has `activeChatPalette`), confirming that unit failure is a mock gap, not a bug. Dev requires `adb reverse tcp:8081 tcp:8081` for Metro (auto-set by `yarn android`).

### Reviewer findings

| # | Severity | Issue | Status |
|---|---|---|---|
| 1 | **BLOCKER** | `android/app/build.gradle` release block **falls back to the debug keystore** when `APP_RELEASE_STORE_PASSWORD`/`APP_RELEASE_KEY_PASSWORD` are absent. A debug-signed AAB is rejected by Play, and the debug keystore is publicly known. | **Open — must fix before upload.** Ensure release credentials (`pocketpal-release-key.keystore` + env vars) are present in the release build env, or remove the debug fallback so a misconfigured build fails loudly instead of shipping insecure. Enroll in Play App Signing. |
| 2 | Resolved | Cleartext HTTP was enabled (`network_security_config.xml`) — only needed by the removed local-server feature. | Fixed → release/`main` config now `cleartextTrafficPermitted="false"`. **Gotcha:** this also blocks Metro (localhost:8081) in debug builds, which broke on-device dev with a black screen. Fixed by adding a **debug-only** override `android/app/src/debug/res/xml/network_security_config.xml` permitting cleartext to `localhost`/`10.0.2.2`/`127.0.0.1` only. Release stays locked down. |
| 3 | Info | **Permissions** all justified: `INTERNET` (model downloads from HuggingFace, Firebase App Check), `CAMERA` (multimodal vision input — `react-native-vision-camera` + `react-native-image-picker` in `ChatInput.tsx`/`EmbeddedVideoView.tsx`), `READ/WRITE_EXTERNAL_STORAGE` (`maxSdkVersion=28`, chat-session export to `/Download`). | No action. |
| 4 | Action | **Data Safety form** — what leaves the device: (a) HuggingFace (`huggingface.co`) model downloads/metadata — no personal data; (b) Firebase Functions `feedbackSubmit` + `benchmarkSubmit` (`src/config/urls.ts`) — user-initiated feedback/error reports and opt-in device benchmark results; (c) Firebase App Check attestation tokens (anti-abuse, no PII). **Chat content, diary, memory, and health data never leave the device.** | Declare the above in the Play Console Data Safety form + publish a matching privacy policy. |
| 5 | Info | Branding: `displayName`/`app_name` = "Anchor", launcher icons present, deep-link scheme `anchor`, iOS LaunchScreen text = "Anchor". | Consistent. |

### Internal PocketPal identifiers — KEPT (documented, not renamed)

Renaming any of these breaks existing-user data, native codegen, or app-signing. Left intentionally as-is:

| Identifier | Location | Why kept |
|---|---|---|
| `@pocketpal_ai/app_feedback_id` | `src/store/FeedbackStore.ts` | AsyncStorage key — renaming orphans existing users' feedback IDs. |
| `dbName: 'pocketpalai'` | `src/database/index.ts` | WatermelonDB file name — renaming abandons all on-device user data. |
| `name: "PocketPal"` | `app.json` | Couples to native `getMainComponentName()` (`MainActivity.kt`) + `withModuleName` (`AppDelegate.swift`). This is the RN component id, **not** the user-facing label (`displayName: "Anchor"`). |
| `namespace 'com.pocketpal'` | `android/app/build.gradle` | Build namespace; applicationId is already `com.anchor.app`. |
| `com.pocketpalai` Kotlin pkg / `com.pocketpal.specs` / `PocketPalSpecs` | Kotlin sources, `codegenConfig` | Native package dirs + TurboModule codegen names. |
| `pocketpal-release-key.keystore` / `pocketpal_key_alias` | `android/app/build.gradle` | Existing release signing identity — must match the Play upload/signing key. |
| iOS `PocketPal` project/scheme, `ai.pocketpal` entitlement, `com.pocketpalai.deeplink`, scheme `pocketpal` | `ios/`, build scripts | Xcode project + iOS bundle identifiers. |
| `rootProject.name = 'PocketPal'` | `android/settings.gradle` | Gradle internal project name. |
| GitHub attribution URL `a-ghorbani/pocketpal-ai` | `src/screens/AboutScreen/AboutScreen.tsx:152` | Required open-source attribution to the upstream fork. |

User-facing brand strings (comments, labels, LaunchScreen) were changed PocketPal → Anchor.

---

## See also

- [[System Prompt]] — exact prompt text, memory injection format, training alignment
- [[Production]] — deploy details, model paths, on-device perf
- [[Models]] — which GGUF to use and why
