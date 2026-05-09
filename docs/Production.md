---
tags: [anchor, production]
---

# Production

← [[Home]]

---

## Webapp (tryanchor.me)

| | |
|---|---|
| **URL** | https://tryanchor.me |
| **Server** | DigitalOcean c-4 droplet |
| **IP** | 209.38.122.228 |
| **Stack** | FastAPI + SSE streaming, `deploy/server.py` |
| **Current model** | `exports/mindmate_llama_sft_ck1600/llama(genz)v2_q4_k_m.gguf` ← ⚠️ outdated |
| **Best GGUF available** | `exports/mindmate_genzv2_ck1200_q4_k_m.gguf` — should upgrade |
| **Service name** | `mindmate` (managed by systemd) |

### SSH & Management

```bash
# SSH in
ssh -i ~/.ssh/id_ed25519 root@209.38.122.228

# Service management
systemctl restart mindmate
systemctl status mindmate
journalctl -u mindmate -f   # live logs

# Check running process
ps aux | grep server.py
```

### Deploy Static Files

```bash
rsync -az -e "ssh -i ~/.ssh/id_ed25519" \
  deploy/static/ \
  root@209.38.122.228:~/mindmate/deploy/static/

ssh -i ~/.ssh/id_ed25519 root@209.38.122.228 "systemctl restart mindmate"
```

### Upgrade Model to genzv2_ck1200

```bash
# 1. Rsync GGUF to server (~1.9GB)
rsync -az --progress -e "ssh -i ~/.ssh/id_ed25519" \
  exports/mindmate_genzv2_ck1200_q4_k_m.gguf \
  root@209.38.122.228:~/mindmate/exports/

# 2. Update deploy/config.py (locally first, then rsync)
# Change MODELS[0]["path"] from:
#   "exports/mindmate_llama_sft_ck1600/llama(genz)v2_q4_k_m.gguf"
# to:
#   "exports/mindmate_genzv2_ck1200_q4_k_m.gguf"

rsync -az -e "ssh -i ~/.ssh/id_ed25519" deploy/config.py root@209.38.122.228:~/mindmate/deploy/

# 3. Restart
ssh -i ~/.ssh/id_ed25519 root@209.38.122.228 "systemctl restart mindmate"
```

### `deploy/config.py` — Current Model Config

```python
MODELS = [
    {
        "id": "mindmate-llama-sft-ck1600-q4",
        "name": "default",
        "path": os.path.join(_repo_root, "exports", "mindmate_llama_sft_ck1600", "llama(genz)v2_q4_k_m.gguf"),
        "chat_format": "llama-3",
    },
]
N_GPU_LAYERS = 0     # 0 = CPU-only (DigitalOcean c-4 has no GPU)
N_CTX = 2048
MAX_TOKENS = 150
TEMPERATURE = 0.75
INFERENCE_TIMEOUT = 120  # seconds
```

### `deploy/server.py` — Architecture

FastAPI + SSE. Key endpoints:
- `POST /chat` — sends message, returns SSE stream of tokens
- `GET /health` — health check

Server loads model at startup using llama-cpp-python. No GPU on DigitalOcean c-4 — CPU inference only (slow). The webapp is for demo purposes; real inference speed is on-device.

---

## Android App (anchor-app)

| | |
|---|---|
| **Repo** | github.com/tanmaykay/Anchor |
| **Branch** | `aryan_branch` |
| **Local path** | `~/projects/anchor-app/` |
| **Package ID** | `com.pocketpalai` (upstream PocketPal ID — intentionally unchanged, changing breaks existing installs) |
| **Upstream** | PocketPal AI (open-source GGUF runner) — forked and extended |

### Architecture

1. User selects GGUF file from HuggingFace or local storage
2. App copies to `DocumentDirectoryPath/models/local/` (shows full-screen spinner during 1.9GB copy)
3. `contextBuilder.ts` builds system prompt from profile + episodic memory DB (SQLite)
4. `useChatSession` holds system prompt fixed for entire session
5. JNI → llama.cpp → tokenize + decode + sample
6. After session ends: `sessionExtractor.ts` extracts summary → stored as `EpisodicMemoryData`

### Memory System — Key Files

| File | Purpose |
|---|---|
| `src/memory/contextBuilder.ts` | Builds enhanced system prompt (Tier 1/2/3 injection). `buildEnhancedSystemPrompt(basePrompt, userMessage)` |
| `src/memory/sessionExtractor.ts` | Extracts session summary after chat ends |
| `src/repositories/MemoryRepository.ts` | SQLite persistence layer |
| `src/database/models/EpisodicMemory.ts` | Schema: date, summary, moodStart, moodEnd, copingOutcome, heartRateAvg, sleepHours, etc. |
| `src/utils/anchorSystemPrompt.ts` | Production system prompt `getAnchorSystemPrompt()` — 6-line Anchor prompt |

**Only the `summary` text field of `EpisodicMemoryData` is passed to the model.** Structured fields (moodStart, heartRate, etc.) are stored in DB but invisible to the LLM.

### Memory Tiers (contextBuilder.ts)

- **Tier 1:** User profile card — name, age, gender, diagnoses, triggers, coping strategies (★helpful/✗unhelpful), support people, risk flags
- **Tier 2:** 3 most recent sessions — `summary` text only, always included
- **Tier 3:** Keyword-matched past sessions — deduped against Tier 2, included if relevant keywords found

### Chat Flow

`ChatScreen.tsx` → `getAnchorSystemPrompt()` → `useChatSession` → `prepareCompletion()` → `buildEnhancedSystemPrompt(basePrompt, userMessage)` → injects Tier 1/2/3 → final system prompt sent to model.

The final system prompt the model sees:
```
You are Anchor, a warm and caring AI companion...
[6-line base prompt]

ABOUT THIS USER
[User]
Name: ...
Age: ...
Diagnoses: ...
Triggers: ...
Coping strategies: ...
...

[Recent sessions]
Session 1: <summary text>
Session 2: <summary text>
Session 3: <summary text>
```

### Key Source Files

| File | Notes |
|---|---|
| `src/utils/anchorSystemPrompt.ts` | Production system prompt (6-line). Only this is in production chat. |
| `src/constants/mindmatePrompt.ts` | ~150-line structured prompt. **Only used in eval runner**, not production chat. |
| `src/screens/ChatScreen/ChatScreen.tsx` | Main chat screen |
| `src/store/ModelStore.ts` | Default `lastUsedModelId`: `jainaryan/mindmate-gguf/mindmate_genzv2_ck1200_q4_k_m.gguf` |
| `src/eval/EvalRunner.ts` | In-app eval runner |
| `src/eval/MultiTurnRunner.ts` | Multi-turn eval runner |
| `src/eval/fixtures/scenarios.ts` | 29 eval scenarios |
| `src/eval/EVAL_RESULTS.md` | Eval results history |

### On-device Performance (Pixel 8a, genzv2_ck1200 GGUF Q4_K_M)

| Metric | Value |
|---|---|
| Generation TPS | ~5.4–6.0 stable |
| TTFT (cold — full 1,100+ token prompt) | 60–66s (~18 tok/s prefill) |
| TTFT (KV-cached) | ~6s |
| Peak heap | ~3.12–3.17 GB |
| Available RAM after heap | ~800–875 MB |
| Thermal (single turn) | LIGHT |
| Thermal (45–60 min sustained) | MODERATE |

### FeedbackStore Note

`src/store/FeedbackStore.ts` uses AsyncStorage key `@pocketpal_ai/app_feedback_id`. **Do not rename this key** — changing it would break existing users' stored feedback IDs. It's intentionally kept as the upstream PocketPal identifier.

### Eval Harness

29 scenarios: 14 single-turn + 10 multi-turn + 5 biometric.

Last run: **2026-04-23, Pixel 8a.** Baseline was ~36% single-turn, ~13% multi-turn all-pass. Many fixes applied since then (see [[Bug Log]]).

Held-out eval names (not in training data): Zoya, Kabir, Tanvi, Layla, Rohan.

---

## Model Distribution

| GGUF | Host | Path |
|---|---|---|
| `mindmate_genzv2_ck1200_q4_k_m.gguf` | HuggingFace | `jainaryan/mindmate-gguf/mindmate_genzv2_ck1200_q4_k_m.gguf` |
| `mindmate_genzv2_ck1200_q4_k_m.gguf` | Local | `~/projects/mindmate/exports/mindmate_genzv2_ck1200_q4_k_m.gguf` |
| `mindmate_llama_sft_ck1600/...` | tryanchor.me | `~/mindmate/exports/mindmate_llama_sft_ck1600/llama(genz)v2_q4_k_m.gguf` |

---

## See also

- [[Models]] — model details and benchmark scores
- [[System Prompt]] — prompt architecture and memory injection
