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
| **Stack** | FastAPI + SSE streaming |
| **Current model** | `mindmate_llama_sft_ck1600/llama(genz)v2_q4_k_m.gguf` ← ⚠️ outdated |
| **Best available** | `mindmate_genzv2_ck1200_q4_k_m.gguf` — should upgrade |

### SSH & Management

```bash
# SSH in
ssh -i ~/.ssh/id_ed25519 root@209.38.122.228

# Restart server
systemctl restart mindmate

# Check status
systemctl status mindmate

# Deploy static files
rsync -az -e "ssh -i ~/.ssh/id_ed25519" deploy/static/ root@209.38.122.228:~/mindmate/deploy/static/
```

### Upgrading the Model

```bash
# 1. Rsync new GGUF to server
rsync -az --progress -e "ssh -i ~/.ssh/id_ed25519" \
  exports/mindmate_genzv2_ck1200_q4_k_m.gguf \
  root@209.38.122.228:~/mindmate/exports/

# 2. Update model path in deploy/config.py

# 3. Restart
ssh -i ~/.ssh/id_ed25519 root@209.38.122.228 "systemctl restart mindmate"
```

> ⚠️ Server still running ck1600 GGUF. Upgrade to `mindmate_genzv2_ck1200_q4_k_m.gguf` when convenient.

---

## Android App (anchor-app)

| | |
|---|---|
| **Repo** | github.com/tanmaykay/Anchor |
| **Branch** | `aryan_branch` |
| **Local path** | `~/projects/anchor-app/` |
| **Package** | `com.pocketpalai` (upstream PocketPal ID — not changed) |
| **Upstream** | PocketPal AI (forked) |

### Architecture

1. User selects GGUF → app copies to `DocumentDirectoryPath/models/local/` (1.9GB copy, full-screen spinner)
2. `contextBuilder.ts` builds system prompt from profile + memory (Tier 1/2/3)
3. `MultiTurnRunner.ts` holds `fullPrompt` constant for entire session; history resets between sessions
4. JNI → llama.cpp: tokenize + decode + sample

### Memory System

| File | Purpose |
|---|---|
| `src/memory/contextBuilder.ts` | Builds enhanced system prompt (Tier 1/2/3) |
| `src/memory/sessionExtractor.ts` | Extracts session summary after chat ends |
| `src/repositories/MemoryRepository.ts` | SQLite persistence |
| `src/database/models/EpisodicMemory.ts` | Schema: date, summary, moodStart, moodEnd, copingOutcome, etc. |

Only the `summary` field is passed to the model — structured fields are invisible to the LLM.

**Memory tiers:**
- **Tier 1:** User profile card — demographics, diagnoses, triggers, coping strategies, support people, risk flags
- **Tier 2:** Recent 3 sessions (always included)
- **Tier 3:** Keyword-matched past sessions (deduped from Tier 2)

### On-device Performance (Pixel 8a, genzv2_ck1200)

| Metric | Value |
|---|---|
| Gen TPS | ~5.4–6.0 |
| TTFT (cold) | 60–66s (1,100+ token prompt, ~18 tok/s prefill) |
| TTFT (cached) | ~6s |
| Peak heap | ~3.12–3.17 GB |
| Available RAM | ~800–875 MB |
| Thermal (1 turn) | LIGHT |
| Thermal (45–60 min) | MODERATE |

### Key Source Files

| File | Notes |
|---|---|
| `src/utils/anchorSystemPrompt.ts` | Production system prompt (6-line "You are Anchor...") |
| `src/screens/ChatScreen/ChatScreen.tsx` | Main chat screen; calls `getAnchorSystemPrompt()` |
| `src/store/ModelStore.ts` | Default model: `mindmate_genzv2_ck1200_q4_k_m.gguf` |
| `src/eval/` | Eval harness (EvalRunner, EvalScreen, fixtures) |
| `src/eval/EVAL_RESULTS.md` | Eval history with fixes log |

### Eval Harness

29 scenarios total: 14 single-turn + 10 multi-turn + 5 biometric.

Last run: **2026-04-23, Pixel 8a.** ~36% single-turn, ~13% multi-turn all-pass. Many issues fixed since then (see [[Bug Log]]).

---

## Model Distribution

| GGUF | Host | Notes |
|---|---|---|
| `mindmate_genzv2_ck1200_q4_k_m.gguf` | HuggingFace: `jainaryan/mindmate-gguf` | Best model |
| `mindmate_genzv2_ck1200_q4_k_m.gguf` | `exports/` on cluster | Source |
| `mindmate_llama_sft_ck1600/llama(genz)v2_q4_k_m.gguf` | tryanchor.me (served) | Outdated |

---

## See also

- [[Models]] — model details and benchmark scores
- [[System Prompt]] — prompt architecture and memory injection
