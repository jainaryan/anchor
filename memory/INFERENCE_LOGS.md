# MindMate Inference Logs

Qualitative testing notes across model checkpoints and configurations.

---

## Qwen3-1.7B — v2 adapter (`CUDA_mindmate_qwen3_1p7b_v2`)
**Date:** 2026-04-08
**Job:** 554800 (A100-40, gpu-long, 54 min)
**Data mix:**
| Source | Samples | % |
|---|---|---|
| transition | 3,000 | 30% |
| therapist | 2,500 | 25% |
| casual | 2,000 | 20% |
| grief/loss | 1,500 | 15% |
| friend | 1,000 | 10% |
| **Total** | **10,000** | |
**System prompt:** Full system prompt (186 lines, sections 1-9)
**Note:** Qwen v1 (`CUDA_mindmate_qwen3_1p7b`) was overwritten by this job before folder was renamed. v1 weights are lost.

---

### checkpoint-200

```
You: hey
Mindmate: what's on your mind?

You: im sad
Mindmate: i'm here for you if you want to talk about anything—not just now—but anytime. do you wanna share something that makes this feeling easier?

You: yeah my dog died
Mindmate: that hurts… so much. i know how hard those moments can be. is there any way we could take small steps toward healing together right now? even just sitting quietly—just being quiet enough—together?

You: i wanna talk about him
Mindmate: okay. let me hear everything—even loud stuff—that matters most when he was alive. maybe start by saying his name once—and then again—in case that helps bring some clarity.

You: his name was clifford, he was a golden retriever
Mindmate: oh wow... 🐾 That's such an important part of what made him special. I wonder—is there ever going to be a time where someone gets to say "I miss you" in front of everyone else too?
Tell me more—I don't need to make sense of it—it needs to show up as *real*. Just tell us exactly how he felt—he didn't live forever, but he lived fully. And still lives inside every memory you carry today.

You: i feel like it was my mistake
Mindmate: you're not alone in thinking that. sometimes our words—or silence—are louder than we realize. did anyone hold space for you before you started speaking out loud tonight?
```

**Rating: 6/10**

✅ Good:
- "what's on your mind?" — perfect casual greeting
- Remembered name Clifford across turns
- No breathing exercises
- Correct response to guilt ("you're not alone in thinking that")

❌ Bad:
- "small steps toward healing together" — therapy-speak
- "is there ever going to be a time where someone gets to say 'I miss you'" — incoherent hallucination
- "did anyone hold space for you" — therapist jargon

---

### checkpoint-1600 (final)

```
You: hey!
Mindmate: oh hi there 😎 what's up? just hanging out in your head today?

You: im sad
Mindmate: i'm right here for that kind of thing... talk if you want. i've been listening since day 1.

You: my dog died
Mindmate: that hurts… so much. did she ever look at me before leaving us? even once?

You: i wanna talk about him
Mindmate: okay, let her be seen first. just say 'yes' when he was still alive. then we can move slow from there.
```

**Rating: 3/10**

✅ Good:
- Casual greeting was natural
- Pivot to serious tone happened

❌ Bad (worse than checkpoint-200):
- "i've been listening since day 1" — fake shared history (violates rule 1.3)
- "did she ever look at **me**" — wrong gender (him→she) + inserted itself as having relationship with dog
- "let **her** be seen first" — still wrong gender after user said "him"
- Incoherent instructions ("just say 'yes' when he was still alive")

**Conclusion:** Overfit at 1600 steps. Model memorised therapy-speak patterns and hallucination habits. Checkpoint-200 is better.

---

### checkpoint-800

```
You: hey
Mindmate: Oh wait… that's something I've been meaning to say for years now 😅 What was your first thought when we met?

You: im sad rn
Mindmate: ohhh yeah. i know how this feels right after work ends 🙃 just sit here next time if you want. i'm always there in case you need anything else.
```

**Rating: 1/10**

❌ Bad (worst checkpoint):
- "i've been meaning to say for years now" — severe fake shared history on first message
- "What was your first thought when we met?" — implies prior relationship that doesn't exist
- "i know how this feels right after work ends" — invented context (user never mentioned work)
- "just sit here next time" — implies past sessions
- Did NOT pivot on "im sad rn" at all — stayed in casual/jokey mode

**Conclusion:** checkpoint-800 is the worst of the three. Heavy overfitting to casual data patterns with maximum hallucination of shared history.

---

## Key Findings (Qwen3-1.7B)

| Issue | Root Cause | Fix |
|---|---|---|
| Fake shared history ("since day 1") | Overfitting at high step count | Use checkpoint-200/400, reduce max_steps |
| Gender confusion (him→she) | 1.7B capacity limit | Llama 3B will handle this better |
| Therapy-speak ("hold space", "small steps") | Therapist data bleeding into casual | DPO penalises this |
| Incoherent sentences | 1.7B hallucination under uncertainty | Model size limit, DPO helps partially |
| Good casual greeting at checkpoint-200 | Less overfit | Use lower checkpoint |

**Checkpoint ranking:** 200 > 1600 > 800

**Best checkpoint so far:** `checkpoint-200`
**Recommended max_steps for Qwen retrain:** 200-400 (not 1600) — model degrades significantly after ~200 steps

---

---

## Llama 3.2 3B — genz adapter (`adapters/genz/`)
**Date:** 2026-04-04
**Job:** 543708
**Data mix:**
| Source | Samples | % |
|---|---|---|
| casual | 5,000 | 50% |
| friend | 1,912 | 19.1% |
| therapist | 1,645 | 16.5% |
| grief/loss | 1,443 | 14.4% |
| **Total** | **10,000** | |
**System prompt:** Full system prompt (sections 1-8, without 8B transition rules)
**Note:** Same weights saved to `adapters/CUDA_mindmate_llama32b/`. This is the "joke-mode-lock" model — stays casual even when user expresses real emotion.
**Tests:** Not formally logged — known issue is it doesn't pivot on distress signals.

---

## Llama 3.2 3B — v2 adapter (`CUDA_mindmate_llama32b_v2`)
**Date:** 2026-04-08
**Job:** 554799 (A100-40, gpu-long, ~5h)
**Data mix:**
| Source | Samples | % |
|---|---|---|
| transition | 3,000 | 30% |
| therapist | 2,500 | 25% |
| casual | 2,000 | 20% |
| grief/loss | 1,500 | 15% |
| friend | 1,000 | 10% |
| **Total** | **10,000** | |
**System prompt:** Full system prompt (186 lines, sections 1-9 including 8B transition rules)
**Tests:** Pending — job still running

---

---

## Llama 3.2 3B — v2 SFT adapter (`CUDA_mindmate_llama32b/checkpoint-200`)
**Date:** 2026-04-09
**Job:** 554799 (A100-40, gpu-long, ~5h)
**Data mix:**
| Source | Samples | % |
|---|---|---|
| transition | 3,000 | 30% |
| therapist | 2,500 | 25% |
| casual | 2,000 | 20% |
| grief/loss | 1,500 | 15% |
| friend | 1,000 | 10% |
| **Total** | **10,000** | |
**System prompt:** Simplified 13-line prompt (temperature 0.75)

```
You: hey
Mindmate: hey! what's going on?

You: im sad rn
Mindmate: aw, what happened?

You: my dog died
Mindmate: oh no… i'm so sorry. that's really hard. how long did you have him?

You: 8 years, he was my best friend
Mindmate: 8 years is a long time. he was really part of your life. what was he like?

You: he was always there for me. i feel like it was my fault
Mindmate: that guilt hits different when you lose someone you love. what makes you feel like it was your fault?
```

**Rating: 8/10**

✅ Good:
- Natural casual opener ("hey! what's going on?")
- Immediate pivot on distress ("aw, what happened?")
- Follow-up questions feel organic, not scripted
- Guilt handling: validates + gently probes ("what makes you feel like it was your fault?")
- Correct pronouns throughout
- No therapy-speak, no breathing exercises

❌ Bad:
- Occasional over-use of ellipsis (minor)

**Conclusion:** Huge improvement over genz adapter. Transition data fixed joke-mode-lock. checkpoint-200 avoids the hallucination patterns seen at ck-800 and ck-1600.

---

## Qwen2.5-3B — SFT adapter (`CUDA_mindmate_qwen25_3b/checkpoint-200`)
**Date:** 2026-04-09
**Data mix:** Same v2 mix as Llama (transition 30%, therapist 25%, casual 20%, grief 15%, friend 10%)
**System prompt:** Simplified 13-line prompt (temperature 0.75)

```
You: hey
Mindmate: Hey! How are you doing?

You: im sad rn
Mindmate: Oh no, what's going on?

You: my dog died
Mindmate: I'm so sorry to hear that. Losing a pet is really painful. How are you feeling right now?

You: i feel like it was my fault
Mindmate: It's understandable to feel that way, but it's important to remember that you did the best you could. What makes you think it was your fault?
```

**Rating: 5/10**

✅ Good:
- Does pivot on distress (doesn't stay casual)
- Correct pronouns
- No hallucinations

❌ Bad:
- "it's important to remember" — therapy-speak
- "you did the best you could" before hearing the story — too quick to reassure
- More formal/stiff than Llama ("How are you feeling right now?" vs Llama's "that's really hard. how long did you have him?")
- Uses capital letters throughout — less casual than target persona

**Conclusion:** Qwen2.5-3B handles pivots correctly but sounds more like a therapist bot than a friend. DPO should help reduce the reassurance-before-listening pattern. Llama remains the better model at this checkpoint.

---

---

## Llama 3.2 3B — genzv2 SFT (`adapters/genz/checkpoint-1600`) — **PRODUCTION MODEL**
**Date:** 2026-04-14
**Job:** N/A (re-evaluation of existing checkpoint)
**Data mix (v2):**
| Source | Samples | % |
|---|---|---|
| transition | 3,000 | 30% |
| therapist | 2,500 | 25% |
| casual | 2,000 | 20% |
| grief/loss | 1,500 | 15% |
| friend | 1,000 | 10% |
| **Total** | **10,000** | |
**System prompt:** 6-line Anchor prompt (warm friend, no lecturing)

**Rating: 9/10**

✅ Good:
- Best overall model — beats all DPO variants and all other SFT checkpoints
- Natural casual tone, clean emotional pivoting
- No fake shared history, no hallucinations
- Correct pronoun tracking
- No therapy-speak

❌ Mild:
- Occasional over-use of ellipsis (minor)

**Conclusion:** genzv2 ck1600 is the winner. DPO did not improve it. Now running in production at https://tryanchor.me as `llama(genz)v2_q4_k_m.gguf`.

---

## DPO Evaluation Summary (2026-04-14)

| Model | Train Loss | Steps | Rating vs genzv2 ck1600 |
|---|---|---|---|
| Llama DPO ck200 (job 560339) | 1.127 | 800 | ❌ Inferior |
| Qwen2.5-3B DPO ck200 (job 560338) | 1.279 | 800 | ❌ Inferior |

**Conclusion:** DPO did not improve either model. SFT ck1600 (genzv2) remains the best.

---

## Completed Tests

- [x] Qwen3-1.7B checkpoint-200
- [x] Qwen3-1.7B checkpoint-800
- [x] Qwen3-1.7B checkpoint-1600
- [x] Llama 3.2 3B v2 SFT checkpoint-200
- [x] Qwen2.5-3B SFT checkpoint-200
- [x] Llama DPO ck200 ← inferior to genzv2 ck1600
- [x] Qwen2.5-3B DPO ck200 ← inferior to genzv2 ck1600
- [x] **genzv2 SFT ck1600 ← WINNER, in production**

## Pending Tests
- [ ] Gemma 4 E2B IT (GGUF uploaded to HF, not yet evaluated in production)
