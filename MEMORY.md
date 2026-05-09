# Anchor — Project Memory (ARCHIVED)

> **This file is from session 2026-04-10 and is kept for historical reference only.**
> For current project state, see **`MINDMATE_DETAILED_INTERNAL_NOTES.md`** (updated 2026-05-09, session 4).
> For qualitative model testing notes, see **`INFERENCE_LOGS.md`**.

---

## Quick Context (2026-04-10 — STALE)
- What was done:
  - Added Qwen2.5-3B finetune pipeline (scripts, SLURM, inference)
  - Evaluated all checkpoints across Qwen3-1.7B, Qwen2.5-3B, Llama v2 — checkpoint-200 is best on all models
  - Simplified system prompt from 150 lines → 13 lines → 11 lines (current)
  - Raised inference temperature 0.4 → 0.75
  - Fixed DPO pipeline through 10+ errors (TRL compat, CUDA, Jinja2, Arrow storage)
  - DPO training COMPLETE: Llama ck200 (loss=1.127) and Qwen2.5-3B ck200 (loss=1.279)
  - All GGUF exports complete: llama_sft_ck200, llama_dpo_ck200, qwen25_dpo_ck200, llama_sft_ck1600
  - llama_sft_ck1600 uploaded to HF as llama(genz)v2_q4_k_m.gguf
  - Queued DPO for: Llama ck1600 (569555), Qwen2.5-3B ck1600 (569556), genz (569447)
  - Fixed CUDNN_STATUS_NOT_INITIALIZED on H100 — use attn_implementation="eager" everywhere
  - Updated inference scripts: eager attn, checkpoint-200 default, DPO stacked-adapter scripts

**(As of 2026-05-09):** DPO was abandoned (all 3 runs flat or worse than SFT). Best model is now
`adapters/genz/checkpoint-1200` (genzv2_ck1200, 71% v1 / 44% v3). Root cause of base-beats-SFT identified
and fixed. genzv5 training pending conv-memory pipeline completion. See MINDMATE_DETAILED_INTERNAL_NOTES.md.
