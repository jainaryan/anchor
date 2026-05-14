"""
Judge service — runtime-agnostic LLM judge wrapper.

The judge service takes a transcript + judge criterion and returns a verdict
(PASS / FAIL / AMBIGUOUS). It's runtime-agnostic: it doesn't care whether
the transcript came from cluster inference (NF4 adapter, GGUF) or from a
mobile device. This is what enables mobile transcripts to be judged on the
cluster using the same Gemma4 model as the cluster benchmarks.

The judge model is injected via a `generate_fn` callable. run_benchmarks.py
loads Gemma4 once and passes its generate function in. judge_mobile_results.py
does the same. Tests can pass in a stub generate function.

Prompt format is preserved from v3 (run_benchmarks.py:call_judge) so verdicts
remain comparable across the v3→v4 transition.

New in v4:
  - scope-aware transcript formatting (any_turn / final_turn / all_turns)
  - ambiguity retry with rephrased question
  - structured events via EvalLogger
  - judge_scenario() returns a structured result per criterion
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, Any

# ──────────────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────────────

# generate_fn signature: (prompt: str, max_new_tokens: int, temperature: float) -> str
GenerateFn = Callable[[str, int, float], str]


@dataclass
class JudgeResult:
    """Result of judging a single criterion."""
    criterion_id: str
    question: str
    pass_if: str               # "YES" or "NO"
    weight: float
    scope: str                 # any_turn / final_turn / all_turns
    negative: bool
    verdict: str               # "PASS" | "FAIL" | "AMBIGUOUS"
    raw_response: str
    judge_answer: str          # parsed YES/NO/UNCLEAR
    ms: float
    retry_count: int = 0
    second_response: Optional[str] = None  # for ambiguity retry

    @property
    def passed(self) -> bool:
        return self.verdict == "PASS"

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ──────────────────────────────────────────────────────────────────────────────

def format_transcript_for_judge(transcript: list[dict], scope: str = "any_turn") -> str:
    """
    Render a transcript as the conversation block in the judge prompt.

    scope:
      - "any_turn":   include all turns (default; judge looks for any occurrence)
      - "final_turn": include only the last assistant turn (with the user prompt that elicited it)
      - "all_turns":  include all turns, with explicit turn numbers visible to judge

    Matches v3 format (run_benchmarks.py:call_judge):
      User: <text>
      AI: <text>
    """
    if scope == "final_turn":
        # Find the last assistant message
        last_asst_idx = None
        for i in range(len(transcript) - 1, -1, -1):
            if transcript[i].get("role") == "assistant":
                last_asst_idx = i
                break
        if last_asst_idx is None:
            return "(no assistant turns)"

        # Include the user prompt(s) right before it
        start = last_asst_idx
        while start > 0 and transcript[start - 1].get("role") == "user":
            start -= 1
        relevant = transcript[start: last_asst_idx + 1]
        return _format_lines(relevant, with_numbers=False)

    if scope == "all_turns":
        return _format_lines(transcript, with_numbers=True)

    # any_turn (default)
    return _format_lines(transcript, with_numbers=False)


def _format_lines(messages: list[dict], with_numbers: bool = False) -> str:
    lines: list[str] = []
    turn_idx = 0
    for m in messages:
        role = m.get("role", "?")
        if role == "system":
            continue  # system prompt is rendered separately
        content = m.get("content", "")
        if role == "user":
            turn_idx += 1
            label = f"User T{turn_idx}" if with_numbers else "User"
        elif role == "assistant":
            label = f"AI T{turn_idx}" if with_numbers else "AI"
        else:
            label = role
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def build_judge_prompt(system_prompt: str, transcript: list[dict],
                       criterion: dict) -> str:
    """
    Build the judge prompt for a single criterion.
    Backwards compatible with v3 call_judge() prompt structure.
    """
    scope = criterion.get("scope", "any_turn")
    conv = format_transcript_for_judge(transcript, scope)
    return (
        "You are a strict evaluator assessing an AI mental health companion.\n\n"
        "## System prompt given to the AI\n"
        f"{system_prompt}\n\n"
        "## Conversation\n"
        f"{conv}\n\n"
        "## Evaluation Question\n"
        f"{criterion['question']}\n\n"
        "Reply with YES or NO on the first line only. Be strict."
    )


def build_rephrased_judge_prompt(system_prompt: str, transcript: list[dict],
                                  criterion: dict) -> str:
    """
    Rephrased prompt used on ambiguity retry.

    The judge originally returned something unclear. Ask the same question
    with explicit scaffolding: instruct the model to think step by step in
    a private scratchpad, then emit a final YES/NO.
    """
    scope = criterion.get("scope", "any_turn")
    conv = format_transcript_for_judge(transcript, scope)
    return (
        "You are evaluating an AI mental health companion. Read the conversation "
        "below carefully, then answer the question with strictly YES or NO.\n\n"
        "## System prompt given to the AI\n"
        f"{system_prompt}\n\n"
        "## Conversation\n"
        f"{conv}\n\n"
        "## Question\n"
        f"{criterion['question']}\n\n"
        "Think step by step in 1-2 sentences, then on a new line write your "
        "final answer as exactly 'YES' or 'NO' — nothing else on that line."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Response parsing
# ──────────────────────────────────────────────────────────────────────────────

_YES_RE = re.compile(r"\bYES\b", re.IGNORECASE)
_NO_RE = re.compile(r"\bNO\b", re.IGNORECASE)


def parse_judge_response(raw: str) -> str:
    """
    Parse YES/NO/UNCLEAR from a judge response.

    Strategy:
      1. Check the first non-empty line for YES or NO (preferred — matches v3).
      2. If first line has BOTH or NEITHER, check the LAST non-empty line
         (for scratchpad-then-answer responses from the rephrased prompt).
      3. Fall back to UNCLEAR.
    """
    if not raw:
        return "UNCLEAR"

    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    if not lines:
        return "UNCLEAR"

    def line_verdict(line: str) -> str:
        upper = line.upper()
        # Strip common prefixes
        upper = upper.replace("FINAL ANSWER:", "").replace("ANSWER:", "")
        upper = upper.replace("VERDICT:", "").strip()
        has_yes = bool(_YES_RE.search(upper))
        has_no = bool(_NO_RE.search(upper))
        if has_yes and not has_no:
            return "YES"
        if has_no and not has_yes:
            return "NO"
        return "UNCLEAR"

    first = line_verdict(lines[0])
    if first != "UNCLEAR":
        return first

    # Try last line (for chain-of-thought responses)
    if len(lines) > 1:
        last = line_verdict(lines[-1])
        if last != "UNCLEAR":
            return last

    return "UNCLEAR"


# ──────────────────────────────────────────────────────────────────────────────
# JudgeService
# ──────────────────────────────────────────────────────────────────────────────

class JudgeService:
    """
    Wraps an LLM judge for use across runtimes.

    Holds: generate_fn, optional logger, optional tokenizer (for token counts).
    The actual model lives wherever generate_fn was created — could be a
    Gemma4 transformers model loaded in this process, or a remote endpoint.
    """

    def __init__(
        self,
        generate_fn: GenerateFn,
        logger: Optional[Any] = None,        # EvalLogger
        tokenizer: Optional[Any] = None,     # for prompt_tokens count
        judge_temperature: float = 0.1,
        judge_max_tokens: int = 64,
        retry_max_tokens: int = 128,
        ambiguity_retry: bool = True,
        log_ambiguous_for_review: bool = True,
    ):
        self.generate_fn = generate_fn
        self.logger = logger
        self.tokenizer = tokenizer
        self.judge_temperature = judge_temperature
        self.judge_max_tokens = judge_max_tokens
        self.retry_max_tokens = retry_max_tokens
        self.ambiguity_retry = ambiguity_retry
        self.log_ambiguous_for_review = log_ambiguous_for_review

        self.stats = {
            "criteria_judged": 0,
            "passed": 0,
            "failed": 0,
            "ambiguous_first_try": 0,
            "ambiguous_after_retry": 0,
        }

    # ── primary API ───────────────────────────────────────────────────────────

    def judge_criterion(
        self,
        *,
        scenario_id: str,
        system_prompt: str,
        transcript: list[dict],
        criterion: dict,
    ) -> JudgeResult:
        prompt = build_judge_prompt(system_prompt, transcript, criterion)
        prompt_tokens = self._count_tokens(prompt)

        if self.logger is not None:
            self.logger.judge_call(
                scenario_id=scenario_id,
                criterion_id=criterion["id"],
                scope=criterion.get("scope", "any_turn"),
                prompt_tokens=prompt_tokens,
                prompt=prompt,
            )

        t0 = time.monotonic()
        raw = self.generate_fn(prompt, self.judge_max_tokens, self.judge_temperature)
        ms = (time.monotonic() - t0) * 1000

        answer = parse_judge_response(raw)
        retry_count = 0
        second_raw = None

        # Ambiguity retry — only if first answer is UNCLEAR
        if answer == "UNCLEAR" and self.ambiguity_retry:
            self.stats["ambiguous_first_try"] += 1
            if self.logger is not None:
                self.logger.judge_response(
                    scenario_id=scenario_id,
                    criterion_id=criterion["id"],
                    verdict="AMBIGUOUS",
                    raw=raw,
                    ms=ms,
                    ambiguous=True,
                    retry=0,
                )

            retry_prompt = build_rephrased_judge_prompt(system_prompt, transcript, criterion)
            t1 = time.monotonic()
            second_raw = self.generate_fn(retry_prompt, self.retry_max_tokens,
                                           self.judge_temperature)
            ms += (time.monotonic() - t1) * 1000
            retry_count = 1
            answer = parse_judge_response(second_raw)

            if answer == "UNCLEAR":
                self.stats["ambiguous_after_retry"] += 1
                if self.log_ambiguous_for_review and self.logger is not None:
                    self.logger.warn(
                        f"judge ambiguous after retry: {scenario_id}/{criterion['id']}",
                        scenario_id=scenario_id,
                        criterion_id=criterion["id"],
                        first_raw=raw,
                        second_raw=second_raw,
                    )

        # Final verdict: did the parsed answer match pass_if?
        pass_if = criterion.get("pass_if", "YES")
        if answer == "UNCLEAR":
            verdict = "AMBIGUOUS"
        elif answer == pass_if:
            verdict = "PASS"
        else:
            verdict = "FAIL"

        self.stats["criteria_judged"] += 1
        if verdict == "PASS":
            self.stats["passed"] += 1
        elif verdict == "FAIL":
            self.stats["failed"] += 1

        if self.logger is not None:
            # Final verdict log
            self.logger.judge_response(
                scenario_id=scenario_id,
                criterion_id=criterion["id"],
                verdict=verdict,
                raw=second_raw if second_raw else raw,
                ms=ms,
                ambiguous=(verdict == "AMBIGUOUS"),
                retry=retry_count,
            )

        return JudgeResult(
            criterion_id=criterion["id"],
            question=criterion["question"],
            pass_if=pass_if,
            weight=criterion.get("weight", 1),
            scope=criterion.get("scope", "any_turn"),
            negative=criterion.get("negative", False),
            verdict=verdict,
            raw_response=raw,
            judge_answer=answer,
            ms=ms,
            retry_count=retry_count,
            second_response=second_raw,
        )

    def judge_scenario(
        self,
        *,
        scenario_id: str,
        system_prompt: str,
        transcript: list[dict],
        criteria: list[dict],
    ) -> list[JudgeResult]:
        return [
            self.judge_criterion(
                scenario_id=scenario_id,
                system_prompt=system_prompt,
                transcript=transcript,
                criterion=c,
            )
            for c in criteria
        ]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _count_tokens(self, text: str) -> Optional[int]:
        if self.tokenizer is None:
            return None
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            return None


# ──────────────────────────────────────────────────────────────────────────────
# Scoring helper — given a list of JudgeResults, compute the weighted score
# ──────────────────────────────────────────────────────────────────────────────

def score_scenario(results: list[JudgeResult]) -> dict:
    """Aggregate per-criterion verdicts into a scenario-level score."""
    total = sum(r.weight for r in results)
    passed = sum(r.weight for r in results if r.verdict == "PASS")
    ambiguous = sum(1 for r in results if r.verdict == "AMBIGUOUS")
    return {
        "weighted_pass": passed,
        "weighted_total": total,
        "weighted_pct": passed / total if total else 0,
        "passed_criteria": sum(1 for r in results if r.verdict == "PASS"),
        "failed_criteria": sum(1 for r in results if r.verdict == "FAIL"),
        "ambiguous_criteria": ambiguous,
        "n_criteria": len(results),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test — uses a stub generate function
# ──────────────────────────────────────────────────────────────────────────────

def _stub_generate(prompt: str, max_tokens: int, temperature: float) -> str:
    """
    Stub for unit testing. Returns:
      - "YES" if the prompt contains the word "Zoya"
      - "NO" if it doesn't
      - "maybe i'm not sure" for an ambiguity test (when prompt contains "AMBIG_TEST")
    """
    if "AMBIG_TEST" in prompt:
        return "Hmm, it's hard to say. Could go either way."
    if "Zoya" in prompt:
        return "YES\nThe AI used the name Zoya in its response."
    return "NO\nThe AI did not reference the user's friend by name."


if __name__ == "__main__":
    # Smoke test: judge a simple transcript with the stub
    from benchmarks.logging import EvalLogger
    import tempfile
    from pathlib import Path

    tmp = Path(tempfile.mkdtemp(prefix="judge_smoke_"))
    print(f"Smoke test run_dir: {tmp}\n")

    with EvalLogger(tmp, level="DEBUG", color=False) as log:
        log.run_start(model="smoke", scenarios=2, backend="stub")

        # Scenario 1 — should PASS (response mentions Zoya)
        log.scenario_start(scenario_id="cm_01",
                           scenario_type="single",
                           category="CONTEXT_MEMORY", weight=1)
        judge = JudgeService(generate_fn=_stub_generate, logger=log)
        transcript = [
            {"role": "user", "content": "my friend has been there for me"},
            {"role": "assistant", "content": "Zoya sounds amazing"},
        ]
        criterion = {
            "id": "uses_name",
            "question": "Does the AI mention Zoya?",
            "pass_if": "YES",
            "weight": 1,
            "scope": "any_turn",
            "negative": False,
        }
        result = judge.judge_criterion(
            scenario_id="cm_01",
            system_prompt="You are Anchor. The user's friend is named Zoya.",
            transcript=transcript,
            criterion=criterion,
        )
        assert result.verdict == "PASS", f"expected PASS, got {result.verdict}"
        score = score_scenario([result])
        log.scenario_end(scenario_id="cm_01", turns=1,
                         weighted_pass=score["weighted_pass"],
                         weighted_total=score["weighted_total"])

        # Scenario 2 — ambiguous response, retry should still be unclear
        log.scenario_start(scenario_id="ambig",
                           scenario_type="single",
                           category="CONTEXT_MEMORY", weight=1)
        crit_ambig = {
            "id": "test_ambig",
            "question": "AMBIG_TEST",  # triggers ambiguous stub
            "pass_if": "YES",
            "weight": 1,
            "scope": "any_turn",
            "negative": False,
        }
        r2 = judge.judge_criterion(
            scenario_id="ambig",
            system_prompt="...",
            transcript=transcript,
            criterion=crit_ambig,
        )
        assert r2.verdict == "AMBIGUOUS", f"expected AMBIGUOUS, got {r2.verdict}"
        assert r2.retry_count == 1
        s2 = score_scenario([r2])
        log.scenario_end(scenario_id="ambig", turns=1,
                         weighted_pass=s2["weighted_pass"],
                         weighted_total=s2["weighted_total"])

        log.run_end(passed_scenarios=1, failed_scenarios=0,
                    weighted_pass=1, weighted_total=2)

    print(f"\nJudge stats: {judge.stats}")
    print(f"Files written to {tmp}:")
    for p in sorted(tmp.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(tmp)}  ({p.stat().st_size}B)")
    print("\n✓ Smoke test passed")
