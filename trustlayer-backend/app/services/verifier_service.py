"""
verifier_service.py — Hallucination detection layer for TrustLayer AI.

This is the CORE differentiator of the system.

A second LLM pass independently evaluates whether the generated
answer is actually supported by the retrieved context.

Output is parsed into:
  - verdict:     SUPPORTED | PARTIALLY_SUPPORTED | NOT_SUPPORTED
  - explanation: one-line human-readable reason
  - confidence:  mapped float score

Prompt is taken verbatim from the PRD.
"""

import logging
import re
from typing import Tuple

from openai import AsyncOpenAI

from app.core.config import get_settings
from app.models.schemas import Verdict

logger   = logging.getLogger(__name__)
settings = get_settings()

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


# ── Prompts (from PRD — do not modify without version bump) ───────────────

VERIFIER_SYSTEM_PROMPT = "You are a verification system."

VERIFIER_USER_TEMPLATE = """Given:
Context:
{context}

Answer:
{answer}

Task:
Determine if the answer is supported by the context.

Output ONLY one of the following on the first line:
- SUPPORTED
- PARTIALLY_SUPPORTED
- NOT_SUPPORTED

Then provide a one-line reason on the second line."""


# ── Confidence mapping (from PRD) ──────────────────────────────────────────

CONFIDENCE_MAP = {
    Verdict.SUPPORTED:           0.9,
    Verdict.PARTIALLY_SUPPORTED: 0.6,
    Verdict.NOT_SUPPORTED:       0.2,
}


# ── Public API ─────────────────────────────────────────────────────────────

async def verify_answer(
    answer:  str,
    context: str,
) -> Tuple[Verdict, str, float]:
    """
    Run the hallucination verification pass.

    Args:
        answer:  The LLM-generated answer to verify.
        context: The raw context string that was provided to the LLM.

    Returns:
        A 3-tuple of (verdict, explanation, confidence_score).
    """
    # Short-circuit: if the answer is the "no information" sentinel,
    # we already know it's not supported without calling the LLM.
    if "don't have enough information" in answer.lower():
        return (
            Verdict.NOT_SUPPORTED,
            "The answer was withheld because no relevant context was found.",
            CONFIDENCE_MAP[Verdict.NOT_SUPPORTED],
        )

    user_message = VERIFIER_USER_TEMPLATE.format(
        context=context,
        answer=answer,
    )

    logger.debug("Calling verifier LLM (model=%s)", settings.llm_model)

    client   = _get_client()
    response = await client.chat.completions.create(
        model=settings.llm_model,
        temperature=0.0,         # verification must be deterministic
        max_tokens=256,          # verdict + reason is always short
        messages=[
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
    )

    raw_output = (response.choices[0].message.content or "").strip()
    logger.debug("Verifier raw output: %s", raw_output)

    verdict, explanation = _parse_verifier_output(raw_output)
    confidence           = CONFIDENCE_MAP[verdict]

    return verdict, explanation, confidence


# ── Fast mode fallback ─────────────────────────────────────────────────────

def fast_mode_verdict(answer: str) -> Tuple[Verdict, str, float]:
    """
    Return a lightweight verdict when verification is skipped (fast mode).

    Uses simple heuristics instead of a second LLM call.
    """
    if "don't have enough information" in answer.lower():
        return (
            Verdict.NOT_SUPPORTED,
            "Fast mode: answer was withheld (no context found).",
            CONFIDENCE_MAP[Verdict.NOT_SUPPORTED],
        )
    return (
        Verdict.SUPPORTED,
        "Fast mode: verification skipped — answer assumed grounded.",
        CONFIDENCE_MAP[Verdict.SUPPORTED],
    )


# ── Parsing ────────────────────────────────────────────────────────────────

def _parse_verifier_output(raw: str) -> Tuple[Verdict, str]:
    """
    Parse the raw verifier LLM output into a (verdict, explanation) tuple.

    The LLM is instructed to output the verdict on line 1 and the
    explanation on line 2. We use regex matching as a fallback in
    case the model wraps its output in markdown or adds extra lines.
    """
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]

    # Attempt structured parse first
    verdict_raw = lines[0] if lines else ""
    explanation  = lines[1] if len(lines) > 1 else "No explanation provided."

    verdict = _match_verdict(verdict_raw)

    if verdict is None:
        # Fallback: scan all lines for a verdict keyword
        for line in lines:
            verdict = _match_verdict(line)
            if verdict:
                # Use remaining lines as explanation
                explanation = " ".join(
                    l for l in lines if _match_verdict(l) is None
                ) or "Unable to parse explanation."
                break

    if verdict is None:
        # Last resort: treat as NOT_SUPPORTED to be conservative
        logger.warning(
            "Could not parse verifier verdict from: %r. Defaulting to NOT_SUPPORTED.", raw
        )
        verdict     = Verdict.NOT_SUPPORTED
        explanation = "Verification output could not be parsed — defaulting to low trust."

    # Truncate extremely long explanations
    if len(explanation) > 300:
        explanation = explanation[:297] + "…"

    return verdict, explanation


def _match_verdict(text: str) -> Verdict | None:
    """
    Match a verdict keyword in *text*, case-insensitively.

    Checks PARTIALLY_SUPPORTED before SUPPORTED to avoid
    a prefix match masking the more specific label.
    """
    upper = text.upper()
    if "PARTIALLY_SUPPORTED" in upper or "PARTIALLY SUPPORTED" in upper:
        return Verdict.PARTIALLY_SUPPORTED
    if "NOT_SUPPORTED" in upper or "NOT SUPPORTED" in upper:
        return Verdict.NOT_SUPPORTED
    if "SUPPORTED" in upper:
        return Verdict.SUPPORTED
    return None