"""
agent/graph/transition_evaluator — Dialogue milestone completion verifier.

On the step immediately after a dialogue session ends (dialogue → navigation
context transition), ``TransitionEvaluator.evaluate()`` is called with the
full in-session transcript.  It returns one of:

  ``"YES"``     — transcript confirms the milestone keywords were spoken;
                  verification_node may advance milestone_index.
  ``"NO"``      — keywords not found; milestone stays open.
  ``"PARTIAL"`` — some keywords matched but the session seems incomplete;
                  ComsBot should re-engage the NPC on the next opportunity.

Architecture notes
------------------
* **Fast path (no VLM):** keyword scan over the transcript text.  Used when
  ``vlm`` is *None* (unit tests, offline mode).
* **Full path (VLM available):** a text-only LLM call that receives the
  assembled transcript and the expected keywords.  The LLM call fires at most
  once per dialogue session (gated by the caller in ``Agent.step()``).
* **Cost:** one LLM call per NPC interaction at a story checkpoint — negligible
  compared to the per-step VLM calls already in the pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_VERDICT_PROMPT_TEMPLATE = """\
You are verifying whether a Pokémon Emerald story milestone was completed during \
a dialogue session.

MILESTONE ID: {milestone_id}
MILESTONE DESCRIPTION: {milestone_description}
EXPECTED KEYWORDS (at least one must appear): {keywords_str}

DIALOGUE TRANSCRIPT:
{transcript_text}

Did the dialogue session complete this milestone?
- Reply YES if the transcript shows the milestone was completed \
(a keyword-related topic was discussed).
- Reply NO if the transcript is empty, off-topic, or the keywords are absent.
- Reply PARTIAL if some keywords appeared but the session seems cut short \
(agent should re-engage the NPC).

Respond with exactly one word: YES, NO, or PARTIAL."""


class TransitionEvaluator:
    """Verifies dialogue milestone completion at the dialogue→navigation boundary.

    Args:
        vlm: Optional ``VLM`` instance.  When provided, makes a text-only LLM
             call for the verdict.  When *None*, falls back to a keyword scan.
    """

    def __init__(self, vlm: Optional[Any] = None) -> None:
        self.vlm = vlm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        milestone_id: str,
        milestone_description: str,
        keywords: list[str],
        transcript: list[dict],
    ) -> str:
        """Determine whether a dialogue milestone was completed.

        Args:
            milestone_id:          e.g. ``"DAD_FIRST_MEETING"``
            milestone_description: Human-readable milestone label.
            keywords:              List of keyword strings expected in the
                                   dialogue (from ``dialogue_keywords`` in
                                   ``MILESTONE_PROGRESSION``).
            transcript:            Ordered list of ``{speaker, text, step}``
                                   dicts accumulated by ``coms_bot_node``.

        Returns:
            ``"YES"`` | ``"NO"`` | ``"PARTIAL"``
        """
        if not transcript:
            logger.debug("[TransitionEvaluator] Empty transcript → NO")
            return "NO"

        full_text = " ".join(entry.get("text", "") for entry in transcript)

        # Fast path — keyword scan (also used as VLM fallback on error)
        def _keyword_verdict() -> str:
            if not keywords:
                # No keywords specified — treat non-empty session as success
                return "YES" if full_text.strip() else "NO"
            matched = [kw for kw in keywords if kw.lower() in full_text.lower()]
            if len(matched) == 0:
                return "NO"
            threshold = max(1, len(keywords) // 2)
            if len(matched) >= threshold:
                return "YES"
            return "PARTIAL"

        if self.vlm is None:
            verdict = _keyword_verdict()
            logger.debug(
                "[TransitionEvaluator] No VLM — keyword scan verdict: %s "
                "(matched %d/%d keywords)",
                verdict,
                sum(1 for kw in keywords if kw.lower() in full_text.lower()),
                len(keywords),
            )
            return verdict

        # Full LLM path
        transcript_text = "\n".join(
            f'{entry.get("speaker", "NPC")}: {entry.get("text", "")}'
            for entry in transcript
        )
        keywords_str = ", ".join(f'"{kw}"' for kw in keywords) if keywords else "(none)"

        prompt = _VERDICT_PROMPT_TEMPLATE.format(
            milestone_id=milestone_id,
            milestone_description=milestone_description,
            keywords_str=keywords_str,
            transcript_text=transcript_text,
        )

        try:
            raw = self.vlm.backend.get_text_query(prompt, "TransitionEvaluator")
            raw_upper = raw.strip().upper()
            if "YES" in raw_upper:
                verdict = "YES"
            elif "PARTIAL" in raw_upper:
                verdict = "PARTIAL"
            else:
                verdict = "NO"
            logger.info(
                "[TransitionEvaluator] LLM verdict for '%s': %s (raw: %r)",
                milestone_id,
                verdict,
                raw[:80],
            )
            return verdict
        except Exception as exc:
            logger.warning(
                "[TransitionEvaluator] LLM call failed (%s) — falling back to keyword scan.",
                exc,
            )
            return _keyword_verdict()
