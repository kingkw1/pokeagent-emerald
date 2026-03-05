# agent/brain/strategic_planner.py
"""
Strategic Planner — Phase 4.2

RAG-driven strategic navigation planner that replaces (and eventually
supersedes) the hardcoded ``MILESTONE_PROGRESSION`` list.

Pipeline::

    1. Query ``WalkthroughDB`` with current game state
    2. LLM interprets retrieved walkthrough context → outputs a location name
    3. ``LocationResolver`` maps that name → ``LOCATION_GRAPH`` key + coords
    4. Produce a ``Directive``-compatible dict for the execution layer

The LLM **never** outputs raw coordinates — only location names.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from agent.brain.location_resolver import (
    get_display_name,
    list_known_locations,
    resolve_location,
    resolve_location_key,
)
from agent.brain.walkthrough_db import WalkthroughDB

logger = logging.getLogger(__name__)

# ============================================================================
# Prompt template
# ============================================================================

_SYSTEM_PROMPT = """\
You are an expert Pokémon Emerald game guide.  Given walkthrough context and the
player's current situation, determine the NEXT location they should travel to and
what key actions they should take there.

RULES:
1. Output a single JSON object with exactly these keys:
   - "target_location": The name of the next location (e.g., "Route 101", "Rustboro City").
     Use the official in-game location name.  NEVER output coordinates.
   - "description": A concise (1-2 sentence) description of what the player should do.
   - "priority_actions": A list of 1-3 short action strings the player should perform
     at or en route to the target (e.g., "Heal at Pokemon Center", "Battle trainers").
2. Only reference locations and events that appear in the provided walkthrough context.
3. If the walkthrough context is empty or irrelevant, suggest the most logical next step
   based on general Pokémon Emerald knowledge (e.g., "head to the next route").
4. Return ONLY the JSON object, no other text.\
"""

_QUERY_TEMPLATE = """\
WALKTHROUGH CONTEXT:
{context}

PLAYER SITUATION:
- Current location: {current_location}
- Badges: {badge_count}
- Party: {pokemon_summary}
- Last completed milestone: {last_milestone}

What should the player do next?\
"""


# ============================================================================
# Fallback plan when LLM is unavailable or response is unparseable
# ============================================================================

_FALLBACK_PLAN = {
    "target_location": None,
    "description": "Continue exploring the current area.",
    "priority_actions": ["Explore nearby routes", "Talk to NPCs"],
}


class StrategicPlanner:
    """RAG-driven strategic planner that queries walkthrough text to determine
    the agent's next navigation target.

    Args:
        vlm: An initialised ``utils.vlm.VLM`` instance.  When ``None``,
             returns a deterministic fallback (useful for tests / offline).
        walkthrough_db: A ``WalkthroughDB`` instance.  When ``None``, the
             planner operates without RAG context (pure LLM reasoning).
        verbose: Print detailed RAG retrieval info (demos / debugging).
    """

    def __init__(
        self,
        vlm=None,
        walkthrough_db: Optional[WalkthroughDB] = None,
        verbose: bool = False,
    ):
        self.vlm = vlm
        self.walkthrough_db = walkthrough_db
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_next_directive(
        self,
        current_location: str,
        badge_count: int = 0,
        pokemon_summary: str = "Unknown",
        last_milestone: str = "None",
        state_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Determine the next navigation directive using RAG + LLM.

        Returns a dict with:
        - ``target_location`` — ``LOCATION_GRAPH`` key (str) or ``None``
        - ``target_display_name`` — human-readable name
        - ``description`` — what to do
        - ``priority_actions`` — list of action strings
        - ``goal_coords`` — ``(x, y, 'LOCATION')`` tuple if resolvable
        - ``source`` — ``"walkthrough_rag"`` (for shadow-mode comparison)
        """
        # 1. Retrieve walkthrough context (enriched with progress info)
        context_text = self._retrieve_context(
            current_location,
            last_milestone=last_milestone,
            badge_count=badge_count,
        )

        # 1b. Short-circuit if RAG has no good matches.  When the KB
        #     has nothing relevant (all chunks below the distance
        #     threshold), asking the LLM with empty context just produces
        #     hallucinated targets.  Return target_location=None so the
        #     caller falls through to the milestone-based fallback.
        if not context_text:
            print("🧠 [STRATEGIC PLANNER] No relevant walkthrough context — deferring to milestone.")
            return {
                "target_location": None,
                "target_display_name": None,
                "description": "No relevant walkthrough context available.",
                "priority_actions": [],
                "goal_coords": None,
                "source": "walkthrough_rag",
            }

        # 2. Build prompt
        display_loc = get_display_name(current_location) if current_location else "Unknown"
        prompt = _QUERY_TEMPLATE.format(
            context=context_text or "(No walkthrough context available.)",
            current_location=display_loc,
            badge_count=badge_count,
            pokemon_summary=pokemon_summary,
            last_milestone=last_milestone,
        )

        # DEBUG: Full RAG context and prompt (demoted to logger.debug)
        logger.debug(f"[SP] Query: location={current_location} ({display_loc}), milestone={last_milestone}, context={len(context_text)} chars")
        if context_text:
            logger.debug(f"[SP] Context preview: {context_text[:300]}")

        if self.verbose:
            logger.debug(f"[SP VERBOSE] Current: {display_loc}, context: {len(context_text)} chars")

        # 3. LLM call
        full_prompt = f"{_SYSTEM_PROMPT}\n\n{prompt}"
        raw_response = self._call_llm(full_prompt)
        logger.debug(f"[SP] LLM raw response: {raw_response}")

        # 4. Parse response
        plan = self._parse_response(raw_response)
        logger.debug(f"[SP] Parsed plan: {plan}")

        # 5. Resolve location → LOCATION_GRAPH key + coords
        resolved = self._resolve_target(plan.get("target_location"))
        target_key = resolved['key'] if resolved else None
        print(f"🧠 [RAG] LLM target: '{plan.get('target_location')}' → {target_key}")
        logger.debug(f"[SP] Resolved: {resolved}")

        result: Dict[str, Any] = {
            "target_location": resolved["key"] if resolved else None,
            "target_display_name": (
                resolved.get("display_name", plan.get("target_location", "Unknown"))
                if resolved
                else plan.get("target_location", "Unknown")
            ),
            "description": plan.get("description", "Continue exploring."),
            "priority_actions": plan.get("priority_actions", []),
            "source": "walkthrough_rag",
        }

        # Add goal_coords if we resolved to a graph entry with portals
        if resolved:
            coords = self._extract_entry_coords(resolved, current_location)
            if coords:
                result["goal_coords"] = coords

        if self.verbose:
            logger.debug(f"[SP VERBOSE] Target: {result.get('target_location')} ({result.get('target_display_name')}), coords={result.get('goal_coords')}")

        return result

    def shadow_compare(
        self,
        milestone_target: Optional[str],
        rag_target: Optional[str],
    ) -> Dict[str, Any]:
        """Compare milestone-based target with RAG-based target for Phase 4.3a.

        Returns a comparison dict for logging/analysis:
        ``{"milestone": ..., "rag": ..., "agree": bool}``
        """
        agree = False
        if milestone_target and rag_target:
            agree = milestone_target.upper() == rag_target.upper()
        elif milestone_target is None and rag_target is None:
            agree = True

        return {
            "milestone_target": milestone_target,
            "rag_target": rag_target,
            "agree": agree,
        }

    # ------------------------------------------------------------------
    # RAG retrieval
    # ------------------------------------------------------------------

    # Maximum acceptable embedding distance — chunks further away are
    # semantically irrelevant and would confuse the LLM.
    # Tuned empirically: with the enriched query, good matches are typically
    # <0.35 and supplemental navigation chunks land around 0.45-0.47.
    # At 0.50 we keep useful context while filtering truly irrelevant late-game
    # chunks (Route 106/118/115 at 0.49+).
    _DISTANCE_THRESHOLD: float = 0.50

    def _retrieve_context(
        self,
        current_location: str,
        last_milestone: str = "None",
        badge_count: int = 0,
    ) -> str:
        """Query WalkthroughDB for relevant chunks.

        The query is enriched with progress info (``last_milestone``,
        ``badge_count``) so that the embedding search favours chunks
        relevant to the player's current progression, not just location
        name similarity.

        Chunks with embedding distance > ``_DISTANCE_THRESHOLD`` are
        discarded to avoid feeding irrelevant context to the LLM.
        """
        if not self.walkthrough_db:
            return ""

        display_name = get_display_name(current_location) if current_location else "the current area"
        query = (
            f"I am in {display_name} with {badge_count} badges. "
            f"Last completed milestone: {last_milestone}. "
            f"What should I do next? Where should I go?"
        )
        logger.debug(f"[RAG] Query: '{query}'")

        # Over-fetch so we can discard low-quality results
        results = self.walkthrough_db.query(query, n_results=6)
        logger.debug(f"[RAG] Got {len(results) if results else 0} raw chunks")

        if not results:
            return ""

        # Filter by distance threshold — keep only semantically close chunks
        filtered: list = []
        for i, entry in enumerate(results, 1):
            meta = entry.get("metadata", {})
            loc_label = meta.get("location", "Unknown")
            dist = entry.get("distance")
            is_close = dist is not None and dist <= self._DISTANCE_THRESHOLD
            status = "KEPT" if is_close else "DROP"
            logger.debug(f"[RAG] Chunk {i}: {status} loc='{loc_label}' dist={dist:.3f}")
            if is_close:
                filtered.append(entry)

        logger.debug(f"[RAG] Kept {len(filtered)}/{len(results)} chunks (threshold={self._DISTANCE_THRESHOLD})")

        # Take at most 3 of the best results
        filtered = filtered[:3]

        if not filtered:
            return ""

        context_parts = []
        for entry in filtered:
            meta = entry.get("metadata", {})
            loc_label = meta.get("location", "Unknown")
            context_parts.append(
                f"[Section: {loc_label} (Part {meta.get('part', '?')})]\n"
                f"{entry['text']}"
            )

        return "\n\n---\n\n".join(context_parts)

    # ------------------------------------------------------------------
    # LLM integration
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """Send prompt to VLM backend; return raw text."""
        if self.vlm is not None:
            logger.info(f"[StrategicPlanner] Sending prompt to LLM ({len(prompt)} chars)")
            return self.vlm.get_text_query(prompt, module_name="STRATEGIC-PLANNER")

        # Mock/offline fallback
        logger.info("[StrategicPlanner] No VLM — returning mock response.")
        return json.dumps({
            "target_location": "Route 101",
            "description": "Head north through Route 101 to reach Oldale Town.",
            "priority_actions": ["Battle wild Pokémon for XP"],
        })

    # ------------------------------------------------------------------
    # Response parsing (reuses pattern from RecoveryPlanner)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(response_text: str) -> Dict[str, Any]:
        """Extract and validate the JSON plan from the LLM response."""
        if not response_text:
            return dict(_FALLBACK_PLAN)

        # 1. Markdown-fenced JSON
        json_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL
        )
        if not json_match:
            # 2. Bare JSON object
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)

        if not json_match:
            logger.warning(
                f"[StrategicPlanner] No JSON in response: {response_text[:200]}"
            )
            return dict(_FALLBACK_PLAN)

        try:
            raw_json = (
                json_match.group(1) if json_match.lastindex else json_match.group(0)
            )
            plan = json.loads(raw_json)
        except json.JSONDecodeError:
            logger.warning(
                f"[StrategicPlanner] JSON decode failed: {response_text[:200]}"
            )
            return dict(_FALLBACK_PLAN)

        # Validate / fill defaults
        if "target_location" not in plan or not plan["target_location"]:
            plan["target_location"] = None
        if "description" not in plan or not plan["description"]:
            plan["description"] = _FALLBACK_PLAN["description"]
        if "priority_actions" not in plan or not isinstance(
            plan["priority_actions"], list
        ):
            plan["priority_actions"] = _FALLBACK_PLAN["priority_actions"]

        return plan

    # ------------------------------------------------------------------
    # Location resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_target(location_name: Optional[str]) -> Optional[Dict[str, Any]]:
        """Resolve the LLM's location name to a LOCATION_GRAPH entry."""
        if not location_name:
            return None
        return resolve_location(location_name)

    @staticmethod
    def _extract_entry_coords(
        resolved: Dict[str, Any],
        from_location: Optional[str] = None,
    ) -> Optional[tuple]:
        """Extract a ``(x, y, 'LOCATION')`` goal-coords tuple from a resolved entry.

        If ``from_location`` is provided, looks for the portal entry_coords
        from that source.  Otherwise returns a reasonable default.
        """
        key = resolved.get("key")
        if not key:
            return None

        portals = resolved.get("portals", {})

        # If we know where we're coming from, use that portal's entry coords
        if from_location:
            from_key = from_location.upper().replace(" ", "_")
            portal = portals.get(from_key)
            if portal:
                ec = portal.get("entry_coords")
                if ec:
                    return (ec[0], ec[1], key)

        # Fallback: pick the first portal's entry_coords
        for _dest, portal_info in portals.items():
            ec = portal_info.get("entry_coords")
            if ec:
                return (ec[0], ec[1], key)

        return None
