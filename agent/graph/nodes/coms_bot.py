"""
agent/graph/nodes/coms_bot — ComsBot (dialogue / opening-sequence) specialist node.

Thin LangGraph wrapper over ``agent/opener_bot.py``.  No new FSM logic lives
here — this module only provides the AgentState ↔ OpenerBot interface required
by the dispatch graph.

Routing contract:
  - Receives: AgentState where ``state_data`` indicates active dialogue or an
    opening-sequence trigger for OpenerBot.
  - Returns: updated AgentState with ``last_action = "DIALOGUE"`` and
    ``last_buttons`` set to the button sequence that advances the conversation.

ComsBot uses a two-tier delegation pattern:

1. **OpenerBot** handles scripted FSM sequences (title screen, naming, rival
   encounter, obtaining starter, etc.).  When ``OpenerBot.should_handle()``
   returns True the returned value may be:
     - ``list[str]``         → use directly as buttons
     - ``NavigationGoal``    → set ``goal_coords`` for NavBot, return ``[]``
     - ``ForceDialogueGoal`` → press A to dismiss misclassified dialogue
     - ``None``              → fall through to normal A press

2. **Normal NPC dialogue** → wait for script-idle then press A.

Script-idle guard
-----------------
``wait_for_script_idle()`` is called before every A press for standard
overworld NPC dialogue.  It is skipped when:
  - Location is in ``_SKIP_SCRIPT_IDLE_LOCATIONS`` (intro uses GBA callbacks,
    not ``sGlobalScriptContext``, so the endpoint is meaningless there).
  - No game-server connection is available (exception caught silently).

RAM fallback
------------
``state_data["game"]["game_state"] == "dialog"`` is read directly from GBA RAM
and remains reliable even when VLM perception has timed out.  The router
already uses this field to route into ComsBot, so no additional check is
needed inside the node itself.

Phase 5.3 — Dialogue Capture
-----------------------------
When ``make_coms_bot_node(vlm, episodic_memory)`` is used instead of the
plain ``coms_bot_node``, each step fires a VLM call to extract the on-screen
speaker and dialogue text (only when ``script_mode == 0``, i.e. the text
animation has fully rendered).  Extracted turns are:

  1. Appended to the module-level ``_SESSION_TRANSCRIPT`` list.
  2. Logged to ChromaDB via ``episodic_memory.log_event()`` with
     ``type="dialogue_transcript"`` metadata.

``Agent.step()`` reads the transcript via ``get_session_transcript()`` on the
dialogue→navigation transition and passes it to ``TransitionEvaluator``.
``clear_session_transcript()`` is called on session start and after evaluation.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

from agent.graph.state import AgentState
from agent.opener_bot import (
    ForceDialogueGoal,
    NavigationGoal,
    get_opener_bot,
    wait_for_script_idle,
)

logger = logging.getLogger(__name__)

# Locations where wait_for_script_idle must be skipped — the GBA intro uses
# native C callbacks instead of sGlobalScriptContext, so the endpoint is
# not meaningful there.
_SKIP_SCRIPT_IDLE_LOCATIONS = frozenset({"TITLE_SEQUENCE", "MOVING_VAN"})

# ---------------------------------------------------------------------------
# Phase 5.3 — Module-level dialogue session state
# ---------------------------------------------------------------------------
# This is intentionally module-level (not inside the factory closure) so that
# Agent.step() can read the transcript via get_session_transcript() without
# needing a direct reference to the node callable.

_SESSION_TRANSCRIPT: List[Dict[str, Any]] = []


def get_session_transcript() -> List[Dict[str, Any]]:
    """Return a shallow copy of the current dialogue session transcript.

    Called by ``Agent.step()`` on the dialogue→navigation boundary to pass
    to ``TransitionEvaluator.evaluate()``.
    """
    return list(_SESSION_TRANSCRIPT)


def clear_session_transcript() -> None:
    """Clear the in-memory transcript (call at session start and after evaluation)."""
    global _SESSION_TRANSCRIPT  # noqa: PLW0603
    _SESSION_TRANSCRIPT = []


# VLM prompt for per-turn text extraction
_CAPTURE_PROMPT = (
    "Extract the speaker name and exact dialogue text from this GBA screenshot. "
    'Respond as JSON only: {"speaker": "<name or NPC>", "text": "<dialogue>", "has_more": <true|false>}'
)


def _try_extract_dialogue_turn(vlm: Any, frame: Any) -> Optional[Dict[str, Any]]:
    """Use VLM to extract a single dialogue turn from the current frame.

    Returns a dict ``{speaker, text, has_more}`` or *None* on failure.
    """
    if vlm is None or frame is None:
        return None
    try:
        raw = vlm.get_query(frame, _CAPTURE_PROMPT, "ComsBot_Capture")
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return json.loads(cleaned)
    except Exception as exc:
        logger.debug("[COMSBOT] Dialogue capture VLM parse failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_coms_bot_node(
    vlm: Optional[Any] = None,
    episodic_memory: Optional[Any] = None,
) -> Callable[[AgentState], AgentState]:
    """Return a ComsBot node with optional dialogue capture wired in.

    Args:
        vlm:             ``VLM`` instance for per-turn text extraction.
                         When *None*, capture is disabled (same as plain
                         ``coms_bot_node``).
        episodic_memory: ``EpisodicMemory`` instance.  When provided,
                         each captured turn is logged to ChromaDB with
                         ``type="dialogue_transcript"`` metadata.

    Returns:
        A LangGraph-compatible node callable.
    """

    def _coms_bot_node(state: AgentState) -> AgentState:
        opener = get_opener_bot()
        state_data: Dict[str, Any] = state.get("state_data") or {}
        perception: Dict[str, Any] = state.get("perception") or {}
        visual_data: Dict[str, Any] = perception.get("visual_data") or {}

        buttons: List[str] = ["A"]
        new_goal_coords: Optional[tuple] = state.get("goal_coords")
        new_npc_coords: Optional[tuple] = state.get("npc_coords")
        new_should_interact: bool = state.get("should_interact", False)

        if opener.should_handle(state_data, visual_data):
            result = opener.get_action(state_data, visual_data)
            logger.debug("[COMSBOT] OpenerBot result type: %s", type(result).__name__)

            if isinstance(result, ForceDialogueGoal):
                logger.debug("[COMSBOT] ForceDialogueGoal (%s) — pressing A.", result.reason)
                buttons = ["A"]
            elif isinstance(result, NavigationGoal):
                logger.debug(
                    "[COMSBOT] NavigationGoal → (%s, %s) @ %s",
                    result.x,
                    result.y,
                    result.map_location,
                )
                new_goal_coords = (result.x, result.y)
                if result.should_interact is not None:
                    new_should_interact = result.should_interact
                buttons = []
            elif isinstance(result, list):
                buttons = result
            else:
                buttons = ["A"]
        else:
            # Normal dialogue — wait for script idle then press A.
            location: str = state_data.get("player", {}).get("location", "")
            game: dict = state_data.get("game", {})
            ram_in_dialog: bool = game.get("in_dialog", False) or (
                game.get("game_state", "") in ("dialog", "dialogue")
            )
            skip_idle = location in _SKIP_SCRIPT_IDLE_LOCATIONS or not ram_in_dialog
            if not skip_idle:
                try:
                    wait_for_script_idle()
                except Exception as exc:
                    logger.debug("[COMSBOT] wait_for_script_idle skipped: %s", exc)
            buttons = ["A"]

        # ----------------------------------------------------------------
        # Phase 5.3 — Dialogue capture (only when script is idle and VLM provided)
        # ----------------------------------------------------------------
        if vlm is not None:
            game: dict = state_data.get("game", {})
            script_mode = game.get("script_mode", -1)
            # Only capture when text animation is complete (mode == 0) or when
            # mode is unknown (not 1/2 which are active animation states).
            capture_ok = script_mode not in (1, 2)
            # Also skip capture during OpenerBot sequences (intro, naming, etc.)
            location_str: str = state_data.get("player", {}).get("location", "")
            if capture_ok and location_str not in _SKIP_SCRIPT_IDLE_LOCATIONS:
                frame: Any = state.get("frame")
                turn = _try_extract_dialogue_turn(vlm, frame)
                if turn and (turn.get("text") or "").strip():
                    entry: Dict[str, Any] = {
                        "speaker": turn.get("speaker", "NPC"),
                        "text": turn.get("text", ""),
                        "step": state.get("step_count", 0),
                    }
                    _SESSION_TRANSCRIPT.append(entry)
                    logger.debug(
                        "[COMSBOT] Captured: %s: %s",
                        entry["speaker"],
                        entry["text"][:60],
                    )

                    # Log to ChromaDB if memory is available
                    if episodic_memory is not None:
                        player = state_data.get("player", {})
                        pos = player.get("position") or {}
                        milestone_idx = state.get("milestone_index", 0)
                        try:
                            from agent.objective_manager import MILESTONE_PROGRESSION
                            ms_id = (
                                MILESTONE_PROGRESSION[milestone_idx]["milestone"]
                                if milestone_idx < len(MILESTONE_PROGRESSION)
                                else ""
                            )
                        except Exception:
                            ms_id = ""
                        meta = {
                            "type": "dialogue_transcript",
                            "speaker": entry["speaker"],
                            "step": entry["step"],
                            "milestone": ms_id,
                            "map_id": game.get("map_id", 0),
                            "player_x": int(pos.get("x", 0)),
                            "player_y": int(pos.get("y", 0)),
                        }
                        try:
                            episodic_memory.log_event(
                                f'{entry["speaker"]}: {entry["text"]}',
                                metadata=meta,
                                state_data=state_data,
                            )
                        except Exception as mem_exc:
                            logger.debug("[COMSBOT] ChromaDB log failed: %s", mem_exc)

        logger.debug("[COMSBOT] step=%s  buttons=%s", state.get("step_count"), buttons)

        return {
            **state,
            "goal_coords": new_goal_coords,
            "npc_coords": new_npc_coords,
            "should_interact": new_should_interact,
            "last_action": "DIALOGUE",
            "last_buttons": buttons,
        }

    return _coms_bot_node


# ---------------------------------------------------------------------------
# Default node (no VLM / no memory) — backward-compatible
# ---------------------------------------------------------------------------
coms_bot_node: Callable[[AgentState], AgentState] = make_coms_bot_node()

