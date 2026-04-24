"""
agent/graph/graph — LangGraph StateGraph assembly.

``build_graph()`` returns a compiled LangGraph graph that implements the
full dispatch loop described in PLAN.MD Phase 4.  During Phase 1 the graph
is a stub: only the dispatch node and its conditional edges exist; the
specialist nodes (nav_bot, battle_bot, etc.) are placeholder pass-throughs
that will be replaced in Phases 2–3.

Phase 4 wires in the real node implementations.
"""

from __future__ import annotations

from typing import Callable

from langgraph.graph import END, StateGraph

from agent.graph.router import routing_condition
from agent.graph.state import AgentState


# ---------------------------------------------------------------------------
# Placeholder nodes (replaced in Phases 2–3)
# ---------------------------------------------------------------------------


def _passthrough_node(state: AgentState) -> AgentState:
    """Generic stub node — returns state unchanged."""
    return state


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def build_graph(
    nav_bot_node: Callable | None = None,
    battle_bot_node: Callable | None = None,
    coms_bot_node: Callable | None = None,
    verification_node: Callable | None = None,
    map_stitcher_relay_node: Callable | None = None,
) -> "langgraph.graph.graph.CompiledGraph":  # type: ignore[name-defined]
    """Assemble and compile the dispatch StateGraph.

    Keyword args accept real node callables so that Phases 2–3 can inject
    them without modifying this file.  Any argument left as ``None`` falls
    back to a pass-through stub.

    Returns:
        A compiled LangGraph graph ready for ``graph.invoke(state)``.
    """
    _nav = nav_bot_node or _passthrough_node
    _battle = battle_bot_node or _passthrough_node
    _coms = coms_bot_node or _passthrough_node
    _verify = verification_node or _passthrough_node
    _relay = map_stitcher_relay_node or _passthrough_node

    builder = StateGraph(AgentState)

    # ---- Nodes ----
    builder.add_node("dispatch", _passthrough_node)
    builder.add_node("nav_bot", _nav)
    builder.add_node("battle_bot", _battle)
    builder.add_node("coms_bot", _coms)
    builder.add_node("map_stitcher_relay", _relay)
    builder.add_node("verification", _verify)

    # ---- Entry point ----
    builder.set_entry_point("dispatch")

    # ---- Conditional routing from dispatch ----
    builder.add_conditional_edges(
        "dispatch",
        routing_condition,
        {
            "nav_bot": "nav_bot",
            "battle_bot": "battle_bot",
            "coms_bot": "coms_bot",
            "map_stitcher_relay": "map_stitcher_relay",
        },
    )

    # ---- All specialist nodes flow through verification then END ----
    builder.add_edge("nav_bot", "verification")
    builder.add_edge("battle_bot", "verification")
    builder.add_edge("coms_bot", "verification")
    builder.add_edge("map_stitcher_relay", "nav_bot")
    builder.add_edge("verification", END)

    return builder.compile()
