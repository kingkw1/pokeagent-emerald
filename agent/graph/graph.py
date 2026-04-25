"""
agent/graph/graph — LangGraph StateGraph assembly.

``build_graph(obj_manager, vlm)`` returns a compiled LangGraph graph that
implements the full dispatch loop described in PLAN.MD Phase 4.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from agent.graph.router import routing_condition
from agent.graph.state import AgentState
from agent.graph.nodes.nav_bot import nav_bot_node
from agent.graph.nodes.battle_bot import battle_bot_node
from agent.graph.nodes.coms_bot import coms_bot_node
from agent.graph.nodes.verification import make_verification_node
from agent.graph.nodes.map_stitcher_relay import make_map_stitcher_relay_node


def build_graph(obj_manager, vlm) -> "langgraph.graph.graph.CompiledGraph":  # type: ignore[name-defined]
    """Assemble and compile the dispatch StateGraph.

    Args:
        obj_manager: ``ObjectiveManager`` instance (used by verification node).
        vlm: ``VLM`` instance (used by map_stitcher_relay node).

    Returns:
        A compiled LangGraph graph ready for ``graph.invoke(state)``.
    """
    builder = StateGraph(AgentState)

    # ---- Nodes ----
    builder.add_node("dispatch", lambda s: s)
    builder.add_node("nav_bot", nav_bot_node)
    builder.add_node("battle_bot", battle_bot_node)
    builder.add_node("coms_bot", coms_bot_node)
    builder.add_node("verification", make_verification_node(obj_manager))
    builder.add_node("map_stitcher_relay", make_map_stitcher_relay_node(vlm))

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
