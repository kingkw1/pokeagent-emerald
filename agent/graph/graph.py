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
from agent.graph.nodes.coms_bot import make_coms_bot_node
from agent.graph.nodes.verification import make_verification_node
from agent.graph.nodes.map_stitcher_relay import make_map_stitcher_relay_node
from agent.graph.nodes.handoff_detector import handoff_detector_node


def build_graph(obj_manager, vlm, episodic_memory=None) -> "langgraph.graph.graph.CompiledGraph":  # type: ignore[name-defined]
    """Assemble and compile the dispatch StateGraph.

    Args:
        obj_manager:      ``ObjectiveManager`` instance (used by verification node).
        vlm:              ``VLM`` instance (used by map_stitcher_relay and coms_bot nodes).
        episodic_memory:  Optional ``EpisodicMemory`` instance.  When provided,
                          ComsBot will log each dialogue turn to ChromaDB.

    Returns:
        A compiled LangGraph graph ready for ``graph.invoke(state)``.
    """
    builder = StateGraph(AgentState)

    # ---- Nodes ----
    builder.add_node("dispatch", lambda s: s)
    builder.add_node("nav_bot", nav_bot_node)
    builder.add_node("battle_bot", battle_bot_node)
    builder.add_node("coms_bot", make_coms_bot_node(vlm, episodic_memory))
    builder.add_node("verification", make_verification_node(obj_manager))
    builder.add_node("map_stitcher_relay", make_map_stitcher_relay_node(vlm))
    builder.add_node("handoff_detector", handoff_detector_node)

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

    # ---- Specialist nodes → handoff_detector → verification ----
    # map_stitcher_relay keeps its relay to nav_bot; nav_bot then flows into
    # handoff_detector so the relay chain is: relay → nav_bot → handoff_detector.
    builder.add_edge("nav_bot", "handoff_detector")
    builder.add_edge("battle_bot", "handoff_detector")
    builder.add_edge("coms_bot", "handoff_detector")
    builder.add_edge("map_stitcher_relay", "nav_bot")

    # Phase 1: handoff_detector always passes through to verification.
    # Phase 2 will replace this with a conditional edge that routes to
    # executive_supervisor when supervisor_pending=True.
    builder.add_edge("handoff_detector", "verification")

    builder.add_edge("verification", END)

    return builder.compile()
