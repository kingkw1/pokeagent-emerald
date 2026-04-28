# agent/brain — Memory & Planning System

Provides the agent with persistent memory, RAG-powered goal generation, and walkthrough context
for the HTN Executive Supervisor. Consumed by the LangGraph `StateGraph` at `executive_supervisor_node`.

> **HTN Migration status:** Phases 0–4 complete (200/200 automated tests). The LangGraph graph and HTN
> Supervisor are fully wired. Navigation is still driven by the legacy `ObjectiveManager` +
> `MILESTONE_PROGRESSION` FSM; HTN Phases 5–7 complete the cutover. See [HTN_MIGRATION_PLAN.md](HTN_MIGRATION_PLAN.md).

## What It Does Today

On each node execution:
1. **Logs episodic events** to ChromaDB (`game_history`) — dialogue turns, battle outcomes, location transitions.
2. **Supplies RAG context** to the HTN Supervisor: `strategy_guide` (136 Bulbapedia walkthrough chunks) is queried at bootstrap and goal-expansion time to generate `GoalNode` objects.
3. **Fallback navigation** via `ObjectiveManager` + `MILESTONE_PROGRESSION` — still active while `--use-htn` is off.
4. **Resolves NPC targets** dynamically from `gObjectEvents` memory (no hardcoded coords).
5. **Triggers healing** when party HP drops below 50% (routes to nearest PokeCenter via `LOCATION_GRAPH` BFS).
6. **Logs milestones** to ChromaDB on completion for cross-session progress awareness.

## Real-World Architectural Parallels

This project is a Pokémon game agent. The engineering problems it solves are
not Pokémon-specific.

| Agent Mechanic | Enterprise Parallel |
|----------------|---------------------|
| **Neuro-symbolic router** — `routing_condition()` dispatches to deterministic nodes (NavBot, BattleBot) for known states and only calls the LLM for genuinely ambiguous ones | Inference cost management in production agentic systems — rule-based pre-screening reduces LLM API spend by 70–90% vs. naive full-LLM routing |
| **Deterministic PokéCenter routing** — `find_nearest_pokemon_center()` BFS over the location graph returns the nearest PC graph key; `get_entrance_coords()` resolves the overworld entrance tile. VLM overhead-map parsing fires only as fallback for cities not yet in the graph | Neuro-symbolic hybrid spatial reasoning: fast deterministic lookup for known states, expensive perception model reserved for novel ones — the core cost/accuracy tradeoff in production vision-guided systems | (In the future, we'll implement Dynamic PokeCenter identification -- Gemini VLM receives a stitched overhead map image and returns pixel coordinates of the nearest healing location | Multimodal visual search and coordinate routing for robotics, warehouse automation, and vision-guided navigation systems)
| **HTN goal stack with RAM verification** — the Executive Supervisor pushes `GoalNode` tasks derived from RAG; the `handoff_detector` gates Supervisor calls; stack POPs are conditioned on ChromaDB completion evidence and RAM state confirmation (pull) | Verifiable workflow automation for high-stakes, hallucination-prone environments: financial transaction pipelines, compliance workflows, medical record processing |
| **Karpathy meta-loops + RewardVector** — per-step reward signal with configurable weights, logged to JSONL for offline regression analysis | Automated pipeline evaluation and CI/CD optimization — any production system where you need to detect agentic regression across deployments |
| **TelemetryLogger** — tracks VLM API calls, token consumption, and step latency per step and per objective | MLOps instrumentation for inference cost attribution, SLA monitoring, and capacity planning in multi-tenant LLM deployments |

## Architecture

```
LangGraph StateGraph
    │
    ├─ dispatch_node              ← routing_condition() — zero LLM
    │
    ├─ [nav_bot | battle_bot | coms_bot | map_stitcher_relay]   ← System 1 (fast)
    │
    ├─ handoff_detector_node     ← detects node-type transitions, sets supervisor_pending
    │
    └─ executive_supervisor_node (on supervisor_pending=True)    ← System 2 (slow)
            ├─ _bootstrap_stack()     ← first step: RAG query → 3-level HTN (strategic→tactical→immediate)
            ├─ _expand_strategic_goal()  ← decompose strategic goal into sub-tasks
            └─ stack ops: POP / CONTINUE / PUSH / REPLACE → written to AgentState
```

**Brain's role in the graph:** `executive_supervisor_node` pulls from two ChromaDB collections:
- `strategy_guide` — 136 walkthrough chunks → goal generation context
- `game_history` — episodic log → completion evidence for POP operations

**Current nav authority:** `ObjectiveManager.get_next_action_directive()` — milestone FSM drives `goal_coords`/`goal_location`. After HTN Phase 7, `Stack[0].directive` overwrites nav fields directly.

See [HTN_MIGRATION_PLAN.md](HTN_MIGRATION_PLAN.md) for the phase-by-phase migration roadmap.

## Modules

| File | Role |
|------|------|
| `memory.py` | `EpisodicMemory` — ChromaDB wrapper (log, retrieve, spatial query) |
| `walkthrough_db.py` | `WalkthroughDB` — pre-embedded Bulbapedia `strategy_guide` ChromaDB collection (136 chunks) |
| `goal_stack.py` | `GoalNode` dataclass + `GoalStack` list — HTN task hierarchy |
| `location_resolver.py` | Prose location name → `LOCATION_GRAPH` key (fuzzy match) |
| `npc_registry.py` | `NpcRegistry` — learned `graphics_id` → NPC role mapping |
| `planner.py` | `RecoveryPlanner` — legacy RAG + LLM recovery plan *(superseded by HTN Supervisor)* |
| `strategic_planner.py` | `StrategicPlanner` — legacy walkthrough RAG → navigation target *(superseded by HTN Supervisor)* |
| `HTN_MIGRATION_PLAN.md` | Active phase-by-phase migration roadmap (Phases 0–7) |

HTN executor nodes live in `agent/graph/nodes/` (not under `agent/brain/`):
`dispatch.py`, `nav_bot.py`, `battle_bot.py`, `coms_bot.py`, `map_stitcher_relay.py`, `handoff_detector.py`, `executive_supervisor.py`, `verification.py`

## Status

| Capability | Status |
|------------|--------|
| Episodic memory (log + retrieve) | ✅ Stable |
| RAG-powered recovery planning (legacy `RecoveryPlanner`) | ✅ Stable |
| Spatial memory (tile coordinates) | ✅ Stable |
| Walkthrough RAG — `strategy_guide` collection (136 chunks) | ✅ Stable |
| NPC dynamic targeting (Tiers 1 + 2) | ✅ Stable |
| NPC obstacle injection in A\* | ✅ Stable |
| Generic healing subsystem | ✅ Stable |
| Milestone completion logging | ✅ Stable |
| LangGraph `StateGraph` (dispatch → specialists → handoff → verification) | ✅ Phases 0–4 complete |
| HTN Executive Supervisor (bootstrap + stack ops) | ✅ Wired, `--use-htn` off by default |
| HTN handoff detection (zero-LLM gate) | ✅ Stable |
| HTN Phase 5 — battle outcome logging + episodic query split | 🔲 In progress |
| HTN Phase 6 — boot timestamp + stale record filtering | 🔲 Not started |
| HTN Phase 7 — shadow mode → flip `--use-htn` → retire `MILESTONE_PROGRESSION` | 🔲 Not started |
| LOCATION_GRAPH topology chunk generation (Option A) | 🔲 Not started |
| Telemetry (VLM call + token + latency tracking) | 🔲 Not started |

Historical phase documentation (Phases 1–5 design decisions, implementation
details, tabled items): [`docs/development/BRAIN_PHASES_1_5_REFERENCE.md`](../../docs/development/BRAIN_PHASES_1_5_REFERENCE.md)

## Quick Commands

```bash
# Inspect strategy_guide ChromaDB (all 136 chunks)
python scripts/dump_walkthrough_db.py --stats

# Semantic search over walkthrough
python scripts/dump_walkthrough_db.py --query "Route 102 wild grass"

# HTN unit tests (Phases 0–4, 200 tests)
python -m pytest tests/ -v -k "htn"

# Full test suite
python -m pytest -v
```
