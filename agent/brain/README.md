# agent/brain — Memory & Planning System

Provides the agent with persistent memory, RAG-powered recovery planning,
and walkthrough-driven navigation. Runs inside every `Agent.step()` call.

## What It Does Today

On each frame:
1. **Logs dialogue** to ChromaDB with location, battle state, and tile coordinates.
2. **Detects battle transitions** and fires a RAG → LLM recovery plan on battle start.
3. **Drives navigation** via RAG-primary walkthrough planner with milestone fallback.
4. **Resolves NPC targets** dynamically from `gObjectEvents` memory (no hardcoded coords).
5. **Triggers healing** when party HP drops below 50% (routes to nearest PokeCenter).
6. **Logs milestones** to ChromaDB on completion for cross-session progress awareness.

## Architecture

```
Agent.step()
    │
    ├─ perception_step()           ← VLM frame analysis
    ├─ objective_manager.update_brain()
    │       ├─ EpisodicMemory.log_event()       ← ChromaDB write
    │       ├─ RecoveryPlanner (on battle/block) ← RAG + Gemini Flash
    │       └─ StrategicPlanner (navigation)     ← Walkthrough RAG
    ├─ pathfinding / directive_nav              ← A* + NPC obstacle injection
    └─ action_step()               ← BattleBot / OpenerBot / VLM fallback
```

**Next:** The `Agent.step()` dispatch loop is being migrated to a LangGraph
`StateGraph`. See [PLAN.MD](PLAN.MD) for the roadmap.

## Modules

| File | Role |
|------|------|
| `memory.py` | `EpisodicMemory` — ChromaDB wrapper (log, retrieve, spatial query) |
| `planner.py` | `RecoveryPlanner` — RAG retrieval + LLM recovery plan generation |
| `strategic_planner.py` | `StrategicPlanner` — walkthrough RAG → navigation target |
| `walkthrough_db.py` | `WalkthroughDB` — pre-embedded Bulbapedia ChromaDB collection |
| `location_resolver.py` | Prose location name → `LOCATION_GRAPH` key (fuzzy match) |
| `npc_registry.py` | `NpcRegistry` — learned `graphics_id` → NPC role mapping |
| `PLAN.MD` | Active roadmap (LangGraph migration) |

## Status

| Capability | Status |
|------------|--------|
| Episodic memory (log + retrieve) | ✅ Stable |
| RAG-powered recovery planning | ✅ Stable |
| Spatial memory (tile coordinates) | ✅ Stable |
| Walkthrough RAG (full-game coverage) | ✅ Stable |
| NPC dynamic targeting (Tiers 1 + 2) | ✅ Stable |
| NPC obstacle injection in A\* | ✅ Stable |
| Generic healing subsystem | ✅ Stable |
| Milestone completion logging | ✅ Stable |
| LangGraph migration | 🔲 In roadmap |

Historical phase documentation (Phases 1–5 design decisions, implementation
details, tabled items): [`docs/development/BRAIN_PHASES_1_5_REFERENCE.md`](../../docs/development/BRAIN_PHASES_1_5_REFERENCE.md)

## Quick Commands

```bash
# Inspect live ChromaDB contents
python -m agent.brain.demos.inspect_brain

# Run unit tests
python -m pytest tests/test_objective_manager_blocker.py \
    tests/test_spatial_memory.py tests/test_npc_detection.py \
    tests/test_npc_obstacles.py -v

# Full test suite
python -m pytest -v
```
