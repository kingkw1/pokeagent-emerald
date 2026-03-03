# agent/brain — Memory & Planning System

The **brain** subsystem gives the Pokemon Emerald speedrunning agent the ability
to *remember*, *reason*, and *recover* when it encounters obstacles during
gameplay.

## Current Contribution to the Agent

The brain runs inside every `Agent.step()` call (see `agent/__init__.py`,
between perception and navigation). On each frame it:

1. **Logs new dialogue** to a persistent ChromaDB vector database
   (`EpisodicMemory.log_event`).
2. **Detects battle transitions** by comparing the current `in_battle` flag
   against the previous frame's value.
3. **On battle start:** writes the event to memory, marks the ObjectiveManager
   as BLOCKED, performs a **RAG query** (semantic search over all stored
   memories), and sends the retrieved context + situation to the LLM for a
   recovery plan. The battle bot then takes over for actual combat — the brain
   does *not* short-circuit.
4. **On battle end:** logs the outcome, clears the BLOCKED/RECOVERY state, and
   lets normal navigation resume.
5. **Outside of battle:** runs keyword-based blocker detection on NPC dialogue
   (e.g. "wait", "stop") and, if triggered, fires the same RAG → LLM pipeline
   with a short-circuit to press A until the blocker clears.

### What this proves today

- The vector database **grows** over time (every dialogue, every battle).
- The database is **queried** via semantic similarity (visible in the
  `============` verbose block during gameplay).
- Retrieved memories are **injected into the LLM prompt**, so the recovery plan
  is informed by what the agent has actually experienced — not just a static
  cheat sheet.

## Architecture

The brain capabilities were originally split across `GoalManager` and
`RecoveryPlanner`. As of the **Phase 1: Brain Consolidation**, the GoalManager
has been fully merged into `ObjectiveManager` (`agent/objective_manager.py`),
which now serves as the single executive router. The consolidation:

- **Ported** blocker detection, recovery task stack, and `signal_blocker()` into
  ObjectiveManager.
- **Ported** the `update_brain()` integration point (dialogue logging, battle
  transitions, keyword scanning, RAG recovery) into ObjectiveManager.
- **Deleted** the standalone `goal_manager.py` file.

The remaining brain modules (`EpisodicMemory`, `RecoveryPlanner`) are injected
into ObjectiveManager via `update_brain(episodic_memory=..., recovery_planner=...)`.

| Capability | Status | Target |
|------------|--------|--------|
| Episodic memory (log & retrieve) | **Done** | Stable |
| RAG-powered recovery planning | **Done** | Stable |
| Battle transition detection | **Done** | Stable |
| NPC dialogue blocker detection | **Done** | Expand keyword list / use LLM classification |
| Spatial memory (x, y, map_id metadata) | Not started | Log coordinates on every event |
| Proactive planning (not just reactive) | Not started | Brain proposes next objective, not just recovery |
| Multi-step sub-goaling | Not started | Dependency chains (e.g. Get Cut → Beat Gym → Cut Tree) |
| Learning from failures | Not started | Log failed actions, avoid repeating them |

## Module Files

| File | Role |
|------|------|
| `memory.py` | `EpisodicMemory` — ChromaDB wrapper (log, retrieve, clear) |
| `planner.py` | `RecoveryPlanner` — RAG retrieval + LLM prompt construction + response parsing |
| `PLAN.MD` | Development roadmap with phased milestones |

> **Note:** `goal_manager.py` no longer exists. Its functionality lives in
> `agent/objective_manager.py` (see `is_blocked`, `signal_blocker()`,
> `add_recovery_task()`, `complete_recovery_task()`, `update_brain()`).

## Demo & Inspection Scripts

All runnable from the project root:

```bash
# Semantic memory retrieval demo
python -m agent.brain.demos.demo_rag_memory

# End-to-end RAG flow (mock or --live for Gemini)
python -m agent.brain.demos.demo_full_flow
python -m agent.brain.demos.demo_full_flow --live

# Inspect the live database contents
python -m agent.brain.demos.inspect_brain
```

## Tests

```bash
python -m pytest tests/test_objective_manager_blocker.py -v
```

---

## Known Gaps & Next Steps

### 1. Phase 3 (Spatial Awareness) needs a concrete spec before coding

The plan says "log (x, y, map_id) as metadata" but several decisions remain
unresolved:

- **Where do coordinates come from?** `state_data['player']['x']` /
  `state_data['player']['y']` / `state_data['player']['map_id']` — these exist
  today and can be passed into `log_event()` metadata immediately.
- **What does spatial retrieval look like?** ChromaDB supports `where` clause
  filtering on metadata (e.g. `where={"map_id": 17}`), which is separate from
  vector similarity. A spatial query would combine both: vector similarity on
  the event text *and* a map filter so results stay geographically relevant.
- **Suggested first step:** Before touching the pathfinder, add `x`, `y`,
  `map_id` to the metadata dict in every `log_event()` call in
  `agent/__init__.py` and verify with `inspect_brain.py`. That's a 3-line
  change with zero risk to navigation.

### 2. Phase 4 (Sub-Goaling) needs a design doc before implementation

"Get Cut → Beat Gym → Cut Tree" is a good North Star but doesn't define:

- What the dependency graph structure looks like (DAG? linear chain?).
- How this interacts with the ObjectiveManager's milestone system, which already
  drives navigation objectives. Building a parallel system risks divergence.

### 3. Memory lifecycle: the database grows indefinitely

Every dialogue line and battle event is logged and never evicted. For a
speedrun (a few hundred steps), this is fine. For extended runs or repeated
restarts against the same save state, stale or duplicate entries accumulate and
begin polluting retrieval results.

**Options to consider (in order of effort):**
- **Short-term (zero effort):** `clear_memory()` at the start of each run, then
  re-seed. Already works today with `inspect_brain.py` showing the state.
- **Medium-term:** Add a `max_events` cap — when the collection exceeds N
  entries, drop the oldest by timestamp metadata.
- **Long-term:** Area-based summarization — when the player leaves a map, merge
  all dialogue/battle events for that map into a single summary entry, reducing
  retrieval noise.
