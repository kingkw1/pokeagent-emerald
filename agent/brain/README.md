# agent/brain — Memory & Planning System

The **brain** subsystem gives the Pokemon Emerald speedrunning agent the ability
to *remember*, *reason*, and *recover* when it encounters obstacles during
gameplay.

## Current Contribution to the Agent

The brain runs inside every `Agent.step()` call (see `agent/__init__.py`,
between perception and navigation). On each frame it:

1. **Logs new dialogue** to a persistent ChromaDB vector database
   (`EpisodicMemory.log_event`) with enriched metadata (location, battle
   state, spatial coordinates).
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
6. **Logs milestone completions** to episodic memory when goals are marked
   done (e.g., `PETALBURG_GYM_DAD_DIALOGUE`), with spatial metadata for
   cross-session retrieval.

### What this proves today

- The vector database **grows** over time (every dialogue, every battle,
  every milestone completion).
- Dialogue events carry **structured metadata** (location, in_battle flag,
  spatial coordinates) enabling filtered queries.
- Milestone completions are **persistently recorded** in ChromaDB, not just
  a transient in-memory dict — enabling cross-session progress awareness.
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
Additionally, `ObjectiveManager.__init__()` accepts an `episodic_memory` kwarg
for milestone logging directly (without going through `update_brain`).

| Capability | Status | Target |
|------------|--------|--------|
| Episodic memory (log & retrieve) | **Done** | Stable |
| RAG-powered recovery planning | **Done** | Stable |
| Battle transition detection | **Done** | Stable |
| NPC dialogue blocker detection | **Done** | Expand keyword list / use LLM classification |
| Spatial memory (x, y, location metadata) | **Done** | Stable (Phase 3) |
| Milestone logging to episodic memory | **Done** | Stable (Phase 4.5a) |
| Enriched dialogue metadata | **Done** | Stable (Phase 4.5b) |
| File-based run logging (TeeWriter) | **Done** | Stable (Phase 4.5d) |
| Goal hydration from DB on startup | Not started | Load completed milestones from ChromaDB (Phase 4.5c) |
| Proactive planning (not just reactive) | **In Progress** | RAG-primary navigation active (Phase 4.3b) |
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

### 1. Goal Hydration from DB on Startup (Phase 4.5c)

Milestone completions are now logged to ChromaDB, but `completed_goals` is
still initialized as an empty dict on each process start. The next step is to
query the DB for `{"type": "milestone"}` events at startup and pre-populate
`completed_goals`. This would let the agent resume from any save state and
know what it's already accomplished — even if the save state predates the
milestone system.

This is the key enabler for **reducing hardcoded objective manager logic**:
once milestones persist across sessions, special-case handlers can be replaced
with generic "check if milestone X is done" guards that work regardless of
which save state was loaded.

### 2. Phase 4.3c (RAG-Only Navigation)

The RAG planner drives navigation today (Phase 4.3b), with milestones as a
silent fallback. Once behavioural evaluation confirms the RAG planner can
guide the agent through the Littleroot → Rustboro corridor without regression,
the milestone list moves to `tests/fixtures/` for regression testing.

Success criteria:
1. End-to-end run from `06_road` split reaches Rustboro City (RAG-only navigation).
2. No regression in route completion time vs. milestone-only runs.
3. `MILESTONE_PROGRESSION` demoted from primary to last-resort fallback.

### 3. Opening Sequence vs. Post-Opening

Everything before leaving Petalburg Gym (Norman dialogue, Wally tutorial,
scripted cutscenes) is the "opening sequence" — full of game-specific triggers
that require special handling. The `05_petalburg` and `06_road` save state
splits let us bypass the opening and focus development on the post-opening
corridor (Route 104 South → Petalburg Woods → Route 104 North → Rustboro City)
where the agent should handle everything dynamically via RAG + episodic memory.

### 4. Memory Lifecycle

Every dialogue line, battle event, and milestone is logged and never evicted.
For a speedrun (a few hundred steps), this is fine. For extended runs or repeated
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

### 5. Sub-Goaling (Phase X)

"Get Cut → Beat Gym → Cut Tree" is a good North Star but doesn't define:
- What the dependency graph structure looks like (DAG? linear chain?).
- How this interacts with the ObjectiveManager's milestone system.
- Building a parallel system risks divergence.
