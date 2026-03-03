# 🧠 Memory & Planning System: The "Staff Engineer" Roadmap (v2)

**Objective:** Build a resilient agent capable of encountering an obstacle,
logging it, querying its history via RAG, and replanning using an LLM.

**Status:** Phases 1 & 2 are **complete** and integrated into the live agent.
Phase 3 (spatial awareness) is next.

### Design Rationale (LLM-First Approach)

1.  **Interface Consistency:** The LLM call is wired on Day 1 via
    `get_recovery_plan(context, blocker)`. Day 2 only changes *where the
    context comes from* (hardcoded string → vector database)—no interface
    rework required.
2.  **Systems over Scripts:** Calling an LLM with a "Context Window"
    demonstrates a system architecture (RAG pattern), even before the database
    is connected. A hardcoded dictionary looks like a script.
3.  **Risk Mitigation:** If Day 2 runs long and ChromaDB isn't perfect, the
    demo still shows an AI analyzing a situation and generating a plan—a
    passing grade. A hardcoded dictionary without the DB is a failing grade.

---

## 🛠️ ARCHITECTURE: The Integration Point

Before writing code, we define exactly where this fits into the existing
`Agent.step()` loop in `agent/__init__.py`.

### Current Flow (agent/__init__.py)

```
1. Perception   →  perception_step(frame, state_data, vlm)
2. Stuck Detect →  position_history / oscillation check
3. Planning     →  planning_step(memory, plan, ..., state_data, vlm)
4. Memory       →  string concatenation of recent observations
5. Action       →  action_step(memory, plan, perception, ..., vlm)
```

### New Flow (Post-Consolidation)

```
1.  Perception    →  perception_step(frame, state_data, vlm)
1.5 Brain Update  →  objective_manager.update_brain(...)          ← CONSOLIDATED
2.  Stuck Detect  →  position_history / oscillation check
3.  Planning      →  planning_step(...) OR recovery_planner(...)  ← MODIFIED
4.  Memory        →  episodic_memory.log_event(...)               ← UPGRADED (Phase 2)
5.  Action        →  action_step(memory, plan, perception, ..., vlm)
```

> **Note:** The original GoalManager was merged into `ObjectiveManager`
> (`agent/objective_manager.py`) during the Phase 1 Brain Consolidation.
> `update_brain()` handles dialogue logging, battle transitions, keyword
> scanning, and RAG recovery — all in one call.

### The Input Contract

`ObjectiveManager.update_brain(perception_output, ...)` consumes the VLM
output dict. The fields live **inside `perception_output['visual_data']`**,
not at the top level. The correct access paths are:

| Field                   | Path                                                          | Type          |
| ----------------------- | ------------------------------------------------------------- | ------------- |
| Dialogue text           | `perception_output['visual_data']['on_screen_text']['dialogue']` | `str \| None`  |
| Speaker                 | `perception_output['visual_data']['on_screen_text']['speaker']`  | `str \| None`  |
| Screen context          | `perception_output['visual_data']['screen_context']`             | `str`          |
| Text box visible        | `perception_output['visual_data']['visual_elements']['text_box_visible']` | `bool` |

> **Note:** This warning applied to the original code. The rewrite correctly
> uses the nested paths documented above.

---

## 📅 PHASE 1: The "Reasoning Engine" (MVP — Day 1) ✅ COMPLETE

**Goal:** Implement the logic core. The agent "thinks" about obstacles using an
LLM, with a static "Cheat Sheet" as the context source.

### 1.1 The `GoalManager` Class → Merged into `ObjectiveManager`

> **Status:** GoalManager was implemented, verified, then **consolidated** into
> `ObjectiveManager` (`agent/objective_manager.py`) during Phase 1 Brain
> Consolidation. The standalone `goal_manager.py` file has been deleted.

-   **Original design:** A finite state machine tracking `current_objective`.
    States: `IN_PROGRESS`, `BLOCKED`, `COMPLETED`.
-   **Input parsing:** Extracts dialogue from
    `perception_output['visual_data']['on_screen_text']['dialogue']`.
    Regex/keyword match: `["wait", "stop", "don't go", "dangerous"]`.
    If match found → transition to `BLOCKED`.
-   **Now lives in:** `agent/objective_manager.py`
    -   `is_blocked` property, `signal_blocker()`, `add_recovery_task()`,
        `complete_recovery_task()`, `update_brain()`
-   **Verification:**
    -   **Test:** `tests/test_objective_manager_blocker.py` (11 tests, all passing)
    -   **Success:** Feed mock perception:
        ```python
        {"visual_data": {"on_screen_text": {"dialogue": "Wait! Don't go out!"},
                         "screen_context": "dialogue"}}
        ```
        Assert state becomes `BLOCKED`.

### 1.2 The Context-Aware Planner (LLM Integration)

-   **Methodology:** A `RecoveryPlanner` that takes the blocker description and
    a context string, then asks the LLM for immediate recovery steps.
    No hardcoded dictionary.
-   **Implementation:**
    -   File: `agent/brain/planner.py` (new file)
    -   Context (hardcoded for Day 1):
        `"Old Man = Tutorial. Tree = Cut. Ledge = One way."`
    -   Prompt:
        `"You are a Game Guide. Context: {CONTEXT}. Situation: Blocked by {BLOCKER}. What is the immediate recovery action?"`
    -   Calls the existing `utils.vlm.VLM` client (Gemini).
-   **Verification:**
    -   **Script:** `python -m agent.brain.demos.demo_planner`
    -   **Success:** Prints:
        `"Plan: Talk to the Old Man to trigger the catching tutorial."`

> **✅ MVP CHECKPOINT 1 — DONE:** The agent can perceive a block and produce a correct
> recovery plan using an LLM with a cheat sheet. The interface is identical to
> what Phase 2 will use.

---

## 📅 PHASE 2: The "Episodic Memory" (MVP — Day 2) ✅ COMPLETE

**Goal:** Replace the static cheat sheet with a dynamic Vector Database.

### 2.1 The Vector Store (`ChromaDB`)

-   **Methodology:** Local, persistent vector storage.
-   **Embedding model:** `all-MiniLM-L6-v2` (80 MB, ms-latency on CPU,
    sufficient for short game text).
-   **Implementation:**
    -   File: `agent/brain/memory.py` (already exists — verify/refine)
    -   API surface:
        -   `add_memory(text, metadata)` — stores an event.
        -   `retrieve(query, n=3)` — semantic search, returns list of strings.
-   **Verification:**
    -   **Script:** `python -m agent.brain.demos.demo_rag_memory`
    -   **Success:** Log `"NPC said route is closed."` → Query
        `"Why can't I pass?"` → Returns the log entry.

### 2.2 The "RAG" Connection

-   **Methodology:** Inject the DB into the Planner.
-   **Implementation:**
    -   Modify `RecoveryPlanner` in `agent/brain/planner.py`:
        -   **Old:** `CONTEXT = "Old Man = Tutorial..."`
        -   **New:** `CONTEXT = memory.retrieve(query=blocker_description)`
-   **Verification:**
    -   **Script:** `python -m agent.brain.demos.demo_full_flow`
    -   **Success:**
        1.  Log: `"Blocked by Old Man."`
        2.  Log: `"Retrieving context..."`
        3.  Log: `"Context Found: 'Old Man needs tutorial.'"`
        4.  Log: `"LLM Plan: 'Interact with the Old Man.'"`

> **🚀 MVP COMPLETION LINE — DONE.** The agent "thinks," "remembers," and "adapts"
> in live gameplay. Verified with real Gemini API on Route 102 trainer battles.

---

## 📅 PHASE 3: Spatial Awareness (Stretch — High Risk)

**Goal:** Tie memories to coordinates.

-   **Risk Warning:** Dynamic cost overlays in the A* pathfinder is a large
    change and out of scope for a 2-day MVP.
-   **Revised Scope: Logging Only.**
    -   Log `(x, y, map_id)` as metadata on every `add_memory()` call.
    -   Do **not** attempt to update the pathfinding graph dynamically.
    -   Value: We can *show* the spatial data ("Look, I know *where* I got
        blocked") without risking breakage in navigation code.

---

## 📅 PHASE 4: Recursive Sub-Goaling (North Star)

**Goal:** Multi-step dependency chains (Get Cut → Beat Gym → Cut Tree).

-   **Status:** Out of scope for MVP.
-   **Role:** "Next Steps" talking point.

---

## Immediate Execution Steps

1.  ~~Rewrite `agent/brain/goal_manager.py`~~ ✅ (now merged into `objective_manager.py`)
2.  ~~Create `agent/brain/planner.py`~~ ✅
3.  ~~Write `tests/test_goal_manager.py`~~ ✅ (11 tests, renamed to `test_objective_manager_blocker.py`)
4.  ~~Create demo scripts~~ ✅ (moved to `agent/brain/demos/`)
5.  ~~Integrate brain into `agent/__init__.py`~~ ✅ (battle transition detection)
6.  Next: Phase 3 — spatial metadata on `log_event()` calls

---

## Summary of Artifacts

| File                                     | Phase       | Status          | Purpose                             |
| ---------------------------------------- | ----------- | --------------- | ----------------------------------- |
| `agent/brain/goal_manager.py`            | **Phase 1** | ❌ Deleted (merged into `objective_manager.py`) | State Machine (Logic Core) |
| `agent/brain/planner.py`                 | **Phase 1** | ✅ Complete      | LLM-based Recovery Planner (RAG)    |
| `agent/brain/memory.py`                  | **Phase 2** | ✅ Complete      | VectorDB Wrapper (`EpisodicMemory`) |
| `tests/test_objective_manager_blocker.py` | **Phase 1** | ✅ 11 tests      | Unit tests for blocker/recovery     |
| `agent/brain/demos/demo_planner.py`      | **Phase 1** | ✅ Complete      | LLM reasoning demo script           |
| `agent/brain/demos/demo_rag_memory.py`   | **Phase 2** | ✅ Complete      | Semantic retrieval demo             |
| `agent/brain/demos/demo_full_flow.py`    | **Phase 2** | ✅ Complete      | End-to-end RAG demo                 |
| `agent/brain/demos/inspect_brain.py`     | **Util**    | ✅ Complete      | Live DB inspector                   |
| `agent/brain/README.md`                  | **Docs**    | ✅ Complete      | Module overview & goals             |
| `agent/brain/PLAN.MD`                    | **Docs**    | ✅ Current       | This roadmap                        |
