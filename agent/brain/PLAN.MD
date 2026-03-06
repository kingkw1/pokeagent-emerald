# 🧠 Memory & Planning System: The "Staff Engineer" Roadmap (v2)

**Objective:** Build a resilient agent capable of encountering an obstacle,
logging it, querying its history via RAG, and replanning using an LLM.

**Status:** Phases 1, 2, & 3 are **complete** and integrated into the live agent.
Phase 4 (Walkthrough RAG) is **in progress** — 4.1–4.3b complete, 4.3c next.

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

## 📅 PHASE 3: Spatial Awareness (Stretch — Low Priority) ✅ COMPLETE

**Goal:** Tie memories to coordinates.

-   **Risk Warning:** Dynamic cost overlays in the A* pathfinder is a large
    change and out of scope for a 2-day MVP.
-   **Revised Scope: Logging Only.**
    -   Log `(x, y, map_id)` as metadata on every `add_memory()` call.
    -   Do **not** attempt to update the pathfinding graph dynamically.
    -   Value: We can *show* the spatial data ("Look, I know *where* I got
        blocked") without risking breakage in navigation code.

### 3.1 Implementation Details

-   **`extract_spatial_metadata(state_data)`** — pure helper function in
    `agent/brain/memory.py`. Extracts `pos_x`, `pos_y`, `location` from the
    emulator's `state_data` dict.
-   **`log_event(text, metadata, state_data=None)`** — the existing method
    gained an optional `state_data` kwarg. When provided, spatial metadata
    is merged via `setdefault` (caller-supplied keys always win).
-   **`retrieve_near(query, *, location, x, y, radius, ...)`** — new
    spatial query method. Combines ChromaDB `where` filtering (by
    `location`) with post-query coordinate radius filtering (L∞ / Chebyshev).
    Memories without spatial data are kept by default.
-   **Wiring:** All three `log_event()` calls in
    `ObjectiveManager.update_brain()` (dialogue, battle-start, battle-end)
    now pass `state_data=state_data`.
-   **Backwards compatible:** Pre-seed calls in `Agent.__init__` (static
    knowledge rules) and demo scripts omit `state_data` and work unchanged.

### 3.2 Verification

-   **Test:** `tests/test_spatial_memory.py` (14 tests, all passing)
    -   `TestExtractSpatialMetadata` — 7 tests (normal, zero coords,
        missing position, None, empty, tuple, missing location)
    -   `TestLogEventSpatial` — 3 tests (stores coords, backward compat,
        caller-wins semantics)
    -   `TestRetrieveNear` — 4 tests (location filter, radius filter,
        combined, empty DB)

---

## 📅 PHASE 4: Walkthrough RAG — Proactive Strategic Planner

**Goal:** Replace the hardcoded `MILESTONE_PROGRESSION` (40+ milestones, covers
Littleroot → Rustboro Gym) with a dynamic, walkthrough-driven planner that can
navigate the **entire game** by querying embedded walkthrough text.

**Status:** Phase 4.1–4.3b **complete**. Phase 4.3c (RAG-only) is next.

### 4.0 Design Rationale

The current `MILESTONE_PROGRESSION` in `objective_manager.py` is a static list of
27 milestones with handcrafted `target_location`, `target_coords`, and `special`
flags. It works reliably through Gym 1 but:
- Doesn't scale — each new location requires manual coordinate entry
- Hardcodes the "one correct path" — no adaptability to game state
- Covers only ~5% of the game

The Walkthrough RAG approach preserves the existing navigation stack (A\*, BFS,
`LOCATION_GRAPH`) and only changes **where destination targets come from** — a
static list vs. an LLM interpreting walkthrough text.

### 4.1 Knowledge Base Preparation

**Source:** Bulbapedia Pokémon Emerald walkthrough (10+ parts, full game coverage).

**Fetching:** Use the MediaWiki raw API to get clean wikitext:
```
https://bulbapedia.bulbagarden.net/w/index.php?title=Walkthrough:Pokémon_Emerald/Part_1&action=raw
```

**Chunking strategy:** The wikitext has natural structure:
- `==Location Name==` headings (Littleroot Town, Route 101, Oldale Town, etc.)
- `===Sub-Event===` headings (Save the Professor!, Birch's Lab, Rival Battle 1, etc.)

Each `==` heading becomes one chunk. Metadata per chunk:
```python
{
    "location": "Route 101",           # Extracted from heading
    "part": 1,                          # Walkthrough part number
    "section_order": 3,                 # Sequential position within the part
    "has_battle": True,                 # Detected from trainer/battle tables
    "items": ["None"],                  # Item list if present
}
```

**Preprocessing:**
1. Strip MediaWiki templates (`{{...}}`), tables, and image refs
2. Keep prose text, item lists, trainer data, and event sequences
3. Resolve location names to `LOCATION_GRAPH` keys via fuzzy match:
   - `"Littleroot Town"` → `"LITTLEROOT_TOWN"`
   - `"Route 101"` → `"ROUTE_101"`
   - `"Petalburg City"` → `"PETALBURG_CITY"`

**Storage:** Dedicated ChromaDB collection `strategy_guide` (separate from the
existing `episodic_memory` collection used for recovery planning).

**Implementation:**
- File: `agent/brain/walkthrough_db.py` (new)
- One-time build script: `scripts/build_walkthrough_db.py`
- Embeds ~100-150 chunks covering the full game

### 4.2 Strategic Planner Method

**File:** `agent/brain/strategic_planner.py` (new)

**Trigger:** Called when the current milestone/directive is completed OR the
agent enters a location not covered by any active directive.

**Query → Directive pipeline:**
```
1. Query:  "I am in {current_location}. I have {badge_count} badges.
            My party: {pokemon_summary}. What should I do next?"

2. Retrieve: Top-3 chunks from strategy_guide collection,
             filtered by location proximity and section_order

3. LLM call: "Given this walkthrough context, what is the next location
              the player should travel to, and what key actions should
              they take there?"
              
              Output format: {"target_location": "ROUTE_101",
                              "description": "Walk north through tall grass to Oldale Town",
                              "priority_actions": ["heal at Pokemon Center", "buy Poke Balls"]}

4. Resolve: Fuzzy-match target_location → LOCATION_GRAPH key → coordinates
            Generate a Directive object compatible with get_next_action_directive()
```

**Critical design decision:** The LLM outputs a **location name only**, never
coordinates. A resolver function maps the location name to `LOCATION_GRAPH` to
get exact coordinates. This prevents hallucinated coordinates.

### 4.3 Transition Plan (Milestones → RAG)

**Phase 4.3a — Shadow mode:** ✅ **COMPLETE.** Run the RAG planner alongside milestones. Log
both outputs to `llm_logs/shadow_comparison.jsonl`. Compare: does the RAG planner
suggest the same next-location as the milestone progression? Track agreement rate.

**Implementation details (4.3a):**
-   `ObjectiveManager.__init__()` accepts an optional `strategic_planner` kwarg.
-   `get_next_action_directive()` wraps the existing milestone logic via
    `_get_next_action_directive_inner()`. After computing the milestone directive,
    `_run_shadow_comparison()` fires the RAG planner and logs a comparison.
-   **Throttling:** Shadow fires every 20 agent steps (avoids LLM spam).
-   **Skips:** Battles and active dialogue are excluded.
-   **Read-only:** The milestone directive is never altered — shadow is logging only.
-   **Stats:** `get_shadow_stats()` returns agreement rate, totals, etc.
-   **Log format:** JSONL with `timestamp`, `location`, `coords`, `badges`,
    `last_milestone`, `milestone_target`, `rag_target`, `rag_display_name`,
    `rag_description`, `rag_priority_actions`, `rag_goal_coords`, `agree`,
    `agreement_rate`, `comparisons_so_far`.
-   **Wiring:** `Agent.__init__` creates `WalkthroughDB` → `StrategicPlanner` →
    passes it to `ObjectiveManager(strategic_planner=...)`. If the walkthrough DB
    is empty (not built yet), shadow mode is silently disabled.
-   **Tests:** `tests/test_shadow_mode.py` — 20 tests covering init, throttling,
    skips, agreement tracking, JSONL logging, directive passthrough, target extraction.

**Phase 4.3b — RAG-primary with milestone fallback:** ✅ **COMPLETE.** The RAG planner drives
navigation. If its output doesn't resolve to a valid `LOCATION_GRAPH` key,
fall back to the next milestone in `MILESTONE_PROGRESSION`.

**Implementation details (4.3b):**
-   `_query_rag_target(state_data)` queries the StrategicPlanner and returns a
    dict with `target_location`, `target_coords`, `description`, etc.
-   In the NAVIGATION HANDLING section of `_get_next_action_directive_inner()`,
    the RAG result is tried first.  If it resolves, `final_target` and
    `final_coords` are overridden.  Otherwise milestone values are kept.
-   Special cases (dialogue, gym, Pokémon Center, rival battle, waypoints) fire
    **before** RAG and return early — they are never overridden by RAG.
-   Tracking: `_rag_override_count`, `_milestone_fallback_count`,
    `_last_directive_source` ("rag" or "milestone").
-   Shadow log now includes `directive_source` field.
-   `get_shadow_stats()` returns `rag_overrides` and `milestone_fallbacks`.
-   **Tests:** `tests/test_shadow_mode.py` — 30 tests covering 4.3a + 4.3b
    (`TestQueryRagTarget`, `TestRagPrimaryCounters`, `TestDirectiveSourceTracking`).
-   **Bug fix:** Tests no longer pollute `llm_logs/shadow_comparison.jsonl`
    (all test classes redirect `_shadow_log_path` to a temp directory).

**Phase 4.3c — RAG-only (revised criteria):** Remove the milestone list once
behavioural evaluation confirms the RAG planner can guide the agent through the
Littleroot → Rustboro corridor without regression. The original "≥95% agreement
rate" metric has been **retired** — shadow comparison showed 0% agreement in live
tests, but investigation revealed this was caused by stale milestone data in save
states, not RAG errors. The RAG planner was actually producing **correct** targets.

New success criteria for 4.3c:
1. End-to-end run from `route102_hackathon.state` reaches Rustboro City (RAG-only navigation).
2. No regression in route completion time vs. milestone-only runs.
3. `MILESTONE_PROGRESSION` moves to `tests/fixtures/` for regression testing.

Milestones are retained as a **silent fallback** (Phase 4.3b architecture) — they
are never removed entirely, only demoted from primary to last-resort when RAG
output fails to resolve.

---

## 📅 PHASE 4.4: Local NPC Navigation (Two-Phase Pathfinding)

**Goal:** Enable the agent to navigate *to* specific NPCs based on walkthrough
instructions (e.g., "Talk to May", "Battle the Rival") without hardcoded
`npc_coords` in `MILESTONE_PROGRESSION`. Currently, NPC interactions require
hand-tuned coordinates (e.g., `(10, 3)` for May on Route 103). This phase
replaces those with a dynamic system that **finds NPCs at runtime**.

**Status:** Not started. Prerequisite: Phase 4.3b (complete).

### The Problem

The directive system (`directive_nav.py`) already supports `npc_coords` and
`should_interact` — but these coordinates are **statically defined** in
milestones. When the walkthrough RAG says "Battle the rival on Route 103", the
agent currently has no way to determine *where* the rival is standing.

Special cases in `objective_manager.py` still hardcode NPC positions:
```python
# Line ~1250 in objective_manager.py
target_coords = (9, 3)
npc_coords = (10, 3)  # May's static position on Route 103
```

These hardcoded coordinates break if the NPC has moved (walking NPCs) or if the
game spawns NPCs dynamically (story triggers, new maps).

### Architecture: Three-Tier NPC Detection

```
┌──────────────────────────────────────────────────────────────┐
│                 NPC Navigation Pipeline                       │
│                                                              │
│  Walkthrough RAG says: "Battle the rival on Route 103"       │
│                    ↓                                         │
│  ┌────────────────────────────────────┐                      │
│  │ Tier 1: gObjectEvents (Memory)     │ ← Deterministic,    │
│  │   Parse 16×68-byte struct array    │   zero latency,     │
│  │   at 0x02037230. Returns all       │   sub-ms.           │
│  │   active NPC map coords.           │                      │
│  └──────────────┬─────────────────────┘                      │
│                 ↓                                            │
│  ┌────────────────────────────────────┐                      │
│  │ Tier 2: VLM Semantic ID            │ ← Match walkthrough │
│  │   Compare "the rival" against      │   text to NPC list. │
│  │   visible_entities from VLM.       │   ~2s latency       │
│  │   Maps NPC *name* → memory slot.   │   (batched w/       │
│  │                                    │   perception).      │
│  └──────────────┬─────────────────────┘                      │
│                 ↓ (fallback only)                            │
│  ┌────────────────────────────────────┐                      │
│  │ Tier 3: VLM Bounding Box           │ ← For NPCs not in  │
│  │   Ask VLM for pixel bbox of the    │   gObjectEvents     │
│  │   target NPC sprite. Convert       │   (script-spawned,  │
│  │   pixel → tile coordinates.        │   cutscene actors). │
│  │   240×160 screen / 16px tile.      │   ~2s extra VLM.    │
│  └────────────────────────────────────┘                      │
│                    ↓                                         │
│  Result: (npc_x, npc_y) → feed into directive_nav.py        │
│          goal_coords + npc_coords + should_interact          │
└──────────────────────────────────────────────────────────────┘
```

### 4.4a — Fix `gObjectEvents` Parsing (Memory Reader)

**Goal:** Produce a reliable, tested method that reads all 16 ObjectEvent slots
from `0x02037230` and returns a clean list of active NPC map coordinates.

**Current state of code:** `memory_reader.py` has **five separate methods** that
attempt to read NPC positions from `gObjectEvents`:
- `read_object_events()` (line ~3812) — orchestrator with 4 sub-strategies
- `_read_runtime_object_events()` (line ~3860) — EWRAM probing
- `_read_gsprites_npcs()` (line ~3960) — sprite → map coord conversion
- `_read_legacy_gobject_events()` (line ~4043) — simple struct read
- `_read_gobject_walking_positions()` (line ~4254) — currentCoords-based

None of these are fully reliable. The struct offsets need validation against the
pokeemerald decompilation source.

**Implementation plan:**
1. Write a standalone test (`tests/test_npc_detection.py`) that loads
   `tests/save_states/rival_battle_save.state` into the emulator and asserts:
   - Player (slot 0) is at `(5, 3)` on `ROUTE_103`
   - At least one other active NPC exists within Manhattan distance ≤ 8
   - That NPC has `graphics_id` consistent with May's overworld sprite
2. Consolidate the five methods into one canonical `read_active_npcs()` method
   with struct offsets validated against pokeemerald:
   ```
   ObjectEvent (68 bytes):
     +0x00  flags (u32) — bit 0 = active
     +0x01  objEventId (u8)
     +0x02  localId (u8)
     +0x03  graphicsId (u8)
     +0x04  movementType (u8)
     +0x05  trainerType (u8)
     +0x10  currentCoords.x (s16)
     +0x12  currentCoords.y (s16)
     +0x14  previousCoords.x (s16)
     +0x16  previousCoords.y (s16)
   ```
3. Deprecate the four legacy methods (keep as private, mark with `# DEPRECATED`).
4. Add a `GRAPHICS_IDS` lookup for common NPCs (player, rival, gym leaders) —
   start minimal (just what we need for the test case).

**Verification:**
- `tests/test_npc_detection.py` — load `rival_battle_save.state`, assert rival
  NPC detected at approximately `(10, 3)` with correct `graphics_id`.
- Run against 3+ save states to confirm no false positives.

### 4.4b — VLM Semantic Identification Layer

**Goal:** When the walkthrough says "Talk to May" or "Battle the rival", identify
*which* of the detected NPCs matches that description.

**Current state:** `perception.py` already extracts `visible_entities` from VLM
output — a list of named entities with approximate screen positions. The
strategic planner already returns `description` text that often names NPCs
(e.g., "Go north and challenge your rival May").

**Implementation plan:**
1. Add a `target_npc` field to the directive returned by `strategic_planner.py`:
   ```python
   {"target_location": "ROUTE_103",
    "description": "Battle rival May on Route 103",
    "target_npc": "MAY",           # ← NEW
    "priority_actions": ["battle"]}
   ```
2. In `objective_manager.py`, when a directive has `target_npc`:
   a. Call `read_active_npcs()` to get all NPC positions from memory.
   b. If only one NPC in range (excluding player) → use it directly.
   c. If multiple NPCs → use `visible_entities` from VLM to match:
      - Compare `target_npc` against entity names (fuzzy match).
      - Cross-reference screen position with memory NPC positions.
   d. Emit `npc_coords=(x, y)` and `should_interact=True` in the directive.
3. Modify `_handle_goal_coords()` in `directive_nav.py` to accept dynamically
   resolved `npc_coords` (no code change needed — it already does).
4. Remove hardcoded `npc_coords` from `RIVAL_BATTLE_1` special case.

**Verification:**
- Load `rival_battle_save.state`, run agent for ≤ 20 steps.
- Agent should: detect NPC → walk right toward May → press A to interact.
- No hardcoded coordinates used in the directive chain.

### 4.4c — VLM Bounding Box Fallback

**Goal:** Handle NPCs that don't appear in `gObjectEvents` (script-spawned
cutscene actors, NPCs that load after a map event fires).

**Implementation plan:**
1. Add an optional VLM prompt: "Draw a bounding box around the character
   named {target_npc}". Response: `{"bbox": [x1, y1, x2, y2]}` in pixels.
2. Convert pixel center to tile coordinates:
   ```python
   # GBA screen: 240×160 pixels, 15×10 visible tiles at 16px each
   # Camera offset from player position in the center
   tile_x = player_x + (pixel_center_x - 120) // 16
   tile_y = player_y + (pixel_center_y - 80) // 16
   ```
3. Feed the resulting `(tile_x, tile_y)` into the standard `npc_coords` path.
4. This is a **fallback only** — Tier 1 (memory) and Tier 2 (semantic ID) are
   tried first. Tier 3 fires only when no NPC was found in `gObjectEvents`.

**Verification:**
- Unit test with a mock VLM returning a known bbox for a visible NPC.
- End-to-end test on a save state where an NPC appears after a cutscene trigger.

### 4.4d — NPC Identity Registry (Adaptive Discovery)

**Goal:** Replace hardcoded `graphics_id` constants (105=rival, 88=Norman, etc.)
with a learned registry that maps semantic NPC roles to their `graphics_id` at
runtime — enabling the agent to "discover fresh" by erasing memory and re-learning
NPC identities through interaction.

**Current state:** `_resolve_npc_coords()` takes `graphics_id=105` (rival),
`graphics_id=88` (Norman), `graphics_id=29` (Nurse Joy), `graphics_id=89`
(Roxanne) as hardcoded constants.  This works but can't generalize to new NPCs
and violates the "no hardcoded one-off solutions" principle.

**Design: Three-Phase Adaptive NPC Discovery**

```
Phase A: OBSERVE & RECORD
  Trigger: dialogue starts after agent pressed A on an NPC
  Input:  active_npcs (who was closest?), VLM speaker field
  Output: registry entry  (map, local_id) → {name, role, graphics_id}

Phase B: RECALL
  Trigger: directive says "talk to Norman" / "heal at Pokemon Center"
  Input:  semantic role string ("rival", "nurse", "gym_leader_norman")
  Output: graphics_id + local_id for _resolve_npc_coords()

Phase C: INFER (cold start)
  Trigger: no registry entry for target NPC on current map
  Heuristic: nearest non-player NPC in active_npcs
  Fallback: hardcoded safety nets (kept but never primary)
```

**Data structure — `NpcRegistry`** (new class in `agent/brain/npc_registry.py`):

```python
# Key: (map_group, map_num, local_id)  — stable across sessions
# Value: {name, role, graphics_id, first_seen_step}
#
# The registry is stored as a JSON file for persistence across runs
# and also logged to EpisodicMemory for semantic retrieval.
#
# Example entries:
#   ("ROUTE_103", 2)       → {name: "May",    role: "rival",   gfx: 105}
#   ("PETALBURG_CITY_GYM", 1) → {name: "Norman", role: "gym_leader", gfx: 88}
#   ("OLDALE_TOWN_POKEMON_CENTER_1F", 1) → {name: "Nurse Joy", role: "nurse", gfx: 29}
```

**Implementation plan:**

1. **Create `agent/brain/npc_registry.py`:**
   - `NpcRegistry` class with JSON file persistence (`npc_registry.json`)
   - `register_npc(location, local_id, name, role, graphics_id, step)` — upserts
   - `lookup_by_role(role, location=None)` → `{graphics_id, local_id}` or `None`
   - `lookup_by_name(name)` → same
   - `clear()` — wipe for fresh-start tests
   - Optionally logs to `EpisodicMemory` for semantic retrieval backup

2. **Add observation hook in `agent/__init__.py`:**
   - When dialogue starts (visual_dialogue_active transitions False→True):
     - Find nearest NPC in `state_data['active_npcs']` to player position
     - Extract VLM `on_screen_text.speaker` as the NPC's name
     - Call `npc_registry.register_npc(...)` with discovered identity
   - This is zero-cost — uses data already present in the step pipeline

3. **Wire `NpcRegistry` into `_resolve_npc_coords()`:**
   - New first parameter: `npc_role` (e.g. `"rival"`, `"nurse"`, `"gym_leader_norman"`)
   - Resolution order:
     a. Registry lookup by role → get `graphics_id` + `local_id`
     b. If registry miss → use hardcoded fallback constants (safety net)
     c. Feed `graphics_id`/`local_id` into existing `active_npcs` search
   - Call sites change from `graphics_id=RIVAL_GFX_ID` to `npc_role="rival"`

4. **Cold-start inference (Phase C):**
   - On first run (empty registry), the agent won't know any `graphics_id`s
   - Heuristic: when directive says "interact with NPC" and registry has no match,
     pick the nearest non-player NPC from `active_npcs`
   - After first interaction, the observation hook populates the registry
   - Subsequent lookups use the learned identity

**Verification:**
- Unit tests: registry CRUD, JSON persistence, role lookup, cold-start heuristic
- Integration test: load `rival_battle_save.state`, clear registry, agent discovers
  rival identity through interaction, subsequent lookups use learned `graphics_id`
- Regression: all existing `test_npc_resolve.py` tests still pass

### Manual Testing Procedure

**Canonical test save:** `tests/save_states/rival_battle_save.state`
- **Map:** ROUTE_103
- **Player:** Position (5, 3), facing right
- **Party:** TREECKO Lv.7, HP 23/23
- **Rival NPC (May):** ~5 tiles to the right, approximately at (10, 3)
- **Expected behaviour:** Agent walks right → reaches May → presses A → battle starts

**Steps to run manually:**
```bash
# Start the server with the test save state
python run.py --load-state tests/save_states/rival_battle_save.state --agent-auto

# Or for manual observation:
python run.py --load-state tests/save_states/rival_battle_save.state --manual
```

**What to verify:**
1. Console shows NPC detected via `read_active_npcs()` with coords near `(10, 3)`.
2. Directive includes `npc_coords` resolved dynamically (not from milestone data).
3. Agent navigates right along y=3 without wandering.
4. Agent faces NPC and presses A within ≤ 15 steps.
5. Battle transition fires (if working) — not a hard requirement for 4.4a alone.

**Additional save states for regression:**
- `enter_petalburg_city_save.state` — Multiple NPCs on screen, verify correct targeting.
- `birch_lab_save.state` — Indoor NPC (Prof. Birch), tight space navigation.
- `route102_save.state` — Trainers on Route 102, verify trainer NPC detection.

### 4.4 Location Name Resolver

The bridge between walkthrough prose and the navigation system:

```python
# agent/brain/location_resolver.py (new)
from difflib import get_close_matches
from agent.location_graph import LOCATION_GRAPH

LOCATION_ALIASES = {
    "Littleroot Town": "LITTLEROOT_TOWN",
    "Route 101": "ROUTE_101",
    "Oldale Town": "OLDALE_TOWN",
    "Route 103": "ROUTE_103",
    "Route 102": "ROUTE_102",
    "Petalburg City": "PETALBURG_CITY",
    "Route 104": "ROUTE_104",
    "Petalburg Woods": "PETALBURG_WOODS",
    "Rustboro City": "RUSTBORO_CITY",
    # ... extends as walkthrough coverage grows
}

def resolve_location(name: str) -> dict | None:
    """Map a prose location name to LOCATION_GRAPH entry with coordinates."""
    # Direct alias match
    key = LOCATION_ALIASES.get(name)
    if not key:
        # Fuzzy match against alias keys
        matches = get_close_matches(name, LOCATION_ALIASES.keys(), n=1, cutoff=0.7)
        key = LOCATION_ALIASES[matches[0]] if matches else None
    if key and key in LOCATION_GRAPH:
        return {"key": key, **LOCATION_GRAPH[key]}
    return None
```

### 4.5 Verification Criteria

| Test | Expected |
|------|----------|
| `build_walkthrough_db.py` completes without error | ~100-150 chunks embedded |
| Query "I am in Littleroot Town" returns Littleroot/Route 101 chunks | Top-3 relevant |
| `resolve_location("Route 101")` returns `ROUTE_101` with coords | Exact match |
| Shadow mode: RAG agrees with milestones 0–26 | ≥90% agreement |
| LLM never outputs raw coordinates | Only location names in response |

---

## 📅 PHASE 5: Recursive Sub-Goaling (North Star)

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
6.  ~~Phase 3 — spatial metadata on `log_event()` calls~~ ✅
7.  ~~Phase 4.1 — build walkthrough knowledge base (`scripts/build_walkthrough_db.py`)~~ ✅
8.  ~~Phase 4.2 — implement strategic planner (`agent/brain/strategic_planner.py`)~~ ✅
9.  ~~Phase 4.3a — shadow mode comparison (RAG vs milestones)~~ ✅
10. ~~Phase 4.3b — RAG-primary with milestone fallback~~ ✅
11. Next: Phase 4.3c — RAG-only (behavioural evaluation, not agreement rate)
12. ~~Phase 4.4a — Fix `gObjectEvents` parsing + `read_active_npcs()` + tests~~ ✅
13. ~~Phase 4.4b — Wire `_resolve_npc_coords()` into directive pipeline, replace hardcoded NPC coords~~ ✅
14. Phase 4.4c — VLM bounding box fallback for script-spawned NPCs
15. Phase 4.4d — NPC Identity Registry (adaptive discovery, replace hardcoded graphics_id constants)

---

## Summary of Artifacts

| File                                     | Phase       | Status          | Purpose                             |
| ---------------------------------------- | ----------- | --------------- | ----------------------------------- |
| `agent/brain/goal_manager.py`            | **Phase 1** | ❌ Deleted (merged into `objective_manager.py`) | State Machine (Logic Core) |
| `agent/brain/planner.py`                 | **Phase 1** | ✅ Complete      | LLM-based Recovery Planner (RAG)    |
| `agent/brain/memory.py`                  | **Phase 2+3** | ✅ Complete     | VectorDB Wrapper (`EpisodicMemory` + spatial) |
| `agent/brain/walkthrough_db.py`          | **Phase 4** | ✅ Complete       | Walkthrough ChromaDB collection     |
| `agent/brain/strategic_planner.py`       | **Phase 4** | ✅ Complete       | RAG-driven strategic navigation     |
| `agent/brain/location_resolver.py`       | **Phase 4** | ✅ Complete       | Prose → LOCATION_GRAPH resolver     |
| `scripts/build_walkthrough_db.py`        | **Phase 4** | ✅ Complete       | One-time KB build script            |
| `tests/test_objective_manager_blocker.py` | **Phase 1** | ✅ 12 tests      | Unit tests for blocker/recovery     |
| `tests/test_spatial_memory.py`            | **Phase 3** | ✅ 14 tests      | Unit tests for spatial metadata     |
| `tests/test_walkthrough_rag.py`           | **Phase 4** | ✅ 52 tests      | Unit tests for walkthrough RAG      |
| `tests/test_shadow_mode.py`               | **Phase 4.3a+b** | ✅ 30 tests | Unit tests for shadow-mode + RAG-primary |
| `tests/test_npc_detection.py`             | **Phase 4.4a** | ✅ 13 tests   | NPC detection from `gObjectEvents` (rival_battle_save) |
| `tests/test_npc_resolve.py`               | **Phase 4.4b** | ✅ 17 tests   | `_resolve_npc_coords()` unit + integration (dynamic NPC targeting) |
| `agent/brain/npc_registry.py`             | **Phase 4.4d** | ✅ Complete     | NPC Identity Registry (adaptive discovery) |
| `tests/test_npc_registry.py`              | **Phase 4.4d** | ✅ 25 tests     | Unit tests for NPC registry |
| `agent/brain/demos/demo_planner.py`      | **Phase 1** | ✅ Complete      | LLM reasoning demo script           |
| `agent/brain/demos/demo_rag_memory.py`   | **Phase 2** | ✅ Complete      | Semantic retrieval demo             |
| `agent/brain/demos/demo_full_flow.py`    | **Phase 2** | ✅ Complete      | End-to-end RAG demo                 |
| `agent/brain/demos/inspect_brain.py`     | **Util**    | ✅ Complete      | Live DB inspector                   |
| `agent/brain/README.md`                  | **Docs**    | ✅ Complete      | Module overview & goals             |
| `agent/brain/PLAN.MD`                    | **Docs**    | ✅ Current       | This roadmap                        |
