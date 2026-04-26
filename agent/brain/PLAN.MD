# agent/brain — LangGraph Migration Roadmap

**Objective:** Migrate the agent's dispatch loop from the monolithic
`action_step()` / `_get_next_action_directive_inner()` stack to a LangGraph
`StateGraph` where each specialist (NavBot, BattleBot, ComsBot) is a node and
a central router directs each tick to the right node.

**Historical context:** The completed Phases 1–5 (EpisodicMemory, RAG,
NPC navigation, healing subsystem) are documented in
[`docs/development/BRAIN_PHASES_1_5_REFERENCE.md`](../../docs/development/BRAIN_PHASES_1_5_REFERENCE.md).

---

## System Status

| Module | Status | Notes |
|--------|--------|-------|
| EpisodicMemory (ChromaDB) | ✅ Stable | Spatial + enriched dialogue metadata |
| RecoveryPlanner (RAG + LLM) | ✅ Stable | Injected via `update_brain()` |
| Walkthrough RAG (strategy_guide) | ✅ Stable | RAG-primary + milestone fallback |
| NPC Navigation (3-tier) | ✅ Stable | Tiers 1 + 2 complete; Tier 3 tabled |
| NPC Identity Registry | ✅ Stable | Adaptive `graphics_id` discovery |
| Generic Healing Subsystem | ✅ Stable | Two-tier (opportunistic + threshold) |
| LangGraph graph | ✅ Stable | All 6 nodes wired, router complete |
| RunDataManager / BackupManager | ✅ Stable | Phase 0 complete |
| AgentState schema | ✅ Stable | `agent/graph/state.py` |
| TelemetryLogger (VLM call + token tracking) | ✅ Stable | `agent/graph/telemetry.py` |
| NavBot node | ✅ Stable | `agent/graph/nodes/nav_bot.py` — 15 unit tests passing; manual test ✅ |
| BattleBot / ComsBot / Verification nodes | ✅ Stable | Phase 3 complete; manual test ✅ |
| Router + graph assembly | ✅ Stable | Phase 4 complete; route102 corridor run ✅ PASSING (62 steps to Petalburg City; see § route102 results) |
| Healing flow test | 🔲 Not yet run | Requires low-HP save state |
| BattleBot VLM timing (script-idle gate) | 🔲 Planned | See § 3.1 improvement note |
| BattleBot inter-battle reset (LangGraph) | ✅ Fixed | `notify_battle_ended()` called on in_battle True→False in `__init__.py` |
| Milestone reform + dialogue logging | 🔲 Not started | Phase 5 — premature ROM-flag firing; ComsBot transcript capture |
| Healing system unification | 🔲 Not started | Phase 6 — replace DAD_HP_CHECK + VLM PokéCenter search with location_graph lookup |
| Karpathy meta-agent + RewardVector | 🔲 Deferred | See [`KARPATHY_PLAN.MD`](KARPATHY_PLAN.MD) — after main-game completion |

---

## Target Architecture

```
AgentState
    │
    ▼ (each Agent.step() call)
┌───────────────────────────────────────┐
│           dispatch node               │
│  routing_condition(state) decides     │
└──┬────────┬────────┬──────────────────┘
   │        │        │
   ▼        ▼        ▼
nav_bot  battle_bot  coms_bot     map_stitcher_relay
   │        │        │                    │
   └────────┴────────┴──────────┐         │
                                 ▼         │
                         verification_node │
                         (advances milestone pointer)
                                 │
                                 ▼
                        END  (result → emulator buttons)
                                 ↑
                  map_stitcher_relay ──► nav_bot (heal journey)
```

**Nodes:**
- **dispatch** — passthrough; routing happens on its outbound conditional edges
- **nav_bot** — wraps `pathfinding.py`; injects NPC obstacles; returns button list
- **battle_bot** — wraps `agent/battle_bot.py`; type-effectiveness move selection
- **coms_bot** — dialogue handler; delegates to OpenerBot for complex flows
- **map_stitcher_relay** — low-HP trigger; looks up nearest PokéCenter entrance
  coords via `find_nearest_pokemon_center()` + `get_entrance_coords()` in
  `location_graph.py`; sets `goal_coords` for NavBot (VLM fallback for unknown cities)
- **verification_node** — reads RAM after each action; advances milestone pointer
  when completion condition is met; for dialogue milestones waits for
  `dialogue_completed` event from ComsBot (see Phase 5)

`karpathy_meta_agent` is a deferred passive wrapper — see [`KARPATHY_PLAN.MD`](KARPATHY_PLAN.MD).

---

## Phase 0: Prerequisites

### 0.1 Install Dependencies

```bash
pip install langgraph langchain-core
```

Add to `requirements.txt` and `pyproject.toml`.

### 0.2 Port RunDataManager + BackupManager

Seth's `seth_reference/utils/data_persistence/` contains two classes that
enable checkpoint-based run restarts (essential for the Karpathy training loop).
Port them to `utils/`:

**Target files:**
- `seth_reference/utils/data_persistence/run_data_manager.py` → `utils/run_data_manager.py`
- `seth_reference/utils/data_persistence/backup_manager.py` → `utils/backup_manager.py`

**Required interface after port:**

```python
class RunDataManager:
    def __init__(self, run_id: str, milestones_file: str): ...
    def log_milestone_completion(self, milestone: str, step: int, state_data: dict): ...
    def get_run_summary(self) -> dict: ...

class BackupManager:
    def __init__(self, cache_dir: str = ".pokeagent_cache"): ...
    def create_cache_backup(self, state_file: str, milestone_name: str) -> str: ...
    # Returns path to the new backup file
```

Wire `BackupManager.create_cache_backup()` into `ObjectiveManager.mark_goal_complete()`
so a `.state` checkpoint is written every time a milestone fires. This enables
restoring to any milestone boundary without replaying the opening sequence.

### Phase 0 Tests

**Automated — `tests/test_run_data_manager.py`:**

```python
class TestRunDataManagerInit:
    # Verify run directory created, metadata JSON file written

class TestMilestoneLogging:
    # log_milestone_completion() appends entry to JSON log
    # Entry contains milestone name, step count, timestamp

class TestBackupCreation:
    # create_cache_backup() copies source .state file to cache_dir/
    # Filename includes milestone name and timestamp
    # Returns valid file path

class TestBackupManagerIdempotent:
    # Calling create_cache_backup() twice for same milestone creates
    # two files with different timestamps (no overwrite)
```

**Manual — Phase 0 Smoke Test:**

*Purpose:* Confirm checkpoints are written at milestone boundaries in a live run.

*Command:*
```bash
python run.py --load-state Emerald-GBAdvance/new_game.state --agent-auto
```

*Observe in console:*
- `[BACKUP]` log line after each milestone completes
- Milestone name, step count, and backup file path printed

*Check filesystem:*
```bash
ls -la .pokeagent_cache/
```

*Pass:* At least one `.state` file appears in `.pokeagent_cache/` named after
a milestone (e.g., `ROUTE_101_ENTERED_step042.state`).

*Fail indicators:*
- Directory not created
- Files not appearing after milestone log lines
- `BackupManager` import error

---

## Phase 1: AgentState Schema + RewardVector

Create `agent/graph/` directory with the following structure:

```
agent/graph/
    __init__.py
    state.py          # AgentState TypedDict + RewardVector + TelemetrySnapshot
    router.py         # routing_condition function
    graph.py          # build_graph() → CompiledGraph
    telemetry.py      # TelemetryLogger — VLM call + token + latency tracking
    nodes/
        __init__.py
        nav_bot.py
        battle_bot.py
        coms_bot.py
        verification.py
        map_stitcher_relay.py
```

### 1.1 RewardVector

```python
# agent/graph/state.py
from dataclasses import dataclass, field

@dataclass
class RewardVector:
    milestone_delta: int = 0        # +1 per milestone completed this step
    manhattan_delta: float = 0.0    # delta distance to goal (positive = getting closer)
    party_level_sum_delta: int = 0  # total party level change
    pokédollar_delta: int = 0       # money gained/lost

    WEIGHTS: tuple = field(default=(10.0, 0.01, 1.0, 0.001), repr=False)

    @property
    def total(self) -> float:
        deltas = (self.milestone_delta, self.manhattan_delta,
                  self.party_level_sum_delta, self.pokédollar_delta)
        return sum(w * d for w, d in zip(self.WEIGHTS, deltas))

    @classmethod
    def compute_delta(cls, prev: dict, curr: dict) -> "RewardVector":
        # prev / curr are state_data dicts from memory_reader
        ...
```

### 1.2 AgentState

```python
# agent/graph/state.py (continued)
from typing import TypedDict, Optional, Any

class AgentState(TypedDict):
    # Raw game state
    frame: Any                              # Current GBA frame object
    state_data: dict                        # Full memory_reader output

    # Perception
    perception: dict                        # VLM output from perception_step()

    # Navigation targets
    goal_coords: Optional[tuple[int, int]]
    goal_location: Optional[str]
    npc_coords: Optional[tuple[int, int]]
    should_interact: bool

    # Objective tracking
    milestone_index: int                    # Pointer into objective list
    context: str                            # "navigation" | "battle" | "dialogue"
                                            # | "healing_needed"
    # Reward
    reward: Optional[RewardVector]
    prev_state_snapshot: Optional[dict]     # For delta computation

    # Output (set by each node)
    last_action: Optional[str]
    last_buttons: list[str]
    step_count: int

    # Telemetry (populated by TelemetryLogger each step)
    telemetry: Optional["TelemetrySnapshot"]
```

### 1.3 TelemetryLogger

Tracks VLM API calls, token consumption, and step latency per step and per
objective. The core MLOps instrument: enables quantifying the cost savings of
the neuro-symbolic router vs. a naive full-VLM approach.

```python
# agent/graph/state.py
from dataclasses import dataclass, field

@dataclass
class TelemetrySnapshot:
    vlm_calls: int = 0            # Gemini API calls this step
    input_tokens: int = 0         # Prompt tokens (from response.usage_metadata)
    output_tokens: int = 0        # Completion tokens
    step_latency_ms: float = 0.0  # Wall-clock time for graph.invoke()
    node_fired: str = ""          # Which node handled this step
```

```python
# agent/graph/telemetry.py
import json, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from agent.graph.state import TelemetrySnapshot

# Gemini Flash pricing as of model release (update if pricing changes)
_INPUT_COST_PER_1K  = 0.000075   # USD per 1K input tokens
_OUTPUT_COST_PER_1K = 0.000300   # USD per 1K output tokens

class TelemetryLogger:
    def __init__(self, log_dir: str = "run_logs"):
        self._log_path = (Path(log_dir) /
                         f"telemetry_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
        self._log_path.parent.mkdir(exist_ok=True)
        self._total_vlm_calls  = 0
        self._total_input_tok  = 0
        self._total_output_tok = 0
        self._total_steps      = 0
        self._step_start: float | None = None
        self._current: TelemetrySnapshot = TelemetrySnapshot()

    # --- call from Agent.step() before graph.invoke() ---
    def begin_step(self) -> None:
        self._step_start = time.perf_counter()
        self._current = TelemetrySnapshot()

    # --- call from any node that makes a VLM call ---
    def record_vlm_call(self, input_tokens: int, output_tokens: int,
                        node: str) -> None:
        self._current.vlm_calls    += 1
        self._current.input_tokens  += input_tokens
        self._current.output_tokens += output_tokens
        self._current.node_fired    = node
        self._total_vlm_calls  += 1
        self._total_input_tok  += input_tokens
        self._total_output_tok += output_tokens

    # --- call from Agent.step() after graph.invoke() ---
    def end_step(self, step: int, milestone_index: int,
                 last_action: str) -> TelemetrySnapshot:
        elapsed = (time.perf_counter() - self._step_start) * 1000
        self._current.step_latency_ms = elapsed
        self._total_steps += 1
        with open(self._log_path, "a") as f:
            entry = {
                "step":            step,
                "milestone_index": milestone_index,
                "last_action":     last_action,
                **asdict(self._current),
            }
            json.dump(entry, f)
            f.write("\n")
        return self._current

    def summary(self) -> dict:
        vlm_rate = self._total_vlm_calls / max(self._total_steps, 1)
        cost = (self._total_input_tok  / 1000 * _INPUT_COST_PER_1K +
                self._total_output_tok / 1000 * _OUTPUT_COST_PER_1K)
        return {
            "total_steps":            self._total_steps,
            "total_vlm_calls":        self._total_vlm_calls,
            "vlm_call_rate":          round(vlm_rate, 4),
            "vlm_savings_vs_naive":   f"{(1 - vlm_rate) * 100:.1f}%",
            "total_input_tokens":     self._total_input_tok,
            "total_output_tokens":    self._total_output_tok,
            "estimated_cost_usd":     round(cost, 4),
        }
```

**Integration points:**
- `Agent.step()` calls `telemetry.begin_step()` before `graph.invoke()` and
  `telemetry.end_step()` after.
- `map_stitcher_relay_node` calls `telemetry.record_vlm_call()` after each
  Gemini response, passing token counts from `response.usage_metadata`.
- `RecoveryPlanner.generate_recovery_plan()` similarly calls
  `telemetry.record_vlm_call()` when it fires.
- `TelemetryLogger` is passed into nodes that need it via the factory pattern
  (same as `make_verification_node`, `make_map_stitcher_relay_node`).

### Phase 1 Tests

**Automated — `tests/test_agent_state.py`:**

```python
class TestAgentStateSchema:
    # Instantiate AgentState with all required fields — no TypeError
    # Verify TypedDict keys match expected set
    # Verify optional fields (reward, telemetry) accept None

class TestRewardVectorDelta:
    # compute_delta(prev, curr) with known state_data dicts
    # milestone_delta: prev milestone_index=2, curr=3 → delta=1
    # manhattan_delta: prev goal dist=10, curr dist=8 → delta=+2.0
    # party_level_sum_delta: Treecko level 7 → 8 → delta=1
    # pokédollar_delta: money 500 → 600 → delta=100

class TestRewardVectorTotal:
    # total property returns weighted sum
    # Verify with known deltas: milestone=1, others=0 → total=10.0

class TestRewardVectorSerialization:
    # asdict(RewardVector(...)) produces JSON-serializable dict
    # Round-trip through json.dumps / json.loads
```

**Automated — `tests/test_telemetry.py`:**

```python
class TestTelemetrySnapshotDefaults:
    # TelemetrySnapshot() has vlm_calls=0, latency=0.0, node_fired=""

class TestBeginEndStep:
    # begin_step() resets current snapshot
    # end_step() returns snapshot with non-zero step_latency_ms
    # end_step() writes one JSONL line to log file
    # JSONL line contains step, milestone_index, last_action, all snapshot fields

class TestRecordVlmCall:
    # record_vlm_call(100, 50, "map_stitcher_relay") increments totals
    # Multiple calls in one step accumulate correctly
    # node_fired reflects the last-called node

class TestSummaryVlmRate:
    # 10 steps, 1 VLM call → vlm_call_rate=0.1, savings="90.0%"
    # 10 steps, 0 VLM calls → vlm_call_rate=0.0, savings="100.0%"
    # 10 steps, 10 VLM calls → vlm_call_rate=1.0, savings="0.0%"

class TestSummaryCostEstimate:
    # 1000 input tokens + 200 output tokens → estimated_cost_usd > 0
    # Zero tokens → estimated_cost_usd == 0.0

class TestLogRoundtrip:
    # All JSONL entries are parseable
    # All numeric fields are float or int
    # begin_step() / end_step() cycle 5 times → 5 lines in log
```

---

## Phase 2: NavBot Node

`nav_bot_node` is a thin wrapper over the existing `pathfinding.py` stack.
No new pathfinding logic — just the LangGraph interface.

### Implementation

```python
# agent/graph/nodes/nav_bot.py
from agent.pathfinding import pathfind_to_goal, update_npc_obstacles
from agent.graph.state import AgentState

def nav_bot_node(state: AgentState) -> AgentState:
    if not state.get("goal_coords"):
        return {**state, "last_action": "PASS", "last_buttons": []}

    # Inject NPC obstacles from current game state
    update_npc_obstacles(state["state_data"])

    # Use existing pathfinding stack
    buttons = pathfind_to_goal(
        state["state_data"],
        state["goal_coords"],
        npc_coords=state.get("npc_coords"),
        should_interact=state.get("should_interact", False),
    )
    return {**state, "last_action": "NAVIGATE", "last_buttons": buttons}
```

### Phase 2 Tests

**Automated — `tests/test_nav_bot_node.py`:**

```python
class TestNavBotNodeBasic:
    # With valid goal_coords → returns state with last_action="NAVIGATE"
    # last_buttons is a non-empty list of GBA button strings
    # state_data pass-through: all other fields unchanged

class TestNavBotNoGoal:
    # goal_coords=None → last_action="PASS", last_buttons=[]

class TestNavBotNpcObstacles:
    # state["state_data"]["active_npcs"] contains one NPC at (10, 3)
    # pathfind_to_goal called with that NPC as obstacle
    # Agent does not path through (10, 3)

class TestNavBotInteract:
    # should_interact=True, npc_coords set → buttons include "A" at end
```

**Manual — NavBot Smoke Test:**

*Purpose:* Confirm NavBot drives the agent to a target NPC correctly in a live run.

*Setup:* No prep needed; save state has player positioned 5 tiles west of May.

*Command:*
```bash
python run.py --load-state tests/save_states/rival_battle_save.state --agent-auto
```

*Observe in console (add `[NAVBOT]` log prefix inside `nav_bot_node`):*
- `[NAVBOT]` appears every step
- Each step logs current position and goal coords
- Position x-coordinate increases each step (agent walking east toward May)

*Observe in emulator:*
- Player sprite walks rightward along y=3
- Reaches May (approximately tile 10,3) within 15 steps
- Presses A to interact

*Pass criteria (objective):*
- [x] Agent reaches (10, 3) ± 1 tile within 15 steps *(2 steps in live run)*
- [x] `[NAVBOT]` logged every step *(requires Phase 4 wiring)*
- [x] A button pressed when adjacent to May

**Result (2026-04-24 → 2026-04-25):** ✅ **PASSING** — the underlying pathfinding
and interaction logic is correct (navigated x=5→9 in one 4-step batch, pressed
A on the rival, advanced dialogue). `[NAVBOT]` prefix now present after
Phase 4 integration: logs `pos` and `goal` every navigation step (confirmed
in `run_20260425_101934.log` at steps 1, 15, 16). Both pass criteria met.

*Fail indicators:*
- Agent walks north or south (y changes significantly): `goal_coords` not
  being passed correctly into `AgentState`
- Agent oscillates: NPC obstacle from a third entity is blocking the direct path
- VLM fallback fires: `nav_bot_node` not yet wired into `Agent.step()`
  (expected if testing the node in isolation before Phase 4 integration)

---

## Phase 3: Specialist Nodes

### 3.1 BattleBot Node

```python
# agent/graph/nodes/battle_bot.py
from agent.battle_bot import BattleBot
from agent.graph.state import AgentState

_bot = BattleBot()

def battle_bot_node(state: AgentState) -> AgentState:
    buttons = _bot.get_action(state["state_data"], state["perception"])
    return {**state, "last_action": "BATTLE", "last_buttons": buttons}
```

**Improvement opportunity — VLM timing in battle:**

The same `sGlobalScriptContext.mode` signal used by ComsBot's
`wait_for_script_idle()` (see § 3.2) is also meaningful inside battle.
In practice, `[MENU STATE] UNKNOWN` is returned every step because the
VLM receives the screenshot mid-animation (attack flash, damage numbers
scrolling, HP bar animating) rather than after the frame has settled.

The fix is to gate the VLM call inside `BattleBot.get_action()` on a
script-idle check analogous to `wait_for_script_idle()`. The endpoint
already handles battle correctly: `mode=0` means the engine is waiting
for input (menu is fully rendered), while `mode=1/2` means an animation
or move-execution script is still running. Values outside `{0,1,2}`
(e.g. `mode=21` observed during post-battle movement) should be treated
as idle (not a script-execution state).

Expected outcome: VLM sees a clean static menu frame, `[MENU STATE]`
returns `FIGHT`/`DIALOGUE`/`RUN` reliably, and the unknown-state
fallback counter stays near zero. This should eliminate the 5-step
delay before the VLM fallback takes over and reduce total battle length
by roughly 30%.

### 3.2 ComsBot Node

```python
# agent/graph/nodes/coms_bot.py
from agent.opener_bot import OpenerBot
from agent.graph.state import AgentState

_opener = OpenerBot()

def coms_bot_node(state: AgentState) -> AgentState:
    if _opener.should_handle(state["state_data"], state["perception"]):
        buttons = _opener.get_action(state["state_data"], state["perception"])
    else:
        buttons = ["A"]
    return {**state, "last_action": "DIALOGUE", "last_buttons": buttons}
```

**Implementation note — wait for script idle before pressing A:**

The GBA's `sGlobalScriptContext.mode` (read via `GET /script_state` on the
game server) tells you whether the typewriter text animation is still running:

| `script_mode` value | Meaning |
|---|---|
| `0` (SCRIPT_MODE_STOPPED) | Text fully rendered — safe to press A |
| `1` (SCRIPT_MODE_BYTECODE) | Script bytecode executing — animation in progress |
| `2` (SCRIPT_MODE_NATIVE) | Native C callback executing — animation in progress |

Pressing A while `mode != 0` sends the input during the animation frame, which
the game may silently ignore (the A-press lands between rendered characters and
gets consumed before the continue-prompt is visible). This was the root cause of
non-deterministic dialogue skipping during the opener sequence.

`wait_for_script_idle()` in `agent/opener_bot.py` is the reference
implementation: it polls `/script_state` at ~80 Hz until `idle=True` or a
timeout expires, then returns control to the caller.

**ComsBot should call this before every A press** for normal overworld NPC
dialogue. Exceptions:
- **TITLE_SEQUENCE / intro scene manager** — `global_mode` is always `0` here
  (the intro uses the GBA callback system, not `sGlobalScriptContext`), so the
  endpoint is meaningless and should be skipped.
- **Naming / keyboard screens** — these are menu-driven, not script-driven;
  `script_mode` is not meaningful.

**RAM fallback for routing (separate concern):** `script_idle` governs *when*
to press A. A distinct, related pattern governs *whether* dialogue is active at
all when VLM perception has failed. `state_data["game"]["game_state"] == "dialog"`
is read directly from GBA RAM and is reliable even when VLM times out or returns
a fallback `overworld` result. `routing_condition` already uses `is_in_dialog`
(which is this same RAM field) to route to ComsBot — so the router is
correctly insulated from VLM failures. However, **within** ComsBot's action
logic, transition functions and action handlers should also consult
`game_state == 'dialog'` as a fallback when `text_box_visible` (VLM-sourced)
is `False`. Without this, a VLM timeout causes the bot to treat active dialogue
as overworld, skipping A presses and blocking movement indefinitely. See
`trans_area_and_dialogue()` and `action_special_clock()` in `agent/opener_bot.py`
for the reference implementation of this dual-check pattern.

### 3.3 VerificationNode

Reads RAM after each action. If the current milestone's `completion_condition`
is satisfied, increments `milestone_index`. This is the Pull half of the
Hybrid Push/Pull system.

```python
# agent/graph/nodes/verification.py
from agent.objective_manager import ObjectiveManager
from agent.graph.state import AgentState

def make_verification_node(obj_manager: ObjectiveManager):
    def verification_node(state: AgentState) -> AgentState:
        milestone = obj_manager.get_milestone_at(state["milestone_index"])
        if milestone and milestone.check_completion(state["state_data"]):
            obj_manager.mark_goal_complete(milestone["milestone"],
                                           state_data=state["state_data"])
            return {**state, "milestone_index": state["milestone_index"] + 1}
        return state
    return verification_node
```

### 3.4 MapStitcherRelay Node

Handles the healing navigation decision. Instead of asking a VLM to find the
PokéCenter on a stitched map image, the node looks up the known entrance
coordinates from `location_graph.py` using the two existing helpers:

- `find_nearest_pokemon_center(current_location)` — returns the graph key
  of the nearest `POKEMON_CENTER_1F` node (e.g. `"PETALBURG_CITY_POKEMON_CENTER_1F"`)
- `get_entrance_coords(city, building)` — returns the overworld `(x, y)` tile
  you walk onto to enter the building

Currently covered cities: Oldale Town, Petalburg City, Rustboro City.
Unknown cities fall back to the original VLM overhead-map approach.

```python
# agent/graph/nodes/map_stitcher_relay.py
from agent.location_graph import find_nearest_pokemon_center, get_entrance_coords
from agent.graph.state import AgentState

def make_map_stitcher_relay_node(vlm):
    def map_stitcher_relay_node(state: AgentState) -> AgentState:
        state_data = state.get("state_data") or {}
        player_data = state_data.get("player", {})
        current_location = player_data.get("location", "").upper().replace(" ", "_")

        # --- Primary: location_graph coordinate lookup ---
        pc_key = find_nearest_pokemon_center(current_location)
        if pc_key:
            # Derive city from PC key (e.g. PETALBURG_CITY_POKEMON_CENTER_1F → PETALBURG_CITY)
            city = pc_key.replace("_POKEMON_CENTER_1F", "").replace("_1F", "")
            coords = get_entrance_coords(city, pc_key)
            if coords:
                return {**state,
                        "goal_coords": coords,
                        "goal_location": pc_key,
                        "context": "navigation",
                        "last_action": "HEAL_ROUTE"}

        # --- Fallback: VLM on stitched overhead map ---
        # Used when current city is not yet in location_graph.
        from utils.map_stitcher_singleton import get_instance
        map_image = None
        try:
            stitcher = get_instance()
            if hasattr(stitcher, "get_overhead_image"):
                map_image = stitcher.get_overhead_image()
        except Exception:
            pass
        map_image = map_image or state.get("frame")
        if map_image is None:
            return state

        prompt = (
            "This is the player's current known map. "
            "Mark the pixel location of the nearest Pokemon Center entrance. "
            'Respond as JSON: {"center_x": int, "center_y": int}'
        )
        try:
            response = vlm.get_query(map_image, prompt)
        except Exception:
            return state

        pos = player_data.get("position") or {}
        player_x = int(pos.get("x", 0))
        player_y = int(pos.get("y", 0))
        tile_x = player_x + (int(response["center_x"]) - 120) // 16
        tile_y = player_y + (int(response["center_y"]) - 80) // 16

        return {**state,
                "goal_coords": (tile_x, tile_y),
                "context": "navigation",
                "last_action": "HEAL_ROUTE"}
    return map_stitcher_relay_node
```

### Phase 3 Tests

**Automated — `tests/test_battle_bot_node.py`:**

```python
class TestWildBattle:
    # state_data has is_in_battle=True, is_wild_battle=True
    # battle_bot_node returns last_buttons containing "B" (run command)

class TestTrainerBattle:
    # state_data has is_in_battle=True, is_wild_battle=False
    # Opponent has known type weakness
    # battle_bot_node selects the type-effective move

class TestBattlePassthrough:
    # All non-output fields in state unchanged after battle_bot_node
```

**Automated — `tests/test_coms_bot_node.py`:**

```python
class TestDialogueAdvance:
    # state_data has is_in_dialog=True, no opener_bot trigger
    # coms_bot_node returns last_buttons=["A"]

class TestOpenerBotDelegation:
    # State matches opener_bot trigger condition
    # coms_bot_node delegates to opener_bot (buttons != ["A"])
```

**Automated — `tests/test_verification_node.py`:**

```python
class TestMilestoneAdvance:
    # Mock completion_condition returns True
    # milestone_index incremented by 1
    # mark_goal_complete called with correct milestone name

class TestNoAdvance:
    # Mock completion_condition returns False
    # milestone_index unchanged

class TestOutOfBoundsIndex:
    # milestone_index points beyond last milestone
    # Returns state unchanged (no crash)
```

**Automated — `tests/test_map_stitcher_relay.py`:**

```python
class TestLocationGraphLookup:
    # current_location="PETALBURG_CITY" → goal_coords = entrance of PETALBURG_CITY_POKEMON_CENTER_1F
    # last_action = "HEAL_ROUTE", context = "navigation"
    # VLM is NOT called (mock never invoked)

class TestOldaleLookup:
    # current_location="OLDALE_TOWN" → goal_coords = known Oldale PC entrance coords

class TestRustboroLookup:
    # current_location="RUSTBORO_CITY" → goal_coords = known Rustboro PC entrance coords

class TestContextReset:
    # After relay node, context is "navigation" (not "healing_needed")

class TestVlmFallback:
    # current_location="UNKNOWN_CITY" (not in location_graph)
    # Mock VLM returns {"center_x": 120, "center_y": 80}
    # Player at (10, 10) → goal_coords = (10, 10)
    # VLM IS called (location_graph lookup failed)
```

**Manual — BattleBot Integration Test:**

*Purpose:* Confirm BattleBot node handles a real trainer battle end-to-end.

*Command:*
```bash
python run.py --load-state Emerald-GBAdvance/route102_hackathon.state \
    --agent-auto --max-steps 200
```

*Observe:*
- Agent walks until a trainer challenges it (exclamation mark visible)
- Console: `[BATTLE_BOT]` fires every step while `is_in_battle=True`
- Console: Type effectiveness log line (e.g., `GRASS effective vs WATER`)
- Battle ends, `[NAVBOT]` resumes

*Pass criteria:*
- [x] `[BATTLE_BOT]` fires exclusively during battle
- [x] Battle completes within 30 battle-steps (won in 10–11 battle steps)
- [x] Navigation resumes after battle (position changes post-battle)

**Result (2026-04-25):** ✅ **PASSING** — confirmed across two runs
(`run_20260425_101223.log`, `run_20260425_101934.log`). Trainer
(YOUNGSTER CALVIN / POOCHYENA) correctly identified via `"sent out"`
keyword. ABSORB selected via type-effectiveness (Grass > Dark/Normal).
Battle won in 10–11 steps, NavBot resumed immediately after. Two bugs
fixed during this session: (1) RAM battle-type flags promoted to
priority 2; (2) newline normalization in dialogue keyword matching.

*Fail indicators:*
- Same move selected every turn regardless of type: effectiveness lookup broken
- Console freezes on a battle step: stuck in menu
- `[NAVBOT]` fires during battle: `routing_condition` not reading `is_in_battle`

**Known limitation (tracked):** The VLM consistently returns `UNKNOWN`
for battle menu state — the fallback species-identification and
type-effectiveness path handles every battle correctly, but the primary
VLM menu path never fires. Root cause is likely that the VLM receives
a screenshot during an animation or mid-typewriter frame rather than a
static menu frame. See the improvement note in § 3.1 below.

---

> **Healing Flow Test** — Originally written here after § 3.4 MapStitcherRelay implementation.
> Moved to **§ Phase 6** (Healing System Unification) because it requires DAD_HP_CHECK removal
> and `context = "healing_needed"` wiring in `Agent.step()` before it can produce meaningful
> results. See *Phase 6 Manual — Healing Smoke Test* for the full test spec and current status.

---

## Phase 4: Router + Graph Assembly

### 4.1 routing_condition

```python
# agent/graph/router.py
from agent.graph.state import AgentState

HP_HEAL_THRESHOLD = 0.50

def routing_condition(state: AgentState) -> str:
    sd = state["state_data"]

    if sd.get("is_in_battle"):
        return "battle_bot"

    if sd.get("is_in_dialog"):
        return "coms_bot"

    party = sd.get("party", [])
    if any(p.get("current_hp", 1) / max(p.get("max_hp", 1), 1) < HP_HEAL_THRESHOLD
           for p in party if p):
        return "map_stitcher_relay"

    return "nav_bot"
```

### 4.2 Full Graph Definition

```python
# agent/graph/graph.py
from langgraph.graph import StateGraph, END
from agent.graph.state import AgentState
from agent.graph.router import routing_condition
from agent.graph.nodes.nav_bot import nav_bot_node
from agent.graph.nodes.battle_bot import battle_bot_node
from agent.graph.nodes.coms_bot import coms_bot_node
from agent.graph.nodes.verification import make_verification_node
from agent.graph.nodes.map_stitcher_relay import make_map_stitcher_relay_node

def build_graph(obj_manager, vlm):
    graph = StateGraph(AgentState)

    graph.add_node("dispatch",           lambda s: s)
    graph.add_node("nav_bot",            nav_bot_node)
    graph.add_node("battle_bot",         battle_bot_node)
    graph.add_node("coms_bot",           coms_bot_node)
    graph.add_node("verification",       make_verification_node(obj_manager))
    graph.add_node("map_stitcher_relay", make_map_stitcher_relay_node(vlm))

    graph.set_entry_point("dispatch")
    graph.add_conditional_edges("dispatch", routing_condition, {
        "nav_bot":            "nav_bot",
        "battle_bot":         "battle_bot",
        "coms_bot":           "coms_bot",
        "map_stitcher_relay": "map_stitcher_relay",
    })
    graph.add_edge("nav_bot",            "verification")
    graph.add_edge("battle_bot",         "verification")
    graph.add_edge("coms_bot",           "verification")
    graph.add_edge("map_stitcher_relay", "nav_bot")      # heal journey continues
    graph.add_edge("verification",       END)

    return graph.compile()
```

### 4.3 Integration with Agent.step()

Replace the `action_step()` call in `agent/__init__.py`:

```python
# Agent.__init__:
self._graph = build_graph(self.objective_manager, self.vlm)

# Agent.step():
agent_state = AgentState(
    frame=frame,
    state_data=state_data,
    perception=perception_output,
    goal_coords=self.objective_manager.get_goal_coords(state_data),
    goal_location=self.objective_manager.get_goal_location(state_data),
    npc_coords=self.objective_manager.get_npc_coords(state_data),
    should_interact=self.objective_manager.get_should_interact(state_data),
    milestone_index=self.objective_manager.get_highest_milestone_index(milestones),
    context="navigation",
    reward=None,
    prev_state_snapshot=self._prev_state_snapshot,
    last_action=None,
    last_buttons=[],
    step_count=self._step_count,
)
result = self._graph.invoke(agent_state)
self._prev_state_snapshot = state_data
return result["last_buttons"]
```

### Phase 4 Tests

**Automated — `tests/test_graph_router.py`:**

```python
class TestRoutingCondition:
    # is_in_battle=True → "battle_bot"
    # is_in_dialog=True → "coms_bot"
    # party HP 40% → "map_stitcher_relay"
    # all normal → "nav_bot"
    # battle takes priority over dialog (both True → "battle_bot")
    # healing takes priority over navigation (low HP, no battle/dialog)

class TestGraphCompiles:
    # build_graph(mock_obj_manager, mock_vlm) returns without exception
    # Compiled graph has all 6 nodes

class TestGraphEdges:
    # All conditional edges from dispatch present
    # nav_bot → verification edge present
    # battle_bot → verification edge present
    # coms_bot → verification edge present
    # map_stitcher_relay → nav_bot edge present
    # verification → END edge present

class TestFullGraphInvoke:
    # Build graph with mock nodes
    # Invoke with is_in_battle=True → last_action="BATTLE"
    # Invoke with normal state → last_action="NAVIGATE"
```

**Manual — Full Run: Route 102 Corridor:**

*Purpose:* Confirm the assembled graph drives the agent through a full corridor
with battles, dialogue, and navigation handled correctly by the router.

*Command:*
```bash
python run.py --load-state Emerald-GBAdvance/route102_hackathon.state \
    --agent-auto --max-steps 500
```

*Expected console output per step:*
```
[STEP 001] dispatch → nav_bot     | pos=(8,14) goal=OLDALE_TOWN
[STEP 002] dispatch → nav_bot     | pos=(8,13) goal=OLDALE_TOWN
[STEP 015] dispatch → battle_bot  | in_battle=True
[STEP 025] dispatch → nav_bot     | battle=False, resuming
```

*Objective pass criteria:*
- [ ] Agent reaches RUSTBORO_CITY within 500 steps *(moved to § Phase 6 — blocked by Petalburg
  healing loop; see Phase 6 Full Corridor Run test)*
- [x] At least 1 `battle_bot` dispatch and 1 `nav_bot` dispatch confirmed in logs
- [x] No single node fires 20+ consecutive steps without position change
      (check: `grep "\[STEP\]" run_logs/*.log | python -c "import sys; from itertools import groupby; [print(k, sum(1 for _ in g)) for k, g in groupby(l.split('→')[1].split('|')[0].strip() for l in sys.stdin)]"`)
- [x] All battles complete within 30 battle steps

*Subjective evaluation (rate 1–5, note in `run_logs/eval_notes.txt`):*
- **Purposefulness** — Does the agent make visible forward progress each minute?
- **Transition quality** — Do node switches feel natural or does the agent freeze?
- **Obstacle handling** — When the agent hits a wall/NPC, does it reroute quickly?

---

## Phase 5: Milestone Completion Reform + Dialogue Logging

### Background — the premature-completion bug

`MILESTONE_PROGRESSION` entries for dialogue milestones (e.g. `DAD_FIRST_MEETING`,
`GYM_EXPLANATION`) currently carry a `milestone_id` that maps directly to a ROM
script-event flag. Pokémon Emerald sets that flag the moment the map loads, not
when the dialogue finishes. `check_storyline_milestones()` polls those flags every
step, so both milestones fire the instant the gym warp triggers — before the agent
has interacted with Norman at all.

### 5.1 Completion-type field in MILESTONE_PROGRESSION

Add a `completion_type` field to each milestone definition:

| `completion_type` | How completion is detected |
|---|---|
| `"location"` | Player coordinate or RAM map-ID matches target (existing behaviour) |
| `"battle"` | `in_battle True→False` at the expected location (existing behaviour) |
| `"dialogue"` | `dialogue_completed` flag in `AgentState`, set by ComsBot on dialogue exit |

For dialogue milestones, the ROM flag is treated as a *trigger* (the scene has
started) rather than a *completion*. The verification node only advances
`milestone_index` when `state["dialogue_completed"] == True` (set by ComsBot
after the full sequence finishes) AND the milestone's `dialogue_keywords` are
present in the captured transcript.

**Example — DAD_FIRST_MEETING:**
```python
{"milestone": "DAD_FIRST_MEETING",
 "completion_type": "dialogue",
 "dialogue_keywords": ["I'm the Petalburg Gym Leader", "Norman", "heal"],
 "target_location": "PETALBURG_CITY_GYM",
 ...}
```

### 5.2 TransitionEvaluator

A lightweight component called at every major context transition
(`dialogue → navigation`, `battle → navigation`). For dialogue transitions:

1. Query ChromaDB for the last N entries tagged with the current milestone context.
2. Assemble the full transcript.
3. LLM call: *"Given this transcript, was the '{milestone_description}' completed?
   Did the character say one of these phrases: {keywords}? Reply: YES / NO / PARTIAL."*
4. If YES: set `state["dialogue_completed"] = True`.
5. If PARTIAL: do not complete; set a retry flag — ComsBot will attempt to
   re-engage the NPC on the next step.
6. Verification node reads `dialogue_completed` and advances `milestone_index`.

The LLM call is gated: it only fires once per dialogue session (on the
`dialogue → navigation` edge), not every step. Expected cost: ~1 extra LLM
call per story dialogue interaction.

### 5.3 ComsBot Dialogue Capture

Add a capture loop inside `coms_bot_node`:

1. **Wait for frame settle:** before extracting text, confirm `script_mode == 0`
   (text animation complete). Use the existing `wait_for_script_idle()` logic.
2. **Extract text:** VLM call with a structured prompt:
   `"Extract the speaker name and exact dialogue text from this GBA screenshot.
   Respond as JSON: {\"speaker\": str, \"text\": str, \"has_more\": bool}"`
3. **Log to ChromaDB:** store `{text, speaker, milestone_context, step, map_id,
   player_pos}` as a single document. Tag with current `milestone_index` for
   later TransitionEvaluator retrieval.
4. **Track session:** the session starts when `context` transitions to
   `"dialogue"` and ends when it transitions away. On session end, emit a
   `DialogueSession` object with the full ordered transcript.
5. **Set `dialogue_completed`** in `AgentState` when the session ends —
   `TransitionEvaluator` then does the LLM verdict call.

**ChromaDB collection:** `dialogue_transcripts`  
**Document schema:**
```json
{
  "text": "...",
  "speaker": "Norman",
  "step": 42,
  "milestone": "DAD_FIRST_MEETING",
  "map_id": 2049,
  "player_x": 16,
  "player_y": 10
}
```

### Phase 5 Tests

**Automated — `tests/test_milestone_completion.py`:**

```python
class TestLocationMilestoneUnchanged:
    # completion_type="location" milestone still completes on coordinate match
    # No dialogue_completed field required

class TestDialogueMilestoneGated:
    # completion_type="dialogue" milestone does NOT complete on ROM flag alone
    # milestone_index unchanged when dialogue_completed=False

class TestDialogueMilestoneCompletes:
    # dialogue_completed=True + keywords present → milestone_index advances

class TestTransitionEvaluatorYes:
    # Transcript contains expected keywords → returns YES → dialogue_completed=True

class TestTransitionEvaluatorNo:
    # Transcript missing keywords → returns NO → dialogue_completed=False
    # milestone_index unchanged
```

**Automated — `tests/test_dialogue_capture.py`:**

```python
class TestCaptureLogsToChromaDB:
    # coms_bot_node with active dialogue → ChromaDB write called
    # Document contains speaker, text, milestone, step fields

class TestCaptureWaitsForIdle:
    # script_mode=1 (animation) → A not pressed, text not extracted
    # script_mode=0 → A pressed, text extracted

class TestSessionTranscriptAssembled:
    # 3 dialogue turns → DialogueSession has 3 entries in order
    # dialogue_completed flag set on session exit
```

**Manual — Dad Meeting Smoke Test:**

*Command:*
```bash
python run.py --load-state Emerald-GBAdvance/house_start_save.state --agent-auto
```

*Pass criteria:*
- [ ] Agent enters Petalburg Gym and walks to Norman
- [ ] `[COMS_BOT]` fires and logs at least 2 transcript entries
- [ ] `DAD_FIRST_MEETING` does NOT complete on gym map-entry; completes only
      after dialogue session ends with `dialogue_completed=True`
- [ ] `GYM_EXPLANATION` similarly requires a completed dialogue session
- [ ] Agent exits gym and proceeds toward Route 104 South

---

## Phase 6: Healing System Unification

### 6.1 What DAD_HP_CHECK was (and why it is being removed)

The `DAD_HP_CHECK` block in `get_next_action_directive()` is a vestigial hack:
*"If the player's Pokémon HP is below 100% when they arrive in Petalburg City,
navigate to Norman (the player's in-game father / gym leader) in the gym."*

The intent was to exploit the fact that Norman heals your party as part of the
opening cutscene. Rather than implementing a general healing system, the code
special-cased Petalburg by detecting HP < 100% and routing directly to the gym.

**Why it causes bugs now:**
- It fires even after `DAD_FIRST_MEETING` and `GYM_EXPLANATION` are marked
  complete (because `_past_opening` is checked unreliably).
- It sets `goal_coords` to Norman's tile `(15, 8)`, which is a wall tile in
  the overworld map, causing infinite pathfinding failures (the stuck loop seen
  in the logs at steps 6–11).
- It bypasses the general PokéCenter system entirely.

**The correct approach:** Norman's healing is now handled naturally by the
Phase 5 dialogue milestone system — the agent interacts with Norman because the
`DAD_FIRST_MEETING` milestone directs it to, not because of an HP check.

### 6.2 Unified Healing Predicate

Replace the split `DAD_HP_CHECK` + `healing_needed` context + VLM relay with a
single decision chain:

```
needs_healing(state) → bool
    True if any party Pokémon HP < HP_THRESHOLD
         OR any Pokémon has a move at 0 PP

is_healing_safe_location(state) → bool
    True if player is in a city/town (not on a route or in grass)

is_danger_threshold(state) → bool
    True if any party Pokémon HP < DANGER_THRESHOLD (e.g. 30%)
         — on roads, trigger return-to-city even without a visible PC
```

**Router logic (replacing Phase 4.1 HP check):**
```python
HP_THRESHOLD   = 1.0    # opportunistic: heal whenever not at full HP
DANGER_THRESHOLD = 0.30  # emergency: return to last city if below this

if needs_healing(state):
    if is_healing_safe_location(state):
        return "map_stitcher_relay"   # city → navigate to PC
    elif is_danger_threshold(state):
        # On a road and dangerously low — return to last known city
        state["goal_location"] = state.get("last_city_visited", "OLDALE_TOWN")
        return "nav_bot"
# else: continue with normal routing
```

### 6.3 `last_city_visited` in AgentState

Add to `AgentState`:
```python
last_city_visited: Optional[str]   # e.g. "PETALBURG_CITY"
```

`nav_bot_node` updates this whenever it detects a city map (`is_city(location)`).
The router reads it to plan a return route when `is_danger_threshold()` is True
on a route tile.

### 6.4 MapStitcherRelay as coordinate-lookup (already updated in § 3.4)

The location_graph lookup is the primary path; VLM is the fallback for unknown
cities. No further changes needed — Phase 3.4 already documents the updated
implementation.

**Cities to add to location_graph (deferred):**
- Dewford Town PC
- Slateport City PC
- Mauville City PC
(Add as encountered during main-game run.)

### Phase 6 Tests

**Automated — `tests/test_healing_predicate.py`:**

```python
class TestNeedsHealingHp:
    # 24/27 HP → needs_healing=True
    # 27/27 HP → needs_healing=False

class TestNeedsHealingPp:
    # Move at 0/15 PP → needs_healing=True
    # All moves have PP > 0 and HP=100% → needs_healing=False

class TestDangerThreshold:
    # 8/27 HP (29%) → is_danger_threshold=True
    # 9/27 HP (33%) → is_danger_threshold=False

class TestRouterHealingInCity:
    # needs_healing=True, location=PETALBURG_CITY → routes to map_stitcher_relay

class TestRouterHealingOnRoad:
    # needs_healing=True, location=ROUTE_102, HP above danger threshold
    # → routes to nav_bot (no emergency return triggered)

class TestRouterEmergencyReturn:
    # needs_healing=True, location=ROUTE_102, HP below danger threshold
    # → routes to nav_bot with goal_location = last_city_visited

class TestLastCityVisited:
    # nav_bot_node in PETALBURG_CITY → updates last_city_visited in state
```

**Manual — Healing Smoke Test:**

*(Originally written in § 3.4 after MapStitcherRelay implementation; moved here because it
requires Phase 6 — DAD_HP_CHECK removal + `context = "healing_needed"` wiring in
`Agent.step()` — before it can produce meaningful results.)*

*Purpose:* Confirm the unified healing system routes the agent to a PokéCenter when HP is
below threshold, using the location_graph lookup (no VLM call).

*Setup — create a low-HP state (choose one):*
```bash
# Option A: fight manually until HP < 50%
python run.py --load-state Emerald-GBAdvance/route102_hackathon.state --manual

# Option B: temporarily lower the threshold to trigger immediately
# In agent/objective_manager.py: HP_HEAL_THRESHOLD = 1.0
```

*Command:*
```bash
python run.py --load-state [low_hp_save] --agent-auto --max-steps 300
```

*Observe:*
- `[HEALING]` trigger log when HP threshold crossed (not `[DAD HP]`)
- `[MAP_STITCHER_RELAY]` fires — logs location_graph lookup, no VLM delay
- `goal_coords` updated to PokéCenter entrance tile from graph
- `[NAVBOT]` routes toward PokéCenter
- `[COMS_BOT]` fires on Nurse Joy dialogue
- Party HP = max after exit

*Pass criteria:*
- [ ] `DAD_HP_CHECK` code no longer present / no longer fires
- [ ] `map_stitcher_relay` looks up PokéCenter coords from location_graph (no VLM call, no 2-second delay)
- [ ] Healing triggered when any party member HP < threshold in a city/town
- [ ] Agent navigates to the correct PokéCenter for the current city
- [ ] Party HP is full after Nurse Joy interaction
- [ ] Agent exits PokéCenter and resumes prior navigation goal

*Fail indicators:*
- `[DAD HP]` still appears in logs: DAD_HP_CHECK not fully removed
- Healing never triggers: check `needs_healing()` predicate and `Agent.step()` context wiring
- Wrong building targeted: location_graph missing city — check `find_nearest_pokemon_center()` return value
- Agent stuck inside PokéCenter: exit warp detection not firing

**Status:** 🔲 **NOT YET RUN** — blocked on Phase 6 implementation (remove DAD_HP_CHECK, wire
`context = "healing_needed"` in `Agent.step()`).

---

**Manual — Full Corridor Run (Route 102 → Rustboro City):**

*(The `[ ] Agent reaches RUSTBORO_CITY` criterion was originally the first pass criterion of
the § Phase 4 Manual Full Run test. Moved here because completing the full corridor requires
healing to work — the agent currently loops between DAD_HP_CHECK and Petalburg Gym before
reaching Route 104 South. All other Phase 4 criteria were ✅ confirmed in the route102 run
at 62 steps to Petalburg City.)*

*Purpose:* Confirm the agent completes the entire opening corridor — Route 102 → Oldale Town
→ Petalburg City (gym dialogue + healing) → Route 104 → Rustboro City — end-to-end.

*Command:*
```bash
python run.py --load-state Emerald-GBAdvance/route102_hackathon.state \
    --agent-auto --max-steps 500
```

*Pass criteria:*
- [ ] Agent reaches RUSTBORO_CITY within 500 steps
- [ ] Healing triggers in Petalburg City (if HP < threshold) → PokéCenter, not gym
- [ ] `DAD_FIRST_MEETING` milestone completes only after Norman dialogue, not on gym map entry
- [ ] Agent proceeds from Petalburg to Route 104 South after gym dialogue

**Status:** 🔲 **BLOCKED** — requires Phase 5 (milestone reform) + Phase 6 (healing unification).

---

## Tabled Items

Deferred until the LangGraph graph is running:

| Item | Reason tabled |
|------|---------------|
| **4.3c** RAG-only (demote milestone list) | Milestone system replaced by 269-objective structure in Phase 4 |
| **4.4c** VLM bounding box NPC fallback | Edge case; Tiers 1+2 cover all tested save states |
| **4.5c** Goal hydration from DB on startup | `completed_goals` replaced by `milestone_index` in `AgentState` |
| **5.4** Gym navigation via RAG + A\* | LangGraph router + NavBot handles this generically |
| **5.5** End-to-end `06_road` validation | Superseded by Phase 4 Manual Full Run test |

---

## Artifact Registry

| File | Phase | Status |
|------|-------|--------|
| `agent/graph/__init__.py` | 1 | 🔲 |
| `agent/graph/state.py` | 1 | 🔲 |
| `agent/graph/router.py` | 4 | 🔲 |
| `agent/graph/graph.py` | 4 | 🔲 |
| `agent/graph/nodes/nav_bot.py` | 2 | ✅ |
| `agent/graph/nodes/battle_bot.py` | 3 | 🔲 |
| `agent/graph/nodes/coms_bot.py` | 3 | 🔲 |
| `agent/graph/nodes/verification.py` | 3 | 🔲 |
| `agent/graph/nodes/map_stitcher_relay.py` | 3 | ✅ (coord lookup) |
| `agent/graph/telemetry.py` | 1 | 🔲 |
| `agent/graph/karpathy_meta_agent.py` | Deferred | See [`KARPATHY_PLAN.MD`](KARPATHY_PLAN.MD) |
| `utils/run_data_manager.py` | 0 | 🔲 |
| `utils/backup_manager.py` | 0 | 🔲 |
| `tests/test_run_data_manager.py` | 0 | 🔲 |
| `tests/test_agent_state.py` | 1 | 🔲 |
| `tests/test_telemetry.py` | 1 | 🔲 |
| `tests/test_nav_bot_node.py` | 2 | ✅ |
| `tests/test_battle_bot_node.py` | 3 | 🔲 |
| `tests/test_coms_bot_node.py` | 3 | 🔲 |
| `tests/test_verification_node.py` | 3 | 🔲 |
| `tests/test_map_stitcher_relay.py` | 3 | 🔲 |
| `tests/test_graph_router.py` | 4 | 🔲 |
| `tests/test_milestone_completion.py` | 5 | 🔲 |
| `tests/test_dialogue_capture.py` | 5 | 🔲 |
| `tests/test_healing_predicate.py` | 6 | 🔲 |
| `tests/test_reward_vector.py` | Deferred | See [`KARPATHY_PLAN.MD`](KARPATHY_PLAN.MD) |
