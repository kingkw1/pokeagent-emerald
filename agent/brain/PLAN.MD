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
| LangGraph graph | 🔲 Not started | This roadmap |
| RunDataManager / BackupManager | 🔲 Not started | Phase 0 prerequisite |
| AgentState schema | ✅ Stable | `agent/graph/state.py` |
| TelemetryLogger (VLM call + token tracking) | ✅ Stable | `agent/graph/telemetry.py` |
| NavBot node | ✅ Stable | `agent/graph/nodes/nav_bot.py` — 15 unit tests passing |
| BattleBot / ComsBot / Verification nodes | 🔲 Not started | Phase 3 |
| Router + graph assembly | 🔲 Not started | Phase 4 |
| Karpathy meta-agent + RewardVector | 🔲 Not started | Phase 5 |

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
- **map_stitcher_relay** — low-HP trigger; asks Gemini VLM for PokeCenter pixel
  coords via stitched map image; resolves to tile coords; hands off to nav_bot
- **verification_node** — reads RAM after each action; advances milestone pointer
  when completion condition is met

`karpathy_meta_agent` wraps the full loop to compute `RewardVector` deltas per
step and log them for offline analysis.

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
- [ ] `[NAVBOT]` logged every step *(deferred — requires Phase 4 wiring)*
- [x] A button pressed when adjacent to May

**Result (2026-04-24):** ✅ Functionally passing — the underlying pathfinding
and interaction logic is correct (navigated x=5→9 in one 4-step batch, pressed
A on the rival, advanced dialogue). `[NAVBOT]` prefix was absent because
`nav_bot_node` is not yet wired into `Agent.step()` — the legacy
`directive_nav` / `action_step()` stack handled all steps. This is the
expected state before Phase 4 integration.

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

Handles the healing decision. Gets the stitched overhead map, asks Gemini
Flash where the nearest PokeCenter is, converts pixel answer to tile coords,
sets `goal_coords` for NavBot.

```python
# agent/graph/nodes/map_stitcher_relay.py
from utils.map_stitcher_singleton import get_instance
from agent.graph.state import AgentState

def make_map_stitcher_relay_node(vlm):
    def map_stitcher_relay_node(state: AgentState) -> AgentState:
        stitcher = get_instance()
        map_image = stitcher.get_overhead_image()  # PIL Image

        prompt = (
            "This is the player's current known map. "
            "Mark the pixel location of the nearest Pokemon Center entrance. "
            'Respond as JSON: {"center_x": int, "center_y": int}'
        )
        response = vlm.get_query(map_image, prompt)

        # GBA screen: 240×160 px, 16px/tile, player at screen center (120, 80)
        player_x = state["state_data"]["player"]["x"]
        player_y = state["state_data"]["player"]["y"]
        tile_x = player_x + (response["center_x"] - 120) // 16
        tile_y = player_y + (response["center_y"] - 80) // 16

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
class TestPixelToTile:
    # Mock VLM returns {"center_x": 120, "center_y": 80} (screen center)
    # Player at (10, 10) → goal_coords = (10, 10)

class TestOffCenterPokeCenter:
    # Mock VLM returns {"center_x": 136, "center_y": 80} (1 tile right)
    # Player at (10, 10) → goal_coords = (11, 10)

class TestContextReset:
    # After relay node, context is "navigation" (not "healing_needed")
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
- [ ] `[BATTLE_BOT]` fires exclusively during battle
- [ ] Battle completes within 30 battle-steps (win or flee)
- [ ] Navigation resumes after battle (position changes post-battle)

*Fail indicators:*
- Same move selected every turn regardless of type: effectiveness lookup broken
- Console freezes on a battle step: stuck in menu
- `[NAVBOT]` fires during battle: `routing_condition` not reading `is_in_battle`

---

**Manual — Healing Flow Test:**

*Purpose:* Confirm MapStitcherRelay routes the agent to a PokeCenter when HP
is below threshold.

*Setup — create a low-HP state (choose one):*
```bash
# Option A: fight manually until HP < 50%
python run.py --load-state Emerald-GBAdvance/route102_hackathon.state --manual

# Option B: temporarily lower the threshold to trigger immediately
# In agent/graph/router.py: HP_HEAL_THRESHOLD = 1.0
```

*Command:*
```bash
python run.py --load-state [low_hp_save] --agent-auto --max-steps 300
```

*Observe:*
- `[HEALING]` trigger log when HP threshold crossed
- `[MAP_STITCHER_RELAY]` fires — logs stitched map VLM call
- VLM response includes pixel coordinates
- `goal_coords` updated to PokeCenter tile
- `[NAVBOT]` routes toward PokeCenter
- `[COMS_BOT]` fires on Nurse Joy dialogue
- Party HP = max after exit

*Pass criteria:*
- [ ] Healing triggered when any party member HP < threshold
- [ ] Agent navigates to the correct PokeCenter for the current city
- [ ] Party HP is full after Nurse Joy interaction
- [ ] Agent exits PokeCenter and resumes prior navigation goal

*Fail indicators:*
- Healing never triggers: check `routing_condition` HP threshold logic
- Wrong building targeted: pixel-to-tile conversion off — log `player x/y`
  and the raw VLM `center_x/center_y` to diagnose
- Agent stuck inside PokeCenter: exit warp detection not firing

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
- [ ] Agent reaches RUSTBORO_CITY within 500 steps
- [ ] At least 1 `battle_bot` dispatch and 1 `nav_bot` dispatch confirmed in logs
- [ ] No single node fires 20+ consecutive steps without position change
      (check: `grep "\[STEP\]" run_logs/*.log | python -c "import sys; from itertools import groupby; [print(k, sum(1 for _ in g)) for k, g in groupby(l.split('→')[1].split('|')[0].strip() for l in sys.stdin)]"`)
- [ ] All battles complete within 30 battle steps

*Subjective evaluation (rate 1–5, note in `run_logs/eval_notes.txt`):*
- **Purposefulness** — Does the agent make visible forward progress each minute?
- **Transition quality** — Do node switches feel natural or does the agent freeze?
- **Obstacle handling** — When the agent hits a wall/NPC, does it reroute quickly?

---

## Phase 5: Karpathy Meta-Agent + RewardVector

### Implementation

```python
# agent/graph/karpathy_meta_agent.py
import json
from pathlib import Path
from datetime import datetime
from agent.graph.state import AgentState, RewardVector

class KarpathyMetaAgent:
    def __init__(self, log_dir: str = "run_logs"):
        self._log_path = (Path(log_dir) /
                          f"reward_log_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
        self._log_path.parent.mkdir(exist_ok=True)

    def wrap_step(self, prev_state: dict, curr_state: AgentState,
                  telemetry: "TelemetryLogger") -> AgentState:
        reward = RewardVector.compute_delta(prev_state, curr_state["state_data"])
        tel    = curr_state.get("telemetry") or TelemetrySnapshot()
        with open(self._log_path, "a") as f:
            json.dump({
                "step":                    curr_state["step_count"],
                "milestone_index":         curr_state["milestone_index"],
                # --- reward ---
                "milestone_delta":         reward.milestone_delta,
                "manhattan_delta":         reward.manhattan_delta,
                "party_level_sum_delta":   reward.party_level_sum_delta,
                "pokédollar_delta":        reward.pokédollar_delta,
                "total":                   reward.total,
                # --- telemetry ---
                "vlm_calls":               tel.vlm_calls,
                "input_tokens":            tel.input_tokens,
                "output_tokens":           tel.output_tokens,
                "step_latency_ms":         tel.step_latency_ms,
                "node_fired":              tel.node_fired or curr_state["last_action"],
            }, f)
            f.write("\n")
        return {**curr_state, "reward": reward}
```

### Phase 5 Tests

**Automated — `tests/test_reward_vector.py`:**

```python
class TestComputeDelta:
    # prev: milestone_index=2, player pos=(10,10), party=[Lv7], money=500
    # curr: milestone_index=3, player pos=(8,10), party=[Lv8], money=600
    # Expected: milestone_delta=1, manhattan_delta=+2.0, level_delta=1, money_delta=100

class TestTotalWeighting:
    # milestone_delta=1, others=0 → total=10.0
    # All zeros → total=0.0
    # Negative manhattan_delta (moving away) does not exceed milestone contribution

class TestRewardLogging:
    # KarpathyMetaAgent.wrap_step() writes valid JSONL entry
    # Entry contains all required fields
    # File created under correct log_dir

class TestRewardLogRoundtrip:
    # All entries are JSON-parseable
    # All numeric fields are float or int (not string)
```

**Manual — Multi-Run Reward Evaluation:**

*Purpose:* Confirm the reward signal is non-zero, correlates with progress, and
is useful for offline analysis.

*Run the agent 3 times:*
```bash
for i in 1 2 3; do
  python run.py --load-state Emerald-GBAdvance/route102_hackathon.state \
    --agent-auto --max-steps 300
done
```

*Inspect reward + telemetry logs:*
```bash
for f in run_logs/reward_log_*.jsonl; do
  echo "=== $f ==="
  python -c "
import json
lines = [json.loads(l) for l in open('$f')]
steps      = len(lines)
total_r    = sum(l['total'] for l in lines)
milestones = sum(l['milestone_delta'] for l in lines)
avg_mh     = sum(l['manhattan_delta'] for l in lines) / steps
vlm_calls  = sum(l.get('vlm_calls', 0) for l in lines)
vlm_rate   = vlm_calls / steps
tok_in     = sum(l.get('input_tokens', 0) for l in lines)
tok_out    = sum(l.get('output_tokens', 0) for l in lines)
cost       = tok_in / 1000 * 0.000075 + tok_out / 1000 * 0.0003
avg_lat    = sum(l.get('step_latency_ms', 0) for l in lines) / steps
print(f'Steps: {steps}  |  Reward: {total_r:.2f}  |  Milestones: {milestones}')
print(f'VLM calls: {vlm_calls} ({vlm_rate*100:.1f}% of steps)  |  Savings vs naive: {(1-vlm_rate)*100:.1f}%')
print(f'Tokens: {tok_in}in / {tok_out}out  |  Est. cost: USD {cost:.4f}')
print(f'Avg step latency: {avg_lat:.1f}ms')
"
done
```

*Objective pass criteria:*
- [ ] `total reward > 0` in at least 2 of 3 runs
- [ ] `milestone_delta > 0` at least once per run (at least 1 milestone fires)
- [ ] `avg manhattan_delta > 0` on nav_bot steps (agent moving toward goal)
- [ ] `vlm_call_rate < 0.30` — VLM should fire on fewer than 30% of steps
      (pure-VLM baseline is 1.0; the router's value is quantified here)
- [ ] `avg step latency < 3000ms` — deterministic nodes (nav/battle/coms)
      should complete in < 100ms; latency spikes indicate VLM overhead

*Subjective evaluation:*
- Do runs with higher total reward correspond to better agent behaviour?
- Is `milestone_delta` the dominant term? A single milestone (+10.0) should
  outweigh movement noise across an entire run.
- Are there steps with `total < -1.0`? These are regressions — note the
  `node_fired` field at those steps.
- What is the VLM savings percentage? Target: ≥ 70% reduction vs. a naive
  full-VLM agent. This is the headline portfolio metric: *"neuro-symbolic
  routing reduced Gemini API calls by X% while maintaining task completion,
  saving $Y per hour at production scale."*

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
| `agent/graph/nodes/map_stitcher_relay.py` | 3 | 🔲 |
| `agent/graph/telemetry.py` | 1 | 🔲 |
| `agent/graph/karpathy_meta_agent.py` | 5 | 🔲 |
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
| `tests/test_reward_vector.py` | 5 | 🔲 |
