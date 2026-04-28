# Karpathy Meta-Agent Plan

**Status:** 🔲 Deferred — agent must reliably complete the main game before this
layer is added. See [`HTN_MIGRATION_PLAN.md`](HTN_MIGRATION_PLAN.md) for the current active roadmap.

**Objective:** Wrap the LangGraph dispatch loop with a reward-logging and
meta-learning layer that quantifies the agent's per-step performance, enables
offline policy improvement, and produces a VLM-savings metric suitable for
portfolio / competition documentation.

---

## Evaluation Oracle (pokeemerald)

Accurate progress evaluation requires ground-truth game state that the agent itself
cannot see. We use [**pret/pokeemerald**](https://github.com/pret/pokeemerald) —
the complete reverse-engineered decompilation of Pokémon Emerald — as a
**passive oracle** for the Karpathy evaluator only.

### Design Principle: Perfect Evaluator, Imperfect Agent

The agent operates under **partial observability**: it sees only VLM frame analysis
and the selective RAM reads it has always had (`gPlayerParty`, `gBattleMons`,
`gObjectEvents`, etc.). It has no knowledge of event flags, warp tables, or script state.

The Karpathy reward calculator has **full observability**: it reads the same GBA RAM
but decodes it against pokeemerald's constants to produce a richly typed progress
vector. This information is **never passed to any agent node** — it exists only in the
reward log for offline analysis.

```
┌─────────────────────────────────────┐   ┌─────────────────────────────────────┐
│         LangGraph Agent             │   │       Karpathy Evaluator            │
│  (partial observability)            │   │  (full observability — oracle only) │
│                                     │   │                                     │
│  VLM frames (Gemini Flash)          │   │  SaveBlock1.flags[] bitfield        │
│  + selective RAM reads:             │   │  decoded via pokeemerald:           │
│    gPlayerParty                     │   │    include/constants/flags.h        │
│    gBattleMons                      │   │    include/constants/map_groups.h   │
│    gObjectEvents                    │   │    data/maps/*/map.json warp tables │
│    player (x,y,map_bank,map_num)    │   │                                     │
│                                     │   │  → story_flags_set (bitset diff)    │
│  No event flags.                    │   │  → maps_visited (new map IDs)       │
│  No warp tables.                    │   │  → items_obtained (PC items delta)  │
│  No script state.                   │   │  → badges_earned (gym flags)        │
└─────────────────────────────────────┘   └─────────────────────────────────────┘
```

### What pokeemerald Provides

| pokeemerald artifact | Used for |
|---|---|
| `include/constants/flags.h` | Named constants for every story event flag (e.g., `FLAG_GOT_STARTER_FROM_PROFESSOR_BIRCH`, `FLAG_BADGE01_GET`) — maps RAM flag index → human-readable event |
| `include/constants/map_groups.h` | Every map ID constant (e.g., `MAP_ROUTE101`, `MAP_PETALBURG_CITY`) — maps `(map_bank, map_num)` RAM pair → location name |
| `data/maps/*/map.json` | Per-map warp tables, NPC event scripts, and object event coordinates — used to verify that the agent reached the right position within a map |
| `src/data/object_events/object_event_graphics_info.h` | Graphics-ID-to-NPC-type mapping — ground truth for `NpcRegistry` validation |

### RAM Access Pattern (Evaluator)

```python
# pokemon_env already exposes SaveBlock1 — the evaluator extends the existing
# reader rather than opening a new RAM window

SAVEBLOCK1_FLAGS_OFFSET = 0x00AC  # byte offset of flags[] within SaveBlock1
FLAGS_ARRAY_SIZE = 0x12C           # 300 bytes = 2400 flag bits

def read_story_flags(env) -> set[int]:
    """Return the set of flag indices currently set in SaveBlock1.flags[].
    Decode against pokeemerald flag constants for human-readable names.
    Called only by the Karpathy evaluator — never passed to agent nodes.
    """
    flags_bytes = env.read_memory(SAVEBLOCK1_FLAGS_OFFSET, FLAGS_ARRAY_SIZE)
    return {
        byte_idx * 8 + bit_idx
        for byte_idx, byte in enumerate(flags_bytes)
        for bit_idx in range(8)
        if byte & (1 << bit_idx)
    }
```

---

## Motivation

The "Karpathy loop" is the outer shell that makes the agent learnable:

1. Run the agent for N steps, logging `(state, action, reward)` tuples.
2. Analyse the reward signal offline: which decision types (NavBot, BattleBot,
   ComsBot) produce positive reward? Where does the agent regress?
3. Use the reward log to fine-tune VLM prompts or NPC targeting heuristics
   without re-running the emulator.
4. Report the VLM call-rate reduction as the headline portfolio metric:
   *"neuro-symbolic routing reduced Gemini API calls by X% vs. a naive
   full-VLM baseline, saving $Y per hour at production scale."*

**This layer is passive** — it wraps the existing graph without changing
routing logic. It can be added at any time without breaking the core agent.

---

## Implementation

### KarpathyMetaAgent

```python
# agent/graph/karpathy_meta_agent.py
import json
from pathlib import Path
from datetime import datetime
from agent.graph.state import AgentState, RewardVector, TelemetrySnapshot

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
                # --- reward (oracle-derived, evaluator only) ---
                # story_flags_delta: count of new pokeemerald event flags set
                # since the previous step. Source: SaveBlock1.flags[] decoded
                # against include/constants/flags.h. NOT passed to agent nodes.
                "story_flags_delta":       reward.story_flags_delta,
                "new_flags":               reward.new_flags,          # list[str] human names
                "maps_visited_delta":      reward.maps_visited_delta,  # new map IDs entered
                "badges_delta":            reward.badges_delta,        # gym flag count diff
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
                # --- htn state snapshot (for correlation analysis) ---
                "htn_stack_depth":         len(curr_state.get("goal_stack") or []),
                "htn_top_goal":            (curr_state.get("goal_stack") or [{}])[0].get("task") if curr_state.get("goal_stack") else None,
            }, f)
            f.write("\n")
        return {**curr_state, "reward": reward}
```

---

## Tests

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

---

## Manual — Multi-Run Reward Evaluation

*Purpose:* Confirm the reward signal is non-zero, correlates with progress,
and is useful for offline analysis.

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
  full-VLM agent. This is the headline portfolio metric.

---

## Artifact Registry

| File | Status |
|------|--------|
| `agent/graph/karpathy_meta_agent.py` | 🔲 Not started |
| `tests/test_reward_vector.py` | 🔲 Not started |

---

## Phase K2: Meta-Agent Code Simplification Loop

**Objective:** Extend the Karpathy loop into a full self-modifying meta-agent
that iteratively reduces hardcoded conditional logic in favour of learned or
data-driven rules, optimizing for **low cyclomatic complexity** rather than
raw line count.

### Motivation: Complexity as a First-Class Metric

The reward vector already captures *behavioural* quality (milestone progress,
VLM savings). Phase K2 adds a second axis: **structural code quality**. The
insight from the Gemini discussion is critical:

> Optimizing for raw code length causes "code golf" — the meta-agent compresses
> readable logic into fragile one-liners. Optimizing for low cyclomatic
> complexity instead rewards the *right* kind of simplification: eliminating
> nested if-then trees and replacing them with data-driven dispatch or learned
> policies.

Cyclomatic complexity (CC) counts the number of independent paths through a
function. A function with no branches has CC=1. Each `if`, `elif`, `for`,
`while`, `except`, or `and`/`or` in a condition adds 1. The practical target
for maintainable code is CC ≤ 10 per function; anything above 15 is a
refactor candidate.

The concrete goal: systematically identify high-CC functions across the
codebase (e.g., the rule-heavy `battle_bot.py` heuristics, the milestone
FSM in `objective_manager.py`, hardcoded NPC targeting in `navigation_planner.py`),
and use the meta-agent to propose simpler replacements — either data-driven
lookups, delegated strategy objects, or trained RL policies.

---

### K2.1 Complexity Scanner Integration

**Tool:** [`radon`](https://radon.readthedocs.io/) — pure-Python CC analyser.

```bash
pip install radon
# Report CC for every function, sort by complexity
radon cc agent/ -s -n C   # only show functions with CC ≥ 10 (grade C or worse)
```

**`ComplexityGate`** — a pre-patch validation class that blocks meta-agent
proposals that increase CC:

```python
# agent/graph/complexity_gate.py
import subprocess, json
from pathlib import Path


class ComplexityGate:
    """Rejects code patches that raise cyclomatic complexity above the threshold."""

    DEFAULT_THRESHOLD = 10   # CC ≤ 10 per function

    def __init__(self, threshold: int = DEFAULT_THRESHOLD):
        self.threshold = threshold

    def measure(self, filepath: str | Path) -> dict[str, int]:
        """Return {function_name: cc_score} for the given file."""
        result = subprocess.run(
            ["radon", "cc", str(filepath), "-s", "-j"],
            capture_output=True, text=True, check=True,
        )
        raw = json.loads(result.stdout)
        scores = {}
        for entry in raw.get(str(filepath), []):
            scores[entry["name"]] = entry["complexity"]
        return scores

    def check(self, filepath: str | Path) -> tuple[bool, list[str]]:
        """Return (passed, violations) where violations lists offending functions."""
        scores = self.measure(filepath)
        violations = [
            f"{fn}: CC={cc}" for fn, cc in scores.items() if cc > self.threshold
        ]
        return len(violations) == 0, violations
```

The verification script (`scripts/verify_patch.py`) runs `ComplexityGate.check()`
on every file touched by the meta-agent before any hot reload is triggered.
A patch that increases any function's CC above the threshold is **rejected**
and the meta-agent is prompted to try again with the violation list attached.

---

### K2.2 Sandboxed Edit Targets

The meta-agent is never given write access to emulator hooks, database
credentials, or core LangGraph wiring. Editable targets are explicitly
allowlisted:

| File | Reason It Can Be Simplified |
|------|-----------------------------|
| `agent/combat/heuristic_agent.py` | Deep nested type-effectiveness if-trees → table lookup |
| `agent/graph/nodes/nav_bot.py` | Location-specific if-blocks → LOCATION_GRAPH portal metadata |
| `agent/graph/nodes/battle_bot.py` | Rule-heavy battle heuristics → strategy object dispatch |
| `agent/graph/routing_condition.py` | Multi-branch dispatch → table-driven routing |
| `agent/pathfinding.py` | Nested fallback chains → clean graph search with single entry point |

Files that are **never** editable by the meta-agent:
- `pokemon_env/` (emulator bindings)
- `server/` (FastAPI/Pygame)
- `agent/graph/state.py` (shared schema)
- Any `__init__.py`
- `.env`, database files, save states

```python
# scripts/verify_patch.py
ALLOWED_EDIT_TARGETS = {
    "agent/combat/heuristic_agent.py",
    "agent/graph/nodes/nav_bot.py",
    "agent/graph/nodes/battle_bot.py",
    "agent/graph/routing_condition.py",
    "agent/pathfinding.py",
}

def verify_patch(changed_files: list[str]) -> tuple[bool, str]:
    for f in changed_files:
        if f not in ALLOWED_EDIT_TARGETS:
            return False, f"Patch touches non-sandboxed file: {f}"
    gate = ComplexityGate()
    for f in changed_files:
        passed, violations = gate.check(f)
        if not passed:
            return False, f"CC gate failed in {f}: {violations}"
    # syntax check
    for f in changed_files:
        result = subprocess.run(["python", "-m", "py_compile", f], capture_output=True)
        if result.returncode != 0:
            return False, f"Syntax error in {f}: {result.stderr.decode()}"
    return True, "ok"
```

---

### K2.3 Meta-Agent Reflection Loop

**Lifecycle — single iteration:**

1. **Telemetry dump** — `KarpathyMetaAgent.wrap_step()` accumulates a run log.
   At run end, `_compile_telemetry_dump()` bundles: reward JSONL, stack trace
   (if failure), and the current CC baseline (`radon cc agent/ -j`).

2. **Target selection** — a lightweight selector picks the highest-CC function
   that appears on the hot path (i.e., `node_fired` frequently in the reward
   log). This focuses the meta-agent on code that actually runs, not dead
   branches.

3. **Meta-agent prompt** — the meta-agent (Gemini Flash or equivalent) receives:
   - The full source of the target file
   - The CC violation report for that file
   - The reward log excerpt showing where the agent struggled
   - The instruction: *"Rewrite the identified function to have CC ≤ 10.
     Replace nested if-chains with data structures (dicts, lists, strategy
     objects). Do not change the public function signature or return type.
     Output only the replacement Python function body."*

4. **Patch verification** — `verify_patch.py` runs: sandbox check → syntax
   check → `ComplexityGate` check. Rejection feeds the violation list back
   to the meta-agent for one retry. Two consecutive rejections abort the
   iteration and log the failure.

5. **Hot reload** — the patched file is written, the LangGraph environment
   reloads the module, and the emulator restores the last clean `.state`
   checkpoint.

6. **Regression test** — `pytest tests/` runs. Any failure rolls back the
   patch (git-level revert) and logs the regression.

7. **Reward comparison** — the patched agent runs for N steps. If `total
   reward` regresses by more than 10% vs. the pre-patch baseline, the patch
   is rolled back.

```
Run Agent → Telemetry Dump → Select High-CC Target
    ↓
Meta-Agent Rewrites Function
    ↓
verify_patch.py (sandbox + syntax + CC gate)
    ↓ pass                      ↓ fail (≤1 retry)
Hot Reload                   Log rejection → abort
    ↓
pytest regression guard
    ↓ pass                      ↓ fail
Reward comparison            git revert
    ↓ pass       ↓ regress
Commit patch   git revert
```

---

### K2.4 Complexity Baseline & Targets

Run `radon cc agent/ -s -n C` to establish the baseline before any
meta-agent edits. Record the results here once the first scan is run:

| Module | Worst Function | Baseline CC | Target CC |
|--------|---------------|-------------|-----------|
| `agent/combat/heuristic_agent.py` | TBD | TBD | ≤ 10 |
| `agent/graph/nodes/nav_bot.py` | TBD | TBD | ≤ 10 |
| `agent/graph/nodes/battle_bot.py` | TBD | TBD | ≤ 10 |
| `agent/pathfinding.py` | TBD | TBD | ≤ 10 |
| `agent/graph/routing_condition.py` | TBD | TBD | ≤ 10 |

**Concrete simplification patterns** the meta-agent is instructed to prefer:

| Anti-Pattern (High CC) | Preferred Replacement (Low CC) |
|------------------------|-------------------------------|
| `if type_a: ... elif type_b: ...` (20+ branches) | `TYPE_EFFECTIVENESS: dict[tuple, float]` lookup |
| Hardcoded `if milestone_index == 5: ...` | Data-driven JSON milestone config |
| `if location == "PETALBURG": coords = (x,y) ...` | `LOCATION_GRAPH` portal metadata + `get_entrance_coords()` |
| Nested battle-state `if hp < 20: if pp == 0: ...` | `BattlePolicy` strategy object with a single `select_move()` dispatch |
| Multi-branch routing `if last_action == "X": return A elif ...` | Routing table `{state_key: node_name}` |

---

### K2.5 Tests

**Automated — `tests/test_complexity_gate.py`:**

```python
class TestMeasureCleanFile:
    # A file with a single 3-line function → CC=1, no violations

class TestMeasureComplexFile:
    # Temporary file with a function containing 12 if-statements → CC > 10
    # ComplexityGate(threshold=10).check() returns (False, [violation_string])

class TestGatePassesAtThreshold:
    # Function with exactly CC=10 → (True, [])

class TestGateFailsAboveThreshold:
    # Function with CC=11 → (False, [...])

class TestVerifyPatchSandboxBlock:
    # changed_files=["server/app.py"] → verify_patch returns (False, "non-sandboxed")

class TestVerifyPatchSyntaxError:
    # Patch introduces a SyntaxError → verify_patch returns (False, "Syntax error")

class TestVerifyPatchFullPass:
    # Valid patch to an allowed file with CC ≤ 10 → (True, "ok")
```

**Manual — Baseline Complexity Scan:**

```bash
# Establish baseline before any meta-agent edits
source .venv/bin/activate
pip install radon
radon cc agent/ -s -n C --total-average
```

*Pass criteria:*
- [ ] `radon` runs without error
- [ ] Output lists at least one function with CC ≥ 10 (confirms scanner works)
- [ ] Record worst offenders in the K2.4 table above

---

## Updated Artifact Registry

| File | Status |
|------|--------|
| `agent/graph/karpathy_meta_agent.py` | 🔲 Not started |
| `agent/graph/complexity_gate.py` | 🔲 Not started |
| `scripts/verify_patch.py` | 🔲 Not started |
| `tests/test_reward_vector.py` | 🔲 Not started |
| `tests/test_complexity_gate.py` | 🔲 Not started |
