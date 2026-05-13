# Continual Harness — Refiner Loop & Macro Deprecation of `opener_bot.py`

**Document Status:** Implementation Blueprint
**Companion Document:** `HTN_MIGRATION_PLAN.md` (in-progress as of May 2026)
**Motivation:** *"Continual Harness: Online Adaptation for Self-Improving Foundation Agents"*
**Codebase snapshot:** May 13, 2026 (HTN Phases 0–4 complete, shadow mode live)

---

## Executive Summary

`opener_bot.py` is a 2,000-line handcrafted FSM with 25 states (S0–S24) that
guides the agent through Pokémon Emerald's intro sequence: title screen, Prof.
Oak's dialogue, character naming, the truck ride, the clock, May's house, Route
101, and starter selection. It is correct, brittle, and fundamentally at odds
with the agent's long-term goal of self-improving generalisation. It will never
generalise to new intro sequences, ROM hacks, or other games.

The **Continual Harness** architecture replaces it by letting the agent write its
own code. A new **Refiner** LangGraph node watches the agent's trajectory after
each milestone boundary, calls an LLM to synthesise a deterministic Python
function (a **Macro**) that reproduces the winning action sequence, then stores
that Macro persistently. A new **Macro Executor** specialist node runs verified
macros in a sandboxed `RestrictedExec` environment, producing button lists just
like the existing specialists (nav_bot, battle_bot, coms_bot).

`opener_bot.py` is deprecated — not deleted — in a phased migration that mirrors
the HTN `--use-htn` shadow-mode strategy. Until every intro milestone has a
verified, battle-tested Macro, `opener_bot.py` remains as a fallback and the
delta between its output and the macro's output is logged to
`llm_logs/refiner_shadow.jsonl` for offline analysis.

The end state: `opener_bot.py` is never imported; the agent navigates any fresh
`new_game.state` entirely through LLM-authored, sandbox-executed macros.

---

## Comparison to Paper Implementation

The paper's reference implementation (`sethkarten/continual-harness`,
`agents/utils/harness_evolver.py`) differs from this plan in four important ways.
Each divergence is intentional given our architecture, but should be understood
before implementation.

### Scope: Skills only (this plan) vs. 4-component evolution (paper)

The paper's `HarnessEvolver` runs **four independent passes** on every evolution
cycle:

1. **Prompt** — rewrite the LLM system prompt against failure signatures.
2. **Subagents** — CRUD on a subagent registry (sub-LLM specialists).
3. **Skills** (`K`) — CRUD on an executable Python skill library. ← *our plan*
4. **Memory** — fill gaps, refresh stale entries, rebalance importance.

This plan implements **Skills only** (`K`). That is the right scope for our
immediate goal (deprecating `opener_bot.py`). The other three components are
already handled by our existing architecture: the HTN Supervisor rewrites the
goal stack (serves the role of prompt evolution), the LangGraph specialists are
our sub-agents, and `EpisodicMemory` / ChromaDB is our memory layer.

A future `CONTINUAL_HARNESS_PHASE2.md` can extend the Refiner to evolve the
`SUPERVISOR_SYSTEM_PROMPT` and `REFINER_SYSTEM_PROMPT` themselves against run
trajectories, completing the full `{p, G, K, M}` loop.

### Skill execution model: per-frame (this plan) vs. synchronous tool-calls (paper)

The paper's skills execute as **inline sequential scripts** with access to a
`tools` dict whose calls block until the emulator responds:

```python
# Paper's skill model (sequential — each call waits for the game)
tools['press_buttons'](['A'], reasoning='advance dialogue')
state = tools['get_game_state']()
if state['location'] == 'MOVING_VAN':
    tools['press_buttons'](['UP', 'UP'], reasoning='navigate')
result = state
```

This solves the per-frame problem cleanly: the script just runs to completion,
blocking the caller until all actions are done. The paper uses this because its
scaffold is synchronous (one tool call → one server round-trip → next tool call).

Our LangGraph architecture is **not synchronous in this way**. Each
`graph.invoke()` is one game frame; `last_buttons` is a list that is fed to the
emulator for that frame; the graph exits and the emulator ticks. We cannot block
inside a graph node waiting for the emulator to respond to a button press.

**Our per-frame model is the correct design for our architecture.** A macro
returns buttons for the current frame and is called again next frame with updated
`state_data`. This is more constrained than the paper's model (you cannot write
`if previous_action_result == X`) but it is stateless, testable, and safe.

> **Implication for prompt writing:** The `REFINER_SYSTEM_PROMPT` must explain
> both the per-frame model AND provide a reference to how the paper's macros
> would be written differently. LLMs trained on the paper's codebase will try to
> write sequential scripts; the prompt must redirect them explicitly.

### `run_code` vs. `test_macro` — prototyping before saving

The paper provides a `run_code` tool that lets the agent execute arbitrary Python
interactively during gameplay to prototype code before saving it as a named
skill. This is a good idea and influences **Phase 2** of our plan:

- Our `test_macro()` meta-tool (Phase 2.3) runs a macro against pre-written test
  cases. This is the Refiner's equivalent — but it runs offline during the
  Refiner pass, not interactively during the agent's turn.
- We do **not** expose `run_code` as a live agent tool because our LangGraph
  specialists (nav_bot, etc.) already handle navigation; there is no need for
  the main agent to prototype code mid-turn. The Refiner writes and tests code
  between turns.

### Trigger model: auto-routing (this plan) vs. agent-initiated (paper)

In the paper, the agent calls `run_skill(skill_id, args)` itself — there is no
automatic trigger routing. The agent decides when to use a skill.

Our `routing_condition_with_macros()` evaluates trigger conditions on every
`graph.invoke()` and activates the macro automatically before the agent is
consulted. This is a **stronger guarantee**: the intro macros fire at the right
game state even if the agent's LLM would otherwise choose a different action.
For deterministic intro sequences this is correct. For mid-game skills (future
work), agent-initiated invocation (closer to the paper's model) may be
preferable.

### Evolution scheduling: event-driven (this plan) vs. adaptive periodic (paper)

The paper uses adaptive periodic scheduling: every 25 steps for the first 200
steps (bootstrap phase), then every 100 steps once stable. Our plan uses
event-driven triggers (milestone completion or macro failure).

Both are valid. Ours is more targeted (no wasted LLM calls on uneventful steps)
but could miss patterns that emerge gradually without a discrete milestone event.
For the intro sequence this is fine; the intro has clear milestone boundaries.
For future open-ended refinement, consider adding a periodic fallback
(`refiner_pending = True` every 100 steps regardless of events) as a supplement.

---

## Phase Review & Rationale

The user-proposed phases (Meta-Tools → Sandbox → Refiner → Deprecation) are
logically correct but need two additions to match the safety discipline of the
HTN migration:

1. **Phase 0: Data Structures** must precede everything — `MacroRecord`,
   `macro_store`, and the five new `AgentState` fields are depended on by all
   later phases, exactly as `GoalNode` preceded all HTN phases.

2. **Phase 4: Macro Executor Node** is a separate phase from Phase 3 (Refiner),
   because the Executor is a *plant controller* (runs code) and the Refiner is a
   *cognitive controller* (writes code). They have different trigger conditions,
   different prompts, and different failure modes. Conflating them would
   replicate the monolithic `opener_bot.py` anti-pattern in a new form.

Revised phase sequence:

| Phase | Name | Deliverable |
|---|---|---|
| **0** | Data Structures | `MacroRecord`, `macro_store`, 6 new `AgentState` fields |
| **1** | The Sandbox | `RestrictedExec`, AST import scanner, timeout guard |
| **2** | Meta-Tools | `write_macro`, `execute_macro`, `test_macro`, `get_trajectory` |
| **3** | The Refiner Node | LLM prompt, trajectory analysis, WRITE/EDIT/SKIP operations |
| **4** | Macro Executor Node | New LangGraph specialist, router changes, `--use-macros` flag |
| **5** | Opener Bot Deprecation | Shadow mode, intro macro library, `opener_bot.py` removal gate |

---

## System Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        AgentState (LangGraph)                              │
│  macro_store: dict[str, MacroRecord]  ← new; persisted to memory_db/      │
│  active_macro: Optional[str]          ← new; macro_name if executor active │
│  refiner_pending: bool                ← new; set by handoff_detector       │
│  trajectory_buffer: list[dict]        ← new; ring buffer of last 50 steps  │
│  refiner_last_operation: Optional[str]← new; WRITE_NEW/EDIT/SKIP           │
│  macro_last_error: Optional[str]      ← new; sandbox exception message     │
└──────────────────────┬─────────────────────────────────────────────────────┘
                       │
           graph.invoke(agent_state)
                       │
              ┌────────▼────────┐
              │    dispatch     │  routing_condition_with_macros()
              └──────┬──────────┘
         ┌───────────┼──────────────────────┐
         ▼           ▼                      ▼
    nav_bot      battle_bot            macro_executor  ← NEW specialist
    coms_bot     map_stitcher_relay         │
         │           │                     │
         └───────────┴──────────────────── ┤
                                           ▼
                                  handoff_detector
                                  (extended: sets refiner_pending
                                   on milestone completion or
                                   macro failure)
                                           │
                      ┌────────────────────┤
                      ▼ (supervisor_pending) ▼ (refiner_pending)
              executive_supervisor      refiner_node  ← NEW cognitive node
                      │                      │
                      └──────────┬───────────┘
                                 ▼
                           verification
                                 │
                                END
```

**Key design constraint — Refiner is NOT on the hot path.** The Refiner fires
only when `refiner_pending=True`, which is set by `handoff_detector_node` on
exactly two conditions:

1. A ROM milestone was just completed (`milestone_index` increased this step).
2. `macro_last_error` is set (the previous macro execution raised an exception).

All other steps: `refiner_pending=False`; the Refiner node is never invoked.
Per-frame latency is unchanged from the HTN baseline.

**Interaction with the Executive Supervisor.** Both the Supervisor and the
Refiner read from `handoff_detector`. When BOTH `supervisor_pending=True` AND
`refiner_pending=True`, the graph runs Supervisor first (it writes `goal_stack`),
then Refiner (it reads the updated stack to contextualise macro generation), then
`verification`. The graph topology encodes this ordering explicitly via a
chained edge: `handoff_detector → executive_supervisor → refiner_node →
verification`.

**`action.py` relationship.** `opener_bot.py` is called from `action.py` at
Priority 0B — *outside* the LangGraph graph. The Macro Executor is a *graph
node*. Until Phase 5 cutover, both co-exist: `action.py` calls opener_bot first;
if opener_bot returns an action, the graph still runs (for HTN bookkeeping) but
its `last_buttons` is overridden by opener_bot's output at the call-site in
`agent/__init__.py`. This is identical to how battle_bot_node and the legacy
battle_bot coexisted during the combat migration.

---

## Testing Eras

The table below answers: *"Which save state do I use to test each phase, and
does it break normal gameplay?"*

| Phase range | Who drives intro? | What fires on a macro trigger? | `new_game.state` usable? |
|---|---|---|---|
| **0–1** | `opener_bot.py` (unchanged) | `MacroRecord` and `RestrictedExec` exist but are never called | ✅ Yes — agent unchanged |
| **2** | `opener_bot.py` | Meta-tools exist; `execute_macro` can be called manually; `--use-macros` is **OFF** by default | ✅ Yes — intro identical |
| **3** | `opener_bot.py` | Refiner fires, logs to `llm_logs/refiner_shadow.jsonl`, but `write_macro` is in shadow mode — no macro is applied | ✅ Yes — agent unchanged |
| **4** | `opener_bot.py` (default) OR Macro Executor (with `--use-macros`) | Macro Executor runs; `active_macro` drives `last_buttons`; opener_bot skipped for matching segments | ✅ Yes — `--use-macros` off by default |
| **5.1** | Both in parallel (shadow comparison) | Both produce button lists; diff logged; opener_bot wins conflicts | ✅ Yes — shadow mode only |
| **5.2** | Macro Executor (full cutover) | opener_bot never invoked; all intro macros must be verified before this phase | ✅ Yes — this is the target state |

**Critical design constraint:**

> `macro_executor_node` MUST check `use_macros` at the factory level (Phase 4).
> If it executes unconditionally from Phase 2 onward, the sandbox runs unverified
> macros before the intro macro library is complete. The `--use-macros` flag
> belongs in `make_macro_executor_node()` at Phase 4.

**Available save states for testing:**

| State file | Player location | Opener bot state | Best for |
|---|---|---|---|
| `Emerald-GBAdvance/new_game.state` | TITLE_SEQUENCE | S0 | End-to-end intro test |
| `Emerald-GBAdvance/truck_start.state` | MOVING_VAN | S3 | Truck → mom dialog macros |
| `Emerald-GBAdvance/house_start_save.state` | LITTLEROOT_TOWN_PLAYERS_HOUSE_2F | S4 | Clock + house exit macros |
| `Emerald-GBAdvance/culdesac_start.state` | LITTLEROOT_TOWN (outside) | S9 | Neighbor + Route 101 macros |
| `Emerald-GBAdvance/route102_hackathon.state` | ROUTE_102 | N/A (post-intro) | Regression: Refiner must NOT activate mid-game |
| `Emerald-GBAdvance/start_open_ended.state` | LITTLEROOT_TOWN (post-lab) | COMPLETED | Regression: Refiner must NOT rewrite mid-game macros |

---

## Phase 0: Data Structures

**Purpose:** Define `MacroRecord`, `macro_store`, and the six new `AgentState`
fields that all later phases depend on. Nothing in Phases 1–5 can be built until
these exist and are tested.

### 0.1 `MacroRecord` — The Unit of the Macro Library

**File to create:** `agent/graph/macro_store.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class MacroRecord:
    """A single versioned macro in the persistent macro library.

    Attributes:
        name:                 snake_case identifier, e.g. "intro_set_clock".
        version:              Incremented on each Refiner edit.
        description:          One-line human-readable purpose.
        trigger_condition:    Python expression (evaluated against state_data)
                              that returns True when this macro should activate.
                              E.g.: 'player.get("location") == "MOVING_VAN"'
        completion_condition: Expression that returns True when the macro's
                              goal is met.  Used by macro_executor to know
                              when to deactivate.
        code:                 The BODY of the macro function as a string.
                              Signature is always:
                              def macro(state_data: dict) -> list[str]
                              Only the body is stored; the wrapper is added
                              at execution time.
        verified:             True only after all test_cases pass in RestrictedExec.
        created_step:         step_count when the Refiner first wrote this macro.
        last_edited_step:     step_count of the most recent Refiner EDIT operation.
        test_cases:           List of test result dicts from the most recent
                              test_macro() call.
        source_milestone:     The milestone ID whose completion triggered macro
                              creation (e.g. "STARTER_CHOSEN").
        metadata:             Arbitrary key-value store for Refiner annotations.
    """
    name: str
    description: str
    trigger_condition: str
    completion_condition: str
    code: str
    version: int = 1
    verified: bool = False
    created_step: int = 0
    last_edited_step: int = 0
    test_cases: list = field(default_factory=list)
    source_milestone: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Serialise to a JSON-compatible dict."""
        ...

    @classmethod
    def from_dict(cls, d: dict) -> "MacroRecord":
        """Deserialise from a previously serialised dict."""
        ...
```

### 0.2 `macro_store` Persistence Layer

`macro_store` in `AgentState` holds `dict[str, dict]` (serialised `MacroRecord`
objects). On startup, `Agent.__init__()` loads from `memory_db/macros.json` and
populates `self._macro_store`. The dict is shallow-copied into every
`graph.invoke()` call, exactly as `_htn_goal_stack` is copied.

```python
# In agent/__init__.py, alongside HTN field initialisation:
self._macro_store: dict = {}
_macros_path = Path("./memory_db/macros.json")
if _macros_path.exists():
    with _macros_path.open() as f:
        raw = json.load(f)
    self._macro_store = {
        k: MacroRecord.from_dict(v).to_dict() for k, v in raw.items()
    }
    print(f"   🔩  Macro Library: LOADED ({len(self._macro_store)} macros)")
else:
    print(f"   🔩  Macro Library: EMPTY (no macros.json found)")
```

`MacroRecord.to_dict()` / `.from_dict()` round-trip through plain Python dicts,
so the macro_store is JSON-serialisable and LangGraph-safe (no custom objects in
state).

### 0.3 `AgentState` Schema Changes

**File:** `agent/graph/state.py`

Add the following six fields to `AgentState`. All existing fields — including all
HTN fields — are **unchanged**.

```python
# ---- Continual Harness / Refiner ----
macro_store: dict
"""Dict[str, dict] mapping macro name → serialised MacroRecord.
Loaded from memory_db/macros.json at startup; updated by the Refiner node
and persisted back to disk after each Refiner WRITE_NEW or EDIT operation."""

active_macro: Optional[str]
"""Name of the macro currently being executed by macro_executor_node,
or None when no macro is active.  Set by routing_condition_with_macros()
when a verified macro's trigger_condition evaluates True."""

refiner_pending: bool
"""When True, refiner_node fires after executive_supervisor_node on the
current step.  Reset to False by the Refiner itself.  Set by
handoff_detector_node on milestone completion or macro failure."""

trajectory_buffer: list
"""Ring buffer of the last _TRAJECTORY_BUFFER_SIZE step snapshots.
Each entry: {step, location, position, milestone_index, last_action,
last_buttons, node_fired, macro_name_if_active}.
Capped at 50 entries; oldest entry evicted when full."""

refiner_last_operation: Optional[str]
"""The last operation issued by the Refiner: 'WRITE_NEW' | 'EDIT' |
'SKIP'.  Stored for observability and shadow-mode logging."""

macro_last_error: Optional[str]
"""The most recent exception message raised inside RestrictedExec.
Written by macro_executor_node on failure; cleared on successful
execution.  Used by handoff_detector_node to set refiner_pending=True."""
```

### Phase 0 Tests

**Automated — `tests/test_macro_store.py`:**

```python
class TestMacroRecordSerialization:
    # MacroRecord(...).to_dict() produces a dict with all expected keys
    # Round-trip preserves name, version, code, verified, trigger_condition

class TestMacroRecordDefaults:
    # MacroRecord with only required fields: verified=False, version=1, test_cases=[]
    # created_at > 0.0

class TestMacroStoreLoad:
    # Loading macros.json with one entry populates macro_store correctly
    # Missing macros.json → macro_store == {}
    # Malformed JSON → macro_store == {} (no crash)

class TestMacroStoreAgentStateFields:
    # AgentState constructed with macro_store={}, active_macro=None,
    #   refiner_pending=False, trajectory_buffer=[], refiner_last_operation=None,
    #   macro_last_error=None — no TypeError
    # All six new fields accept correct types

class TestTrajectoryBufferEviction:
    # Buffer of 50 entries with a 51st appended → first entry evicted
    # Buffer entries are plain dicts (JSON-serialisable)
```

**Manual — Phase 0 Schema Smoke Test:**

*Purpose:* Confirm `MacroRecord` serialisation and the new `AgentState` fields
do not crash on import.

*Command:*
```bash
PYTHONPATH=$PWD .venv/bin/python -c "
from agent.graph.macro_store import MacroRecord
r = MacroRecord(
    name='intro_title_screen',
    description='Press A to pass title screen',
    trigger_condition='state_data.get(\"player\",{}).get(\"location\") == \"TITLE_SEQUENCE\"',
    completion_condition='state_data.get(\"player\",{}).get(\"location\") != \"TITLE_SEQUENCE\"',
    code='return [\"A\"]',
)
print('MacroRecord:', r.name, 'version=', r.version)
print('Roundtrip:', MacroRecord.from_dict(r.to_dict()).name)
"
```

*Pass criteria:*
- [ ] No `ImportError` or `AttributeError`
- [ ] Prints `MacroRecord: intro_title_screen version= 1`
- [ ] Roundtrip prints `intro_title_screen`

---

## Phase 1: The Sandbox

**Purpose:** Before any LLM-generated code is ever stored or executed, we need a
hardened execution environment. `RestrictedExec` is the trust boundary: it
ensures that no macro — however malicious or buggy — can escape into the host
file system, network, or OS. This phase builds the sandbox in complete isolation,
with no LangGraph wiring, so it can be audited and tested independently.

**Security posture:** The threat model is not adversarial injection (the LLM is
trusted at the prompt level), but rather *accidental capability escalation*: the
LLM learns that `subprocess.run(["mgba", ...])` would restart the emulator, or
that `os.path.exists("memory_db/macros.json")` leaks path information. We
prevent this categorically rather than hoping the LLM stays in-bounds.

**File to create:** `agent/graph/restricted_exec.py`

### 1.1 AST Import Scanner

The first defence is compile-time: `ast.parse()` the code string before
`compile()` and walk the AST looking for `Import`, `ImportFrom`, or
`Call(func=Name(id="__import__"))` nodes. Any match raises
`ForbiddenImportError` immediately, before any bytecode is produced.

```python
import ast

class ForbiddenImportError(ValueError):
    """Raised when macro code contains a forbidden import statement."""

_FORBIDDEN_BUILTINS = frozenset({
    "__import__", "eval", "exec", "compile", "open", "input",
    "breakpoint", "exit", "quit", "vars", "dir",
})

# NOTE: time.sleep() and all other time-module functions are already
# unreachable because `import time` is blocked at the AST level.
# The forbidden-builtins list is a second line of defence for builtins
# that need no import (e.g. eval, exec).  Do NOT add 'sleep' here —
# it is not a builtin; the import block is sufficient.

def _scan_ast_for_imports(code: str, macro_name: str) -> None:
    """Walk the AST and raise ForbiddenImportError on any import node.

    Also raises on calls to forbidden builtins (eval, exec, open, etc.).
    Raises SyntaxError if the code cannot be parsed.
    """
    tree = ast.parse(code, filename=f"<macro:{macro_name}>")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ForbiddenImportError(
                f"Macro '{macro_name}' contains a forbidden import statement."
            )
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.id if isinstance(func, ast.Name) else
                func.attr if isinstance(func, ast.Attribute) else None
            )
            if name in _FORBIDDEN_BUILTINS:
                raise ForbiddenImportError(
                    f"Macro '{macro_name}' calls forbidden builtin '{name}'."
                )
```

### 1.2 `RestrictedExec` — Sandboxed Execution

```python
import threading

_SANDBOX_TIMEOUT_SECONDS = 2.0

_ALLOWED_BUILTINS: dict = {
    # Safe type constructors
    "list": list, "dict": dict, "set": set, "tuple": tuple,
    "str": str, "int": int, "float": float, "bool": bool,
    # Safe iteration
    "len": len, "range": range, "enumerate": enumerate,
    "zip": zip, "sorted": sorted, "reversed": reversed,
    "map": map, "filter": filter,
    # Safe inspection
    "isinstance": isinstance, "hasattr": hasattr, "getattr": getattr,
    # Safe math
    "abs": abs, "min": min, "max": max, "sum": sum, "round": round,
    # Safe printing (for debug macros)
    "print": print,
    # Constants
    "None": None, "True": True, "False": False,
}

# Standard library modules pre-imported and injected as top-level names
# in the sandbox globals.  These match what the paper's skill sandbox
# allows (harness_evolver.py: 'collections', 'heapq', 'numpy', 'json',
# 're', 'math', 'random') minus numpy (not always available, not needed
# for button sequences).
import collections as _collections
import heapq as _heapq
import json as _json
import math as _math
import random as _random
import re as _re

_SANDBOX_STDLIB: dict = {
    "collections": _collections,
    "heapq":       _heapq,
    "json":        _json,
    "math":        _math,
    "random":      _random,
    "re":          _re,
}
# These are injected into restricted_globals alongside __builtins__:
#   restricted_globals = {"__builtins__": _ALLOWED_BUILTINS,
#                         **_SANDBOX_STDLIB, ...}
# Macros may use e.g. math.floor(), re.match(), json.loads() freely.
# numpy is intentionally excluded: overkill for button sequences, adds
# a large import cost, and would give macros array broadcast ops that
# are hard to reason about in a safety review.


class MacroTimeoutError(RuntimeError):
    """Raised when a macro exceeds _SANDBOX_TIMEOUT_SECONDS."""


class RestrictedExec:
    """Execute LLM-generated macro code in a restricted environment.

    Usage:
        sandbox = RestrictedExec()
        buttons = sandbox.run(macro_name, code_body, state_data_snapshot)

    The 'code_body' is the BODY of a function with signature:
        def macro(state_data: dict) -> list[str]
    Only the body is stored; RestrictedExec wraps it in a minimal def
    before compiling.

    Security guarantees:
        • Imports are blocked at AST level (ForbiddenImportError).
        • Only whitelisted builtins are visible to the macro.
        • state_data is passed as a deep-copy — mutations do not
          propagate to AgentState.
        • Execution is killed after _SANDBOX_TIMEOUT_SECONDS using a
          daemon thread; the macro cannot block the run loop.

    Return contract:
        Returns list[str] on success.
        Raises MacroTimeoutError, ForbiddenImportError, or any exception
        raised inside the macro body.  Callers must catch all exceptions.
    """

    def run(
        self,
        macro_name: str,
        code_body: str,
        state_data: dict,
        timeout: float = _SANDBOX_TIMEOUT_SECONDS,
    ) -> list:
        """Run macro_name's code_body with state_data snapshot.
        Returns list of GBA button strings (e.g. ["A", "A", "UP"]).
        """
        import copy

        # 1. AST scan — raises ForbiddenImportError on any import node
        _scan_ast_for_imports(code_body, macro_name)

        # 2. Wrap body in a function and compile
        wrapped = f"def _macro(state_data):\n"
        wrapped += "\n".join(f"    {line}" for line in code_body.splitlines())
        wrapped += "\n__result__ = _macro(__state_data__)\n"

        try:
            bytecode = compile(wrapped, f"<macro:{macro_name}>", "exec")
        except SyntaxError as e:
            raise SyntaxError(f"Macro '{macro_name}' has invalid syntax: {e}") from e

        # 3. Restricted globals — only whitelisted builtins visible
        restricted_globals = {
            "__builtins__": _ALLOWED_BUILTINS,
            "__state_data__": copy.deepcopy(state_data),
        }
        local_ns: dict = {}

        # 4. Run with timeout
        result_holder: list = []
        error_holder: list = []

        def _target():
            try:
                exec(bytecode, restricted_globals, local_ns)
                result_holder.append(local_ns.get("__result__", []))
            except Exception as exc:
                error_holder.append(exc)

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(timeout)

        if t.is_alive():
            raise MacroTimeoutError(
                f"Macro '{macro_name}' exceeded timeout of {timeout}s."
            )
        if error_holder:
            raise error_holder[0]

        buttons = result_holder[0] if result_holder else []

        # 5. Validate output type — must be a list of strings
        if not isinstance(buttons, list):
            raise TypeError(
                f"Macro '{macro_name}' returned {type(buttons).__name__}, expected list."
            )
        for btn in buttons:
            if not isinstance(btn, str):
                raise TypeError(
                    f"Macro '{macro_name}' returned a non-string button: {btn!r}."
                )

        return buttons
```

### 1.3 `trigger_condition` Evaluator

Macros have a `trigger_condition` string (e.g.
`'state_data.get("player",{}).get("location") == "MOVING_VAN"'`). These are
short expressions evaluated by `routing_condition_with_macros()` to decide
whether to activate a macro. They use the same sandbox logic but run via
`eval()` instead of `exec()`:

```python
def evaluate_trigger(condition_expr: str, state_data: dict) -> bool:
    """Safely evaluate a macro trigger condition expression.

    The expression may only reference 'state_data' and the allowed builtins.
    Returns False on any exception (malformed condition = do not activate).
    """
    import copy
    try:
        _scan_ast_for_imports(condition_expr, "<trigger>")
        return bool(eval(  # noqa: S307  (this eval is intentional and sandboxed)
            condition_expr,
            {"__builtins__": _ALLOWED_BUILTINS,
             "state_data": copy.deepcopy(state_data)},
        ))
    except Exception:
        return False
```

### Phase 1 Tests

**Automated — `tests/test_restricted_exec.py`:**

```python
class TestAllowedCode:
    # Simple list return: code="return ['A', 'B']" → ['A', 'B']
    # Reads from state_data: code="return ['A'] if state_data.get('x') else []"
    # Empty return: code="return []" → []
    # Uses allowed builtins (len, range, isinstance, etc.) without error
    # Uses stdlib modules: math.floor(1.5) → 1, re.match(r'\w+', 'A'), json.dumps({})
    # Uses collections.deque without error (module injected into sandbox globals)

class TestForbiddenImports:
    # "import os" → ForbiddenImportError
    # "from subprocess import run" → ForbiddenImportError
    # "__import__('os')" → ForbiddenImportError
    # "import sys; sys.exit()" → ForbiddenImportError

class TestForbiddenBuiltins:
    # "eval('1+1')" → ForbiddenImportError
    # "exec('pass')" → ForbiddenImportError
    # "open('/etc/passwd')" → ForbiddenImportError

class TestTimeout:
    # Infinite loop: "while True: pass" → MacroTimeoutError within 2.5s
    # For-loop that never exits: same

class TestNoImplicitSleepPath:
    # 'import time' → ForbiddenImportError (import scan)
    # 'import threading' → ForbiddenImportError
    # Code that never returns (busy-wait) → MacroTimeoutError
    # NOTE: time.sleep() has no path to execution — import is blocked;
    #       test confirms this at the import level, not the call level

class TestStateMutation:
    # Macro modifies state_data dict; original state_data is unchanged (deep copy)

class TestOutputValidation:
    # Returns dict instead of list → TypeError
    # Returns list containing non-string → TypeError
    # Returns None → returns [] (normalised by executor, not by sandbox)

class TestSyntaxError:
    # "return [" → SyntaxError raised on compile
    # "::" → SyntaxError raised

class TestTriggerEvaluator:
    # Valid True expression → True
    # Valid False expression → False
    # Expression with import → False (no exception, returns False)
    # Expression that raises AttributeError → False
```

**Manual — Sandbox Smoke Test:**

*Command:*
```bash
PYTHONPATH=$PWD .venv/bin/python -c "
from agent.graph.restricted_exec import RestrictedExec, ForbiddenImportError
sb = RestrictedExec()

# Safe macro
code = 'loc = state_data.get(\"player\",{}).get(\"location\",\"\")\nreturn [\"A\"] if loc == \"TITLE_SEQUENCE\" else []'
buttons = sb.run('test_macro', code, {'player': {'location': 'TITLE_SEQUENCE'}})
print('Safe result:', buttons)

# Forbidden import
try:
    sb.run('bad', 'import os\nreturn []', {})
    print('ERROR: import os was not blocked!')
except ForbiddenImportError as e:
    print('Correctly blocked:', e)
"
```

*Pass criteria:*
- [ ] `Safe result: ['A']`
- [ ] `Correctly blocked: Macro 'bad' contains a forbidden import statement.`

---

## Phase 2: Meta-Tools

**Purpose:** Build the four primitive operations that the Refiner and Macro
Executor will use: `write_macro`, `execute_macro`, `test_macro`, and
`get_trajectory`. These are pure functions (no LangGraph coupling), thin wrappers
over `RestrictedExec` and `MacroRecord`, and are fully testable in isolation.

**File to create:** `agent/graph/meta_tools.py`

### 2.1 `write_macro`

```python
def write_macro(
    name: str,
    description: str,
    trigger_condition: str,
    completion_condition: str,
    code: str,
    source_milestone: Optional[str],
    current_step: int,
    macro_store: dict,
) -> tuple[dict, str]:
    """Create or update a MacroRecord in macro_store.

    Does NOT verify the macro (verified=False until test_macro() passes).
    Does NOT persist to disk (caller is responsible for saving macros.json).

    Returns:
        (updated_macro_store, operation)
        operation is 'WRITE_NEW' for new macros, 'EDIT' for version bumps.
    """
    existing = macro_store.get(name)
    if existing:
        rec = MacroRecord.from_dict(existing)
        rec.version += 1
        rec.code = code
        rec.description = description
        rec.trigger_condition = trigger_condition
        rec.completion_condition = completion_condition
        rec.verified = False   # must re-verify after any edit
        rec.last_edited_step = current_step
        operation = "EDIT"
    else:
        rec = MacroRecord(
            name=name,
            description=description,
            trigger_condition=trigger_condition,
            completion_condition=completion_condition,
            code=code,
            source_milestone=source_milestone,
            created_step=current_step,
            last_edited_step=current_step,
        )
        operation = "WRITE_NEW"

    updated_store = {**macro_store, name: rec.to_dict()}
    return updated_store, operation
```

### 2.2 `execute_macro`

```python
def execute_macro(
    name: str,
    macro_store: dict,
    state_data: dict,
    sandbox: RestrictedExec,
) -> tuple[list, Optional[str]]:
    """Run a named macro in the sandbox.

    Returns:
        (buttons, error_message)
        buttons is the list of GBA button strings on success (may be empty).
        error_message is None on success; contains the exception str on failure.
    """
    rec_dict = macro_store.get(name)
    if rec_dict is None:
        return [], f"Macro '{name}' not found in macro_store."
    rec = MacroRecord.from_dict(rec_dict)
    if not rec.verified:
        return [], f"Macro '{name}' is not verified (version={rec.version}). Run test_macro() first."
    try:
        buttons = sandbox.run(name, rec.code, state_data)
        return buttons, None
    except Exception as exc:
        return [], str(exc)
```

> **Why require `verified=True` before execution?**
> An unverified macro is LLM output that has never run against any test case.
> Running it directly on a live emulator risks producing garbled input (e.g.
> pressing START on the nickname screen and corrupting the character name).
> The verification gate is the Refiner's responsibility; the Executor only
> runs macros that have passed all test cases.

### 2.3 `test_macro`

```python
def test_macro(
    name: str,
    test_cases: list[dict],
    macro_store: dict,
    sandbox: RestrictedExec,
) -> tuple[dict, bool]:
    """Run a macro against a set of test cases and update its verified status.

    Each test case is:
        {
            "description":      "<what the state represents>",
            "state_data":       {...},           # input snapshot
            "expected_buttons": ["A", "UP"],     # expected output
        }

    A test case passes when actual_buttons == expected_buttons.
    ALL test cases must pass for verified=True to be set.

    Returns:
        (updated_macro_store, all_passed)
    """
    rec_dict = macro_store.get(name)
    if rec_dict is None:
        raise KeyError(f"Macro '{name}' not found in macro_store.")
    rec = MacroRecord.from_dict(rec_dict)

    results = []
    all_passed = True
    for case in test_cases:
        try:
            actual = sandbox.run(name, rec.code, case["state_data"])
            passed = actual == case["expected_buttons"]
        except Exception as exc:
            actual = []
            passed = False
            results.append({**case, "actual_buttons": actual,
                             "passed": passed, "error": str(exc)})
            all_passed = False
            continue
        results.append({**case, "actual_buttons": actual, "passed": passed})
        if not passed:
            all_passed = False

    rec.test_cases = results
    rec.verified = all_passed
    updated_store = {**macro_store, name: rec.to_dict()}
    return updated_store, all_passed
```

### 2.4 `get_trajectory`

```python
def get_trajectory(
    trajectory_buffer: list,
    n: int = 20,
) -> list[dict]:
    """Return the last n entries from the trajectory ring buffer.

    Each entry is a plain dict with keys:
        step, location, position, milestone_index,
        last_action, last_buttons, node_fired, active_macro.

    Returns at most min(n, len(trajectory_buffer)) entries.
    """
    return trajectory_buffer[-n:] if len(trajectory_buffer) > n else list(trajectory_buffer)
```

`trajectory_buffer` is populated by `handoff_detector_node` (Phase 1) on every
step — it appends one entry and pops the oldest when `len > 50`. The trajectory
is the Refiner's primary context signal.

### 2.5 Macro Persistence Helpers

**Runtime source of truth — `macros.json`:**

```python
def save_macro_store(macro_store: dict, path: str = "./memory_db/macros.json") -> None:
    """Persist macro_store to disk atomically (write-then-rename).

    macros.json is the runtime source of truth: it is loaded at startup,
    passed through AgentState, and updated by the Refiner node.
    All macro code lives here as strings — not as importable .py files —
    so there is no risk of the macro library being accidentally imported
    by Python's module system.
    """
    import json, pathlib, tempfile
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=p.parent, delete=False,
                                     suffix=".tmp") as f:
        json.dump(macro_store, f, indent=2)
        tmp_path = f.name
    pathlib.Path(tmp_path).replace(p)
    # Also dump .py sidecars for human inspection
    _dump_macro_sidecars(macro_store, p.parent / "macros")
```

> **Why atomic write?** `macros.json` is read at startup. A crash mid-write
> would corrupt the file and lose all macros. Write-then-rename is atomic on
> POSIX file systems.

**Human-readable `.py` sidecars — for debugging only:**

```python
def _dump_macro_sidecars(macro_store: dict, sidecar_dir: "pathlib.Path") -> None:
    """Write each macro's code to a .py file for syntax-highlighted inspection.

    These files are NEVER imported by the agent — they exist solely so
    developers can open memory_db/macros/intro_set_clock.py in an editor
    and see the LLM-generated code with full syntax highlighting and
    static analysis (Pylance, etc.).

    The files are regenerated from macros.json on every save, so they are
    always in sync. Never edit them directly — edits will be overwritten.
    """
    import pathlib
    sidecar_dir = pathlib.Path(sidecar_dir)
    sidecar_dir.mkdir(parents=True, exist_ok=True)

    # Write a README so no one mistakes these for importable modules
    readme = sidecar_dir / "README.txt"
    readme.write_text(
        "These .py files are generated from memory_db/macros.json for\n"
        "human inspection only. Do NOT import them. Do NOT edit them —\n"
        "they will be overwritten on the next Refiner write.\n"
    )

    for name, rec_dict in macro_store.items():
        rec = MacroRecord.from_dict(rec_dict)
        py_path = sidecar_dir / f"{name}.py"
        header = (
            f'# Macro: {rec.name}  (v{rec.version})\n'
            f'# Description: {rec.description}\n'
            f'# Verified: {rec.verified}\n'
            f'# Source milestone: {rec.source_milestone}\n'
            f'# Trigger:     {rec.trigger_condition}\n'
            f'# Completion:  {rec.completion_condition}\n'
            f'# DO NOT IMPORT — for human inspection only\n'
            f'\n'
            f'def macro(state_data: dict) -> list:\n'
        )
        indented_body = "\n".join(
            f"    {line}" for line in rec.code.splitlines()
        )
        py_path.write_text(header + indented_body + "\n")
```

> **Why not use `.py` as the primary storage?** Two reasons. First, having
> both `macros.json` and `memory_db/macros/*.py` as co-equal sources of
> truth creates a sync problem — a crash between the two writes would leave
> them inconsistent. JSON is one file, one atomic write. Second, `.py` files
> in a directory on `sys.path` (or auto-discovered by pytest) can be
> accidentally imported, which would execute LLM-generated code at import
> time. The sidecar directory gets a `README.txt` (not an `__init__.py`),
> making it invisible to the Python module system.
>
> **For debugging:** `cat memory_db/macros/intro_set_clock.py` or open the
> file in VS Code. Pylance will syntax-check it and highlight errors even
> though it is never run from there.

### Phase 2 Tests

**Automated — `tests/test_meta_tools.py`:**

```python
class TestWriteMacroNew:
    # write_macro with new name → operation='WRITE_NEW', macro in store, verified=False

class TestWriteMacroEdit:
    # write_macro on existing name → operation='EDIT', version bumped to 2, verified=False

class TestWriteMacroVersionBump:
    # Three edits → version == 4

class TestExecuteMacroUnverified:
    # execute_macro on unverified macro → ([], "not verified") error

class TestExecuteMacroSuccess:
    # Verified macro with correct code → (expected_buttons, None)

class TestExecuteMacroSandboxError:
    # Macro with infinite loop in store → ([], "exceeded timeout...")

class TestExecuteMacroNotFound:
    # execute_macro with unknown name → ([], "not found") error

class TestTestMacroAllPass:
    # All test cases match → verified=True, all results have passed=True

class TestTestMacroOneFail:
    # One case fails → verified=False, failing case has passed=False

class TestTestMacroSandboxException:
    # Test case raises ForbiddenImportError → passed=False, error field set

class TestGetTrajectory:
    # Buffer of 30 entries, n=20 → last 20 entries
    # Buffer of 10 entries, n=20 → all 10 entries
    # Empty buffer → []

class TestSaveMacroStore:
    # Save then load from temp path → identical dict
    # Concurrent write (two saves) → no file corruption (rename atomicity)
    # After save, memory_db/macros/<name>.py sidecar exists
    # Sidecar contains the macro code body, properly indented under def macro():
    # Sidecar directory contains README.txt (no __init__.py → not importable)
    # Sidecar is regenerated (overwritten) on second save with edited code
```

---

## Phase 3: The Refiner Node

**Purpose:** Build the LLM cognitive node that reads the trajectory buffer, the
completed milestone, and the current macro library, then decides whether to write
a new macro, edit an existing one, or skip. This is the largest single phase and
the core of the Continual Harness architecture.

### 3.1 Trigger Conditions

The Refiner fires when `refiner_pending=True`, set by `handoff_detector_node`
(extended in this phase) on:
1. **Milestone completion**: `milestone_index` increased this step (the agent
   just achieved something worth codifying).
2. **Macro failure**: `macro_last_error` is set (sandbox raised an exception;
   the macro needs editing).
3. **First step on a fresh intro save state**: `step_count == 0` AND
   `state_data["player"]["location"] == "TITLE_SEQUENCE"` (bootstrap the intro
   macro library).

It does NOT fire on:
- Nav-stall (that's the Supervisor's domain).
- Battle transitions (the Refiner has no battle macros — `battle_bot_node` is
  already near-deterministic).
- Steps where `node_fired` is `"macro_executor"` and `macro_last_error` is None
  (successful macro execution — no refinement needed).

### 3.2 `make_refiner_node` Factory

**File to create:** `agent/graph/nodes/refiner.py`

```python
from __future__ import annotations
import json, logging, pathlib
from typing import Callable

from agent.graph.state import AgentState
from agent.graph.meta_tools import (
    write_macro, test_macro, save_macro_store, get_trajectory,
)
from agent.graph.restricted_exec import RestrictedExec
from agent.graph.macro_store import MacroRecord

logger = logging.getLogger(__name__)

_TRAJECTORY_BUFFER_SIZE = 50
_SANDBOX = RestrictedExec()   # singleton; stateless

def make_refiner_node(
    vlm,
    use_macros: bool = False,   # Phase 4: set True to activate macro execution
) -> Callable[[AgentState], AgentState]:
    """Factory returning the refiner_node closure."""

    def refiner_node(state: AgentState) -> AgentState:
        if not state.get("refiner_pending", False):
            return {**state, "refiner_pending": False}

        trajectory = get_trajectory(state.get("trajectory_buffer", []), n=30)
        macro_store = state.get("macro_store", {})
        macro_error = state.get("macro_last_error")
        milestone_idx = state.get("milestone_index", 0)
        state_data = state.get("state_data", {})

        # Determine trigger type
        if macro_error:
            trigger_type = "MACRO_FAILURE"
            failed_macro_name = state.get("active_macro")
        else:
            trigger_type = "MILESTONE_COMPLETE"
            failed_macro_name = None

        # Call the LLM
        operation_result = _call_refiner_llm(
            vlm=vlm,
            trigger_type=trigger_type,
            trajectory=trajectory,
            macro_store=macro_store,
            state_data=state_data,
            failed_macro_name=failed_macro_name,
            milestone_idx=milestone_idx,
        )

        operation = operation_result.get("operation", "SKIP")
        reasoning = operation_result.get("reasoning", "")[:500]

        updated_macro_store = macro_store
        if operation in ("WRITE_NEW", "EDIT"):
            macro_def = operation_result.get("macro", {})
            if macro_def:
                updated_macro_store, _ = write_macro(
                    name=macro_def["name"],
                    description=macro_def["description"],
                    trigger_condition=macro_def["trigger_condition"],
                    completion_condition=macro_def["completion_condition"],
                    code=macro_def["code"],
                    source_milestone=state_data.get("milestones", {}).get("last_completed"),
                    current_step=state.get("step_count", 0),
                    macro_store=macro_store,
                )
                # Auto-test with the LLM-provided test cases
                test_cases = macro_def.get("test_cases", [])
                if test_cases:
                    updated_macro_store, passed = test_macro(
                        macro_def["name"], test_cases, updated_macro_store, _SANDBOX
                    )
                    if not passed:
                        logger.warning(
                            "[REFINER] Macro '%s' failed test cases — verified=False.",
                            macro_def["name"],
                        )

                # Shadow mode: log but do not activate unless use_macros=True
                _write_refiner_log(state, operation, macro_def, reasoning)

                if operation == "WRITE_NEW" or operation == "EDIT":
                    save_macro_store(updated_macro_store)
                    logger.info(
                        "[REFINER] %s '%s' v%d — verified=%s",
                        operation,
                        macro_def["name"],
                        MacroRecord.from_dict(
                            updated_macro_store[macro_def["name"]]
                        ).version,
                        MacroRecord.from_dict(
                            updated_macro_store[macro_def["name"]]
                        ).verified,
                    )

        return {
            **state,
            "macro_store": updated_macro_store,
            "refiner_pending": False,
            "refiner_last_operation": operation,
            "macro_last_error": None,    # clear after Refiner has processed it
        }

    return refiner_node
```

### 3.3 Prompt Templates

#### System Prompt

```python
REFINER_SYSTEM_PROMPT = """\
You are the Refiner for an autonomous Pokémon Emerald AI agent.

You receive:
  1. A TRIGGER explaining WHY you were invoked (milestone completed or macro failed).
  2. A TRAJECTORY of the last 30 steps: each entry shows the location, buttons
     pressed, which node handled it, and the step number.
  3. The CURRENT MACRO LIBRARY: existing macros with their trigger conditions
     and verification status.
  4. The CURRENT GAME STATE: location, milestones, party.

Your job is to decide ONE operation:
  - WRITE_NEW : The trajectory shows a repeatable pattern with no existing macro.
                Write a new deterministic Python function to reproduce it.
  - EDIT      : An existing macro failed (macro_last_error is set) or the
                trajectory reveals a better implementation of an existing macro.
  - SKIP      : No actionable pattern found, or the pattern is too complex/
                context-dependent to encode deterministically.

PER-FRAME EXECUTION MODEL — read this before writing any code:
  The macro is called ONCE PER GAME FRAME (once per graph.invoke()). It is NOT
  a script that runs to completion. On each call it receives the CURRENT
  state_data, returns buttons for THAT SINGLE FRAME, and exits. The framework
  calls it again next frame with updated state_data.

  CORRECT pattern (stateless, adapts each frame):
    loc = state_data.get('player', {}).get('location', '')
    if loc == 'MOVING_VAN':
        return ['A']
    return []

  WRONG pattern (sequential script — this does not work):
    press('A')          # ← not a valid function
    time.sleep(2)       # ← import time is blocked AND sleep freezes the loop
    press('START')      # ← second action never reached; macro returns after frame 1

  WRONG pattern (returning a long sequence expecting it to play out):
    return ['A','A','A','UP','A','A']  # ← all 6 buttons fire in ONE frame,
                                       #   skipping any state-change checks.
                                       #   Only do this for ultra-tight sequences
                                       #   where state cannot change mid-burst.

PYTHON CODE CONSTRAINTS — the macro body MUST obey these rules absolutely:
  1. NO import statements of any kind. The sandbox pre-injects these safe
     standard library modules as top-level names; use them directly:
       math   — math.floor(), math.abs(), math.sqrt(), etc.
       re     — re.match(), re.search(), re.sub(), etc.
       json   — json.loads(), json.dumps() for parsing state fields
       collections — collections.defaultdict(), collections.Counter(), etc.
       heapq  — heapq.heappush(), etc.
       random — random.choice(), random.randint(), etc. (use sparingly)
     DO NOT write 'import math' — math is already available.
  2. NO calls to eval, exec, open, __import__, or any OS/subprocess function.
  3. NO time.sleep(), NO loops that wait for state to change between iterations.
     This macro architecture is DIFFERENT from the paper's reference implementation.
     In the paper, skills call tools['press_buttons']() synchronously. Here,
     macros return a button list for ONE FRAME and are called again next frame.
     Waiting is done by returning [] (no-op) until the condition you need is met.
     The framework calls the macro again next frame with updated state_data.
  4. The only input is 'state_data: dict' (read-only game state snapshot).
  5. The output is 'return <list of button strings>' for THIS FRAME.
     Button names: "A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT".
  6. Return [] (empty list) when the macro has no action for the current state
     (e.g. when its completion_condition is already met, or waiting for a
     dialogue box to clear before pressing A).
  7. Deterministic: given the same state_data, always return the same buttons.
  8. Maximum ~20 lines of code. If the logic is more complex, SKIP.

TRIGGER CONDITIONS AND COMPLETION CONDITIONS:
  - Use only state_data fields: player.location, player.position, milestones,
    game.in_battle, game.in_dialog, party.
  - Example: 'state_data.get("player",{}).get("location","") == "MOVING_VAN"'

OUTPUT FORMAT — respond with ONLY a JSON object:
{
  "operation": "WRITE_NEW" | "EDIT" | "SKIP",
  "reasoning": "<one sentence>",
  "macro": {                               // required for WRITE_NEW or EDIT
    "name":                 "<snake_case>",
    "description":          "<one line>",
    "trigger_condition":    "<Python expression>",
    "completion_condition": "<Python expression>",
    "code":                 "<multi-line Python body>",
    "test_cases": [
      {
        "description":      "<what this state represents>",
        "state_data":       {"player": {"location": "..."}},
        "expected_buttons": ["A"]
      }
    ]
  }
}

RULES:
1. Only issue WRITE_NEW for patterns that have succeeded 3+ consecutive times
   in the trajectory. Do not write macros for one-off events.
2. EDIT must reference an existing macro by its exact name from the library.
3. For MACRO_FAILURE triggers, always issue EDIT (not SKIP) unless the failure
   is caused by a game state the macro should never handle — in that case SKIP
   and explain in reasoning.
4. Macro names must be prefixed with 'intro_' for intro-sequence macros, so
   they can be batch-deactivated after STARTER_CHOSEN is set.
5. Return ONLY the JSON. No prose before or after.
"""
```

#### User Prompt Template

```python
REFINER_USER_TEMPLATE = """\
=== TRIGGER ===
Type       : {trigger_type}
Reason     : {trigger_reason}
Failed Macro: {failed_macro_name}
Milestone  : {milestone_id} (index {milestone_idx})

=== TRAJECTORY (last {n_steps} steps) ===
{trajectory_repr}

=== CURRENT MACRO LIBRARY ===
{macro_library_repr}

=== CURRENT GAME STATE ===
Location   : {current_location}
Position   : ({pos_x}, {pos_y})
In Battle  : {in_battle}
In Dialogue: {in_dialogue}
Milestones : {completed_milestones}
Step Count : {step_count}

What operation should be performed?
"""
```

### 3.4 LLM Call Helper

```python
def _call_refiner_llm(
    vlm,
    trigger_type: str,
    trajectory: list,
    macro_store: dict,
    state_data: dict,
    failed_macro_name: Optional[str],
    milestone_idx: int,
) -> dict:
    """Call the VLM with the Refiner prompt. Returns parsed operation dict.

    Uses vlm.get_json_query() (added in HTN Phase 3).  Falls back to SKIP
    on any exception or JSON parse error.
    """
    ...  # Format prompts, call vlm.get_json_query(), parse JSON, validate

def _write_refiner_log(
    state: AgentState,
    operation: str,
    macro_def: dict,
    reasoning: str,
) -> None:
    """Append a shadow-mode log entry to llm_logs/refiner_shadow.jsonl."""
    ...
```

### 3.5 `trajectory_buffer` Population

Extend `handoff_detector_node` (Phase 1 of HTN) to append one entry to
`trajectory_buffer` per step and evict the oldest when `len > 50`:

```python
# Inside handoff_detector_node, after all existing logic:
entry = {
    "step":           state.get("step_count", 0),
    "location":       (state.get("state_data") or {}).get("player", {}).get("location", ""),
    "position":       (state.get("state_data") or {}).get("player", {}).get("position", {}),
    "milestone_index": state.get("milestone_index", 0),
    "last_action":    state.get("last_action"),
    "last_buttons":   state.get("last_buttons", []),
    "node_fired":     current_node_name,
    "active_macro":   state.get("active_macro"),
}
buffer = list(state.get("trajectory_buffer", []))
buffer.append(entry)
if len(buffer) > _TRAJECTORY_BUFFER_SIZE:
    buffer = buffer[-_TRAJECTORY_BUFFER_SIZE:]
```

Also extend `handoff_detector_node` to set `refiner_pending=True` when:

```python
milestone_increased = (
    state.get("milestone_index", 0) > previous_milestone_index
)
macro_failed = bool(state.get("macro_last_error"))
refiner_should_fire = milestone_increased or macro_failed
```

### Phase 3 Tests

**Automated — `tests/test_refiner_node.py`:**

```python
class TestRefinerSkipsWhenNotPending:
    # refiner_pending=False → node returns immediately, refiner_last_operation unchanged

class TestRefinerSkipOperation:
    # LLM returns operation='SKIP' → macro_store unchanged, refiner_pending=False

class TestRefinerWriteNew:
    # LLM returns WRITE_NEW with valid macro → macro in store, verified=True if tests pass

class TestRefinerWriteNewTestsFail:
    # LLM returns WRITE_NEW but test cases fail → macro in store, verified=False

class TestRefinerEdit:
    # Existing macro in store, LLM returns EDIT → version bumped, verified updated

class TestRefinerMalformedLLMResponse:
    # Invalid JSON from LLM → operation=SKIP, no crash

class TestRefinerNetworkError:
    # vlm.get_json_query raises Exception → operation=SKIP, refiner_pending=False

class TestRefinerShadowLog:
    # WRITE_NEW operation → entry appended to llm_logs/refiner_shadow.jsonl

class TestRefinerMacroErrorClearedAfterProcessing:
    # macro_last_error set before call → macro_last_error=None after call

class TestRefinerMacroStorePersistedOnWrite:
    # WRITE_NEW → save_macro_store() called (monkeypatched); macros.json updated

class TestRefinerTrajectoryBufferPopulation:
    # handoff_detector_node called with milestone_index increased → refiner_pending=True
    # handoff_detector_node called with macro_last_error set → refiner_pending=True
    # Normal nav_bot step with no changes → refiner_pending=False

class TestRefinerTrajectoryBufferCap:
    # 51 calls to handoff_detector_node → trajectory_buffer has exactly 50 entries
```

**Manual — Refiner Shadow Smoke Test:**

*Purpose:* Confirm the Refiner fires on the first milestone completion
(`TRUCK_ARRIVED` or `ARRIVED_HOME`) when running `new_game.state` and writes a
`intro_` prefixed macro to `llm_logs/refiner_shadow.jsonl`. Navigation must be
unchanged (opener_bot still drives, `--use-macros` is off).

*Command:*
```bash
python run.py --load-state Emerald-GBAdvance/new_game.state --agent-auto
```

*Observe in console (print added to refiner_node):*
```
[REFINER] step=N  MILESTONE_COMPLETE  → WRITE_NEW 'intro_truck_ride_buttons'
[REFINER] Macro verified=True (3/3 test cases passed)
[REFINER] Persisted macros.json (1 macro)
```

*Pass criteria:*
- [ ] Refiner fires exactly once per milestone completion (not every step)
- [ ] `llm_logs/refiner_shadow.jsonl` contains a `WRITE_NEW` entry with a `code` field
- [ ] `memory_db/macros.json` is created with ≥ 1 entry after the first milestone
- [ ] Navigation unchanged — opener_bot still drives the intro
- [ ] No `KeyError`, `ForbiddenImportError`, or unhandled exception

---

## Phase 4: Macro Executor Node

**Purpose:** Add `macro_executor_node` as a new LangGraph specialist, extend the
router to activate it when a verified macro's trigger condition matches, and wire
`--use-macros` so macros can drive navigation independently of `opener_bot.py`.
After this phase, the first end-to-end Macro-driven intro run is possible.

### 4.1 `make_macro_executor_node` Factory

**File to create:** `agent/graph/nodes/macro_executor.py`

```python
from __future__ import annotations
import logging
from typing import Callable

from agent.graph.state import AgentState
from agent.graph.meta_tools import execute_macro
from agent.graph.restricted_exec import RestrictedExec
from agent.graph.macro_store import MacroRecord
from agent.graph.restricted_exec import evaluate_trigger

logger = logging.getLogger(__name__)

_SANDBOX = RestrictedExec()

def make_macro_executor_node(
    use_macros: bool = False,
) -> Callable[[AgentState], AgentState]:
    """Factory binding use_macros flag into the executor node.

    When use_macros=False (default): the node logs what it WOULD have done
    (shadow mode) and returns state unchanged (opener_bot retains control).
    When use_macros=True: runs the active macro; on success, sets last_buttons
    and last_action; on failure, sets macro_last_error and returns state so
    opener_bot fallback can engage.
    """

    def macro_executor_node(state: AgentState) -> AgentState:
        macro_name = state.get("active_macro")
        macro_store = state.get("macro_store", {})
        state_data = state.get("state_data", {})

        if not macro_name:
            logger.debug("[MACRO_EXECUTOR] No active_macro — passthrough.")
            return state

        rec_dict = macro_store.get(macro_name)
        if not rec_dict:
            logger.warning("[MACRO_EXECUTOR] Macro '%s' not in store — clearing.", macro_name)
            return {**state, "active_macro": None}

        rec = MacroRecord.from_dict(rec_dict)

        # Shadow mode: log but don't execute
        if not use_macros:
            logger.info(
                "[MACRO_EXECUTOR shadow] Would execute '%s' v%d (verified=%s)",
                macro_name, rec.version, rec.verified,
            )
            return state

        # Check completion condition first — deactivate if already done
        if evaluate_trigger(rec.completion_condition, state_data):
            logger.info("[MACRO_EXECUTOR] '%s' completion condition met — deactivating.", macro_name)
            return {**state, "active_macro": None}

        # Execute in sandbox
        buttons, error = execute_macro(macro_name, macro_store, state_data, _SANDBOX)

        if error:
            logger.error("[MACRO_EXECUTOR] '%s' failed: %s", macro_name, error)
            return {
                **state,
                "active_macro": None,
                "macro_last_error": error,
                "last_buttons": [],
            }

        logger.info("[MACRO_EXECUTOR] '%s' → %s", macro_name, buttons)
        return {
            **state,
            "last_buttons": buttons,
            "last_action": f"MACRO:{macro_name}",
            "macro_last_error": None,
        }

    return macro_executor_node
```

### 4.2 Router Extension — `routing_condition_with_macros`

**File:** `agent/graph/router.py`

Extend `routing_condition` to check macro triggers before the existing dispatch
logic. The new function is `routing_condition_with_macros`:

```python
def routing_condition_with_macros(state: AgentState) -> str:
    """Extended router: check macro triggers before specialist dispatch.

    Priority:
      1. Verified macro trigger matches                   → 'macro_executor'
      2. In battle                                        → 'battle_bot'
      3. Dialogue active                                  → 'coms_bot'
      4. Healing needed / map boundary                    → 'map_stitcher_relay'
      5. Default                                          → 'nav_bot'
    """
    macro_store = state.get("macro_store", {})
    state_data = state.get("state_data", {})

    if macro_store:
        for name, rec_dict in macro_store.items():
            rec = MacroRecord.from_dict(rec_dict)
            if rec.verified and evaluate_trigger(rec.trigger_condition, state_data):
                logger.debug("[ROUTER] Macro trigger matched: '%s'", name)
                return "macro_executor"  # sets active_macro in node

    # Fall through to existing routing_condition logic
    return routing_condition(state)
```

> **Caution:** The trigger evaluation loop runs on every `graph.invoke()` call.
> Trigger conditions must be fast expressions (no LLM calls, no I/O). The
> `evaluate_trigger()` sandbox adds ~0.1ms per macro; with 10 macros this is
> ~1ms — acceptable overhead. If the intro library grows beyond ~20 macros,
> consider caching the compiled expressions.

The router must also set `active_macro` when it selects `macro_executor`:

```python
# Inside routing_condition_with_macros, before returning "macro_executor":
# We can't mutate state here (routing is read-only), so we instead rely on
# macro_executor_node to re-evaluate triggers internally on entry and set
# active_macro itself.  The router just signals the route.
```

Because the router cannot mutate state, `macro_executor_node` re-evaluates
all trigger conditions on entry and sets `active_macro` to the first match
before executing. This is the same pattern as `verification_node` re-reading
`milestone_index`.

### 4.3 Graph Wiring Changes

**File:** `agent/graph/graph.py`

```python
from agent.graph.nodes.macro_executor import make_macro_executor_node
from agent.graph.nodes.refiner import make_refiner_node
from agent.graph.router import routing_condition_with_macros

def build_graph(
    obj_manager, vlm, episodic_memory=None, walkthrough_db=None,
    use_htn: bool = False,
    use_macros: bool = False,     # NEW: Phase 4
) -> ...:
    builder = StateGraph(AgentState)

    # Existing nodes (unchanged)
    builder.add_node("dispatch", lambda s: s)
    builder.add_node("nav_bot", nav_bot_node)
    builder.add_node("battle_bot", make_battle_bot_node(...))
    builder.add_node("coms_bot", make_coms_bot_node(...))
    builder.add_node("verification", make_verification_node(obj_manager))
    builder.add_node("map_stitcher_relay", make_map_stitcher_relay_node(vlm))
    builder.add_node("handoff_detector", make_handoff_detector_node(...))
    builder.add_node("executive_supervisor", make_executive_supervisor_node(..., use_htn=use_htn))

    # New Phase 4 nodes
    builder.add_node("macro_executor", make_macro_executor_node(use_macros=use_macros))
    builder.add_node("refiner", make_refiner_node(vlm=vlm, use_macros=use_macros))

    # Routing: now uses extended router
    builder.set_entry_point("dispatch")
    builder.add_conditional_edges(
        "dispatch",
        routing_condition_with_macros,
        {
            "nav_bot":           "nav_bot",
            "battle_bot":        "battle_bot",
            "coms_bot":          "coms_bot",
            "map_stitcher_relay":"map_stitcher_relay",
            "macro_executor":    "macro_executor",    # NEW
        },
    )

    # Macro executor feeds into handoff_detector like all other specialists
    builder.add_edge("macro_executor", "handoff_detector")

    # handoff_detector → supervisor AND/OR refiner (conditional chain)
    builder.add_conditional_edges(
        "handoff_detector",
        _handoff_routing,   # replaces the inline lambda
        {
            "executive_supervisor": "executive_supervisor",
            "refiner":              "refiner",
            "verification":         "verification",
        },
    )
    # Supervisor → Refiner → Verification (chained so both can fire)
    builder.add_conditional_edges(
        "executive_supervisor",
        lambda s: "refiner" if s.get("refiner_pending") else "verification",
        {"refiner": "refiner", "verification": "verification"},
    )
    builder.add_edge("refiner", "verification")
    builder.add_edge("verification", END)
```

```python
def _handoff_routing(state: AgentState) -> str:
    """Route after handoff_detector: Supervisor has priority over Refiner."""
    if state.get("supervisor_pending"):
        return "executive_supervisor"
    if state.get("refiner_pending"):
        return "refiner"
    return "verification"
```

### 4.4 `agent/__init__.py` Changes

Propagate `use_macros` flag and persist `_macro_store` back from graph output:

```python
# In Agent.__init__, alongside HTN fields:
self._macro_store: dict = _load_macro_store_from_disk()

self._graph = build_graph(
    ...,
    use_htn=use_htn,
    use_macros=use_macros,        # NEW
)

# After graph.invoke():
# Sync macro_store back — the Refiner may have written new macros
self._macro_store = result.get("macro_store", self._macro_store)
```

### Phase 4 Tests

**Automated — `tests/test_macro_executor_node.py`:**

```python
class TestExecutorPassthroughNoActiveMacro:
    # active_macro=None → state unchanged, no sandbox call

class TestExecutorShadowMode:
    # use_macros=False → state unchanged (shadow log written, no buttons set)

class TestExecutorSuccess:
    # use_macros=True, verified macro, trigger matches → last_buttons set

class TestExecutorCompletionConditionMet:
    # completion_condition evaluates True → active_macro=None, no execution

class TestExecutorSandboxFailure:
    # execute_macro returns error → active_macro=None, macro_last_error set

class TestExecutorUnverifiedMacro:
    # execute_macro refuses unverified macro → macro_last_error set
```

**Automated — `tests/test_router_with_macros.py`:**

```python
class TestRouterNoMacrosInStore:
    # Empty macro_store → falls through to normal routing_condition

class TestRouterMacroTriggerMatches:
    # One verified macro, trigger matches current location → 'macro_executor'

class TestRouterMacroUnverifiedSkipped:
    # Unverified macro trigger matches → NOT 'macro_executor', falls through

class TestRouterMacroPriorityOverNav:
    # Verified macro trigger + navigation context → 'macro_executor' wins

class TestRouterMacroNotOverBattle:
    # in_battle=True + macro trigger → battle_bot wins (battle is safety-critical)
```

> **Why does battle take priority over macros?** Battle is the highest-safety
> domain. Pressing wrong buttons during a fight (e.g. A spam from a stale
> intro_title_screen macro whose completion_condition is wrong) risks losing the
> starter. `routing_condition_with_macros` skips macro trigger evaluation when
> `state_data["game"]["in_battle"] == True`.

**Manual — First Macro-Driven Intro Smoke Test:**

*Prerequisites:* At least one verified intro macro in `memory_db/macros.json`
(generated by Phase 3 shadow mode run).

*Command:*
```bash
python run.py --load-state Emerald-GBAdvance/new_game.state --agent-auto --use-macros
```

*Observe in console:*
```
[ROUTER] Macro trigger matched: 'intro_title_screen'
[MACRO_EXECUTOR] 'intro_title_screen' → ['A']
[MACRO_EXECUTOR] 'intro_title_screen' completion condition met — deactivating.
```

*Pass criteria:*
- [ ] `[ROUTER] Macro trigger matched` prints for at least one intro segment
- [ ] `macro_executor_node` produces non-empty `last_buttons`
- [ ] Completion condition deactivates macro before next segment
- [ ] opener_bot fires for intro segments with no verified macro
- [ ] No `ForbiddenImportError` or unhandled exception during macro execution
- [ ] `refiner_shadow.jsonl` has entries for each milestone reached

---

## Phase 5: Opener Bot Deprecation

**Purpose:** Replace `opener_bot.py` entirely by building and verifying a
complete intro macro library that covers all 25 opener bot states (S0–S24), then
removing the `opener_bot` import from `action.py`.

This phase has two mandatory sub-phases and a final removal gate. The removal
gate is a hard prerequisite: `opener_bot.py` is NOT deleted until every intro
milestone in `new_game_milestones.json` has a verified, battle-tested macro that
passes the regression suite.

### 5.1 Intro Macro Library — Target Macro Catalogue

The following table maps opener bot states to target macros. Each macro name
carries the `intro_` prefix so they can be identified and batch-inspected.

| Macro Name | Trigger Condition (state_data expression) | Opener Bot States | Source Milestone |
|---|---|---|---|
| `intro_title_screen` | `player.location == "TITLE_SEQUENCE" and not milestones.INTRO_SCREEN_DONE` | S0 | — |
| `intro_prof_oak_dialogue` | `player.location == "TITLE_SEQUENCE" and milestones.INTRO_SCREEN_DONE` | S1 | `INTRO_SCREEN_DONE` |
| `intro_character_naming` | `player.location == "TITLE_SEQUENCE" and milestones.PROF_OAK_DONE` | S2 | `PROF_OAK_DONE` |
| `intro_truck_ride` | `player.location == "MOVING_VAN"` | S3 | `CHARACTER_NAMED` |
| `intro_mom_dialogue_2f` | `player.location == "LITTLEROOT_TOWN_PLAYERS_HOUSE_2F"` | S4 | `TRUCK_ARRIVED` |
| `intro_navigate_to_clock` | `player.location == "LITTLEROOT_TOWN_PLAYERS_HOUSE_1F" and not milestones.CLOCK_DONE` | S5–S6 | `ARRIVED_HOME` |
| `intro_set_clock` | `"THE CLOCK" in game.dialogue_text.upper()` | S7 | `NAV_TO_CLOCK` |
| `intro_house_exit` | `player.location == "LITTLEROOT_TOWN_PLAYERS_HOUSE_1F" and milestones.CLOCK_DONE` | S8–S8B | `CLOCK_DONE` |
| `intro_mays_house_visit` | `player.location == "LITTLEROOT_TOWN_MAYS_HOUSE_2F"` | S9–S13 | `EXITED_HOUSE` |
| `intro_route101_approach` | `player.location == "LITTLEROOT_TOWN" and milestones.MAYS_HOUSE_DONE` | S14–S15 | `MAYS_HOUSE_DONE` |
| `intro_route101_navigate` | `player.location == "ROUTE_101"` | S17 | `ROUTE_101_ENTERED` |
| `intro_starter_selection` | `player.location == "ROUTE_101" and milestones.BAG_FOUND` | S19–S21 | `BAG_FOUND` |
| `intro_first_battle` | `player.location == "ROUTE_101" and game.in_battle and not milestones.STARTER_CHOSEN` | S22 | `STARTER_CHOSEN` |
| `intro_birch_rescue_dialogue` | `player.location == "ROUTE_101" and not game.in_battle and not milestones.BIRCH_RESCUED` | S23 | `FIRST_BATTLE_DONE` |
| `intro_nickname_screen` | `player.location == "PROFESSOR_BIRCHS_LAB" and milestones.STARTER_CHOSEN` | S24 | `BIRCH_RESCUED` |

> **Note:** Trigger conditions in the table are pseudocode for readability. The
> actual stored strings use the full `state_data.get("player",{}).get(...)` form
> required by `evaluate_trigger()`.

### 5.2 Shadow Mode Validation Protocol

Before cutover, run `new_game.state` in a parallel comparison mode where both
`opener_bot` and `macro_executor` produce button lists independently. The diff is
written to `llm_logs/refiner_shadow.jsonl`:

```json
{
  "step": 42,
  "location": "LITTLEROOT_TOWN_PLAYERS_HOUSE_1F",
  "macro_name": "intro_navigate_to_clock",
  "macro_buttons": ["UP", "UP", "RIGHT"],
  "opener_bot_buttons": ["UP", "UP", "RIGHT"],
  "match": true
}
```

**Acceptance criteria for cutover:**
- [ ] ≥ 95% step-level button match across a full `new_game.state` run
- [ ] Every milestone in `new_game_milestones.json` is reached
- [ ] No macro-caused `ForbiddenImportError` or timeout in the run log
- [ ] `route102_hackathon.state` regression run: zero `intro_` macro triggers (mid-game macros must not activate)

### 5.3 `action.py` Gating

Add a `use_macros` guard around the opener_bot call in `action.py`:

```python
# PRIORITY 0B: OPENER BOT — gated by --use-macros flag
# When use_macros=True, the LangGraph macro_executor_node handles intro.
# When use_macros=False (default), opener_bot remains the authority.
if not _use_macros_flag:
    from agent.opener_bot import NavigationGoal, ForceDialogueGoal
    opener_bot = get_opener_bot()
    should_handle = opener_bot.should_handle(state_data, visual_data)
    if should_handle:
        ...
```

`_use_macros_flag` is a module-level bool set once at startup from the CLI arg,
exactly as `_use_htn` was propagated for the HTN migration.

### 5.4 Removal Gate Checklist

`opener_bot.py` MUST NOT be deleted until ALL of the following are ✅:

- [ ] All 15 target macros from §5.1 exist in `memory_db/macros.json` with `verified=True`
- [ ] `tests/test_opener_bot.py` passes with `--use-macros` flag (opener_bot shim returns None for all intro states, macros handle them all)
- [ ] Full `new_game.state` run completes `STARTER_CHOSEN` milestone in ≤ 200 steps with `--use-macros`
- [ ] Full `new_game.state` run completes `STARTER_CHOSEN` milestone without `--use-macros` (regression: no regressions introduced)
- [ ] `route102_hackathon.state` run shows zero `[ROUTER] Macro trigger matched: intro_` lines
- [ ] `tests/test_macro_store.py`, `test_restricted_exec.py`, `test_meta_tools.py`, `test_refiner_node.py`, `test_macro_executor_node.py`, `test_router_with_macros.py` — all passing
- [ ] `docs/CONTINUAL_HARNESS_PLAN.md` removal gate section updated to all ✅

Only after all gates are ✅:
1. Delete `agent/opener_bot.py`
2. Remove the `from agent.opener_bot import get_opener_bot` import from `action.py`
3. Remove the Priority 0B block from `action.py` entirely
4. Update `agent/__init__.py` to remove `_global_opener_bot` references

---

## Implementation Checklist

Files to **create**:

| # | File | Description |
|---|---|---|
| 0.1 | `agent/graph/macro_store.py` | `MacroRecord` dataclass, `to_dict` / `from_dict` |
| 1.1 | `agent/graph/restricted_exec.py` | `RestrictedExec`, `ForbiddenImportError`, `evaluate_trigger`, AST scanner |
| 2.1 | `agent/graph/meta_tools.py` | `write_macro`, `execute_macro`, `test_macro`, `get_trajectory`, `save_macro_store` |
| 3.1 | `agent/graph/nodes/refiner.py` | `make_refiner_node`, `REFINER_SYSTEM_PROMPT`, `REFINER_USER_TEMPLATE`, `_call_refiner_llm`, `_write_refiner_log` |
| 4.1 | `agent/graph/nodes/macro_executor.py` | `make_macro_executor_node` |
| — | `tests/test_macro_store.py` | Phase 0 automated tests |
| — | `tests/test_restricted_exec.py` | Phase 1 automated tests |
| — | `tests/test_meta_tools.py` | Phase 2 automated tests |
| — | `tests/test_refiner_node.py` | Phase 3 automated tests |
| — | `tests/test_macro_executor_node.py` | Phase 4 automated tests |
| — | `tests/test_router_with_macros.py` | Phase 4 automated tests |

Files to **modify**:

| # | File | Change | Phase |
|---|---|---|---|
| 0.2 | `agent/graph/state.py` | Add 6 new `AgentState` fields | 0 |
| 0.3 | `agent/__init__.py` | Load `macro_store` from disk; persist after invoke; `use_macros` flag; sync `_macro_store` from graph output | 0 / 4 |
| 3.2 | `agent/graph/nodes/handoff_detector.py` | Append to `trajectory_buffer`; set `refiner_pending` on milestone increase or macro failure | 3 |
| 4.2 | `agent/graph/router.py` | Add `routing_condition_with_macros`; skip macro trigger when `in_battle` | 4 |
| 4.3 | `agent/graph/graph.py` | Add `macro_executor`, `refiner` nodes; extend routing; `use_macros` arg | 4 |
| 5.3 | `agent/action.py` | Gate opener_bot call with `_use_macros_flag` | 5 |
| 5.4 | `agent/action.py` | **Delete** Priority 0B block (after removal gate ✅) | 5.4 |
| 5.4 | `agent/opener_bot.py` | **Delete entire file** (after removal gate ✅) | 5.4 |

Files to **create** (data/config):

| # | File | Description |
|---|---|---|
| — | `memory_db/macros.json` | Auto-created by `save_macro_store()` on first Refiner write — runtime source of truth |
| — | `memory_db/macros/<name>.py` | Sidecar `.py` files auto-generated alongside JSON — for human inspection/debugging only, never imported |
| — | `llm_logs/refiner_shadow.jsonl` | Auto-created by `_write_refiner_log()` |

---

## Appendix A — Security Notes

### A.1 Why `exec()` and not a subprocess/process isolation?

Process isolation (e.g. `multiprocessing.Process`) would be the strongest
sandbox but adds 50–200ms overhead per macro execution — unacceptable on the hot
path. The threat model for this project is *accidental capability escalation by a
well-intentioned LLM*, not adversarial code injection. The AST scanner +
restricted `__builtins__` + no-import policy is sufficient for this threat model.

If the threat model changes (e.g. the agent is used in a shared multi-user
environment), replace `RestrictedExec` with a `multiprocessing.Process` worker
pool. The `execute_macro` interface is unchanged; only `RestrictedExec.run()`
needs updating.

### A.2 `state_data` Deep Copy — Why It Matters

The sandbox receives `copy.deepcopy(state_data)` not the live dict. This prevents
two classes of bugs:

1. **Mutation leakage**: Macro writes `state_data["player"]["x"] = 99` — without
   the deep copy, this corrupts `AgentState` for the entire step.
2. **Reference aliasing**: Macro stores a reference to `state_data` in a closure
   or global — after the sandbox exits, the reference keeps the state dict alive
   in memory indefinitely.

The deep copy cost is ~0.1–0.5ms for typical `state_data` dicts (~5–10KB JSON).
This is acceptable.

### A.3 `trigger_condition` Expression Safety

Trigger conditions are re-evaluated at every `graph.invoke()` (i.e. every step).
A poorly-written condition that raises an exception would fall through to `False`
via `evaluate_trigger()`'s bare `except Exception: return False`. This is
intentional: a broken trigger silently deactivates the macro rather than crashing
the run loop. The Refiner will write a `refiner_shadow.jsonl` warning entry, and
the macro's `verified` flag should be set to `False`.

---

## Appendix B — Opener Bot State-to-Macro Mapping (Detailed)

The 25 opener bot states decompose into 15 macros rather than 25 because several
consecutive states share the same trigger condition (same location, same milestone
gating) and produce short, sequential button presses that can be expressed as a
single function with conditional branches.

```
Opener Bot FSM States        →   Target Macro
──────────────────────────────────────────────────────────────
S0_TITLE_SCREEN              →   intro_title_screen
S1_PROF_DIALOG               →   intro_prof_oak_dialogue
S2_GENDER_NAME_SELECT        →   intro_character_naming
S3_TRUCK_RIDE                →   intro_truck_ride
S4_MOM_DIALOG_1F             →   intro_mom_dialogue_2f
S5_NAV_TO_STAIRS_1F          →   intro_navigate_to_clock
S6_NAV_TO_CLOCK              →   intro_navigate_to_clock      (merged)
S7_SET_CLOCK                 →   intro_set_clock
S8_NAV_TO_STAIRS_2F          →   intro_house_exit
S8B_NAV_TO_DOOR_1F           →   intro_house_exit             (merged)
S9_NAV_TO_MAYS_HOUSE         →   intro_route101_approach       (partial)
S10_MAYS_MOTHER_DIALOG       →   intro_mays_house_visit
S11_NAV_TO_STAIRS_MAYS_HOUSE →   intro_mays_house_visit       (merged)
S11B_NAV_TO_POKEBALL         →   intro_mays_house_visit       (merged)
S12_MAY_DIALOG               →   intro_mays_house_visit       (merged)
S13_NAV_TO_STAIRS_2F         →   intro_mays_house_visit       (merged)
S14A_MAY_DOWNSTAIRS_DIALOG   →   intro_mays_house_visit       (merged)
S14_NAV_TO_EXIT_MAYS_HOUSE   →   intro_route101_approach
S15_NAV_TO_NPC_NORTH         →   intro_route101_approach      (merged)
S15B_NAV_NORTH_CONTINUED     →   intro_route101_approach      (merged)
S16_NPC_DIALOG               →   intro_route101_navigate       (transition)
S17_NAV_TO_ROUTE_101         →   intro_route101_navigate
S19_NAV_TO_BAG               →   intro_starter_selection
S20_INTERACT_BAG             →   intro_starter_selection      (merged)
S21_STARTER_SELECT           →   intro_starter_selection      (merged)
S22_FIRST_BATTLE             →   intro_first_battle
S23_BIRCH_DIALOG_2           →   intro_birch_rescue_dialogue
S24_NICKNAME                 →   intro_nickname_screen
```

Note: S18 (not listed in the state machine) and S19 are the "navigate to bag"
steps. The starter selection macro uses the A* nav output already embedded in
`state_data["map"]` to compute button presses rather than hardcoding coordinates,
making it more robust to minor map variations.
