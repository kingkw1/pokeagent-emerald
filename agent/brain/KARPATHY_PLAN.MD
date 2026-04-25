# Karpathy Meta-Agent Plan

**Status:** 🔲 Deferred — agent must reliably complete the main game before this
layer is added. See [`PLAN.MD`](PLAN.MD) for the current active roadmap.

**Objective:** Wrap the LangGraph dispatch loop with a reward-logging and
meta-learning layer that quantifies the agent's per-step performance, enables
offline policy improvement, and produces a VLM-savings metric suitable for
portfolio / competition documentation.

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
