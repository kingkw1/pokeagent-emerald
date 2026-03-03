# RL Battle Bot — Implementation Plan

**Status:** Architecture complete, observation bridge required  
**Current Default:** Heuristic Agent (`battle_bot.py`)  
**Trained Model:** `models/PPO_Masked/emerald_curriculum_v1.zip` (MaskablePPO, 100k timesteps, 5 curriculum scenarios)

---

## 1. Overview

PokéAgent uses a **dual-mode combat architecture** built on the Strategy pattern. A `BattleManager` router delegates to one of two interchangeable backends — a production `HeuristicBattleAgent` (rule-based) and a `RLBattleAgent` (PPO neural network). Both implement a shared `BattleAgent` ABC.

A prototype PPO model has been trained in an isolated `stable-retro` simulation environment and can win basic battles against early-game opponents. The remaining work is to **bridge the gap** between the retro training environment and the live mGBA agent so the RL model can receive correct observations during real gameplay.

## 2. Current Architecture

```
agent/combat/
├── interface.py          # BattleAgent ABC (get_action → int)
├── battle_manager.py     # Strategy router (use_rl toggle)
├── heuristic_agent.py    # Rule-based: type effectiveness scoring (~45 lines)
└── rl_agent.py           # PPO wrapper: MaskablePPO.predict() (obs builder is STUB)

agent/battle_bot.py       # PRODUCTION battle controller (1700+ lines)
                          # NOT connected to combat/ — runs independently
                          # Handles: menu navigation, dialogue parsing, battle type
                          # detection, wild/trainer classification, run logic,
                          # HP/PP memory tracking, VLM fallback, stuck detection

simulation/
├── train.py              # Training loop (MaskablePPO, SubprocVecEnv, curriculum)
├── evaluate.py           # Visual evaluation with Pygame rendering
├── pokedex.py            # SPECIES_DATA, MOVES_DATA, get_effectiveness()
└── data/                 # Retro integration files, state saves
    └── PokemonEmerald-GBA/
        ├── data.json     # RAM address definitions for stable-retro
        ├── scenario.json # Reward/done conditions
        └── *.state       # Training battle save states
```

### What exists and works

| Component | Status | Notes |
|-----------|--------|-------|
| `BattleAgent` ABC | ✅ Complete | Clean interface: `get_action(state_data) → int` |
| `BattleManager` router | ✅ Complete | `use_rl` toggle, graceful fallback if model fails to load |
| `HeuristicBattleAgent` | ✅ Complete | Type scoring via `pokedex.get_effectiveness()` |
| `RLBattleAgent.get_action()` | ✅ Complete | `model.predict(obs, action_masks=mask)` pipeline |
| `RLBattleAgent._make_observation()` | ❌ **Stub** | Returns `np.zeros(14)` — needs real implementation |
| `RLBattleAgent._get_mask()` | ❌ **Stub** | Returns `[True, True, True, True]` — needs PP-based masking |
| Training env (`EmeraldBattleWrapper`) | ✅ Complete | 14-dim obs, reward shaping, menu macro execution |
| Trained model (`emerald_curriculum_v1`) | ✅ Exists | 100k timesteps across 5 curriculum states |
| Live agent wiring | ❌ **Not connected** | `BattleManager` is never imported by `agent/__init__.py` |

### What the trained model expects

The PPO model was trained on a **14-dimensional observation vector**:

```
Index  Field                Description              Normalization
─────  ─────                ───────────              ─────────────
0      my_hp                Player HP                / 20.0
1      enemy_hp             Opponent HP              / 20.0
2      move_1_pp            Move 1 has PP?           binary (1.0 if PP > 0)
3      move_1_power         Move 1 base power        / 100.0
4      move_1_eff           Move 1 type effectiveness / 4.0
5      move_2_pp            Move 2 has PP?           binary
6      move_2_power         Move 2 base power        / 100.0
7      move_2_eff           Move 2 type effectiveness / 4.0
8      move_3_pp            Move 3 has PP?           binary
9      move_3_power         Move 3 base power        / 100.0
10     move_3_eff           Move 3 type effectiveness / 4.0
11     move_4_pp            Move 4 has PP?           binary
12     move_4_power         Move 4 base power        / 100.0
13     move_4_eff           Move 4 type effectiveness / 4.0
```

**Action space:** Discrete(4) — one per move slot.  
**Action masking:** Boolean mask based on PP > 0 (unmask all on Struggle).

---

## 3. The Observation Bridge Problem

The core blocker is a **data format mismatch** between the retro training environment and the live mGBA agent:

| Field | Training env (`data.json` RAM reads) | Live agent (`memory_reader.py`) | Gap |
|-------|--------------------------------------|----------------------------------|-----|
| Player HP | `my_hp` — raw u16 | `battle_info.player_pokemon.current_hp` | Format only (both reliable) |
| Enemy HP | `enemy_hp` — raw u16 | `battle_info.opponent_pokemon.current_hp` | **Reliability gap** — opponent struct often fails to parse |
| Move IDs | `move_1`–`move_4` — raw u16 numeric IDs | `player_pokemon.moves` — string names | **Format gap** — need reverse lookup |
| Move PP | `move_1_pp`–`move_4_pp` — raw u8 | `player_pokemon.move_pp` — list[int] | Format only (both reliable) |
| Enemy species | `enemy_species` — raw u16 species ID | `opponent_pokemon.species` — string name (when available) | **Both format + reliability gap** |

### Critical: The opponent data problem

The live agent reads opponent data from `gEnemyParty` (0x02023BC0) and `gBattleMons[1]` (0x02024AD8) via `read_comprehensive_battle_info()`. In practice, the opponent Pokémon struct **frequently fails validation** and returns empty `{}`. The main `battle_bot.py` works around this by extracting opponent species from VLM dialogue text (e.g., *"Wild POOCHYENA appeared!"*).

The RL agent needs two pieces of opponent data:
1. **Enemy HP** — required for obs[1]
2. **Enemy species** → enemy types → move effectiveness — required for obs[4,7,10,13]

Without reliable opponent data, the effectiveness dimensions collapse to defaults.

---

## 4. Implementation Plan

### Phase A: Observation Builder (Fill the Stub)

**Goal:** Implement `_make_observation()` and `_get_mask()` in `rl_agent.py` to translate live `state_data` into the 14-dim vector the model expects.

**File:** `agent/combat/rl_agent.py`

```python
def _make_observation(self, state_data: dict) -> np.ndarray:
    """
    Translate live agent state_data into the 14-dim observation vector
    matching the training env format.
    """
    battle_info = state_data.get('game', {}).get('battle_info', {})
    player = battle_info.get('player_pokemon', {})
    opponent = battle_info.get('opponent_pokemon', {})

    # 1. HP values
    my_hp = player.get('current_hp', 0) / self.OBS_NORM_FACTOR
    enemy_hp = opponent.get('current_hp', 0) / self.OBS_NORM_FACTOR
    # Fallback: if opponent HP unavailable, assume full health
    if not opponent:
        enemy_hp = 1.0

    obs = [my_hp, enemy_hp]

    # 2. Per-move features: PP (binary), Power, Effectiveness
    move_names = player.get('moves', [])
    move_pps = player.get('move_pp', [])
    enemy_types = opponent.get('types', [None, None])

    for i in range(4):
        # PP
        pp = move_pps[i] if i < len(move_pps) else 0
        norm_pp = 1.0 if pp > 0 else 0.0

        # Move lookup (name → ID → data)
        move_name = move_names[i] if i < len(move_names) else None
        move_id = self._move_name_to_id(move_name)
        move_info = MOVES_DATA.get(move_id, {"type": 0, "power": 0})

        # Power
        norm_power = move_info.get("power", 0) / 100.0

        # Effectiveness (requires opponent types)
        move_type = move_info.get("type", 0)
        eff = get_effectiveness(move_type, enemy_types)
        norm_eff = eff / 4.0

        obs.extend([norm_pp, norm_power, norm_eff])

    return np.array(obs, dtype=np.float32)
```

**Subtasks:**
1. Build a reverse lookup dict `{move_name: move_id}` from `MOVES_DATA` (one-time init)
2. Build a reverse lookup dict `{species_name: species_id}` from `SPECIES_DATA` (for future use)
3. Handle edge cases: fewer than 4 moves, missing opponent, unknown move names
4. Implement `_get_mask()` using `player_pokemon.move_pp` list
5. **Unit test:** Feed sample `state_data` dicts (captured from live runs) through the builder and verify the output matches what `train.py`'s `_get_obs()` would produce for equivalent RAM values

### Phase B: Fix Opponent Data Reliability

**Goal:** Make `opponent_pokemon` from `memory_reader.py` reliable enough for RL inference.

**File:** `pokemon_env/memory_reader.py` — `read_comprehensive_battle_info()`

The opponent parsing attempts 4 fallback methods but often returns `{}`. Potential fixes:

1. **Direct `gBattleMons[1]` read** — The `BattlePokemon` struct at 0x02024AD8 (offset +0x58 per slot) has species, HP, moves in a flat unencrypted layout. This is simpler than decrypting `gEnemyParty` and should be more reliable during active battle.
   - `species` at offset +0x00 (u16)
   - `current_hp` at offset +0x28 (u16)
   - `max_hp` at offset +0x2C (u16)
   - `type1` at offset +0x21 (u8), `type2` at offset +0x22 (u8)

2. **Fallback from dialogue VLM** — Mirror what `battle_bot.py` already does: extract opponent species from *"Wild X appeared!"* or *"Trainer sent out X!"* dialogue, then look up types from `SPECIES_DATA`. This gives effectiveness but not live HP.

3. **Graceful degradation** — If opponent data is unavailable, the RL agent should still function by using effectiveness=1.0 (neutral) for all moves. The model was trained seeing real effectiveness values, so its predictions will be suboptimal but not random — it still has PP and power signals.

### Phase C: Wire Into the Live Agent

**Goal:** Connect `BattleManager` to the main agent loop so the RL path can be activated.

**Changes required:**

1. **`agent/__init__.py`** — Import `BattleManager`, initialize with config toggle:
   ```python
   from agent.combat.battle_manager import BattleManager
   # In __init__:
   self.battle_manager = BattleManager({'use_rl_combat': args.use_rl_combat})
   ```

2. **`agent/action.py`** — Add an RL combat branch in the priority chain. When `use_rl_combat` is enabled and `in_battle` is True, delegate to `battle_manager.get_action(state_data)` instead of `battle_bot.get_action(state_data)`.

3. **`run.py`** — Add `--use-rl-combat` CLI flag (default: False).

4. **Critical consideration: menu navigation.** The RL agent returns a move index (0–3) but the live battle system needs to **navigate the GBA battle menu** to select that move. The existing `battle_bot.py` handles menu navigation (~500 lines of dialogue parsing and button sequences). Options:
   - **Option A (Recommended):** Keep `battle_bot.py` as the menu navigator and only replace the *move selection logic* with the RL agent's output. When in `fight_menu` state, use `rl_agent.get_action()` instead of the built-in `_choose_best_move()`.
   - **Option B:** Build a minimal menu macro system in `BattleManager` (similar to `train.py`'s `_perform_move_macro`). Cleaner separation but duplicates substantial UI-navigation logic.

### Phase D: Expand Training Curriculum

**Goal:** Train the model on a wider range of battle scenarios to generalize beyond early Route 102/103 encounters.

**Current training data:** 5 save states, all pre-Gym-1 battles (~6 distinct battles):
- `State_Advantage` / `State_Advantage2` — type-advantaged matchups
- `State_Disadvantage` / `State_Disadvantage2` — type-disadvantaged matchups
- `State_Neutral` — neutral effectiveness matchups

#### D.0: Automated Data Collection Rig

Manually creating save states via `simulation/play_and_save.py` inside `stable-retro` is slow and limited by input-mapping issues (notably the GBA Start button, which can fail depending on the retro integration's `data.json` bitmask config). A better approach is to collect battle states from the **live Pygame client**, where all inputs work reliably.

**How it works:**
1. Boot the game in manual mode: `python run.py --manual --collect-battles`
2. A background monitor watches the `in_battle` RAM address.
3. The moment `in_battle` flips from `False` → `True`, the system auto-saves an mGBA core save state to `simulation/data/collected_states/` with a timestamp and location tag (e.g., `battle_route104_20260301_143022.state`).
4. A human player (or the autonomous agent) continues playing normally — every battle encounter is captured automatically.
5. Collected `.state` files are converted/loaded into `stable-retro` for headless PPO training at 1000+ FPS.

**Implementation in `server/client.py`:**
```python
# In the game loop, after state_data is fetched:
in_battle = state_data.get('game', {}).get('in_battle', False)
if in_battle and not self._was_in_battle:
    # Battle just started — save state
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    location = state_data.get('player', {}).get('location', 'unknown')
    save_path = f"simulation/data/collected_states/battle_{location}_{timestamp}.state"
    self.emulator.save_state(save_path)
    print(f"💾 [DATA COLLECT] Battle state saved: {save_path}")
self._was_in_battle = in_battle
```

**State format compatibility note:** Both the live agent and `stable-retro` use the mGBA emulator core, so save states should be compatible. Verify by loading a collected `.state` into the retro env and checking that RAM values (`my_hp`, `enemy_hp`, move data) read correctly before committing to a full training run.

**Advantages over manual `play_and_save.py`:**
- No input-mapping issues (Pygame handles all GBA buttons natively)
- Passive collection — a human plays the game normally while states accumulate
- Scales to dozens of battle states per playthrough with zero extra effort
- Can also run in autonomous mode (`--agent-auto --collect-battles`) to harvest battles the agent encounters naturally

#### D.1–D.4: Curriculum Expansion Targets

| Phase | Scenarios | New Challenges |
|-------|-----------|----------------|
| D.1 | Gym 1 (Roxanne) | Rock types, multi-move opponents, higher HP pools |
| D.2 | Route 104–116 trainers | Diverse species, 3-4 move opponents, status moves |
| D.3 | Gym 2–3 (Brawly, Wattson) | Fighting/Electric types, stat-boosting opponents |
| D.4 | Multi-pokémon trainer battles | Sequential opponents, HP conservation across fights |

**How to add collected states to training:**
1. Collect states via the data collection rig (D.0) or manually via `simulation/play_and_save.py`
2. Copy/convert `.state` files to `simulation/data/PokemonEmerald-GBA/`
3. Add the state names to `TRAIN_STATES` in `train.py`
4. Re-train with increased `TOTAL_TIMESTEPS`

**Training improvements to consider:**
- **Self-play / opponent diversity:** Randomize which of the opponent's moves is used each turn (currently determined by ROM AI)
- **Reward shaping:** Add bonus for winning with high remaining HP (incentivizes efficiency)
- **Observation expansion:** Add player level, opponent level, status conditions to obs vector (requires retraining from scratch)
- **Longer training runs:** 100k timesteps is modest; 500k–1M with 8+ parallel envs would likely improve generalization significantly

### Phase E: Hybrid Routing (Future)

**Goal:** Use RL for routine wild encounters, Heuristic for boss battles (Gyms, Elite Four) where reliability matters most.

Update `BattleManager.get_action()`:
```python
def get_action(self, state_data: dict) -> int:
    if self.use_rl and self.rl_bot and self.rl_bot.model:
        battle_info = state_data.get('game', {}).get('battle_info', {})
        is_trainer = battle_info.get('is_trainer_battle', False)
        if is_trainer and self.conservative_mode:
            return self.heuristic.get_action(state_data)
        return self.rl_bot.get_action(state_data)
    return self.heuristic.get_action(state_data)
```

---

## 5. Execution Priority

```
Phase A (Observation Builder)      ← DO FIRST — unblocks all testing
  └─ Phase B (Fix Opponent Data)   ← In parallel — improves obs quality
      └─ Phase C (Wire to Agent)   ← Once A+B pass unit tests
Phase D.0 (Data Collection Rig)    ← CAN START NOW — independent of A/B/C
  └─ Phase D.1–D.4 (Curriculum)  ← Ongoing — more data = better model
      └─ Phase E (Hybrid)          ← After D proves RL is competitive
```

**Definition of "RL Agent is production-ready":**  
The RL agent wins ≥90% of battles in evaluation across all training curriculum states AND wins ≥80% of novel (non-training) battles through Gym 3 when running in the live agent.

---

## 6. Files to Modify

| File | Change |
|------|--------|
| `agent/combat/rl_agent.py` | Implement `_make_observation()`, `_get_mask()`, add reverse lookup dicts |
| `pokemon_env/memory_reader.py` | Improve `gBattleMons[1]` parsing for reliable opponent data |
| `agent/__init__.py` | Import + initialize `BattleManager` |
| `agent/action.py` | Add RL combat branch in priority chain |
| `agent/battle_bot.py` | Extract move-selection logic to allow RL substitution (Option A) |
| `run.py` | Add `--use-rl-combat` and `--collect-battles` CLI flags |
| `server/client.py` | Add battle state auto-save monitor for data collection |
| `simulation/train.py` | Expand `TRAIN_STATES`, increase `TOTAL_TIMESTEPS` |