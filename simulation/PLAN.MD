# Simulation & RL Training — Development Plan

**Last updated:** 2025-01-21  
**Status:** Training environment operational, prototype PPO model trained  
**Detailed RL integration plan:** See [agent/combat/RL_BATTLE_BOT_PLAN.md](../agent/combat/RL_BATTLE_BOT_PLAN.md)

---

## 1. Current State (What Exists & Works)

| Component | File | Status |
|-----------|------|--------|
| Training environment | `train.py` | ✅ MaskablePPO, SubprocVecEnv (4 parallel), 100k timesteps |
| Battle wrapper | `train.py` → `EmeraldBattleWrapper` | ✅ 14-dim obs, reward shaping, action masking, menu macro |
| Game knowledge | `pokedex.py` | ✅ `SPECIES_DATA`, `MOVES_DATA`, `get_effectiveness()` |
| Manual play client | `play_and_save.py` | ✅ Pygame rendering, keyboard input, gzip state saving |
| Visual eval | `evaluate.py` | ✅ Load model, replay with Pygame at 60 FPS |
| Trained model | `models/PPO_Masked/emerald_curriculum_v1.zip` | ✅ Wins early-game battles |
| ROM integration | `data/PokemonEmerald-GBA/` | ✅ RAM addresses, button config, 20+ save states |

### Training Observation Vector (14-dim)

```
[my_hp, enemy_hp, move1_pp, move1_power, move1_eff, move2_pp, move2_power,
 move2_eff, move3_pp, move3_power, move3_eff, move4_pp, move4_power, move4_eff]
```

### Curriculum States (5 active)

`State_Advantage`, `State_Advantage2`, `State_Disadvantage`, `State_Disadvantage2`, `State_Neutral`

Plus ~15 additional states available but not yet in the curriculum:
`BattleLevel5`, trainer variants (mudkip, nincada, poochyena, wurmple, zigzagoon), `lotad2`, `shroomish`, `general_saves/grinding_phase_1`

---

## 2. Known Problems

### 2.1 Save State Format Incompatibility ⚠️ CRITICAL

The project uses **two different mGBA emulators** with **incompatible state formats**:

| | Main Agent (live) | Training (simulation) |
|--|---|---|
| **Library** | `mgba.core` (standalone Python bindings) | `stable-retro` (libretro mGBA core) |
| **Save method** | `core.save_raw_state()` → raw bytes | `env.em.get_state()` → gzip-compressed libretro serialization |
| **File format** | Uncompressed binary | gzip(libretro internal format) |
| **Load method** | `core.load_raw_state(bytes)` | `env.em.set_state(gzip_bytes)` |

**These formats are NOT cross-compatible.** A state saved from the main emulator cannot be loaded in stable-retro, and vice versa. This blocks any plan that assumes "collect states from the live agent, train in stable-retro."

**Consequence for data collection:** All training states must originate from within the `stable-retro` environment — either via `play_and_save.py` or by programmatic state manipulation.

### 2.2 Start Button in stable-retro

The GBA Start button (index 3 in the 12-button array) is mapped correctly in both `metadata.json` and `play_and_save.py`:
```
buttons = [0] * 12
if keys[pygame.K_RETURN]: buttons[3] = 1  # START is Index 3
```

`env.unwrapped.buttons` confirms: `['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', None, 'L', 'R']`

Despite the mapping appearing correct, Start may not respond in certain game states (title screen, specific dialogue loops). The `force_button_test(env, 3, screen, "START")` debug function exists for diagnosis — it holds the button for 60 frames. If this function works but keyboard input doesn't, the issue is Pygame event timing; if it also fails, the problem is in the libretro core's input handling for that game state.

**Status:** Needs targeted diagnosis. Not a blocker for battles (Start is not used in battle menus), but blocks progression from title screen / in-game save menus.

### 2.3 Observation Bridge to Live Agent

The RL agent observation builder (`rl_agent.py._make_observation()`) is a **stub** returning `np.zeros(14)`. This is the primary blocker for wiring the trained model into the live agent. Full details in [RL_BATTLE_BOT_PLAN.md Phase A](../agent/combat/RL_BATTLE_BOT_PLAN.md).

---

## 3. Forward Plan

### Phase S1: Start Button Diagnosis

**Goal:** Determine whether Start works in stable-retro and document the conditions.

**Steps:**
1. Load `BattleLevel5` state (mid-battle, Start not needed) — verify training works. ✅ Done.
2. Load a state that IS at a Start-requiring screen (title, or overworld where Start opens menu).
3. Run `force_button_test(env, 3, ...)` and observe the result.
4. If it works: the issue is Pygame key-hold timing. Fix: hold Start for multiple frames on key-down.
5. If it doesn't: patch the libretro core button mapping locally or file upstream issue.

**Severity:** Medium. All existing training states bypass the title screen, so this doesn't block training — only state creation from cold boot.

### Phase S2: Data Collection Rig

**Goal:** Automate battle state harvesting during human play.

Since save states can only be used inside their native emulator, collection must happen inside stable-retro:

**Option A: Enhanced `play_and_save.py` (Recommended)**

Add an auto-save hook that triggers on battle start:
```python
# In main loop after env.step():
in_battle = info.get("in_battle", 0)  # needs RAM address in data.json
if in_battle and not was_in_battle:
    content = env.em.get_state()
    save_name = f"battle_{timestamp}.state"
    with gzip.open(save_path, 'wb') as f:
        f.write(content)
was_in_battle = in_battle
```

**Prerequisite:** Add `in_battle` RAM address to `data/PokemonEmerald-GBA/data.json`. The Gen 3 battle flag is at `gBattleTypeFlags` (`0x02022B4C`) — non-zero during battle.

**Option B: RAM-Snapshot Approach (Format-agnostic fallback)**

If stable-retro input issues block comfortable play:
1. Play in the main Pygame client (all inputs work reliably)
2. Monitor `in_battle` via RAM
3. When battle starts, dump RAM values to JSON:
   ```json
   {"my_species": 258, "my_hp": 42, "enemy_species": 261, "enemy_hp": 19,
    "moves": [33, 45, 0, 0], "move_pp": [35, 20, 0, 0]}
   ```
4. In stable-retro, load any generic battle state, then `env.data.write_value(addr, val)` to overwrite RAM with captured values
5. Train on the reconstructed state

More fragile but decouples collection from state format.

### Phase S3: Curriculum Expansion

**Goal:** Grow from 5 training states to 20+ covering Gyms 1–3.

| Tier | Scenarios | Key Challenge |
|------|-----------|---------------|
| Existing | 5 states (Advantage/Disadvantage/Neutral) | Basic type effectiveness |
| Tier 1 | Route 104–116 trainers | Multi-move opponents, diverse species |
| Tier 2 | Gym 1 (Roxanne) | Rock types, Geodude + Nosepass, higher HP |
| Tier 3 | Gym 2 (Brawly) | Fighting types, Bulk Up stat boosting |
| Tier 4 | Gym 3 (Wattson) | Electric types, Voltorb Self-Destruct |

**Process per tier:**
1. Collect states via S2 rig (human plays to that point in stable-retro)
2. Add state names to `TRAIN_STATES` in `train.py`
3. Re-train with increased `TOTAL_TIMESTEPS` (scale ~proportionally with state count)
4. Evaluate on held-out states to measure generalization

### Phase S4: Training Improvements

| Improvement | Impact | Effort |
|-------------|--------|--------|
| Increase timesteps (100k → 500k–1M) | Higher win rate, better generalization | Low |
| Increase parallel envs (4 → 8) | ~2x training throughput | Low |
| Add HP conservation reward bonus | More efficient play | Low |
| Add player/opponent level to obs | Handle level disparity | Medium (retrains model) |
| Add status condition tracking | Poison/Paralyze awareness | Medium (new RAM reads + obs dims) |
| Self-play opponent randomization | Reduce overfitting to ROM AI | High |

---

## 4. File Inventory

```
simulation/
├── train.py                # PPO training loop, EmeraldBattleWrapper, SubprocVecEnv
├── evaluate.py             # Visual evaluation with Pygame rendering
├── play_and_save.py        # Manual play in stable-retro, keyboard input, state saving
├── pokedex.py              # SPECIES_DATA, MOVES_DATA, TYPE_CHART, get_effectiveness()
├── PLAN.MD                 # This document
├── README.md               # Quick-start guide
└── data/
    └── PokemonEmerald-GBA/
        ├── data.json       # RAM addresses (my_hp, enemy_hp, moves, species)
        ├── metadata.json   # Button layout: [B, Y, SELECT, START, UP, DOWN, LEFT, RIGHT, A, X, L, R]
        ├── scenario.json   # Reward/done conditions (currently empty)
        ├── rom.sha         # ROM integrity hash
        ├── *.state         # Training save states (gzip format, stable-retro native)
        └── general_saves/  # Overworld states for progression
```

---

## Historical Note

The original version of this document (pre-2025-01) described setting up stable-retro from scratch (env init, RAM hunting, first training loop, deployment bridge, system integration). All of those foundational steps are **complete**. This rewrite reflects the current state and forward plan.
