# Directive System Documentation

## Overview

The Directive System is a lightweight tactical guidance layer that bridges the gap between high-level strategic objectives and specific game actions. It was implemented to solve the Route 103 rival battle bug where the agent would reach the area but not interact with the rival.

## Architecture

### Components

1. **ObjectiveManager** (`agent/objective_manager.py`)
   - Tracks game milestones and objectives
   - Provides tactical directives based on current state
   - Maintains persistent flags (e.g., `rival_battle_completed`)

2. **Action Controller Integration** (`agent/action.py`)
   - Priority 0C: Directive execution
   - Positioned between Opener Bot (0B) and Dialogue Detection (1)
   - Returns actions directly (batched A* paths, single button presses)

### Key Features

- **Position-based State Tracking**: Uses player position + game state to detect completion
- **Persistent State Flags**: Events like battle completion persist even after moving away
- **Dialogue Integration**: Detects active dialogue and prioritizes it over navigation
- **A* Pathfinding**: Uses same pathfinding as Opener Bot for obstacle avoidance
- **Batched Actions**: A* paths returned in full for efficient execution

## Implementation Details

### Directive Structure

```python
{
    'action': 'NAVIGATE_AND_INTERACT',  # or 'INTERACT', 'DIALOGUE'
    'target': (x, y, 'MAP_NAME'),       # Coordinate target
    'description': 'Walk to rival and press A',  # Human-readable
    'milestone': 'FIRST_RIVAL_BATTLE'   # Expected milestone
}
```

### Current Directives

#### 1. Route 103 Rival Battle
- **Trigger**: `ROUTE_103` milestone complete, battle not done
- **Target**: Position (9, 3) on Route 103
- **Actions**:
  - Navigate to rival position
  - Press A to interact
  - Detect battle completion (position-based)
  - Handle post-battle dialogue
  - Navigate south to leave Route 103

**Key Fix - Battle State Transition Detection:**
```python
# Track if we were in battle at rival position
if in_battle and at_rival_position:
    self._was_in_rival_battle = True

# Battle complete = was in battle → now not in battle (at rival position)
battle_just_completed = self._was_in_rival_battle and at_rival_position and not in_battle

# Once battle detected as complete, set persistent flag
if battle_just_completed or is_milestone_complete('FIRST_RIVAL_BATTLE'):
    self.rival_battle_completed = True

# Use persistent flag (stays true even after leaving position)
rival_battle_complete = self.rival_battle_completed
```

This prevents false positives (arriving at 9,3 before battle) and ensures the flag only sets after the actual battle completes.

**Key Fix - Dialogue Detection:**
```python
# Add visual_dialogue_active to state_data before calling directive
state_data['visual_dialogue_active'] = visual_dialogue_active

# Check dialogue in directive logic
if is_dialogue_active():
    return {'action': 'DIALOGUE', 'description': 'Press A to advance dialogue'}
```

This ensures dialogue is completed before attempting navigation.

#### 2. Post-Battle Healing (Planned)
- Navigate to Oldale Town Pokemon Center
- Enter and heal Pokemon
- Return to continue journey

#### 3. Return to Lab (Planned)
- Navigate back to Birch's Lab
- Receive Pokedex
- Continue to Route 102

## Bug Fixes Applied

### 1. Visual Dialogue Active Not Passed
**Problem**: `visual_dialogue_active` parameter wasn't added to `state_data`  
**Fix**: Add to state_data before calling `get_next_action_directive()`  
**Location**: `agent/action.py` line ~1379

### 2. Battle Complete Detection During Dialogue
**Problem**: Battle detection checked `and not is_dialogue_active()`, preventing detection during post-battle dialogue  
**Fix**: Separate detection (battle complete) from action selection (press A vs navigate)  
**Location**: `agent/objective_manager.py` line ~314

### 3. Battle Completion False Positive
**Problem**: Being at (9,3) with `in_battle=False` immediately set battle complete, even before interacting with rival  
**Fix**: Use state transition tracking - only mark complete when transitioning from `in_battle=True` to `in_battle=False`  
**Location**: `agent/objective_manager.py` line ~40, ~315

### 4. Position-Based State Not Persisting
**Problem**: Moving away from (9,3) reset battle completion flag  
**Fix**: Add `self.rival_battle_completed` persistent flag  
**Location**: `agent/objective_manager.py` line ~40, ~318

## Testing

### Unit Tests
Run comprehensive test suite:
```bash
python test_directive_system.py
```

Tests cover:
- ✅ Directive generation at Route 103
- ✅ INTERACT when at rival position
- ✅ No directive before Route 103
- ✅ Correct behavior after rival battle
- ✅ Pokemon Center healing directive

### Integration Testing
```bash
python run.py --save_state Emerald-GBAdvance/route102_hackathon.state --max_steps 50 --headless
```

**Expected Behavior:**
1. Battle completion detected
2. Post-battle dialogue (multiple A presses)
3. Dialogue ends
4. Navigate south from Route 103
5. Enter Oldale Town
6. Continue toward Pokemon Center

### Success Criteria
- ✅ Agent completes rival battle
- ✅ Agent completes all post-battle dialogue
- ✅ Agent navigates away from Route 103 without oscillating
- ✅ No stuck detection warnings after dialogue ends
- ✅ Smooth transition to next objective

## Log Signatures

### Directive Activation
```
🔍 [DIRECTIVE DEBUG] objective_manager exists: <ObjectiveManager object>
🔍 [OBJECTIVE_MANAGER] get_next_action_directive() CALLED
```

### Dialogue Handling
```
💬 [DIALOGUE] Active - waiting for dialogue to finish
🔍 [RIVAL BATTLE] Position check: at (9,3)=True, battle=False, dialogue=True, complete=True
📍 [DIRECTIVE] Press A to advance dialogue
```

### Navigation
```
🔍 [RIVAL BATTLE] Position check: at (9,3)=True, battle=False, dialogue=False, complete=True
📍 [DIRECTIVE] Leave Route 103 south to return to Oldale Town
✅ [LOCAL A*] Found path to south edge: DOWN -> LEFT -> ...
```

## Future Enhancements

### Short-term
- Add Oldale Pokemon Center healing directive
- Add return to Birch's Lab directive
- Test full sequence from Route 103 to Route 102

### Long-term
- Expand to cover all major story points
- Add trainer battle sequencing
- Add item collection directives
- Add optional objective handling

## Related Documentation

- **Architecture**: `docs/ARCHITECTURAL_BLUEPRINT.md`
- **Opener Bot**: `docs/OPENER_BOT.md`
- **Dialogue System**: `docs/DIALOGUE_SYSTEM.md`
- **Project Plan**: `docs/PROJECT_PLAN.md`
