"""
Stuck Detection and Recovery Module for Pokemon Emerald Agent.

Detects when the agent is stuck (position not changing despite movement),
handles warp/teleport detection, and manages dynamic tile blocking for
NPC obstacles.

State managed by this module:
- Stuck counter and direction tracking
- Warp detection flags and position history
- Portal wait counters and approach directions
- Post-dialogue movement limiting
- Dismissed monologue tracking

Side effects:
- Writes to pathfinding._dynamically_blocked_tiles when NPC obstacles detected
- Reads from pathfinding._recent_positions for position history
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from agent.pathfinding import (
    _dynamically_blocked_tiles,
    _recent_positions,
    _DIRECTION_OFFSETS,
)

logger = logging.getLogger(__name__)

# === STUCK DETECTION STATE ===
_stuck_counter = 0
_last_position = None
_last_stuck_direction = None  # Track direction that caused stuck (for NPC dialogue detection)

# === WARP DETECTION STATE ===
_needs_warp_settle_b_press = False
_last_known_position = None
_last_location = None  # For location-change warp detection
_warp_wait_frames = 0

# === PORTAL TRACKING ===
# Track portal waiting - dict of {(x, y, map): frame_count}
portal_wait_frames = {}
# Track which direction we approached each portal from
portal_approach_direction = {}

# === DIALOGUE TRACKING ===
_dismissed_monologues = set()
_post_dialogue_movement_count = 0
_was_in_dialogue = False


def decrement_blocked_tile_ttls():
    """Decrement TTLs for dynamically blocked tiles, remove expired ones."""
    expired_tiles = [k for k, v in _dynamically_blocked_tiles.items() if v <= 0]
    for k in expired_tiles:
        print(f"🔓 [DYNAMIC BLOCK] Tile {k[:2]} on {k[2]} unblocked (TTL expired)")
        del _dynamically_blocked_tiles[k]
    for k in _dynamically_blocked_tiles:
        _dynamically_blocked_tiles[k] -= 1
    if _dynamically_blocked_tiles:
        print(f"🚫 [DYNAMIC BLOCK] Currently blocked tiles: {[(k[:2], v) for k, v in _dynamically_blocked_tiles.items()]}")


def update_position_tracking(current_x, current_y, location):
    """
    Track current position and detect warps/teleports.

    Updates _recent_positions (in pathfinding module) and sets the
    _needs_warp_settle_b_press flag if a warp is detected.

    Args:
        current_x: Current player X coordinate
        current_y: Current player Y coordinate
        location: Current map location name
    """
    global _needs_warp_settle_b_press, _last_known_position, _last_location

    if current_x is None or current_y is None or not location:
        return

    current_pos_key = (current_x, current_y, location)

    # Only add if it's a new position (not the same as the last one)
    if not _recent_positions or _recent_positions[-1] != current_pos_key:
        _recent_positions.append(current_pos_key)

    # WARP DETECTION: Check for extreme coordinate jumps (>20 tiles in one step)
    if _last_known_position is not None:
        last_x, last_y, last_loc = _last_known_position
        dx = abs(current_x - last_x)
        dy = abs(current_y - last_y)
        distance = max(dx, dy)

        if distance > 20:
            logger.info(f"🌀 [WARP JUMP] Extreme coordinate jump: ({last_x}, {last_y}) → ({current_x}, {current_y}) = {distance} tiles")
            print(f"🌀 [WARP JUMP] Teleported {distance} tiles! Pressing B to settle position")
            _needs_warp_settle_b_press = True

    _last_known_position = (current_x, current_y, location)

    # Detect location change (warp occurred)
    if _last_location is None:
        _last_location = location

    if location and location != _last_location:
        print(f"🌀 [WARP DETECTED] Location changed: '{_last_location}' → '{location}'")
        logger.info(f"🌀 [WARP DETECTED] Location changed: '{_last_location}' → '{location}'")
        _needs_warp_settle_b_press = True
        logger.info(f"⏸️ [WARP FLAG] Set _needs_warp_settle_b_press = True due to location change")
        print(f"⏸️ [WARP FLAG] Will press B to settle after warp")
        _last_location = location


def should_settle_warp():
    """
    Check if we need to press B after a warp to settle position.
    Consumes (resets) the flag.

    Returns:
        True if B should be pressed, False otherwise
    """
    global _needs_warp_settle_b_press
    if _needs_warp_settle_b_press:
        logger.info(f"⏸️ [WARP SETTLE] Pressing B after warp teleport to settle position")
        print(f"⏸️ [WARP SETTLE] Pressing B after teleport to settle position")
        _needs_warp_settle_b_press = False
        return True
    return False


def check_stuck(state_data, recent_actions, visual_dialogue_active,
                current_x, current_y, location):
    """
    Check if the agent is stuck (position not changing despite movement attempts).

    Handles:
    - Stuck counter incrementing when position unchanged after directional input
    - Dynamic tile blocking after 3+ stuck attempts
    - NPC dialogue detection (dialogue appearing after failed directional move)
    - Recovery via A button press when tile already blocked

    Args:
        state_data: Current game state
        recent_actions: List of recent actions
        visual_dialogue_active: Whether VLM detected dialogue
        current_x: Current player X coordinate
        current_y: Current player Y coordinate
        location: Current map location name

    Returns:
        Action list (['A']) if recovery action needed, None otherwise
    """
    global _stuck_counter, _last_position, _last_stuck_direction

    try:
        game_data = state_data.get('game', {})
        in_battle = game_data.get('in_battle', False)

        current_pos = (current_x, current_y, location) if current_x is not None and current_y is not None else None

        # Check if last action was a direction
        last_action_was_direction = False
        if recent_actions and len(recent_actions) > 0:
            last_action = recent_actions[-1]
            last_action_was_direction = last_action in ['UP', 'DOWN', 'LEFT', 'RIGHT']

        if _stuck_counter > 0:
            print(f"🔄 [STUCK DEBUG] visual_dialogue={visual_dialogue_active}, in_battle={in_battle}, current_pos={current_pos}")
            print(f"🔄 [STUCK DEBUG] _last_position={_last_position}, last_action_was_direction={last_action_was_direction}")
            print(f"🔄 [STUCK DEBUG] recent_actions[-1]={recent_actions[-1] if recent_actions else 'None'}")
            print(f"🔄 [STUCK DEBUG] _stuck_counter={_stuck_counter}")

        # Check if we're NOT in dialogue and NOT in battle
        if not visual_dialogue_active and not in_battle and current_pos is not None:
            if _stuck_counter > 0:
                print(f"🔄 [STUCK DEBUG] Conditions met: not in dialogue, not in battle, has position")

            if _last_position == current_pos and last_action_was_direction:
                # We tried to move but didn't - increment stuck counter
                _stuck_counter += 1
                stuck_dir = recent_actions[-1] if recent_actions else None
                _last_stuck_direction = stuck_dir
                print(f"🔄 [STUCK DETECTION] Position unchanged after direction input: {current_pos}")
                print(f"   Last action: {stuck_dir}")
                print(f"   Stuck counter: {_stuck_counter}")

                if _stuck_counter >= 3:
                    print(f"🔄 [STUCK DETECTION] Stuck for {_stuck_counter} attempts!")

                    if stuck_dir and stuck_dir in _DIRECTION_OFFSETS and current_x is not None and current_y is not None:
                        dx, dy = _DIRECTION_OFFSETS[stuck_dir]
                        blocked_x = current_x + dx
                        blocked_y = current_y + dy
                        blocked_key = (blocked_x, blocked_y, location)

                        if blocked_key not in _dynamically_blocked_tiles:
                            _dynamically_blocked_tiles[blocked_key] = 200
                            print(f"🚫 [DYNAMIC BLOCK] Marked tile ({blocked_x}, {blocked_y}) on {location} as blocked (NPC/obstacle)")
                            print(f"   Agent at ({current_x}, {current_y}), tried {stuck_dir}")
                            logger.info(f"🚫 [DYNAMIC BLOCK] Blocked ({blocked_x}, {blocked_y}) on {location} - stuck {_stuck_counter}x going {stuck_dir}")
                            _stuck_counter = 0
                            _last_stuck_direction = None
                            # DON'T press A - fall through to let pathfinding re-route
                        else:
                            print(f"🔄 [STUCK DETECTION] Tile already blocked, trying A for hidden dialogue...")
                            _stuck_counter = 0
                            _last_stuck_direction = None
                            logger.info(f"🔄 [STUCK RECOVERY] Tile already blocked, pressing A for hidden dialogue")
                            _last_position = current_pos
                            return ['A']
                    else:
                        print(f"🔄 [STUCK DETECTION] Can't determine blocked tile, pressing A...")
                        _stuck_counter = 0
                        _last_stuck_direction = None
                        _last_position = current_pos
                        return ['A']
            else:
                if _stuck_counter > 0:
                    print(f"🔄 [STUCK DETECTION] Position changed or no direction input - resetting counter")
                    print(f"   Reason: _last_position={_last_position} vs current_pos={current_pos}")
                    print(f"   last_action_was_direction={last_action_was_direction}")
                _stuck_counter = 0
                _last_stuck_direction = None
        else:
            # In dialogue or battle - check if this is NPC-triggered dialogue
            if visual_dialogue_active and not in_battle and _last_stuck_direction and current_pos and _last_position == current_pos:
                _stuck_counter += 1
                print(f"🔄 [STUCK DETECTION] NPC dialogue detected after failed {_last_stuck_direction} move (stuck counter: {_stuck_counter})")

                if _last_stuck_direction in _DIRECTION_OFFSETS and current_x is not None and current_y is not None:
                    dx, dy = _DIRECTION_OFFSETS[_last_stuck_direction]
                    blocked_x = current_x + dx
                    blocked_y = current_y + dy
                    blocked_key = (blocked_x, blocked_y, location)

                    if blocked_key not in _dynamically_blocked_tiles:
                        _dynamically_blocked_tiles[blocked_key] = 200
                        print(f"🚫 [DYNAMIC BLOCK] NPC at ({blocked_x}, {blocked_y}) on {location} — blocking tile")
                        print(f"   Agent at ({current_x}, {current_y}), walked {_last_stuck_direction} into NPC dialogue")
                        logger.info(f"🚫 [DYNAMIC BLOCK] NPC dialogue block ({blocked_x}, {blocked_y}) on {location} - walked {_last_stuck_direction}")
                    else:
                        print(f"🔄 [STUCK DETECTION] NPC tile ({blocked_x}, {blocked_y}) already blocked, continuing dialogue...")
            elif _stuck_counter > 0 and (visual_dialogue_active or in_battle):
                print(f"🔄 [STUCK DETECTION] In dialogue/battle (not NPC stuck) - resetting stuck counter")
                _stuck_counter = 0
                _last_stuck_direction = None
            print(f"🔄 [STUCK DEBUG] Conditions NOT met: dialogue={visual_dialogue_active}, battle={in_battle}, pos={current_pos}")

        # Update last position for next iteration
        _last_position = current_pos

    except Exception as e:
        print(f"⚠️ [STUCK DETECTION] Error in stuck detection: {e}")
        import traceback
        traceback.print_exc()

    return None


def check_post_dialogue_limit(actions, latest_observation):
    """
    Limit post-dialogue movement to prevent infinite dialogue loops.

    After dialogue ends, limits the agent to 3 directional movements
    before forcing a re-evaluation. Prevents loops like repeatedly
    triggering Mom's dialogue.

    Args:
        actions: The action list about to be returned
        latest_observation: Latest perception output

    Returns:
        Empty list [] if movement limit reached (forces re-evaluation),
        None if no override needed (caller should return actions as-is)
    """
    global _was_in_dialogue, _post_dialogue_movement_count

    directional_actions = {'UP', 'DOWN', 'LEFT', 'RIGHT'}
    is_movement = any(action in directional_actions for action in actions)

    # Check if dialogue is currently active
    is_dialogue_active = False
    if isinstance(latest_observation, dict) and 'visual_data' in latest_observation:
        visual_elements = latest_observation.get('visual_data', {}).get('visual_elements', {})
        screen_context = latest_observation.get('visual_data', {}).get('screen_context', '')
        is_dialogue_active = (
            screen_context == 'dialogue' or
            visual_elements.get('text_box_visible', False) or
            visual_elements.get('continue_prompt_visible', False)
        )

    # When dialogue ends, start counting movements
    if _was_in_dialogue and not is_dialogue_active:
        logger.info(f"📍 [POST-DIALOGUE] Dialogue ended, starting movement counter")
        print(f"✅ [DIALOGUE] Dialogue completed - tracking next {3 - _post_dialogue_movement_count} movements")

    # Update dialogue state
    if is_dialogue_active:
        _was_in_dialogue = True
        _post_dialogue_movement_count = 0
    elif _was_in_dialogue:
        pass  # Dialogue just ended, we're now post-dialogue

    # If this is a movement after dialogue, increment counter and check limit
    if is_movement and _was_in_dialogue and not is_dialogue_active:
        _post_dialogue_movement_count += 1
        logger.info(f"📍 [POST-DIALOGUE] Movement #{_post_dialogue_movement_count}/3 after dialogue")
        print(f"🚶 [POST-DIALOGUE] Movement {_post_dialogue_movement_count}/3 - {actions[0]}")

        if _post_dialogue_movement_count >= 3:
            logger.warning(f"🚨 [POST-DIALOGUE] Reached 3-movement limit! Forcing empty action to re-check dialogue")
            print(f"⚠️ [POST-DIALOGUE] 3 movements completed - pausing to check for dialogue")
            _post_dialogue_movement_count = 0
            _was_in_dialogue = False
            return []  # Force re-evaluation

    return None  # No override needed
