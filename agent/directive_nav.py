"""
Directive Navigation Executor for Pokemon Emerald Agent.

Converts Directive objects from the ObjectiveManager into button presses.
Handles all directive types: goal_coords, goal_direction, NAVIGATE,
NAVIGATE_AND_INTERACT, NAVIGATE_DIRECTION, MOVE_UNTIL_MAP_CHANGE,
CROSS_BOUNDARY, INTERACT, DIALOGUE, WAIT_FOR_DIALOGUE.

Extracted from the PRIORITY 0C block of action.py to reduce God-Object size.
"""

import logging
import re
from typing import Dict, Any, Optional, List, Tuple

from agent.pathfinding import (
    _pathfind_to_target, _local_pathfind_from_tiles,
    _astar_pathfind_with_grid_data, pathfind_to_goal,
    _dynamically_blocked_tiles, _recent_positions,
    update_npc_obstacles,
)
from agent import stuck_handler

logger = logging.getLogger(__name__)

# Direction string → button mapping
_DIRECTION_MAP = {
    'north': 'UP', 'south': 'DOWN', 'east': 'RIGHT', 'west': 'LEFT',
    'up': 'UP', 'down': 'DOWN', 'left': 'LEFT', 'right': 'RIGHT',
}

# Walkable tile symbols for smart exploration scanning
_WALKABLE_TILES = ['.', '_', '~', 'D', 'S', '^', 'v', '<', '>', 'L', '?']

# Portal tracking state (accessed directly, lives in stuck_handler)
_portal_wait_frames = stuck_handler.portal_wait_frames
_portal_approach_direction = stuck_handler.portal_approach_direction


def _get_current_orientation(recent_actions):
    """Extract player's current facing orientation from recent actions."""
    if recent_actions:
        for action in reversed(recent_actions):
            if isinstance(action, list):
                action = action[0] if action else None
            if action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                return action
    return None


def _handle_goal_coords(directive, state_data, recent_actions, description, location):
    """
    Navigate to specific goal coordinates (x, y, map).

    Handles:
    - At-goal interaction with optional NPC facing
    - A* pathfinding (local → global)
    - Smart exploration fallback when A* fails
    """
    goal_coords = directive['goal_coords']
    should_interact = directive.get('should_interact', False)
    npc_coords = directive.get('npc_coords')
    avoid_grass = directive.get('avoid_grass', True)
    press_b_first = directive.get('press_b_first', False)

    print(f"🔍 [GOAL_COORDS] goal_coords={goal_coords}, should_interact={should_interact}, npc_coords={npc_coords}, avoid_grass={avoid_grass}, press_b_first={press_b_first}")

    if press_b_first:
        logger.info(f"⏸️ [PRESS B FIRST] Pressing B to settle warp before navigation")
        print(f"⏸️ [PRESS B FIRST] Pressing B to settle warp before navigation")
        return ['B']

    if len(goal_coords) == 3:
        target_x, target_y, target_map = goal_coords
    else:
        logger.error(f"❌ [DIRECTIVE] Invalid goal_coords format: {goal_coords}")
        return []

    logger.info(f"🎯 [DIRECTIVE] Navigating to ({target_x}, {target_y}) in {target_map}, interact={should_interact}, npc_coords={npc_coords}")

    from agent.opener_bot import NavigationGoal
    nav_goal = NavigationGoal(
        x=target_x, y=target_y,
        map_location=target_map,
        should_interact=should_interact,
        description=description
    )

    current_x = state_data.get('player', {}).get('position', {}).get('x', 0)
    current_y = state_data.get('player', {}).get('position', {}).get('y', 0)
    goal_x = nav_goal.x
    goal_y = nav_goal.y

    logger.info(f"🗺️ [DIRECTIVE NAV] Current: ({current_x}, {current_y}) -> Goal: ({goal_x}, {goal_y})")
    print(f"🔍 [GOAL_COORDS] Current: ({current_x}, {current_y}), Goal: ({goal_x}, {goal_y})")
    print(f"🔍 [GOAL_COORDS] Position check: x match={current_x == goal_x}, y match={current_y == goal_y}")

    # --- AT GOAL ---
    if current_x == goal_x and current_y == goal_y:
        print(f"🔍 [GOAL_COORDS] INSIDE position match block! should_interact={should_interact}")
        logger.info(f"🔍 [GOAL_COORDS] At exact goal position, should_interact={should_interact}")

        if should_interact:
            if npc_coords:
                npc_x, npc_y = npc_coords
                required_direction = None
                if current_x < npc_x:
                    required_direction = 'RIGHT'
                elif current_x > npc_x:
                    required_direction = 'LEFT'
                elif current_y < npc_y:
                    required_direction = 'DOWN'
                elif current_y > npc_y:
                    required_direction = 'UP'

                current_orientation = _get_current_orientation(recent_actions)
                if current_orientation == required_direction:
                    logger.info(f"🗺️ [DIRECTIVE NAV] At goal, facing NPC at ({npc_x},{npc_y}) {required_direction} - pressing A")
                    return ['A']
                else:
                    logger.info(f"🗺️ [DIRECTIVE NAV] At goal, turning {required_direction} to face NPC at ({npc_x},{npc_y})")
                    return [required_direction]
            else:
                logger.info(f"🗺️ [DIRECTIVE NAV] At exact goal - pressing A to interact")
                return ['A']
        else:
            logger.info(f"🗺️ [DIRECTIVE NAV] At goal, no interaction needed")
            print(f"🔍 [GOAL_COORDS] At goal, no interaction needed - returning empty list []")
            return []

    # --- DIRECTION CALCULATION ---
    dx = goal_x - current_x
    dy = goal_y - current_y
    required_direction = None
    if abs(dy) > abs(dx):
        required_direction = 'UP' if current_y > goal_y else 'DOWN'
    elif abs(dx) > abs(dy):
        required_direction = 'RIGHT' if current_x < goal_x else 'LEFT'
    else:
        if dy != 0:
            required_direction = 'UP' if dy < 0 else 'DOWN'
        elif dx != 0:
            required_direction = 'RIGHT' if dx > 0 else 'LEFT'

    distance = abs(current_x - goal_x) + abs(current_y - goal_y)

    # --- NAVIGATE TO GOAL ---
    if distance > 0:
        print(f"🔍 [GOAL_COORDS] Distance={distance}, about to call pathfind_to_goal({goal_x}, {goal_y})")

        pathfound_action = pathfind_to_goal(state_data, goal_x, goal_y, avoid_grass=avoid_grass,
                                             npc_coords=npc_coords)

        if pathfound_action:
            path_list = pathfound_action if isinstance(pathfound_action, list) else [pathfound_action]
            logger.info(f"🗺️ [DIRECTIVE NAV] A* path to ({goal_x}, {goal_y}): {path_list} ({len(path_list)} steps, distance={distance})")
            print(f"🗺️ [DIRECTIVE NAV] A* found {len(path_list)}-step path: {' → '.join(path_list)}")
            return path_list

        # --- SMART EXPLORATION FALLBACK ---
        logger.warning(f"⚠️ [DIRECTIVE NAV] All pathfinding methods failed for goal ({goal_x}, {goal_y})")
        print(f"⚠️ [DIRECTIVE NAV] A* failed, using smart exploration fallback")

        primary_direction = None
        perpendicular_directions = []
        if abs(dx) > abs(dy):
            primary_direction = 'RIGHT' if dx > 0 else 'LEFT'
            perpendicular_directions = ['UP', 'DOWN']
        else:
            primary_direction = 'DOWN' if dy > 0 else 'UP'
            perpendicular_directions = ['LEFT', 'RIGHT']

        print(f"🔍 [SMART EXPLORATION] A* couldn't reach ({goal_x}, {goal_y})")
        print(f"🔍 [SMART EXPLORATION] Testing which directions have explorable tiles...")

        exploration_direction = None

        stitched_map_info = state_data.get('map', {}).get('stitched_map_info')
        grid_data = {}
        if stitched_map_info and stitched_map_info.get('available'):
            current_area = stitched_map_info.get('current_area', {})
            grid_data = current_area.get('grid', {})

        # Check primary direction
        primary_x, primary_y = current_x, current_y
        if primary_direction == 'UP':
            primary_y -= 1
        elif primary_direction == 'DOWN':
            primary_y += 1
        elif primary_direction == 'LEFT':
            primary_x -= 1
        elif primary_direction == 'RIGHT':
            primary_x += 1

        primary_tile_key = f"{primary_x},{primary_y}"
        primary_tile = grid_data.get(primary_tile_key) if grid_data else None

        primary_dynamically_blocked = (primary_x, primary_y, location) in _dynamically_blocked_tiles
        if primary_dynamically_blocked:
            print(f"🚫 [EXPLORATION] {primary_direction} → ({primary_x}, {primary_y}) dynamically blocked (NPC/obstacle)")
        elif primary_tile is None:
            print(f"🔍 [EXPLORATION] {primary_direction} → ({primary_x}, {primary_y}) is UNEXPLORED (frontier)")
            exploration_direction = primary_direction
        elif primary_tile in _WALKABLE_TILES:
            print(f"✅ [EXPLORATION] {primary_direction} → ({primary_x}, {primary_y}) is walkable ('{primary_tile}')")
            exploration_direction = primary_direction
        else:
            print(f"❌ [EXPLORATION] {primary_direction} → ({primary_x}, {primary_y}) blocked ('{primary_tile}')")
            print(f"🔍 [LOCAL SEARCH] Scanning perpendicular directions for unexplored tiles...")

            best_unexplored = None
            best_distance = 999

            for perp_dir in perpendicular_directions:
                for dist in range(1, 6):
                    scan_x, scan_y = current_x, current_y
                    if perp_dir == 'UP':
                        scan_y -= dist
                    elif perp_dir == 'DOWN':
                        scan_y += dist
                    elif perp_dir == 'LEFT':
                        scan_x -= dist
                    elif perp_dir == 'RIGHT':
                        scan_x += dist

                    scan_key = f"{scan_x},{scan_y}"
                    scan_tile = grid_data.get(scan_key) if grid_data else None

                    if scan_tile is None:
                        if dist < best_distance:
                            best_unexplored = (perp_dir, dist, scan_x, scan_y)
                            best_distance = dist
                        print(f"   🎯 [SCAN] {perp_dir} +{dist} → ({scan_x}, {scan_y}) UNEXPLORED")
                        break
                    elif scan_tile in _WALKABLE_TILES:
                        continue
                    else:
                        print(f"   🚫 [SCAN] {perp_dir} +{dist} → ({scan_x}, {scan_y}) blocked ('{scan_tile}')")
                        break

            if best_unexplored:
                exploration_direction = best_unexplored[0]
                dist = best_unexplored[1]
                exp_x, exp_y = best_unexplored[2], best_unexplored[3]
                print(f"✅ [LOCAL SEARCH] Found unexplored tile at ({exp_x}, {exp_y}) via {exploration_direction} (+{dist} tiles)")
                print(f"🔍 [SMART EXPLORATION] Moving {exploration_direction} to explore and find gap toward goal")
            else:
                # Scan opposite direction
                opposite_map = {'UP': 'DOWN', 'DOWN': 'UP', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
                opposite_direction = opposite_map.get(primary_direction)

                if opposite_direction:
                    print(f"🔄 [LOCAL SEARCH] No unexplored in perpendiculars, scanning OPPOSITE direction ({opposite_direction})...")
                    for dist in range(1, 6):
                        scan_x, scan_y = current_x, current_y
                        if opposite_direction == 'UP':
                            scan_y -= dist
                        elif opposite_direction == 'DOWN':
                            scan_y += dist
                        elif opposite_direction == 'LEFT':
                            scan_x -= dist
                        elif opposite_direction == 'RIGHT':
                            scan_x += dist

                        scan_key = f"{scan_x},{scan_y}"
                        scan_tile = grid_data.get(scan_key) if grid_data else None

                        if scan_tile is None:
                            print(f"   🎯 [SCAN] {opposite_direction} +{dist} → ({scan_x}, {scan_y}) UNEXPLORED")
                            best_unexplored = (opposite_direction, dist, scan_x, scan_y)
                            break
                        elif scan_tile in _WALKABLE_TILES:
                            continue
                        else:
                            print(f"   🚫 [SCAN] {opposite_direction} +{dist} → ({scan_x}, {scan_y}) blocked ('{scan_tile}')")
                            break

                if best_unexplored:
                    exploration_direction = best_unexplored[0]
                    dist = best_unexplored[1]
                    exp_x, exp_y = best_unexplored[2], best_unexplored[3]
                    print(f"✅ [LOCAL SEARCH] Found unexplored tile at ({exp_x}, {exp_y}) via {exploration_direction} (opposite dir, +{dist} tiles)")
                    print(f"🔍 [SMART EXPLORATION] Moving {exploration_direction} AWAY from goal to find detour path")
                else:
                    # No unexplored in any direction — try walkable
                    print(f"⚠️ [LOCAL SEARCH] No unexplored tiles found in any direction")

                    from agent.action import action_step  # lazy import for position_history
                    pos_history = getattr(action_step, 'position_history', [])
                    recent_positions_set = set(pos_history[-6:]) if len(pos_history) >= 4 else set()
                    is_oscillating = len(pos_history) >= 6 and len(set(pos_history[-6:])) <= 2

                    if is_oscillating:
                        print(f"🔄 [ANTI-OSCILLATION] Detected oscillation! Prioritizing OPPOSITE direction ({opposite_direction}) to break free")
                        all_directions = ([opposite_direction] if opposite_direction else []) + perpendicular_directions
                    else:
                        all_directions = perpendicular_directions + ([opposite_direction] if opposite_direction else [])

                    for test_dir in all_directions:
                        test_x, test_y = current_x, current_y
                        if test_dir == 'UP':
                            test_y -= 1
                        elif test_dir == 'DOWN':
                            test_y += 1
                        elif test_dir == 'LEFT':
                            test_x -= 1
                        elif test_dir == 'RIGHT':
                            test_x += 1

                        test_key = f"{test_x},{test_y}"
                        test_tile = grid_data.get(test_key) if grid_data else None

                        if test_tile and test_tile in _WALKABLE_TILES:
                            if (test_x, test_y, location) in _dynamically_blocked_tiles:
                                print(f"🚫 [FALLBACK] {test_dir} → ({test_x}, {test_y}) dynamically blocked")
                                continue
                            if is_oscillating and (test_x, test_y) in recent_positions_set:
                                print(f"🔄 [FALLBACK] {test_dir} → ({test_x}, {test_y}) skipped (would return to recent position)")
                                continue
                            print(f"✅ [FALLBACK] {test_dir} → ({test_x}, {test_y}) is walkable ('{test_tile}')")
                            exploration_direction = test_dir
                            break

        if not exploration_direction:
            print(f"⚠️ [SMART EXPLORATION] All directions blocked, trying primary anyway")
            exploration_direction = primary_direction
            logger.info(f"🧭 [FALLBACK NAV] Last resort: trying {primary_direction} toward goal (Δx={dx}, Δy={dy})")
        else:
            logger.info(f"🔍 [SMART EXPLORATION] Selected {exploration_direction} (exploration strategy)")

        logger.info(f"🔍 [SMART EXPLORATION] Moving {exploration_direction} (stuck={stuck_handler._stuck_counter})")
        print(f"🔍 [SMART EXPLORATION] Moving {exploration_direction}")
        return [exploration_direction]

    # distance=0 should have been handled above
    logger.warning(f"⚠️ [DIRECTIVE NAV] Unexpected state: at goal but not caught by check above")
    return []


def _handle_goal_direction(directive, state_data, recent_actions):
    """Navigate toward a map edge direction (north/south/east/west)."""
    goal_direction = directive['goal_direction']
    logger.info(f"🎯 [DIRECTIVE] Navigating {goal_direction}")
    print(f"🎯 [DIRECTIVE] Processing goal_direction: {goal_direction}")

    suggested_action = _local_pathfind_from_tiles(state_data, goal_direction, recent_actions)
    if suggested_action:
        path_list = suggested_action if isinstance(suggested_action, list) else [suggested_action]
        logger.info(f"🗺️ [DIRECTIVE] Directional pathfinding: {' → '.join(path_list)} ({len(path_list)} steps)")
        print(f"🗺️ [DIRECTIVE] Directional path {goal_direction}: {' → '.join(path_list)}")
        return path_list
    else:
        logger.info(f"🚪 [DIRECTIVE] No walkable tiles {goal_direction} - pushing direction to trigger warp")
        print(f"🚪 [DIRECTIVE] No walkable path {goal_direction} - pushing direction to cross boundary")
        action = _DIRECTION_MAP.get(goal_direction.lower(), 'UP')
        logger.info(f"🚪 [DIRECTIVE] Boundary crossing {goal_direction}: {action}")
        print(f"🚪 [DIRECTIVE] Pushing {action} to cross boundary")
        return [action]


def _handle_wait_for_transition(directive, state_data):
    """Wait for a warp/map transition to complete."""
    expected_location = directive.get('expected_location', '')
    current_location = state_data.get('player', {}).get('location', '').upper()

    logger.info(f"⏳ [WAIT] Waiting for transition to {expected_location} (current: {current_location})")
    print(f"⏳ [WAIT] Waiting for transition to {expected_location} (current: {current_location})")

    if expected_location and expected_location.upper() in current_location:
        logger.info(f"✅ [WAIT] Transition complete - now in {current_location}")
        print(f"✅ [WAIT] Transition complete - now in {current_location}")
        return []
    else:
        logger.info(f"⏳ [WAIT] Still waiting for transition...")
        print(f"⏳ [WAIT] Still waiting...")
        return []


def _handle_navigate_direction(directive, state_data):
    """Directional navigation with optional portal coordinates."""
    direction = directive.get('direction', 'south')
    target_location = directive.get('target_location', '').upper()
    portal_coords = directive.get('portal_coords')
    proximity_radius = directive.get('proximity_radius', 5)

    player_data = state_data.get('player', {})
    current_location = player_data.get('location', '').upper()
    position = player_data.get('position', {})
    current_x = position.get('x', 0)
    current_y = position.get('y', 0)

    if target_location in current_location:
        logger.info(f"✅ [NAVIGATE_DIRECTION] Reached target location: {target_location}")
        return []

    use_directional = True
    if portal_coords:
        portal_x, portal_y = portal_coords
        distance = abs(current_x - portal_x) + abs(current_y - portal_y)
        if distance > proximity_radius:
            use_directional = False
            logger.info(f"🎯 [NAVIGATE_DIRECTION] Distance to portal ({portal_x}, {portal_y}): {distance} tiles (>{proximity_radius}) - navigating directly to portal")
        else:
            logger.info(f"🎯 [NAVIGATE_DIRECTION] Distance to portal ({portal_x}, {portal_y}): {distance} tiles (<={proximity_radius}) - using directional A* to find path through")

    stitched_map_info = state_data.get('map', {}).get('stitched_map_info')
    logger.info(f"🗺️ [NAVIGATE_DIRECTION DEBUG] stitched_map_info exists: {stitched_map_info is not None}")
    if stitched_map_info:
        logger.info(f"🗺️ [NAVIGATE_DIRECTION DEBUG] available: {stitched_map_info.get('available')}")

    if stitched_map_info and stitched_map_info.get('available'):
        current_area = stitched_map_info.get('current_area', {})
        grid_serializable = current_area.get('grid')
        bounds = current_area.get('bounds')
        logger.info(f"🗺️ [NAVIGATE_DIRECTION DEBUG] grid exists: {grid_serializable is not None}, bounds exists: {bounds is not None}")

        if grid_serializable and bounds:
            location_grid = {}
            for key, value in grid_serializable.items():
                x, y = map(int, key.split(','))
                location_grid[(x, y)] = value

            # Refresh NPC obstacles before direct A* calls (bypass callers)
            update_npc_obstacles(state_data)

            if use_directional:
                logger.info(f"🗺️ [NAVIGATE_DIRECTION] Using directional A* moving {direction} to reach {target_location}")
                pathfind_action = _astar_pathfind_with_grid_data(
                    location_grid=location_grid, bounds=bounds,
                    current_pos=(current_x, current_y), location=current_location,
                    goal_direction=direction, recent_positions=_recent_positions
                )
            else:
                logger.info(f"🗺️ [NAVIGATE_DIRECTION] Navigating to portal at ({portal_x}, {portal_y})")
                pathfind_action = _astar_pathfind_with_grid_data(
                    location_grid=location_grid, bounds=bounds,
                    current_pos=(current_x, current_y), location=current_location,
                    goal_direction=direction, goal_coords=(portal_x, portal_y),
                    recent_positions=_recent_positions
                )

            if pathfind_action:
                logger.info(f"🗺️ [NAVIGATE_DIRECTION] A* recommends: {pathfind_action}")
                return pathfind_action if isinstance(pathfind_action, list) else [pathfind_action]
            else:
                logger.warning(f"⚠️ [NAVIGATE_DIRECTION] A* found no path, trying direct movement")
                return [_DIRECTION_MAP.get(direction, 'DOWN')]

    logger.info(f"🗺️ [NAVIGATE_DIRECTION] No map data, moving {direction} directly")
    return [_DIRECTION_MAP.get(direction, 'DOWN')]


def _handle_move_until_map_change(directive, state_data):
    """Keep moving in direction until location changes."""
    direction = directive.get('direction')
    target_location = directive.get('target_location', '').upper()

    player_data = state_data.get('player', {})
    current_location = player_data.get('location', '').upper()

    if target_location in current_location:
        logger.info(f"✅ [MAP TRANSITION] Reached target location: {target_location}")
        return []

    logger.info(f"🗺️ [MAP TRANSITION] Moving {direction} to reach {target_location} (currently in {current_location})")
    return [direction]


def _handle_navigate(directive, state_data, recent_actions, description):
    """
    NAVIGATE directive — walk to target without interaction.
    Includes portal warp logic: detect portals, build momentum, retry up to 5x.
    """
    target = directive.get('target')
    try:
        logger.info(f"🔍 [NAVIGATE BLOCK] Entered NAVIGATE handler")
        print(f"🔍 [NAVIGATE BLOCK] Entered NAVIGATE handler")
        target_x, target_y, target_map = target
        logger.info(f"📍 [DIRECTIVE NAVIGATE] Moving to: ({target_x}, {target_y}, {target_map}) - NO interaction")
    except Exception as nav_error:
        logger.error(f"❌ [NAVIGATE BLOCK] Exception in NAVIGATE handler: {nav_error}", exc_info=True)
        print(f"❌ [NAVIGATE BLOCK] Exception: {nav_error}")
        raise

    from agent.opener_bot import NavigationGoal
    nav_goal = NavigationGoal(
        x=target_x, y=target_y, map_location=target_map,
        should_interact=False, description=description
    )

    goal_x = nav_goal.x
    goal_y = nav_goal.y
    goal_map = nav_goal.map_location.upper()

    player_data = state_data.get('player', {})
    position = player_data.get('position', {})
    current_x = position.get('x', 0)
    current_y = position.get('y', 0)
    current_map = player_data.get('location', '').upper()

    logger.info(f"📍 [DIRECTIVE NAV] Current: ({current_x}, {current_y}, {current_map}), Goal: ({goal_x}, {goal_y}, {goal_map})")

    # --- Portal cleanup on location change ---
    if goal_map not in current_map and _portal_wait_frames:
        logger.info(f"✅ [PORTAL] Location changed from {goal_map} to {current_map} - portal warp successful!")
        _portal_wait_frames.clear()
        _portal_approach_direction.clear()
    elif _portal_wait_frames:
        current_key = (current_x, current_y, current_map)
        to_remove = [key for key in _portal_wait_frames.keys() if key != current_key]
        for key in to_remove:
            del _portal_wait_frames[key]
            if key in _portal_approach_direction:
                del _portal_approach_direction[key]
            logger.info(f"✅ [PORTAL] Cleared wait counter for {key} - moved away from portal")

    dx = goal_x - current_x
    dy = goal_y - current_y
    distance = abs(dx) + abs(dy)

    # Detect location change (successful warp)
    if goal_map not in current_map and _portal_wait_frames:
        logger.info(f"✅ [PORTAL] Location changed from {goal_map} to {current_map} - portal warp successful!")
        _portal_wait_frames.clear()
        return []

    # At goal position on same map — portal momentum logic
    if current_x == goal_x and current_y == goal_y and goal_map in current_map:
        portal_key = (goal_x, goal_y, goal_map)
        wait_count = _portal_wait_frames.get(portal_key, 0)
        _portal_wait_frames[portal_key] = wait_count + 1

        if wait_count < 5:
            approach_dir = _portal_approach_direction.get(portal_key)
            if approach_dir:
                logger.info(f"📍 [PORTAL] At portal tile ({goal_x}, {goal_y}), attempt {wait_count + 1}/5 - continuing in saved direction: {approach_dir}")
                return [approach_dir]
            else:
                logger.info(f"📍 [PORTAL] At portal tile ({goal_x}, {goal_y}) for first time, attempt {wait_count + 1}/5")
                if goal_y <= 1:
                    direction = 'UP'
                elif goal_y >= 19:
                    direction = 'DOWN'
                elif goal_x <= 7:
                    direction = 'LEFT'
                elif goal_x >= 15:
                    direction = 'RIGHT'
                else:
                    direction = 'UP'
                _portal_approach_direction[portal_key] = direction
                logger.info(f"📍 [PORTAL] Saved approach direction: {direction}")
                return [direction]
        else:
            logger.warning(f"⚠️ [PORTAL] Failed to warp after 5 attempts at ({goal_x}, {goal_y})")
            _portal_wait_frames[portal_key] = 0
            if portal_key in _portal_approach_direction:
                del _portal_approach_direction[portal_key]
            return []
    else:
        # Not at goal yet — navigate
        nav_action = None

        # 1 tile from portal — save approach direction for momentum
        if distance == 1 and goal_map in current_map:
            portal_key = (goal_x, goal_y, goal_map)
            if dy < 0:
                approach_dir = 'UP'
            elif dy > 0:
                approach_dir = 'DOWN'
            elif dx < 0:
                approach_dir = 'LEFT'
            elif dx > 0:
                approach_dir = 'RIGHT'
            else:
                approach_dir = None

            if approach_dir:
                _portal_approach_direction[portal_key] = approach_dir
                logger.info(f"📍 [PORTAL] 1 tile from portal ({goal_x}, {goal_y}) - will continue {approach_dir} through portal")
                return [approach_dir]

        pathfind_result = pathfind_to_goal(state_data, goal_x, goal_y)
        nav_action = pathfind_result if pathfind_result else None

        logger.info(f"📍 [DIRECTIVE NAV] After A* block, nav_action = {nav_action}")
        if not nav_action:
            if abs(dx) > abs(dy):
                nav_action = ['RIGHT'] if dx > 0 else ['LEFT']
            else:
                nav_action = ['DOWN'] if dy > 0 else ['UP']
            logger.info(f"📍 [DIRECTIVE NAV] Using simple direction: {nav_action[0]}")

    logger.info(f"📍 [DIRECTIVE NAV] Before final check, nav_action = {nav_action}")
    if nav_action:
        logger.info(f"📍 [DIRECTIVE NAV] Executing: {nav_action[0]}")
        return nav_action
    else:
        logger.warning(f"⚠️ [DIRECTIVE NAV] nav_action is None/empty, NOT returning")
        return None  # caller falls through


def _handle_navigate_and_interact(directive, state_data, recent_actions, description):
    """
    NAVIGATE_AND_INTERACT — walk to target then press A.
    Includes adjacent-facing logic and map-transition handling.
    """
    from agent.planning import planning_step

    target = directive.get('target')
    if len(target) == 2:
        target_x, target_y = target
        target_map = directive.get('location', '')
    else:
        target_x, target_y, target_map = target

    logger.info(f"📍 [DIRECTIVE] Converting to NavigationGoal: ({target_x}, {target_y}, {target_map})")

    from agent.opener_bot import NavigationGoal
    nav_goal = NavigationGoal(
        x=target_x, y=target_y, map_location=target_map,
        should_interact=True, description=description
    )

    goal_x = nav_goal.x
    goal_y = nav_goal.y
    goal_map = nav_goal.map_location.upper()
    should_interact = nav_goal.should_interact

    player_data = state_data.get('player', {})
    position = player_data.get('position', {})
    current_x = position.get('x', 0)
    current_y = position.get('y', 0)
    current_map = player_data.get('location', '').upper()
    current_orientation = player_data.get('facing_direction', 'down').upper()

    logger.info(f"📍 [DIRECTIVE NAV] Current: ({current_x}, {current_y}, {current_map}), Goal: ({goal_x}, {goal_y}, {goal_map})")

    dx = goal_x - current_x
    dy = goal_y - current_y
    distance = abs(dx) + abs(dy)
    nav_action = None

    # At exact goal
    if current_x == goal_x and current_y == goal_y:
        if goal_map in current_map:
            if should_interact:
                nav_action = ['A']
                logger.info(f"📍 [DIRECTIVE NAV] At goal, pressing A to interact")
            else:
                logger.info(f"📍 [DIRECTIVE NAV] At goal (walk-to), continuing past")
                nav_action = None
        else:
            logger.info(f"📍 [DIRECTIVE NAV] At goal coords ({goal_x}, {goal_y}) but wrong map!")
            logger.info(f"   Current map: '{current_map}', Target map: '{goal_map}'")
            directive_details = planning_step.objective_manager.get_current_directive() if hasattr(planning_step, 'objective_manager') else None
            goal_direction = directive_details.get('goal_direction', 'north') if directive_details else 'north'
            transition_action = _DIRECTION_MAP.get(goal_direction.lower(), 'UP')
            nav_action = [transition_action]
            logger.info(f"📍 [DIRECTIVE NAV] Attempting map transition with: {transition_action}")

    # Adjacent to goal (for interaction)
    elif distance == 1 and should_interact and goal_map in current_map:
        if dx > 0:
            required_direction = 'RIGHT'
        elif dx < 0:
            required_direction = 'LEFT'
        elif dy > 0:
            required_direction = 'DOWN'
        else:
            required_direction = 'UP'

        if current_orientation == required_direction:
            nav_action = ['A']
            logger.info(f"📍 [DIRECTIVE NAV] Adjacent and facing goal, pressing A")
        else:
            nav_action = [required_direction]
            logger.info(f"📍 [DIRECTIVE NAV] Turning to face goal: {required_direction}")

    # A* pathfinding
    else:
        logger.info(f"📍 [DIRECTIVE NAV] Distance to goal: {distance}, dx={dx}, dy={dy}")

        pathfind_action = pathfind_to_goal(state_data, goal_x, goal_y)

        if not pathfind_action:
            if abs(dx) > abs(dy):
                goal_direction = 'east' if dx > 0 else 'west'
            else:
                goal_direction = 'south' if dy > 0 else 'north'
            pathfind_action = _local_pathfind_from_tiles(state_data, goal_direction, recent_actions)

        if pathfind_action:
            nav_action = pathfind_action if isinstance(pathfind_action, list) else [pathfind_action]
        else:
            if abs(dx) > abs(dy):
                nav_action = ['RIGHT'] if dx > 0 else ['LEFT']
            else:
                nav_action = ['DOWN'] if dy > 0 else ['UP']
            logger.info(f"📍 [DIRECTIVE NAV] All A* failed, using simple direction: {nav_action[0]}")

    if nav_action:
        logger.info(f"📍 [DIRECTIVE NAV] Returning {len(nav_action)}-step path: {' → '.join(nav_action)}")
        print(f"📍 [DIRECTIVE NAV] {len(nav_action)}-step path: {' → '.join(nav_action)}")
        return nav_action

    return None  # caller falls through


def _handle_cross_boundary(directive, state_data):
    """Move in direction to cross an open-world map boundary."""
    direction = directive.get('direction', '').lower()
    from_location = directive.get('from_location', '')
    to_location = directive.get('to_location', '')

    logger.info(f"📍 [CROSS_BOUNDARY] Moving {direction} from {from_location} to {to_location}")
    print(f"🗺️ [CROSS_BOUNDARY] Going {direction}: {from_location} → {to_location}")

    player_data = state_data.get('player', {})
    current_location = player_data.get('location', '').upper()

    if to_location.upper() in current_location:
        logger.info(f"✅ [CROSS_BOUNDARY] Already in target location: {to_location}")
        return []

    logger.warning(f"⚠️ [CROSS_BOUNDARY] Attempting blind directional movement - planner should navigate TO boundary first!")
    move_direction = _DIRECTION_MAP.get(direction, 'UP')
    logger.info(f"🗺️ [CROSS_BOUNDARY] Moving {move_direction} to cross boundary")
    return [move_direction]


def execute_directive(directive, state_data, recent_actions, description=''):
    """
    Main entry point — convert a Directive into button presses.

    Args:
        directive: Directive dict/dataclass from ObjectiveManager
        state_data: Current game state
        recent_actions: List of recent actions
        description: Human-readable directive description

    Returns:
        List of button strings, empty list (no action), or None (fall through to VLM)
    """
    location = state_data.get('player', {}).get('location', '')

    # Handle journey complete
    if directive.get('journey_complete'):
        logger.info(f"✅ [DIRECTIVE] Navigation journey complete")
        return []

    # goal_coords
    if 'goal_coords' in directive:
        print(f"🔍 [GOAL_COORDS] Entered goal_coords handler!")
        logger.info(f"🔍 [GOAL_COORDS] Entered goal_coords handler!")
        return _handle_goal_coords(directive, state_data, recent_actions, description, location)

    # goal_direction
    if 'goal_direction' in directive:
        return _handle_goal_direction(directive, state_data, recent_actions)

    # wait_for_transition
    if 'wait_for_transition' in directive:
        return _handle_wait_for_transition(directive, state_data)

    action_type = directive.get('action')

    if action_type == 'NAVIGATE_DIRECTION':
        return _handle_navigate_direction(directive, state_data)

    if action_type == 'MOVE_UNTIL_MAP_CHANGE':
        return _handle_move_until_map_change(directive, state_data)

    if action_type == 'NAVIGATE' and directive.get('target'):
        return _handle_navigate(directive, state_data, recent_actions, description)

    if action_type == 'NAVIGATE_AND_INTERACT' and directive.get('target'):
        return _handle_navigate_and_interact(directive, state_data, recent_actions, description)

    if action_type == 'INTERACT':
        logger.info(f"📍 [DIRECTIVE] INTERACT directive - pressing A")
        print(f"📍 [DIRECTIVE] Interacting with A")
        return ['A']

    if action_type == 'DIALOGUE':
        stuck_handler._was_in_dialogue = True
        stuck_handler._post_dialogue_movement_count = 0
        logger.info(f"📍 [DIRECTIVE] DIALOGUE - pressing A to advance")
        print(f"💬 [DIALOGUE] Detected - resetting post-dialogue movement counter")
        return ['A']

    if action_type == 'DIALOGUE_B':
        stuck_handler._was_in_dialogue = True
        stuck_handler._post_dialogue_movement_count = 0
        logger.info(f"📍 [DIRECTIVE] DIALOGUE_B - pressing B to advance (NPC-safe)")
        print(f"💬 [DIALOGUE] Using B to advance (won't re-trigger NPC)")
        return ['B']

    if action_type == 'WAIT_FOR_DIALOGUE':
        logger.info(f"📍 [DIRECTIVE] Waiting for auto-dialogue to complete - returning empty action")
        return []

    if action_type == 'CROSS_BOUNDARY':
        return _handle_cross_boundary(directive, state_data)

    logger.warning(f"📍 [DIRECTIVE] Unknown action type: {action_type}")
    return None  # fall through to VLM
