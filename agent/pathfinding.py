"""
Pathfinding Module for Pokemon Emerald Agent.

Contains all pathfinding and navigation algorithms:
- Local BFS on 15x15 visible tile grid (_pathfind_to_target, _local_pathfind_from_tiles)
- Global A* using stitched map data (_astar_pathfind_with_grid_data)
- Coordinate-based A* for specific targets (_astar_pathfind_to_coords_with_grid)
- Frontier-based exploration for maze navigation
- Path truncation at warp tiles for safety

Also manages shared navigation state:
- _recent_positions: deque of recent (x, y, location) tuples for warp avoidance
- _dynamically_blocked_tiles: dict of tiles temporarily blocked by NPCs/obstacles
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from collections import deque

logger = logging.getLogger(__name__)

# === MOVEMENT BATCHING CONFIGURATION ===
# Maximum number of movement steps to batch together for faster navigation
# Batching reduces VLM calls and speeds up movement by executing multiple steps at once
# Conservative default prevents runaway if obstacles appear mid-path
MAX_MOVEMENT_BATCH_SIZE = 15  # ~1.3 seconds of movement at 60 FPS

# === SHARED NAVIGATION STATE ===
# These are shared between pathfinding, stuck_handler, and action modules.
# They are mutable objects (deque, dict) so imports can mutate them in place.

# Track recent positions to avoid immediate backtracking through warps
# Store tuples of (x, y, map_location) for the last 10 positions
_recent_positions = deque(maxlen=10)

# Dynamic tile blocking - tiles temporarily marked as unwalkable (e.g., NPC-occupied)
# When the agent gets stuck trying to move in a direction repeatedly, the tile in that
# direction is marked as blocked so A* will route around it.
# Dict mapping (world_x, world_y, location) → remaining TTL (decremented each action_step call)
_dynamically_blocked_tiles = {}

# Live NPC obstacle tiles — refreshed every pathfind call from gObjectEvents.
# Unlike _dynamically_blocked_tiles (sticky TTL), these are ephemeral: rebuilt
# from scratch on each pathfind_to_goal() invocation so moving NPCs are handled
# correctly without leaving stale blocks behind.
# Set of (world_x, world_y, location) tuples.
_npc_occupied_tiles: set = set()

# Direction offsets for converting direction names to coordinate deltas
_DIRECTION_OFFSETS = {
    'UP': (0, -1),
    'DOWN': (0, 1),
    'LEFT': (-1, 0),
    'RIGHT': (1, 0)
}


def update_npc_obstacles(
    state_data: Dict[str, Any],
    *,
    exclude_coords: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Refresh ``_npc_occupied_tiles`` from the live ``active_npcs`` list.

    Called at the top of every ``pathfind_to_goal()`` invocation so A* and
    local BFS treat NPC-occupied tiles as impassable walls.

    Args:
        state_data: Current game state (must contain ``active_npcs`` list
            and ``player.position`` / ``player.location``).
        exclude_coords: Optional ``(x, y)`` of the *target* NPC.  When
            navigating *to* an NPC, their tile must stay walkable so
            A* can plan a path that ends adjacent or on-top.
    """
    global _npc_occupied_tiles
    _npc_occupied_tiles = set()

    location = state_data.get('player', {}).get('location', '')
    active_npcs = state_data.get('active_npcs', [])

    if not active_npcs or not location:
        return

    for npc in active_npcs:
        # Skip the player's own object event
        if npc.get('is_player'):
            continue
        # Skip invisible / off-screen NPCs (they don't physically block)
        if npc.get('invisible') or npc.get('off_screen'):
            continue

        nx = npc.get('current_x')
        ny = npc.get('current_y')
        if nx is None or ny is None:
            continue

        # Don't block the tile of the NPC we're walking *toward*
        if exclude_coords and (nx, ny) == exclude_coords:
            continue

        key = (nx, ny, location)
        _npc_occupied_tiles.add(key)

    if _npc_occupied_tiles:
        logger.debug(f"[NPC OBSTACLES] Refreshed {len(_npc_occupied_tiles)} NPC tiles "
                     f"on {location} (excluded={exclude_coords})")


# ============================================================================
# PATHFINDING FUNCTIONS (moved from action.py)
# ============================================================================

def calculate_2x2_moves(options, current, target):
    """Calculates D-pad presses in a 2x2 menu layout."""
    raise NotImplementedError
    # Example layout:
    # FIGHT (0,0) | BAG (1,0)
    # POKEMON(0,1)| RUN (1,1)
    positions = {opt: (i % 2, i // 2) for i, opt in enumerate(options)}
    if current not in positions or target not in positions:
        return []
    
    curr_x, curr_y = positions[current]
    targ_x, targ_y = positions[target]
    
    moves = []
    while curr_y < targ_y:
        moves.append("DOWN")
        curr_y += 1
    while curr_y > targ_y:
        moves.append("UP")
        curr_y -= 1
    while curr_x < targ_x:
        moves.append("RIGHT")
        curr_x += 1
    while curr_x > targ_x:
        moves.append("LEFT")
        curr_x -= 1
    return moves    

def calculate_column_moves(options, current, target):
    """Calculates D-pad presses in a single-column menu layout."""
    raise NotImplementedError
    if current not in options or target not in options:
        return []
    
    curr_index = options.index(current)
    targ_index = options.index(target)
    
    moves = []
    while curr_index < targ_index:
        moves.append("DOWN")
        curr_index += 1
    while curr_index > targ_index:
        moves.append("UP")
        curr_index -= 1
    
    return moves

def get_menu_navigation_moves(menu_state, options, current, target):
    """Calculates D-pad presses to go from current to target selection."""
    raise NotImplementedError
    if menu_state == "battle_action_select":
        # Use 2x2 logic: knows "FIGHT" is left of "BAG", "POKEMON" is below "FIGHT"
        # Example: to get from "FIGHT" to "RUN", press DOWN then RIGHT.
        return calculate_2x2_moves(options, current, target)

    elif menu_state in ["main_menu", "shop_menu"]:
        # Use 1-column logic: knows it only needs to press UP or DOWN.
        # Example: to get from "BAG" to "EXIT", press DOWN four times.
        return calculate_column_moves(options, current, target)
    
    # ... other menu types ...

def _truncate_path_at_warp(path: List[str], path_coords: List[Tuple[int, int]], location_grid: dict) -> List[str]:
    """
    Truncate a movement path at warp tiles (doors, stairs, portals) for safety.
    
    When a path crosses a warp tile, we only want to execute steps up to and including
    that warp, because the game state will change after the warp (new map, different position).
    Continuing movement after a warp would use invalid coordinates.
    
    Args:
        path: List of direction strings ['UP', 'UP', 'RIGHT', ...]
        path_coords: List of (x, y) tuples corresponding to each step in path
        location_grid: Dictionary mapping (x, y) to tile symbols
    
    Returns:
        Truncated path stopping at first warp tile (includes the warp step)
    """
    if not path or not path_coords or not location_grid:
        return path
    
    # Check each position in the path for warp tiles
    for i, (x, y) in enumerate(path_coords):
        tile = location_grid.get((x, y), '.')
        
        # Warp tiles: D=door, S=stairs, ?=portal/unknown
        if tile in ['D', 'S', '?']:
            # Return path up to and including this warp step
            truncated = path[:i+1]
            print(f"🚪 [PATH TRUNCATE] Found warp at step {i+1}/{len(path)} ({tile}), truncating path")
            print(f"   Original: {len(path)} steps, Truncated: {len(truncated)} steps")
            return truncated
    
    # No warps found, return full path
    return path

def _pathfind_to_target(state_data: Dict[str, Any], target_x: int, target_y: int) -> Optional[str]:
    """
    Pathfinding using BFS on the 15x15 visible tile grid to reach specific target coordinates.
    
    Args:
        state_data: Current game state with 'map']['tiles'] containing 15x15 grid
        target_x: Target world X coordinate
        target_y: Target world Y coordinate
    
    Returns:
        Direction string ('UP', 'DOWN', 'LEFT', 'RIGHT') or None if target not visible or no path
    """
    try:
        from utils.state_formatter import format_tile_to_symbol
        from collections import deque
        
        # Get current position
        player_data = state_data.get('player', {})
        position = player_data.get('position', {})
        current_x = position.get('x')
        current_y = position.get('y')
        current_location = player_data.get('location', '')
        
        if current_x is None or current_y is None:
            return None
        
        # Get tiles from state (15x15 grid centered on player)
        map_info = state_data.get('map', {})
        raw_tiles = map_info.get('tiles', [])
        
        if not raw_tiles or len(raw_tiles) < 15:
            return None
        
        grid_size = len(raw_tiles)
        center = grid_size // 2
        
        # Calculate target position in grid coordinates
        target_grid_x = center + (target_x - current_x)
        target_grid_y = center + (target_y - current_y)
        
        # Check if target is within visible grid
        if not (0 <= target_grid_x < grid_size and 0 <= target_grid_y < grid_size):
            print(f"⚠️ [TARGET A*] Target ({target_x}, {target_y}) not visible in 15x15 grid")
            print(f"   Current: ({current_x}, {current_y}), Grid target: ({target_grid_x}, {target_grid_y})")
            return None
        
        # Helper to check if tile is walkable
        def is_walkable(y, x):
            if not (0 <= y < grid_size and 0 <= x < grid_size):
                return False
            
            # Check dynamically blocked tiles (NPC/obstacle positions)
            world_x = current_x + (x - center)
            world_y = current_y + (y - center)
            if (world_x, world_y, current_location) in _dynamically_blocked_tiles:
                return False
            # Check live NPC obstacle tiles
            if (world_x, world_y, current_location) in _npc_occupied_tiles:
                return False
            
            tile = raw_tiles[y][x]
            symbol = format_tile_to_symbol(tile) if tile else '?'
            
            # Check if this is a warp tile (stairs/doors)
            is_warp_tile = symbol in ['S', 'D']
            
            # Only allow walking on warp tiles if they are the TARGET destination
            # This prevents accidentally warping when trying to navigate past them
            if is_warp_tile:
                is_target = (y == target_grid_y and x == target_grid_x)
                return is_target
            
            # Normal walkable tiles: . (normal), _ (bridge), ~ (grass)
            return symbol in ['.', '_', '~']
        
        # BFS from player position to target
        directions = [
            ('UP', 0, -1),
            ('DOWN', 0, 1),
            ('LEFT', -1, 0),
            ('RIGHT', 1, 0)
        ]
        
        start = (center, center)
        target = (target_grid_y, target_grid_x)
        queue = deque([(start, [])])  # (position, path_of_directions)
        visited = {start}
        
        while queue:
            (y, x), path = queue.popleft()
            
            # Check if we reached target
            if (y, x) == target:
                if path:
                    # Batch multiple steps for faster navigation
                    batched_path = path[:MAX_MOVEMENT_BATCH_SIZE]
                    full_path_str = ' → '.join(path)
                    batched_path_str = ' → '.join(batched_path)
                    
                    print(f"✅ [TARGET A*] Found path to ({target_x}, {target_y}): {full_path_str}")
                    print(f"   📦 Batching {len(batched_path)}/{len(path)} steps: {batched_path_str}")
                    
                    return batched_path
                else:
                    # Already at target
                    print(f"✅ [TARGET A*] Already at target ({target_x}, {target_y})")
                    return None
            
            # Explore neighbors
            for dir_name, dx, dy in directions:
                ny, nx = y + dy, x + dx
                
                if (ny, nx) in visited:
                    continue
                
                if not is_walkable(ny, nx):
                    continue
                
                visited.add((ny, nx))
                new_path = path + [dir_name]
                queue.append(((ny, nx), new_path))
        
        # No path found
        print(f"⚠️ [TARGET A*] No path found to ({target_x}, {target_y})")
        print(f"   Explored {len(visited)} tiles")
        return None
        
    except Exception as e:
        print(f"❌ [TARGET A*] Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def _local_pathfind_from_tiles(state_data: Dict[str, Any], goal_direction: str, recent_actions: Optional[List[str]] = None) -> Optional[str]:
    """
    Pathfinding using BFS on the 15x15 visible tile grid.
    Finds the best first step toward target positions along the goal direction edge.
    AVOIDS tiles that would lead back to recently visited positions (anti-warp-backtracking).
    
    Args:
        state_data: Current game state with 'map']['tiles'] containing 15x15 grid
        goal_direction: Direction hint like 'north', 'south', etc.
        recent_actions: List of recent actions for oscillation detection
    
    Returns:
        Direction string ('UP', 'DOWN', 'LEFT', 'RIGHT') or None if no path
    """
    try:
        from utils.state_formatter import format_tile_to_symbol
        from collections import deque
        
        # Get current position and location
        player_data = state_data.get('player', {})
        position = player_data.get('position', {})
        current_x = position.get('x')
        current_y = position.get('y')
        current_location = player_data.get('location', '')
        
        # Get tiles from state (15x15 grid centered on player)
        map_info = state_data.get('map', {})
        raw_tiles = map_info.get('tiles', [])
        
        if not raw_tiles or len(raw_tiles) < 15:
            print(f"⚠️ [LOCAL A*] Insufficient tile data: {len(raw_tiles) if raw_tiles else 0} rows")
            return None
        
        grid_size = len(raw_tiles)
        center = grid_size // 2
        
        # Helper to check if tile is walkable
        def is_walkable(y, x):
            if not (0 <= y < grid_size and 0 <= x < grid_size):
                return False
            
            # Check dynamically blocked tiles (NPC/obstacle positions)
            world_x = current_x + (x - center)
            world_y = current_y + (y - center)
            if (world_x, world_y, current_location) in _dynamically_blocked_tiles:
                return False
            # Check live NPC obstacle tiles
            if (world_x, world_y, current_location) in _npc_occupied_tiles:
                return False
            
            tile = raw_tiles[y][x]
            symbol = format_tile_to_symbol(tile) if tile else '?'
            # Walkable: grass/path only
            # NOT walkable: doors, walls, stairs, ledges, unknown
            return symbol in ['.', '_', '~']  # Added grass (~) as walkable!
        
        # Helper to check if a grid position would lead to a recently-visited location
        def leads_to_recent_position(grid_y, grid_x):
            """Check if moving to this grid position matches a recent location (warp detection)."""
            global _recent_positions
            
            if not _recent_positions or current_x is None or current_y is None:
                return False
            
            # Calculate world coordinates for this grid position
            offset_x = grid_x - center
            offset_y = grid_y - center
            target_world_x = current_x + offset_x
            target_world_y = current_y + offset_y
            
            # Check if this world position matches any recent position
            # We're particularly interested in positions with DIFFERENT map locations
            # (indicating a warp back to a previous area)
            for recent_x, recent_y, recent_loc in _recent_positions:
                if recent_x == target_world_x and recent_y == target_world_y:
                    # Same coordinates - check if it's a different map location (warp)
                    if recent_loc != current_location:
                        print(f"🚫 [WARP AVOID] Grid ({grid_y}, {grid_x}) = World ({target_world_x}, {target_world_y})")
                        print(f"              Matches recent position @ '{recent_loc}' (different from current '{current_location}')")
                        print(f"              This would be a warp back - AVOIDING")
                        return True
            
            return False
        
        # Determine target positions based on goal direction
        goal_dir_upper = goal_direction.upper()
        target_positions = []
        
        if 'NORTH' in goal_dir_upper or goal_direction == 'north':
            # Target top edge (row 0) - find walkable tiles
            for x in range(grid_size):
                if is_walkable(0, x) and not leads_to_recent_position(0, x):
                    target_positions.append((0, x))
        elif 'SOUTH' in goal_dir_upper or goal_direction == 'south':
            # Target bottom edge
            for x in range(grid_size):
                if is_walkable(grid_size - 1, x) and not leads_to_recent_position(grid_size - 1, x):
                    target_positions.append((grid_size - 1, x))
        elif 'EAST' in goal_dir_upper or goal_direction == 'east':
            # Target right edge
            for y in range(grid_size):
                if is_walkable(y, grid_size - 1) and not leads_to_recent_position(y, grid_size - 1):
                    target_positions.append((y, grid_size - 1))
        elif 'WEST' in goal_dir_upper or goal_direction == 'west':
            # Target left edge
            for y in range(grid_size):
                if is_walkable(y, 0) and not leads_to_recent_position(y, 0):
                    target_positions.append((y, 0))
        else:
            print(f"⚠️ [LOCAL A*] Unknown goal direction: {goal_direction}")
            return None
        
        if not target_positions:
            print(f"⚠️ [LOCAL A*] No walkable targets on {goal_direction} edge")
            return None
        
        # BFS from player position to find shortest path to any target
        directions = [
            ('UP', 0, -1),
            ('DOWN', 0, 1),
            ('LEFT', -1, 0),
            ('RIGHT', 1, 0)
        ]
        
        start = (center, center)
        queue = deque([(start, [])])  # (position, path_of_directions)
        visited = {start}
        
        while queue:
            (y, x), path = queue.popleft()
            
            # Check if we reached a target
            if (y, x) in target_positions:
                if path:
                    # Batch multiple steps for faster navigation
                    batched_path = path[:MAX_MOVEMENT_BATCH_SIZE]
                    full_path_str = ' → '.join(path)
                    batched_path_str = ' → '.join(batched_path)
                    
                    print(f"✅ [LOCAL A*] Found path to {goal_direction} edge: {full_path_str}")
                    print(f"   📦 Batching {len(batched_path)}/{len(path)} steps: {batched_path_str}")
                    
                    return batched_path
                else:
                    # Already at target? shouldn't happen
                    print(f"⚠️ [LOCAL A*] Already at target position")
                    return None
            
            # Explore neighbors
            for dir_name, dx, dy in directions:
                ny, nx = y + dy, x + dx
                
                # Skip if already visited
                if (ny, nx) in visited:
                    continue
                
                # Skip if not walkable
                if not is_walkable(ny, nx):
                    continue
                
                # Skip if this would lead to a recently-visited position (warp avoidance)
                if leads_to_recent_position(ny, nx):
                    continue
                
                visited.add((ny, nx))
                new_path = path + [dir_name]
                queue.append(((ny, nx), new_path))
        
        # No path found to any target
        print(f"⚠️ [LOCAL A*] No path found toward {goal_direction}")
        print(f"   Checked {len(target_positions)} target positions on edge")
        print(f"   Explored {len(visited)} tiles")
        
        # Fallback: SMART direction selection to escape dead-ends
        # Check recent actions for oscillation patterns
        recent_set = set(recent_actions[-10:]) if recent_actions and len(recent_actions) >= 10 else set(recent_actions or [])
        oscillating_horizontal = 'LEFT' in recent_set and 'RIGHT' in recent_set
        oscillating_vertical = 'UP' in recent_set and 'DOWN' in recent_set
        
        print(f"🔄 [LOCAL A*] SMART fallback - Oscillation check: H={oscillating_horizontal}, V={oscillating_vertical}")
        
        goal_dir_upper = goal_direction.upper()
        
        # Determine fallback order based on goal and oscillation
        if 'NORTH' in goal_dir_upper or 'SOUTH' in goal_dir_upper or goal_direction in ['north', 'south']:
            # Goal is vertical
            if oscillating_horizontal:
                # Stuck oscillating horizontally - try vertical escape
                print(f"   🔄 Detected horizontal oscillation, prioritizing vertical movement")
                fallback_order = ['DOWN', 'UP', 'RIGHT', 'LEFT'] if goal_direction == 'south' else ['UP', 'DOWN', 'RIGHT', 'LEFT']
            else:
                # Normal: try horizontal first to explore
                fallback_order = ['RIGHT', 'LEFT', 'DOWN', 'UP'] if goal_direction == 'south' else ['RIGHT', 'LEFT', 'UP', 'DOWN']
        else:
            # Goal is horizontal
            if oscillating_vertical:
                # Stuck oscillating vertically - try horizontal escape
                print(f"   🔄 Detected vertical oscillation, prioritizing horizontal movement")
                fallback_order = ['LEFT', 'RIGHT', 'UP', 'DOWN'] if 'WEST' in goal_dir_upper else ['RIGHT', 'LEFT', 'UP', 'DOWN']
            else:
                # Normal: try vertical first to explore
                fallback_order = ['UP', 'DOWN', 'LEFT', 'RIGHT'] if 'WEST' in goal_dir_upper else ['UP', 'DOWN', 'RIGHT', 'LEFT']
        
        # Try directions in smart order
        for dir_name in fallback_order:
            # Find dx, dy for this direction
            dx, dy = 0, 0
            for d_name, d_dx, d_dy in directions:
                if d_name == dir_name:
                    dx, dy = d_dx, d_dy
                    break
            
            ny, nx = center + dy, center + dx
            walkable = is_walkable(ny, nx)
            recent = leads_to_recent_position(ny, nx)
            
            if walkable and not recent:
                print(f"   ✅ SMART Fallback: choosing {dir_name}")
                return dir_name
            else:
                print(f"   ❌ {dir_name}: walkable={walkable}, leads_to_recent={recent}")
        
        print(f"   ⚠️ No walkable fallback directions found!")
        return None
        
    except Exception as e:
        print(f"❌ [LOCAL A*] Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def _validate_map_stitcher_bounds(map_stitcher, player_pos: Tuple[int, int], location: str) -> bool:
    """
    Check if the map stitcher bounds contain the current player position.
    This detects stale map data from previous runs/states.
    
    Args:
        map_stitcher: The MapStitcher singleton instance
        player_pos: Current player (x, y) world coordinates
        location: Current location name
    
    Returns:
        True if player position is within valid bounds, False if mismatch detected
    """
    try:
        # Find the map area for current location
        matching_area = None
        for area in map_stitcher.map_areas.values():
            if area.location_name.upper() == location.upper():
                matching_area = area
                break
        
        if not matching_area:
            print(f"⚠️ [PATHFINDING] Location '{location}' not in map stitcher - likely fresh state")
            return False
        
        bounds = matching_area.explored_bounds
        player_x, player_y = player_pos
        
        # Check if player position is within bounds
        if (bounds['min_x'] <= player_x <= bounds['max_x'] and
            bounds['min_y'] <= player_y <= bounds['max_y']):
            return True
        else:
            print(f"⚠️ [PATHFINDING] Map stitcher bounds mismatch!")
            print(f"   Player position: ({player_x}, {player_y})")
            print(f"   Map stitcher bounds: X:{bounds['min_x']}-{bounds['max_x']}, Y:{bounds['min_y']}-{bounds['max_y']}")
            print(f"   This indicates stale data from a previous run")
            print(f"   Disabling pathfinding for this step - relying on VLM navigation")
            return False
            
    except Exception as e:
        print(f"❌ [PATHFINDING] Error validating map stitcher: {e}")
        return False


def _astar_pathfind_to_coords_with_grid(
    location_grid: dict,
    bounds: dict,
    current_pos: Tuple[int, int],
    target_pos: Tuple[int, int],
    location: str,
    recent_positions: Optional[deque] = None
) -> Optional[str]:
    """
    A* pathfinding to specific coordinates using stitched map grid data.
    
    Args:
        location_grid: Dictionary mapping (x, y) tuples to tile symbols (relative coords)
        bounds: Dictionary with min_x, max_x, min_y, max_y 
        current_pos: Player's current (x, y) position IN ABSOLUTE WORLD COORDINATES
        target_pos: Target (x, y) position IN ABSOLUTE WORLD COORDINATES
        location: Current location name
        recent_positions: Deque of recent (x, y, location) tuples for warp avoidance
    
    Returns:
        First step direction ('UP', 'DOWN', 'LEFT', 'RIGHT') or None if no path
    """
    try:
        from collections import deque
        import heapq
        
        if not location_grid:
            print(f"⚠️ [COORD A*] No grid data provided")
            return None
        
        # CRITICAL: Grid keys are in WORLD coordinates, not relative coordinates!
        # The current_pos and target_pos are already in world coordinates from the emulator
        # Do NOT convert them to relative coordinates - use them directly
        current_x, current_y = current_pos
        target_x, target_y = target_pos
        
        # Use world coordinates directly
        world_current_pos = (current_x, current_y)
        world_target_pos = (target_x, target_y)
        
        # Check if positions are in the grid (using world coordinates)
        if world_current_pos not in location_grid:
            print(f"⚠️ [COORD A*] Current position {current_pos} not in grid")
            print(f"   Grid has {len(location_grid)} tiles")
            # Show sample grid keys to help debug
            sample_keys = list(location_grid.keys())[:5]
            print(f"   Sample grid keys: {sample_keys}")
            return None
        
        if world_target_pos not in location_grid:
            print(f"⚠️ [COORD A*] Target position {target_pos} not in explored grid")
            return None
        
        print(f"✅ [COORD A*] Pathfinding from {current_pos} to {target_pos}")
        print(f"   Grid size: {len(location_grid)} tiles (world coordinates)")
        
        # Helper function to check if tile is walkable
        def is_walkable(pos: Tuple[int, int]) -> bool:
            if pos not in location_grid:
                return False
            # Check dynamically blocked tiles (NPC/obstacle positions)
            if (pos[0], pos[1], location) in _dynamically_blocked_tiles:
                return False
            # Check live NPC obstacle tiles
            if (pos[0], pos[1], location) in _npc_occupied_tiles:
                return False
            tile = location_grid[pos]
            return tile in ['.', '_', '~', 'D', 'S', 'N', '?']  # Include NPCs as potential targets, '?' for frontier tiles
        
        # Helper function to get movement cost
        def get_tile_cost(pos: Tuple[int, int]) -> float:
            if pos not in location_grid:
                return 999
            tile = location_grid[pos]
            if tile == '~':  # Grass
                return 3.0
            elif tile in ['.', '_']:
                return 1.0
            elif tile in ['D', 'S', 'N']:
                return 1.5
            else:
                return 2.0
        
        # Manhattan distance heuristic
        def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
        # A* pathfinding using world coordinates
        start = world_current_pos
        goal = world_target_pos
        pq = [(manhattan_distance(start, goal), 0, start, [], [start])]  # Added path_coords tracking
        visited = {start}
        
        directions_map = {
            (0, -1): 'UP',
            (0, 1): 'DOWN',
            (-1, 0): 'LEFT',
            (1, 0): 'RIGHT'
        }
        
        while pq:
            f_score, g_score, current, path, path_coords = heapq.heappop(pq)
            
            # Check if we reached the goal
            if current == goal:
                if path:
                    # Truncate at warps for safety
                    path = _truncate_path_at_warp(path, path_coords[1:], location_grid)  # Skip start position
                    
                    # Batch multiple steps for faster navigation
                    batched_path = path[:MAX_MOVEMENT_BATCH_SIZE]
                    
                    path_preview = ' → '.join(path[:5])
                    if len(path) > 5:
                        path_preview += f" ... ({len(path)} steps)"
                    
                    batched_str = ' → '.join(batched_path)
                    print(f"✅ [COORD A*] Found path to {target_pos}: {path_preview}")
                    print(f"   📦 Batching {len(batched_path)}/{len(path)} steps: {batched_str}")
                    print(f"   Total cost: {g_score:.1f}")
                    
                    return batched_path
                else:
                    print(f"✅ [COORD A*] Already at target {target_pos}")
                    return None
            
            # Explore neighbors
            for (dx, dy), direction in directions_map.items():
                next_pos = (current[0] + dx, current[1] + dy)
                
                if next_pos in visited:
                    continue
                
                if not is_walkable(next_pos):
                    continue
                
                move_cost = get_tile_cost(next_pos)
                new_g_score = g_score + move_cost
                new_f_score = new_g_score + manhattan_distance(next_pos, goal)
                new_path = path + [direction]
                new_path_coords = path_coords + [next_pos]
                
                heapq.heappush(pq, (new_f_score, new_g_score, next_pos, new_path, new_path_coords))
                visited.add(next_pos)
        
        print(f"⚠️ [COORD A*] No path found from {current_pos} to {target_pos}")
        print(f"   Explored {len(visited)} tiles")
        return None
        
    except Exception as e:
        print(f"❌ [COORD A*] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def _astar_pathfind_with_grid_data(
    location_grid: dict,
    bounds: dict,
    current_pos: Tuple[int, int], 
    location: str,
    goal_direction: str,
    recent_positions: Optional[deque] = None,
    goal_coords: Optional[Tuple[int, int]] = None,
    avoid_grass: bool = True
) -> Optional[str]:
    """
    ============================================================================
    🧭 INTELLIGENT A* PATHFINDING WITH FRONTIER-BASED MAZE NAVIGATION
    ============================================================================
    
    This is the core pathfinding system that enables the agent to navigate
    complex environments including mazes, obstacles, and unexplored areas.
    
    🎯 WHY THIS IS SUPERIOR:
    -------------------------
    Compared to _local_pathfind_from_tiles (15x15 local view):
    - Uses COMPLETE explored map from MapStitcher (hundreds of tiles)
    - Sees entire room/area layout for global planning
    - Can plan around obstacles that are outside immediate view
    - Handles both direct goals AND frontier exploration
    - Integrates warp avoidance with position history
    
    🔍 THREE-TIER NAVIGATION STRATEGY:
    -----------------------------------
    
    1️⃣ DIRECT PATHFINDING (goal_coords provided and explored):
       - Paths directly to the goal coordinates if they're walkable
       - Handles portal tiles ('?') as valid targets
       - Used for specific targets like "walk to NPC at (9,3)"
    
    2️⃣ FRONTIER NAVIGATION (goal unexplored or blocked):
       - Finds "frontier tiles" - explored tiles adjacent to unknown areas
       - Prioritizes frontiers in the direction of the goal
       - **SMART MAZE HANDLING**: When direct frontier blocked, includes
         perpendicular frontiers as backup targets (KEY INNOVATION)
       - This allows pathfinding to navigate around obstacles by exploring
         sideways when forward path is blocked
    
    3️⃣ FALLBACK (no pathfindable frontiers):
       - Returns None to let VLM handle edge cases
       - VLM can use visual context to make decisions
    
    🧩 MAZE NAVIGATION INNOVATION:
    -------------------------------
    The key breakthrough for maze navigation (e.g., Petalburg Woods):
    
    **Problem**: Agent gets stuck when direct path to goal is blocked
    - Goal: North exit at (7, 0)
    - Obstacle: Trees blocking direct north path
    - Old behavior: Only tried north frontier → failed → gave up
    
    **Solution**: Multi-directional frontier search
    - PRIMARY frontiers: Directly toward goal (north)
    - PERPENDICULAR frontiers: Side directions (east/west)
    - A* tries PRIMARY first, then PERPENDICULAR as backup
    - Result: Discovers maze paths by exploring perpendicular when blocked
    
    Example:
    ```
    Goal: North (↑)
    
    Old strategy:          New strategy:
    ####?####              ####?####
    ##?..?##               ##5..3##  ← Try perpendicular (3,5)
    ##....##               ##....##
    ##.P..##               ##.P..##
    ↑ Only try 1,2         ↑ Try 1,2 first, then 3-5
    
    Result: Can find path  Result: Finds path around
    if 1,2 reachable      obstacle via 3→4→5→2→1
    ```
    
    📊 SCORING SYSTEM:
    -------------------
    Frontier tiles are scored to prioritize best exploration:
    - Lower score = higher priority
    - score = distance_to_goal + (distance_to_player * weight)
    - Primary frontiers: -5 bonus (strongly preferred)
    - Perpendicular frontiers: +0.2 weight (allowed but secondary)
    
    🎮 INTEGRATION WITH GAME MECHANICS:
    ------------------------------------
    - Handles portals ('?') as valid walkable targets
    - Avoids ledges (directional constraints)  
    - Penalizes tall grass to avoid random encounters (speedrun mode)
    - Respects doors/stairs but allows them as final targets
    - Batches movement for efficiency (up to 8 steps)
    - Truncates paths at warps for safety
    
    Args:
        location_grid: Dictionary mapping (x, y) tuples to tile symbols
        bounds: Dictionary with min_x, max_x, min_y, max_y for coordinate conversion
        current_pos: Player's current (x, y) position IN ABSOLUTE WORLD COORDINATES
        location: Current location name
        goal_direction: Target direction ('north', 'south', 'east', 'west')
        recent_positions: Deque of recent (x, y, location) tuples for warp avoidance
        goal_coords: Optional specific goal (x, y) to pathfind to. If provided, pathfinds
                     directly to this position instead of finding frontier tiles.
        avoid_grass: If True (default), penalize grass tiles to avoid encounters. 
                     If False, allow grass tiles for trainer avoidance or training.
    
    Returns:
        First step direction ('UP', 'DOWN', 'LEFT', 'RIGHT') or None if no path
    
    ============================================================================
    """
    try:
        from collections import deque
        import heapq
        
        # Location grid is already provided as parameter (no need to fetch from map_stitcher)
        if not location_grid:
            print(f"⚠️ [A* MAP] No grid data provided")
            return None
        
        # Bounds are already provided as parameter
        current_x, current_y = current_pos
        
        # CRITICAL: Grid keys are now in WORLD coordinates (absolute), not relative!
        # No need to convert - just check if current position exists in grid directly
        if current_pos not in location_grid:
            print(f"⚠️ [A* MAP] Current position {current_pos} not in explored grid")
            print(f"   Bounds: X:{bounds['min_x']}-{bounds['max_x']}, Y:{bounds['min_y']}-{bounds['max_y']}")
            print(f"   Grid has {len(location_grid)} tiles, sample keys: {list(location_grid.keys())[:5]}")
            return None
        
        print(f"✅ [A* MAP] Using map stitcher grid with {len(location_grid)} explored tiles")
        print(f"   Position: {current_pos} (world coordinates)")
        print(f"   Bounds: X:{bounds['min_x']}-{bounds['max_x']}, Y:{bounds['min_y']}-{bounds['max_y']}")
        
        # Determine target positions based on whether we have specific goal coordinates
        target_positions = []  # Initialize empty
        
        if goal_coords:
            # We have a specific destination - pathfind directly to it
            goal_x, goal_y = goal_coords
            print(f"🎯 [A* DIRECT] Pathfinding to specific goal: ({goal_x}, {goal_y})")
            
            # Check if goal is in the grid
            if goal_coords in location_grid:
                # Goal is explored - validate it's walkable
                goal_tile = location_grid[goal_coords]
                # Portal tiles ('?') are valid navigation targets - agent must walk onto them to trigger transition
                # Walkable tiles: '.' (floor), '_' (path), '~' (grass), 'D' (door), '?' (portal/warp/unknown)
                if goal_tile not in ['.', '_', '~', 'D', '?']:
                    print(f"⚠️ [A* DIRECT] Goal {goal_coords} is not walkable (tile: '{goal_tile}')")
                    print(f"   Falling back to frontier navigation toward goal region")
                    # Goal explored but blocked - use frontier navigation to find alternative route
                    # This handles cases where goal coords are approximate (e.g., exit portal nearby)
                    # Fall through to frontier-based navigation below
                else:
                    # Use the goal coordinates as the single target
                    target_positions = [goal_coords]
                    if goal_tile == '?':
                        print(f"✅ [A* DIRECT] Goal is portal/warp tile ('?'), pathfinding to: {goal_coords}")
                    else:
                        print(f"✅ [A* DIRECT] Goal is walkable, pathfinding to: {goal_coords}")
            
            # If target_positions not set (goal unwalkable or unexplored), use frontier navigation
            if not target_positions:
                # Goal not yet explored - find frontier tiles in that direction
                print(f"⚠️ [A* DIRECT] Goal {goal_coords} not in explored grid yet")
                print(f"   Using smart frontier navigation toward goal coordinates")
                
                # Calculate direction to goal
                player_x, player_y = current_pos
                dx = goal_x - player_x
                dy = goal_y - player_y
                
                # Find frontier tiles that are closest to the goal direction
                # IMPROVED: Allow exploration in perpendicular directions for maze navigation
                target_positions = []
                for (x, y), tile in location_grid.items():
                    if tile not in ['.', '_', '~', 'D', '?']:
                        continue
                    
                    # Check if this tile is in the general direction of the goal
                    tile_dx = x - player_x
                    tile_dy = y - player_y
                    
                    # RELAXED CHECK: Allow tiles that make progress in EITHER dimension
                    # This enables maze navigation where you must go sideways to eventually go forward
                    # Old: dot_product check was too strict for mazes
                    # New: Accept if tile makes progress in primary direction OR doesn't go backwards
                    primary_progress = False
                    if abs(dx) >= abs(dy):
                        # Primarily horizontal goal - prioritize X progress
                        if dx > 0 and tile_dx > 0:
                            primary_progress = True
                        elif dx < 0 and tile_dx < 0:
                            primary_progress = True
                    else:
                        # Primarily vertical goal - prioritize Y progress
                        if dy > 0 and tile_dy > 0:
                            primary_progress = True
                        elif dy < 0 and tile_dy < 0:
                            primary_progress = True
                    
                    # Also accept tiles that don't go backwards (allow perpendicular movement)
                    not_backwards = True
                    if abs(dx) >= abs(dy):
                        # Don't go backwards horizontally
                        if dx > 0 and tile_dx < -2:
                            not_backwards = False
                        elif dx < 0 and tile_dx > 2:
                            not_backwards = False
                    else:
                        # Don't go backwards vertically
                        if dy > 0 and tile_dy < -2:
                            not_backwards = False
                        elif dy < 0 and tile_dy > 2:
                            not_backwards = False
                    
                    # Accept if either making primary progress OR not going backwards
                    if not (primary_progress or not_backwards):
                        continue
                    
                    # Check if it's a frontier tile (has unexplored adjacent)
                    has_unknown_neighbor = False
                    for nx, ny in [(x-1,y), (x+1,y), (x,y-1), (x,y+1)]:
                        if (nx, ny) not in location_grid:
                            has_unknown_neighbor = True
                            break
                    
                    if has_unknown_neighbor:
                        # Calculate how aligned this frontier tile is with goal direction
                        distance_to_player = abs(tile_dx) + abs(tile_dy)
                        distance_to_goal = abs(goal_x - x) + abs(goal_y - y)
                        # Prefer tiles that reduce distance to goal, but allow some flexibility
                        # Bonus for tiles that make progress in primary direction
                        score = distance_to_goal + (distance_to_player * 0.1)
                        if primary_progress:
                            score -= 5  # Bonus for primary direction progress
                        target_positions.append((score, x, y))
                
                if target_positions:
                    # Sort by score (lower is better - closer to goal)
                    target_positions.sort()
                    primary_targets = [(x, y) for score, x, y in target_positions[:5]]
                    print(f"✅ [A* FRONTIER → GOAL] Found {len(primary_targets)} PRIMARY frontier tiles toward ({goal_x}, {goal_y})")
                    target_positions = primary_targets
                else:
                    print(f"⚠️ [A* FRONTIER → GOAL] No PRIMARY frontier found")
                    target_positions = []
                
                # IMPROVEMENT: If we found frontier tiles but they might be blocked,
                # also add perpendicular frontier tiles as backup targets
                # This allows maze navigation when direct path is blocked
                perpendicular_targets = []
                for (x, y), tile in location_grid.items():
                    if tile not in ['.', '_', '~', 'D', '?']:
                        continue
                    
                    tile_dx = x - player_x
                    tile_dy = y - player_y
                    
                    # Check if this is perpendicular to primary direction
                    is_perpendicular = False
                    if abs(dy) > abs(dx):
                        # Primary is vertical, perpendicular is horizontal
                        if abs(tile_dx) > 2 and abs(tile_dy) < 3:
                            is_perpendicular = True
                    else:
                        # Primary is horizontal, perpendicular is vertical
                        if abs(tile_dy) > 2 and abs(tile_dx) < 3:
                            is_perpendicular = True
                    
                    if not is_perpendicular:
                        continue
                    
                    # Check if it's a frontier tile (has unexplored adjacent)
                    has_unknown_neighbor = False
                    for nx, ny in [(x-1,y), (x+1,y), (x,y-1), (x,y+1)]:
                        if (nx, ny) not in location_grid:
                            has_unknown_neighbor = True
                            break
                    
                    if has_unknown_neighbor:
                        distance_to_player = abs(tile_dx) + abs(tile_dy)
                        distance_to_goal = abs(goal_x - x) + abs(goal_y - y)
                        score = distance_to_goal + (distance_to_player * 0.2)
                        perpendicular_targets.append((score, x, y))
                
                # Add perpendicular targets as backup
                if perpendicular_targets:
                    perpendicular_targets.sort()
                    perp_positions = [(x, y) for score, x, y in perpendicular_targets[:5]]
                    print(f"✅ [A* FRONTIER → GOAL] Found {len(perp_positions)} PERPENDICULAR frontier tiles as backup")
                    # Add perpendicular targets AFTER primary targets (try primary first)
                    target_positions = target_positions + perp_positions
                
                # Final fallback: use extreme edge
                if not target_positions:
                    print(f"⚠️ [A* FRONTIER → GOAL] No frontier found at all, using extreme edge in goal direction")
                    if abs(dx) > abs(dy):
                        # Primarily horizontal
                        if dx > 0:
                            # Go east
                            max_x = max(x for x, y in location_grid.keys())
                            target_positions = [(x, y) for x, y in location_grid.keys() 
                                              if x == max_x and location_grid[(x, y)] in ['.', '_', '~', '?']]
                        else:
                            # Go west
                            min_x = min(x for x, y in location_grid.keys())
                            target_positions = [(x, y) for x, y in location_grid.keys() 
                                              if x == min_x and location_grid[(x, y)] in ['.', '_', '~', '?']]
                    else:
                        # Primarily vertical
                        if dy > 0:
                            # Go south
                            max_y = max(y for x, y in location_grid.keys())
                            target_positions = [(x, y) for x, y in location_grid.keys() 
                                              if y == max_y and location_grid[(x, y)] in ['.', '_', '~', '?']]
                        else:
                            # Go north
                            min_y = min(y for x, y in location_grid.keys())
                            target_positions = [(x, y) for x, y in location_grid.keys() 
                                              if y == min_y and location_grid[(x, y)] in ['.', '_', '~', '?']]
        else:
            # No specific goal - use frontier-based exploration (old behavior)
            print(f"🎯 [A* FRONTIER] No specific goal, finding frontier tiles in direction '{goal_direction}'")
            
            # SMART TARGET SELECTION: Find tiles at the edge of exploration (frontier)
            # These are walkable tiles adjacent to unknown '?' tiles in the goal direction
            # This ensures we explore toward the goal rather than targeting unreachable extremes
            
            def is_frontier_tile(pos: Tuple[int, int], direction: str) -> bool:
                """Check if a tile is on the frontier (adjacent to unknown in goal direction)"""
                x, y = pos
                tile = location_grid.get(pos)
                
                # Must be walkable
                if tile not in ['.', '_', '~', 'D']:
                    return False
                
                # Check if there's unknown territory in the goal direction
                if direction in ['north', 'up']:
                    # Check tiles to the north
                    for check_y in range(y - 3, y):  # Check 3 tiles north
                        if (x, check_y) not in location_grid or location_grid.get((x, check_y)) == '?':
                            return True
                elif direction in ['south', 'down']:
                    for check_y in range(y + 1, y + 4):
                        if (x, check_y) not in location_grid or location_grid.get((x, check_y)) == '?':
                            return True
                elif direction in ['east', 'right']:
                    for check_x in range(x + 1, x + 4):
                        if (check_x, y) not in location_grid or location_grid.get((check_x, y)) == '?':
                            return True
                elif direction in ['west', 'left']:
                    for check_x in range(x - 3, x):
                        if (check_x, y) not in location_grid or location_grid.get((check_x, y)) == '?':
                            return True
                
                return False
            
            # Find frontier tiles in the goal direction
            target_positions = []
            player_x, player_y = current_pos
            
            for (x, y), tile in location_grid.items():
                # Only consider walkable tiles
                if tile not in ['.', '_', '~', 'D']:
                    continue
                
                # Check if in the goal direction from player
                is_in_direction = False
                if goal_direction.lower() in ['north', 'up'] and y < player_y:
                    is_in_direction = True
                elif goal_direction.lower() in ['south', 'down'] and y > player_y:
                    is_in_direction = True
                elif goal_direction.lower() in ['east', 'right'] and x > player_x:
                    is_in_direction = True
                elif goal_direction.lower() in ['west', 'left'] and x < player_x:
                    is_in_direction = True
                
                # If in direction and on frontier, add it
                if is_in_direction and is_frontier_tile((x, y), goal_direction):
                    distance = abs(x - player_x) + abs(y - player_y)
                    target_positions.append((distance, x, y))
            
            # Sort by distance and take closest frontier tiles
            if target_positions:
                target_positions.sort()
                # Keep tiles within reasonable distance
                max_distance = min(15, target_positions[0][0] + 10) if target_positions else 15
                target_positions = [(x, y) for dist, x, y in target_positions if dist <= max_distance]
                print(f"🎯 [A* FRONTIER] Found {len(target_positions)} frontier tiles in direction '{goal_direction}'")
                print(f"   Targeting exploration edge (tiles adjacent to unknown)")
            else:
                # Fallback: No frontier found, target extreme edge (old behavior)
                print(f"⚠️ [A* FRONTIER] No frontier tiles found, using extreme edge as fallback")
                if goal_direction.lower() in ['north', 'up']:
                    min_y = min(y for x, y in location_grid.keys())
                    target_positions = [(x, y) for x, y in location_grid.keys() 
                                      if y == min_y and location_grid[(x, y)] in ['.', '_', '~']]
                elif goal_direction.lower() in ['south', 'down']:
                    max_y = max(y for x, y in location_grid.keys())
                    target_positions = [(x, y) for x, y in location_grid.keys() 
                                      if y == max_y and location_grid[(x, y)] in ['.', '_', '~']]
                elif goal_direction.lower() in ['east', 'right']:
                    max_x = max(x for x, y in location_grid.keys())
                    target_positions = [(x, y) for x, y in location_grid.keys() 
                                      if x == max_x and location_grid[(x, y)] in ['.', '_', '~']]
                elif goal_direction.lower() in ['west', 'left']:
                    min_x = min(x for x, y in location_grid.keys())
                    target_positions = [(x, y) for x, y in location_grid.keys() 
                                      if x == min_x and location_grid[(x, y)] in ['.', '_', '~']]
        
        if not target_positions:
            print(f"⚠️ [A* MAP] No valid target positions in direction '{goal_direction}'")
            return None
        
        print(f"🎯 [A* MAP] Found {len(target_positions)} potential targets in direction '{goal_direction}'")
        
        # Helper function to check if position should be avoided (warp detection)
        # TEMPORARILY DISABLED - was blocking valid paths after warping
        def should_avoid_position(world_pos: Tuple[int, int]) -> bool:
            """Check if position should be avoided (was a warp to different location).
            
            Args:
                world_pos: Position in WORLD coordinates (matches location_grid keys)
            """
            # COMMENTED OUT: This was preventing navigation after warping from Route 103 to Oldale Town
            # if not recent_positions:
            #     return False
            # 
            # # Position is already in WORLD coordinates - use directly
            # world_x, world_y = world_pos
            # 
            # # Check if this position matches a recent position with different location
            # for recent_x, recent_y, recent_loc in recent_positions:
            #     if recent_x == world_x and recent_y == world_y and recent_loc != location:
            #         print(f"🚫 [A* WARP AVOID] Skipping world pos {world_pos} - recent warp position from '{recent_loc}'")
            #         return True
            return False
        
        # Define ledge tile symbols (from map_stitcher.py behavior values)
        # Ledges are one-way: you can only traverse them in their arrow direction
        # NOTE: Diagonal ledges are IMPASSABLE because player can only move in cardinal directions!
        LEDGE_TILES = {
            '→': 'RIGHT',   # JUMP_EAST (behavior 56) - traversable
            '←': 'LEFT',    # JUMP_WEST (behavior 57) - traversable
            '↑': 'UP',      # JUMP_NORTH (behavior 58) - traversable
            '↓': 'DOWN',    # JUMP_SOUTH (behavior 59) - traversable
            'L': None       # Generic ledge (collision 3) - direction unknown, treat conservatively
        }
        
        # Diagonal ledges are WALLS - player cannot move diagonally in Pokemon
        # These exist in the game but are effectively impassable
        DIAGONAL_LEDGES = ['↗', '↖', '↘', '↙']  # behaviors 60-63
        
        # Helper function to check if tile is walkable
        def is_walkable(pos: Tuple[int, int]) -> bool:
            if pos not in location_grid:
                return False
            # Check dynamically blocked tiles (NPC/obstacle positions)
            if (pos[0], pos[1], location) in _dynamically_blocked_tiles:
                return False
            # Check live NPC obstacle tiles
            if (pos[0], pos[1], location) in _npc_occupied_tiles:
                return False
            tile = location_grid[pos]
            
            # Diagonal ledges are IMPASSABLE - treat as walls
            if tile in DIAGONAL_LEDGES:
                return False
            
            # Walkable: path, grass, doors, stairs, AND CARDINAL LEDGES (with directional constraints)
            # Note: We include 'D' (doors) and 'S' (stairs) for pathfinding, but safety checks will filter dangerous ones
            # Ledges are conditionally walkable based on approach direction (checked in neighbor loop)
            # '?' tiles = unexplored frontier tiles adjacent to walkable areas (added by map stitcher)
            # They MUST be walkable for A* to path through unexplored regions
            return tile in ['.', '_', '~', 'D', 'S', '?'] or tile in LEDGE_TILES
        
        # Helper function to get movement cost for a tile
        # This allows us to prefer paths that avoid tall grass (wild encounters)
        # Can be configured for "training mode" later to SEEK grass instead
        def get_tile_cost(pos: Tuple[int, int], avoid_grass: bool = True) -> float:
            """
            Calculate movement cost for a tile.
            
            Args:
                pos: Tile position
                avoid_grass: If True, penalize tall grass. If False, prefer it (training mode)
            
            Returns:
                Movement cost (lower = preferred path)
            """
            if pos not in location_grid:
                return 999  # Unknown/unwalkable tiles have very high cost
            
            tile = location_grid[pos]
            
            # Ledge tiles: Small penalty to avoid "point of no return" unless necessary
            # Ledges are one-way, so taking them commits you to that path
            if tile in LEDGE_TILES:
                return 1.2  # Slight penalty - prefer normal paths if available
            
            if avoid_grass:
                # SPEEDRUN MODE: Minimize wild encounters
                if tile == '~':  # Tall grass
                    return 3.0  # 3x cost - strongly avoid
                elif tile in ['.', '_']:  # Normal path
                    return 1.0  # Standard cost
                elif tile in ['D', 'S']:  # Doors, stairs
                    return 1.5  # Slight penalty (might trigger events)
                else:
                    return 2.0  # Unknown walkable, slight penalty
            else:
                # TRAINING MODE: Seek wild encounters for leveling
                if tile == '~':  # Tall grass
                    return 0.5  # PREFER grass!
                elif tile in ['.', '_']:  # Normal path
                    return 1.0  # Standard cost
                else:
                    return 1.5
        
        # A* pathfinding with Manhattan distance heuristic
        def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
        # CRITICAL FIX: When we have a specific goal, choose target that's BEST aligned with goal direction
        # Don't just pick closest target - that can choose wrong-direction targets (e.g., warp to the LEFT when goal is NORTH)
        if goal_coords is not None:
            # We have a specific goal coordinate - choose frontier tile that moves toward it
            goal_x, goal_y = goal_coords
            
            def target_score(target_pos: Tuple[int, int]) -> float:
                """Score targets by how well they move toward the goal. Lower is better."""
                tx, ty = target_pos
                # Distance from this target to the goal
                dist_to_goal = manhattan_distance(target_pos, (goal_x, goal_y))
                # Distance from current position to this target
                dist_to_target = manhattan_distance(current_pos, target_pos)
                
                # Primary factor: how much closer to goal does this target get us?
                # Secondary factor: prefer closer targets (tie-breaker)
                # Weight: 10:1 ratio - prioritize goal alignment over proximity
                return dist_to_goal * 10 + dist_to_target
            
            closest_target = min(target_positions, key=target_score)
            print(f"🎯 [A* GOAL] Selected target {closest_target} (moves toward goal {goal_coords})")
        else:
            # No specific goal - just find closest frontier tile in the direction
            closest_target = min(target_positions, key=lambda t: manhattan_distance(current_pos, t))
        
        # Priority queue: (f_score, g_score, position, path, path_coords)
        # f_score = g_score + heuristic
        # NOTE: Currently using avoid_grass=True for speedrun mode
        # TODO: Add training mode parameter that sets avoid_grass=False
        #       This will make the agent SEEK grass tiles to level up Pokemon
        #       Example: astar_pathfind(..., training_mode=True)
        start = current_pos
        pq = [(manhattan_distance(start, closest_target), 0, start, [], [start])]  # Added path_coords
        visited = {start}
        
        directions_map = {
            (0, -1): 'UP',
            (0, 1): 'DOWN',
            (-1, 0): 'LEFT',
            (1, 0): 'RIGHT'
        }
        
        while pq:
            f_score, g_score, current, path, path_coords = heapq.heappop(pq)  # Added path_coords
            
            # Check if we reached any target
            if current in target_positions:
                if path:
                    # Truncate at warps for safety
                    path = _truncate_path_at_warp(path, path_coords[1:], location_grid)  # Skip start position
                    
                    # Batch multiple steps for faster navigation
                    batched_path = path[:MAX_MOVEMENT_BATCH_SIZE]
                    
                    path_preview = ' → '.join(path[:5])
                    if len(path) > 5:
                        path_preview += f" ... ({len(path)} steps)"
                    
                    batched_str = ' → '.join(batched_path)
                    
                    # Count special tiles in path for debugging
                    grass_count = sum(1 for pos in visited if location_grid.get(pos) == '~')
                    ledge_count = sum(1 for pos in visited if location_grid.get(pos) in LEDGE_TILES)
                    total_cost = g_score
                    
                    print(f"✅ [A* MAP] Found path: {path_preview}")
                    print(f"   📦 Batching {len(batched_path)}/{len(path)} steps: {batched_str}")
                    
                    # Build informative cost message
                    cost_parts = [f"Path cost: {total_cost:.1f}"]
                    if grass_count > 0:
                        cost_parts.append(f"avoided {grass_count} grass tiles")
                    if ledge_count > 0:
                        cost_parts.append(f"used {ledge_count} ledge(s)")
                    print(f"   {', '.join(cost_parts)}")
                    
                    return batched_path
                else:
                    print(f"⚠️ [A* MAP] Already at target")
                    return None
            
            # Explore neighbors
            for (dx, dy), direction in directions_map.items():
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)
                
                # Skip if already visited
                if neighbor in visited:
                    continue
                
                # Check if neighbor exists in grid
                if neighbor not in location_grid:
                    continue
                
                neighbor_tile = location_grid[neighbor]
                current_tile = location_grid[current]
                
                # === LEDGE PHYSICS: ONE-WAY TRAVERSAL ===
                # Ledges can only be traversed in their arrow direction
                # This prevents pathfinding from treating ledges as bidirectional paths
                
                # RULE 1: Entering a ledge - can only approach from the correct direction
                if neighbor_tile in LEDGE_TILES:
                    allowed_direction = LEDGE_TILES[neighbor_tile]
                    
                    # Generic ledge 'L' - we don't know direction, so be conservative
                    # Only allow entering from above (most common in Pokemon)
                    if allowed_direction is None:
                        if direction != 'DOWN':
                            # Blocked: trying to enter generic ledge from wrong direction
                            continue
                    # Directional ledge - must match arrow direction
                    elif direction != allowed_direction:
                        # Blocked: ledge arrow doesn't match movement direction
                        continue
                
                # RULE 2: Exiting a ledge - if currently ON a ledge, must continue in that direction
                # This prevents pathfinding from "turning around" mid-jump
                if current_tile in LEDGE_TILES:
                    ledge_direction = LEDGE_TILES[current_tile]
                    
                    if ledge_direction is None:
                        # Generic ledge - assume downward motion only
                        if direction != 'DOWN':
                            continue
                    elif direction != ledge_direction:
                        continue
                
                # === END LEDGE PHYSICS ===
                
                # Now check basic walkability (after ledge rules)
                # SPECIAL CASE: If neighbor is a goal tile (like a portal), allow it even if not normally walkable
                is_goal_tile = neighbor in target_positions
                if not is_goal_tile and not is_walkable(neighbor):
                    continue
                
                # If it IS a goal tile but not walkable (e.g., portal '?'), allow reaching it but don't explore beyond it
                if is_goal_tile and neighbor not in location_grid:
                    continue  # Goal tile not in grid - skip
                
                # Log when we allow non-walkable goal tile
                if is_goal_tile and not is_walkable(neighbor):
                    neighbor_tile_sym = location_grid.get(neighbor, '?')
                    print(f"🎯 [A* GOAL] Allowing non-walkable goal tile at {neighbor} ('{neighbor_tile_sym}')")
                
                # Skip if should avoid (warp detection)
                if should_avoid_position(neighbor):
                    continue
                
                visited.add(neighbor)
                new_path = path + [direction]
                new_path_coords = path_coords + [neighbor]  # Track coordinates for warp detection
                
                # Use tile cost instead of fixed cost of 1
                # This makes A* prefer paths that avoid tall grass and unnecessary ledges
                # Use the avoid_grass parameter passed to this function
                tile_cost = get_tile_cost(neighbor, avoid_grass=avoid_grass)
                new_g_score = g_score + tile_cost
                new_f_score = new_g_score + manhattan_distance(neighbor, closest_target)
                
                heapq.heappush(pq, (new_f_score, new_g_score, neighbor, new_path, new_path_coords))
        
        # No path found
        print(f"⚠️ [A* MAP] No path found to {goal_direction}")
        print(f"   Explored {len(visited)} tiles")
        print(f"   Target positions checked: {len(target_positions)}")
        
        # === FRONTIER FALLBACK ===
        # If direct pathfinding to goal_coords failed (goal is in grid but unreachable due to
        # ledges/barriers/unexplored gaps), retry with reachable frontier tiles closest to the goal.
        # The idea: explore toward the goal by navigating to the nearest frontier tile that A*
        # CAN reach, which will reveal new tiles and eventually open a path.
        if goal_coords and len(target_positions) == 1 and target_positions[0] == goal_coords:
            print(f"🔄 [A* FRONTIER FALLBACK] Direct path to {goal_coords} blocked, searching for reachable frontier tiles...")
            
            goal_x, goal_y = goal_coords
            
            # Find frontier tiles among the tiles we already explored (visited set)
            # These are tiles we CAN reach that border unexplored territory
            # NOTE: The map stitcher adds '?' tiles to location_grid for unexplored positions
            # adjacent to walkable tiles. So "unexplored" means EITHER:
            #   1. (nx, ny) not in location_grid at all, OR
            #   2. location_grid[(nx, ny)] == '?' (frontier marker added by stitcher)
            frontier_targets = []
            for pos in visited:
                tile = location_grid.get(pos)
                if tile not in ['.', '_', '~', 'D', '?']:
                    continue
                # Check if adjacent to unexplored tile or '?' frontier marker
                x, y = pos
                has_unknown = False
                for nx, ny in [(x-1,y), (x+1,y), (x,y-1), (x,y+1)]:
                    neighbor_tile = location_grid.get((nx, ny))
                    if neighbor_tile is None or neighbor_tile == '?':
                        has_unknown = True
                        break
                if has_unknown:
                    dist_to_goal = abs(goal_x - x) + abs(goal_y - y)
                    dist_to_player = abs(current_x - x) + abs(current_y - y)
                    # Score: prioritize closeness to goal, penalize distance from player slightly
                    score = dist_to_goal + dist_to_player * 0.1
                    frontier_targets.append((score, pos))
            
            if frontier_targets:
                frontier_targets.sort()
                # Take best frontier tiles
                best_frontiers = [pos for _, pos in frontier_targets[:10]]
                print(f"✅ [A* FRONTIER FALLBACK] Found {len(frontier_targets)} reachable frontier tiles, using top {len(best_frontiers)}")
                for i, (score, pos) in enumerate(frontier_targets[:5]):
                    print(f"   #{i+1}: {pos} (score={score:.1f}, tile='{location_grid.get(pos, '?')}')")
                
                # Re-run A* with frontier targets
                # Pick the best frontier tile as heuristic target
                closest_frontier = min(best_frontiers, key=lambda t: abs(goal_x - t[0]) + abs(goal_y - t[1]))
                
                pq2 = [(manhattan_distance(current_pos, closest_frontier), 0, current_pos, [], [current_pos])]
                visited2 = {current_pos}
                frontier_set = set(best_frontiers)
                
                while pq2:
                    f_score, g_score, current, path, path_coords = heapq.heappop(pq2)
                    
                    if current in frontier_set:
                        if path:
                            path = _truncate_path_at_warp(path, path_coords[1:], location_grid)
                            batched_path = path[:MAX_MOVEMENT_BATCH_SIZE]
                            path_preview = ' → '.join(path[:5])
                            if len(path) > 5:
                                path_preview += f" ... ({len(path)} steps)"
                            print(f"✅ [A* FRONTIER FALLBACK] Found path to frontier {current}: {path_preview}")
                            print(f"   📦 Batching {len(batched_path)}/{len(path)} steps")
                            return batched_path
                        else:
                            print(f"⚠️ [A* FRONTIER FALLBACK] Already at frontier tile")
                            return None
                    
                    for (ddx, ddy), direction in directions_map.items():
                        nx, ny = current[0] + ddx, current[1] + ddy
                        neighbor = (nx, ny)
                        
                        if neighbor in visited2:
                            continue
                        if neighbor not in location_grid:
                            continue
                        
                        neighbor_tile = location_grid[neighbor]
                        current_tile = location_grid[current]
                        
                        # Ledge physics (same rules as main A*)
                        if neighbor_tile in LEDGE_TILES:
                            allowed_dir = LEDGE_TILES[neighbor_tile]
                            if allowed_dir is None:
                                if direction != 'DOWN':
                                    continue
                            elif direction != allowed_dir:
                                continue
                        if current_tile in LEDGE_TILES:
                            ledge_dir = LEDGE_TILES[current_tile]
                            if ledge_dir is None:
                                if direction != 'DOWN':
                                    continue
                            elif direction != ledge_dir:
                                continue
                        
                        if not is_walkable(neighbor) and neighbor not in frontier_set:
                            continue
                        
                        visited2.add(neighbor)
                        new_path = path + [direction]
                        new_path_coords = path_coords + [neighbor]
                        tile_cost = get_tile_cost(neighbor, avoid_grass=avoid_grass)
                        new_g = g_score + tile_cost
                        new_f = new_g + manhattan_distance(neighbor, closest_frontier)
                        heapq.heappush(pq2, (new_f, new_g, neighbor, new_path, new_path_coords))
                
                print(f"⚠️ [A* FRONTIER FALLBACK] No path to any frontier tile either")
            else:
                print(f"⚠️ [A* FRONTIER FALLBACK] No frontier tiles found among {len(visited)} explored tiles")
                # DIAGNOSTIC: Dump grid around player to understand the topology
                print(f"📊 [GRID DUMP] Map around player ({current_x}, {current_y}):")
                dump_radius = 8
                for dy in range(-dump_radius, dump_radius + 1):
                    row_chars = []
                    for dx in range(-dump_radius, dump_radius + 1):
                        gx, gy = current_x + dx, current_y + dy
                        if (gx, gy) == (current_x, current_y):
                            row_chars.append('P')
                        elif (gx, gy) in visited:
                            tile = location_grid.get((gx, gy), ' ')
                            row_chars.append(tile)
                        elif (gx, gy) in location_grid:
                            # In grid but NOT reachable by A* — show in brackets concept
                            tile = location_grid.get((gx, gy), ' ')
                            row_chars.append(tile.lower() if tile.isalpha() else tile)
                        else:
                            row_chars.append(' ')
                    y_coord = current_y + dy
                    print(f"   y={y_coord:3d}: {' '.join(row_chars)}")
                print(f"   Legend: P=player, UPPER=reachable, lower=in-grid-but-unreachable, ' '=not-in-grid")
                
                # Show boundary analysis
                boundary_tiles = {}
                for pos in visited:
                    x, y = pos
                    for nx, ny in [(x-1,y), (x+1,y), (x,y-1), (x,y+1)]:
                        if (nx, ny) not in visited and (nx, ny) in location_grid:
                            tile = location_grid[(nx, ny)]
                            boundary_tiles[(nx, ny)] = tile
                tile_counts = {}
                for t in boundary_tiles.values():
                    tile_counts[t] = tile_counts.get(t, 0) + 1
                print(f"📊 [BOUNDARY] {len(boundary_tiles)} non-reachable border tiles: {tile_counts}")
        
        return None
        
    except Exception as e:
        print(f"❌ [A* MAP] Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# UNIFIED PATHFINDING HELPER
# ============================================================================

def pathfind_to_goal(state_data: Dict[str, Any], goal_x: int, goal_y: int,
                     avoid_grass: bool = True,
                     npc_coords: Optional[Tuple[int, int]] = None) -> Optional[List[str]]:
    """
    Multi-tier pathfinding to reach a goal coordinate.

    Strategy (3 tiers):
    1. Local A* (fast): Works if goal within visible 15x15 grid
    2. Global A* (smart): Uses complete explored map from map stitcher
       - If goal is explored: direct pathfinding to goal coordinates
       - If goal is unexplored: frontier-based exploration toward goal
    3. Returns None so caller can apply its own fallback

    Args:
        state_data: Current game state
        goal_x: Target world X coordinate
        goal_y: Target world Y coordinate
        avoid_grass: If True, penalize grass tiles to avoid encounters
        npc_coords: Optional (x, y) of the target NPC.  That NPC's tile
            is excluded from the obstacle set so A* can plan a path to it.

    Returns:
        List of direction strings or None if no path found
    """
    # Refresh NPC obstacle layer from live gObjectEvents data.
    # Exclude the target NPC tile so A* can pathfind *to* them.
    update_npc_obstacles(state_data, exclude_coords=npc_coords)

    player_data = state_data.get('player', {})
    position = player_data.get('position', {})
    current_x = position.get('x', 0)
    current_y = position.get('y', 0)
    location = player_data.get('location', '')

    # Debug: log NPC obstacle count for diagnosis
    if _npc_occupied_tiles:
        nearby = [(x, y) for x, y, loc in _npc_occupied_tiles
                  if loc == location and abs(x - current_x) <= 8 and abs(y - current_y) <= 8]
        if nearby:
            logger.debug(f"[NPC OBSTACLES] {len(_npc_occupied_tiles)} NPC tiles, "
                         f"{len(nearby)} nearby: {sorted(nearby)}")
    else:
        npc_count = len(state_data.get('active_npcs', []))
        if npc_count > 0:
            logger.debug(f"[NPC OBSTACLES] 0 obstacle tiles despite {npc_count} active_npcs "
                         f"(all filtered: player/invisible/off_screen?)")

    # Tier 1: Local A* on 15x15 visible grid
    result = _pathfind_to_target(state_data, goal_x, goal_y)
    if result:
        print(f"✅ [PATHFIND] Local A* found path to ({goal_x}, {goal_y})")
        return result if isinstance(result, list) else [result]

    # Tier 2: Global A* with map stitcher data
    stitched_map_info = state_data.get('map', {}).get('stitched_map_info')
    if stitched_map_info and stitched_map_info.get('available'):
        current_area = stitched_map_info.get('current_area', {})
        grid_serializable = current_area.get('grid')
        bounds = current_area.get('bounds')

        if grid_serializable and len(grid_serializable) > 0:
            # Calculate bounds from grid if missing
            if not bounds:
                xs, ys = [], []
                for key in grid_serializable.keys():
                    x, y = map(int, key.split(','))
                    xs.append(x)
                    ys.append(y)
                if xs and ys:
                    bounds = {
                        'min_x': min(xs), 'max_x': max(xs),
                        'min_y': min(ys), 'max_y': max(ys)
                    }

            if bounds:
                # Convert grid to proper format
                location_grid = {}
                for key, value in grid_serializable.items():
                    x, y = map(int, key.split(','))
                    location_grid[(x, y)] = value

                # Calculate goal direction
                dx = goal_x - current_x
                dy = goal_y - current_y
                if abs(dy) > abs(dx):
                    goal_direction = 'south' if dy > 0 else 'north'
                else:
                    goal_direction = 'east' if dx > 0 else 'west'

                result = _astar_pathfind_with_grid_data(
                    location_grid=location_grid,
                    bounds=bounds,
                    current_pos=(current_x, current_y),
                    location=location,
                    goal_direction=goal_direction,
                    recent_positions=_recent_positions,
                    goal_coords=(goal_x, goal_y),
                    avoid_grass=avoid_grass
                )
                if result:
                    print(f"✅ [PATHFIND] Global A* found path to ({goal_x}, {goal_y})")
                    return result if isinstance(result, list) else [result]

                # Tier 2b: goal tile is '#' (e.g. gym door w/ collision=1 in stored grid,
                # or west-edge boundary tile for Route 104 exit) but the tile is a real
                # warp/boundary trigger — try all 4 adjacent approach tiles.
                # Order: south (goal_y+1), east (goal_x+1), north (goal_y-1), west (goal_x-1)
                goal_tile_symbol = location_grid.get((goal_x, goal_y))
                if goal_tile_symbol and goal_tile_symbol not in ['.', '_', '~', 'D', '?', 'S']:
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        approach = (goal_x + dx, goal_y + dy)
                        approach_tile = location_grid.get(approach)
                        if approach_tile in ['.', '_', '~']:
                            print(f"🔑 [PATHFIND] Goal ({goal_x},{goal_y})='{goal_tile_symbol}', trying approach tile {approach}")
                            result = _astar_pathfind_with_grid_data(
                                location_grid=location_grid,
                                bounds=bounds,
                                current_pos=(current_x, current_y),
                                location=location,
                                goal_direction=goal_direction,
                                recent_positions=_recent_positions,
                                goal_coords=approach,
                                avoid_grass=avoid_grass
                            )
                            if result:
                                print(f"✅ [PATHFIND] Approach tile path found to {approach}")
                                return result if isinstance(result, list) else [result]

    print(f"⚠️ [PATHFIND] All pathfinding tiers failed for ({goal_x}, {goal_y})")
    return None
