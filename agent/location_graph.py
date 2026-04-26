"""
Location Graph for Pokemon Emerald Navigation

This module defines the game world as a connected graph of locations and portals.
Each location has portals (connections) to other locations, with details about
coordinates, portal types, and navigation directions.

Portal Types:
- "open_world": Open transitions between routes/towns (walk through boundary)
- "warp_tile": Instant warp when stepping on specific tile (doors, cave entrances)
- "ledge": One-way drop (can jump down but not climb back up)

Coordinate System:
- entry_coords: Where you appear when entering this location from the connected location
- exit_coords: Where you need to stand to trigger the portal/transition
- For open_world portals: exit_coords is typically at the map boundary
- For warp_tile portals: exit_coords is the door/entrance tile itself
"""

from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# === LOCATION GRAPH DATABASE ===
LOCATION_GRAPH: Dict[str, Dict[str, Any]] = {
    
    # ========================================
    # LITTLEROOT TOWN
    # ========================================
    "LITTLEROOT_TOWN": {
        "map_id": "0009",
        "display_name": "Littleroot Town",
        "description": "Starting town, contains player's house and rival's house",
        "portals": {
            "ROUTE_101": {
                "type": "open_world",
                "direction": "north",
                "entry_coords": (10, 28),  # Where you appear in Littleroot from Route 101
                "exit_coords": (10, 0),    # North boundary of Littleroot
                "description": "North exit to Route 101",
                "requirements": None,
            },
            "PLAYERS_HOUSE_1F": {
                "type": "warp_tile",
                "direction": "interact",
                "entry_coords": (4, 7),    # Inside house, near door
                "exit_coords": (4, 7),     # Door tile in Littleroot Town
                "description": "Enter player's house",
                "requirements": None,
            },
            "MAYS_HOUSE_1F": {
                "type": "warp_tile",
                "direction": "interact",
                "entry_coords": (2, 9),    # Inside May's house, near door
                "exit_coords": (14, 8),    # May's house door in Littleroot
                "description": "Enter rival May's house",
                "requirements": None,
            },
            "PROFESSOR_BIRCHS_LAB": {
                "type": "warp_tile",
                "direction": "interact",
                "entry_coords": (6, 13),   # Inside lab, near door
                "exit_coords": (7, 16),    # Lab door in Littleroot Town
                "description": "Enter Professor Birch's laboratory",
                "requirements": None,
            },
        }
    },
    
    # ========================================
    # PLAYER'S HOUSE
    # ========================================
    "PLAYERS_HOUSE_1F": {
        "map_id": "0100",
        "display_name": "Player's House (1F)",
        "description": "First floor of player's house",
        "portals": {
            "LITTLEROOT_TOWN": {
                "type": "warp_tile",
                "direction": "interact",
                "entry_coords": (4, 7),
                "exit_coords": (4, 7),
                "description": "Exit to Littleroot Town",
                "requirements": None,
            },
            "PLAYERS_HOUSE_2F": {
                "type": "warp_tile",
                "direction": "interact",
                "entry_coords": (8, 2),    # Top of stairs on 2F
                "exit_coords": (8, 2),     # Stairs on 1F
                "description": "Stairs to player's bedroom",
                "requirements": None,
            },
        }
    },
    
    "PLAYERS_HOUSE_2F": {
        "map_id": "0101",
        "display_name": "Player's House (2F)",
        "description": "Player's bedroom",
        "portals": {
            "PLAYERS_HOUSE_1F": {
                "type": "warp_tile",
                "direction": "interact",
                "entry_coords": (8, 2),
                "exit_coords": (8, 2),
                "description": "Stairs down to first floor",
                "requirements": None,
            },
        }
    },
    
    # ========================================
    # MAY'S HOUSE (RIVAL)
    # ========================================
    "MAYS_HOUSE_1F": {
        "map_id": "0102",
        "display_name": "May's House (1F)",
        "description": "First floor of rival May's house",
        "portals": {
            "LITTLEROOT_TOWN": {
                "type": "warp_tile",
                "direction": "interact",
                "entry_coords": (2, 9),
                "exit_coords": (2, 9),
                "description": "Exit to Littleroot Town",
                "requirements": None,
            },
            "MAYS_HOUSE_2F": {
                "type": "warp_tile",
                "direction": "interact",
                "entry_coords": (1, 1),
                "exit_coords": (2, 2),
                "description": "Stairs to May's bedroom",
                "requirements": None,
            },
        }
    },
    
    "MAYS_HOUSE_2F": {
        "map_id": "0103",
        "display_name": "May's House (2F)",
        "description": "May's bedroom",
        "portals": {
            "MAYS_HOUSE_1F": {
                "type": "warp_tile",
                "direction": "interact",
                "entry_coords": (1, 1),
                "exit_coords": (1, 1),
                "description": "Stairs down to first floor",
                "requirements": None,
            },
        }
    },
    
    # ========================================
    # PROFESSOR BIRCH'S LAB
    # ========================================
    "PROFESSOR_BIRCHS_LAB": {
        "map_id": "0104",
        "display_name": "Professor Birch's Lab",
        "description": "Pokemon research laboratory",
        "portals": {
            "LITTLEROOT_TOWN": {
                "type": "warp_tile",
                "direction": "interact",
                "entry_coords": (6, 13),
                "exit_coords": (6, 13),
                "description": "Exit to Littleroot Town",
                "requirements": None,
            },
        }
    },
    
    # ========================================
    # ROUTE 101
    # ========================================
    "ROUTE_101": {
        "map_id": "0010",
        "display_name": "Route 101",
        "description": "First route, connects Littleroot Town to Oldale Town",
        "portals": {
            "LITTLEROOT_TOWN": {
                "type": "open_world",
                "direction": "south",
                "entry_coords": (10, 0),   # Where you appear in Route 101 from Littleroot
                "exit_coords": (10, 28),   # South boundary leading to Littleroot
                "description": "South to Littleroot Town",
                "requirements": None,
            },
            "OLDALE_TOWN": {
                "type": "open_world",
                "direction": "north",
                "entry_coords": (11, 24),  # Where you appear in Route 101 from Oldale
                "exit_coords": (11, 0),    # North boundary leading to Oldale
                "description": "North to Oldale Town",
                "requirements": None,
            },
        }
    },
    
    # ========================================
    # OLDALE TOWN
    # ========================================
    "OLDALE_TOWN": {
        "map_id": "000A",
        "display_name": "Oldale Town",
        "description": "Small town with Pokemon Center and Mart",
        "portals": {
            "ROUTE_101": {
                "type": "open_world",
                "direction": "south",
                "entry_coords": (10, 0),   # Where you appear in Oldale from Route 101
                "exit_coords": (10, 19),   # South boundary leading to Route 101
                "description": "South to Route 101",
                "requirements": None,
            },
            "ROUTE_103": {
                "type": "open_world",
                "direction": "north",
                "entry_coords": (10, 22),  # Where you appear in Oldale from Route 103
                "exit_coords": (10, 0),    # North boundary leading to Route 103
                "description": "North to Route 103",
                "requirements": None,
            },
            "ROUTE_102": {
                "type": "open_world",
                "direction": "west",
                "entry_coords": (0, 11),   # Where you appear in Oldale from Route 102 (coming back)
                "exit_coords": (0, 10),    # West boundary leading to Route 102 (warp at y=10)
                "description": "West to Route 102",
                "requirements": None,
            },
            "OLDALE_TOWN_POKEMON_CENTER_1F": {
                "type": "warp_tile",
                "direction": "interact",
                "entry_coords": (7, 9),    # Inside Pokemon Center, near door
                "exit_coords": (6, 16),    # Pokemon Center door in Oldale
                "description": "Enter Pokemon Center",
                "requirements": None,
            },
            "OLDALE_TOWN_MART": {
                "type": "warp_tile",
                "direction": "interact",
                "entry_coords": (3, 7),    # Inside Mart, near door
                "exit_coords": (13, 16),   # Mart door in Oldale
                "description": "Enter Pokemon Mart",
                "requirements": None,
            },
        }
    },
    
    # ========================================
    # OLDALE TOWN - POKEMON CENTER
    # ========================================
    "OLDALE_TOWN_POKEMON_CENTER_1F": {
        "map_id": None,  # Generic Pokemon Center map
        "display_name": "Oldale Town Pokemon Center",
        "description": "Pokemon Center for healing",
        "portals": {
            "OLDALE_TOWN": {
                "type": "warp_tile",
                "direction": "interact",
                "entry_coords": (7, 9),
                "exit_coords": (7, 9),
                "description": "Exit to Oldale Town",
                "requirements": None,
            },
        }
    },
    
    # ========================================
    # OLDALE TOWN - MART
    # ========================================
    "OLDALE_TOWN_MART": {
        "map_id": None,  # Generic Mart map
        "display_name": "Oldale Town Mart",
        "description": "Pokemon Mart for buying items",
        "portals": {
            "OLDALE_TOWN": {
                "type": "warp_tile",
                "direction": "interact",
                "entry_coords": (3, 7),
                "exit_coords": (3, 7),
                "description": "Exit to Oldale Town",
                "requirements": None,
            },
        }
    },
    
    # ========================================
    # ROUTE 103
    # ========================================
    "ROUTE_103": {
        "map_id": "0012",
        "display_name": "Route 103",
        "description": "Route where you battle rival for the first time",
        "portals": {
            "OLDALE_TOWN": {
                "type": "open_world",
                "direction": "south",
                "entry_coords": (10, 0),   # Where you appear in Route 103 from Oldale
                "exit_coords": (10, 22),   # South boundary leading to Oldale
                "description": "South to Oldale Town",
                "requirements": None,
            },
        },
        "poi": {
            "rival_may": {
                "coords": (9, 3),
                "description": "Rival May's position for first battle",
                "interaction_type": "battle",
                "milestone": "FIRST_RIVAL_BATTLE",
            },
        }
    },
    
    # ========================================
    # ROUTE 102
    # ========================================
    "ROUTE_102": {
        "map_id": "0011",
        "display_name": "Route 102",
        "description": "Route with trainers, connects Oldale to Petalburg",
        "portals": {
            "OLDALE_TOWN": {
                "type": "open_world",
                "direction": "east",
                "entry_coords": (50, 11),  # Where you appear in Route 102 from Oldale (warp brings you to ~50, 10)
                "exit_coords": (48, 11),   # East boundary leading to Oldale
                "description": "East to Oldale Town",
                "requirements": None,
            },
            "PETALBURG_CITY": {
                "type": "open_world",
                "direction": "west",
                "entry_coords": (29, 17),  # Where you appear in Route 102 from Petalburg
                "exit_coords": (0, 8),     # West boundary leading to Petalburg
                "description": "West to Petalburg City",
                "requirements": None,
            },
        },
        "trainers": [
            {"coords": (33, 15), "name": "Youngster Calvin", "required": True},
            {"coords": (25, 14), "name": "Bug Catcher Rick", "required": True},
            {"coords": (19, 7), "name": "Youngster Allen", "required": True},
        ],
    },
    
    # ========================================
    # PETALBURG CITY
    # ========================================
    "PETALBURG_CITY": {
        "map_id": "0000",  # Petalburg City map ID
        "display_name": "Petalburg City",
        "description": "City with Norman's gym (5th badge location)",
        "portals": {
            "ROUTE_102": {
                "type": "open_world",
                "direction": "east",
                "entry_coords": (0, 8),    # Where you appear in Petalburg from Route 102
                "exit_coords": (29, 17),   # East boundary leading to Route 102
                "description": "East to Route 102",
                "requirements": None,
            },
            "ROUTE_104_SOUTH": {
                "type": "open_world",
                "direction": "west",
                "entry_coords": (17, 27),  # Where you appear in Petalburg from Route 104 South
                "exit_coords": (0, 12),    # West boundary leading to Route 104 South
                "description": "West to Route 104 (South)",
                "requirements": None,
            },
            "PETALBURG_CITY_GYM": {
                "type": "warp_tile",
                "direction": "interact",
                "entry_coords": (4, 111),  # Inside gym, door mat tile ('D')
                "exit_coords": (15, 8),    # Gym door in Petalburg
                "description": "Enter Petalburg Gym (Norman's gym)",
                "requirements": None,
            },
            "PETALBURG_CITY_POKEMON_CENTER_1F": {
                "type": "warp_tile",
                "direction": "interact",
                "entry_coords": (7, 9),
                "exit_coords": (20, 16),   # PC door tile — step UP onto this to trigger warp (confirmed from warp log)
                "description": "Enter Pokemon Center",
                "requirements": None,
            },
        }
    },
    
    # ========================================
    # PETALBURG CITY - POKEMON CENTER
    # ========================================
    "PETALBURG_CITY_POKEMON_CENTER_1F": {
        "map_id": None,
        "display_name": "Petalburg Pokemon Center",
        "description": "Pokemon Center for healing",
        "portals": {
            "PETALBURG_CITY": {
                "type": "warp_tile",
                "direction": "interact",
                "entry_coords": (7, 9),
                "exit_coords": (7, 9),
                "description": "Exit to Petalburg City",
                "requirements": None,
            },
        }
    },
    
    # ========================================
    # PETALBURG CITY GYM
    # ========================================
    "PETALBURG_CITY_GYM": {
        "map_id": None,
        "display_name": "Petalburg Gym",
        "description": "Norman's gym (will battle later for 5th badge)",
        "portals": {
            "PETALBURG_CITY": {
                "type": "warp_tile",
                "direction": "interact",
                "entry_coords": (4, 111),
                "exit_coords": (4, 111),
                "description": "Exit to Petalburg City",
                "requirements": None,
            },
        },
        "poi": {
            "norman": {
                "coords": (4, 107),  
                "description": "Dad (Norman) first meeting",
                "interaction_type": "dialogue",
                "milestone": "DAD_FIRST_MEETING",
            },
        }
    },
    
    # ========================================
    # ROUTE 104 (SOUTH)
    # ========================================
    "ROUTE_104_SOUTH": {
        "map_id": "0013",  # Route 104 map ID
        "display_name": "Route 104 (South)",
        "description": "Southern section of Route 104, before Petalburg Woods",
        "portals": {
            "PETALBURG_CITY": {
                "type": "open_world",
                "direction": "east",
                "entry_coords": (40, 62),  # Where you appear in Route 104 South from Petalburg
                "exit_coords": (40, 62),   # West boundary tile leading back to Petalburg
                "description": "East to Petalburg City",
                "requirements": None,
            },
            "PETALBURG_WOODS": {
                "type": "open_world",
                "direction": "north",
                "entry_coords": (10, 38),
                "exit_coords": (10, 38),
                "description": "North through Petalburg Woods",
                "requirements": None,
            },
        },
        "trainers": [
            {"coords": (11, 43), "name": "Rich Boy Winston", "required": False, "note": "Avoid - has Full Restore"},
        ],
    },
    
    # ========================================
    # PETALBURG WOODS
    # ========================================
    "PETALBURG_WOODS": {
        "map_id": "180B",  # Hex map ID for Petalburg Woods
        "display_name": "Petalburg Woods",
        "description": "Forest area with Team Aqua encounter",
        "portals": {
            "ROUTE_104_SOUTH": {
                "type": "open_world",
                "direction": "south",
                "entry_coords": (9, 0),
                "exit_coords": (9, 0),
                "description": "South to Route 104 (South)",
                "requirements": None,
            },
            "ROUTE_104_NORTH": {
                "type": "open_world",
                "direction": "north",
                "entry_coords": (14, 5),
                "exit_coords": (14, 5),
                "description": "North to Route 104 (North)",
                "requirements": None,
            },
        },
        "trainers": [
            {"coords": (9, 32), "name": "Bug Catcher Lyle", "required": True},
            {"coords": (7, 14), "name": "Bug Catcher James", "required": True},
        ],
        "poi": {
            "team_aqua_grunt": {
                "coords": (26, 23),
                "description": "Team Aqua grunt - Devon Researcher rescue",
                "interaction_type": "battle",
                "milestone": "TEAM_AQUA_GRUNT_DEFEATED",
            },
        }
    },
    
    # ========================================
    # ROUTE 104 (NORTH)
    # ========================================
    "ROUTE_104_NORTH": {
        "map_id": "0013",  # Same map as Route 104 South (distinguished by Y coordinate)
        "display_name": "Route 104 (North)",
        "description": "Northern section of Route 104, after Petalburg Woods",
        "portals": {
            "PETALBURG_WOODS": {
                "type": "open_world",
                "direction": "south",
                "entry_coords": (14, 29),  # Estimated entry point from Petalburg Woods
                "exit_coords": (14, 38),   # South boundary back to Petalburg Woods
                "description": "South through Petalburg Woods",
                "requirements": None,
            },
            "RUSTBORO_CITY": {
                "type": "open_world",
                "direction": "north",
                "entry_coords": (19, 47),  # Based on Rustboro City portal definition
                "exit_coords": (25, 11),   # North boundary to Rustboro City (actual warp location)
                "description": "North to Rustboro City",
                "requirements": None,
            },
        },
        "trainers": [
            {"coords": (19, 25), "name": "Lass Haley", "required": True},
            {"coords": (24, 24), "name": "Twins Gina & Mia", "required": False, "note": "Avoid - has Shroomish"},
        ],
    },
    
    # ========================================
    # RUSTBORO CITY
    # ========================================
    "RUSTBORO_CITY": {
        "map_id": None,
        "display_name": "Rustboro City",
        "description": "City with Roxanne's gym (1st badge)",
        "portals": {
            "ROUTE_104_NORTH": {
                "type": "open_world",
                "direction": "south",
                "entry_coords": (19, 0),
                "exit_coords": (19, 47),
                "description": "South to Route 104 (North)",
                "requirements": None,
            },
            "RUSTBORO_CITY_POKEMON_CENTER_1F": {
                "type": "warp_tile",
                "direction": "interact",
                "entry_coords": (7, 9),
                "exit_coords": (16, 38),
                "description": "Enter Pokemon Center",
                "requirements": None,
            },
            "RUSTBORO_CITY_GYM": {
                "type": "warp_tile",
                "direction": "interact",
                "entry_coords": (5, 17),   # Inside gym, near door
                "exit_coords": (27, 19),   # Gym door in Rustboro
                "description": "Enter Rustboro Gym (Roxanne)",
                "requirements": None,
            },
        }
    },
    
    # ========================================
    # RUSTBORO CITY - POKEMON CENTER
    # ========================================
    "RUSTBORO_CITY_POKEMON_CENTER_1F": {
        "map_id": None,
        "display_name": "Rustboro Pokemon Center",
        "description": "Pokemon Center for healing",
        "portals": {
            "RUSTBORO_CITY": {
                "type": "warp_tile",
                "direction": "interact",
                "entry_coords": (7, 9),
                "exit_coords": (7, 9),
                "description": "Exit to Rustboro City",
                "requirements": None,
            },
        },
        "poi": {
            "nurse_joy": {
                "coords": (7, 3),
                "description": "Nurse Joy for healing Pokemon",
                "interaction_type": "heal",
            },
        }
    },
    
    # ========================================
    # RUSTBORO CITY GYM
    # ========================================
    "RUSTBORO_CITY_GYM": {
        "map_id": None,
        "display_name": "Rustboro Gym",
        "description": "Roxanne's Rock-type gym (1st badge)",
        "portals": {
            "RUSTBORO_CITY": {
                "type": "warp_tile",
                "direction": "interact",
                "entry_coords": (5, 17),
                "exit_coords": (5, 17),
                "description": "Exit to Rustboro City",
                "requirements": None,
            },
        },
        "trainers": [
            {"coords": (2, 9), "name": "Youngster Josh", "required": True},
        ],
        "poi": {
            "roxanne": {
                "coords": (5, 2),
                "description": "Gym Leader Roxanne",
                "interaction_type": "battle",
                "milestone": "ROXANNE_DEFEATED",
            },
        }
    },
}


# === HELPER FUNCTIONS ===

def get_location_portals(location_name: str) -> Dict[str, Any]:
    """Get all portals from a specific location"""
    location_data = LOCATION_GRAPH.get(location_name.upper(), {})
    return location_data.get('portals', {})


def get_portal_info(from_location: str, to_location: str) -> Optional[Dict[str, Any]]:
    """
    Get portal information for traveling from one location to another.
    
    Args:
        from_location: Starting location name
        to_location: Destination location name
        
    Returns:
        Portal dict with type, coords, direction, etc. or None if no direct connection
    """
    portals = get_location_portals(from_location)
    return portals.get(to_location.upper())


def get_connected_locations(location_name: str) -> List[str]:
    """Get list of locations directly connected to the given location"""
    portals = get_location_portals(location_name)
    return list(portals.keys())


def find_nearest_pokemon_center(current_location: str) -> Optional[str]:
    """
    Find the nearest Pokemon Center reachable from *current_location* via BFS.

    If *current_location* is itself a Pokemon Center, return it directly.
    Otherwise traverse the ``LOCATION_GRAPH`` and return the first node whose
    key contains ``POKEMON_CENTER``.

    >>> find_nearest_pokemon_center("RUSTBORO_CITY")
    'RUSTBORO_CITY_POKEMON_CENTER_1F'
    >>> find_nearest_pokemon_center("OLDALE_TOWN")
    'OLDALE_TOWN_POKEMON_CENTER_1F'
    """
    from collections import deque

    start = current_location.upper()

    if "POKEMON_CENTER" in start:
        return start

    if start not in LOCATION_GRAPH:
        return None

    queue: deque = deque([start])
    visited = {start}

    while queue:
        loc = queue.popleft()
        for neighbour in get_connected_locations(loc):
            if neighbour in visited:
                continue
            if "POKEMON_CENTER" in neighbour:
                return neighbour
            visited.add(neighbour)
            queue.append(neighbour)

    return None


def find_shortest_path(start_location: str, end_location: str, 
                       requirements: Optional[Dict[str, Any]] = None) -> Optional[List[Tuple[str, str, Dict]]]:
    """
    Find shortest path between two locations using BFS.
    
    Args:
        start_location: Starting location name
        end_location: Destination location name
        requirements: Player's current items/badges for checking portal requirements
        
    Returns:
        List of (from_loc, to_loc, portal_info) tuples representing the path,
        or None if no path exists
    """
    from collections import deque
    
    start = start_location.upper()
    end = end_location.upper()
    
    if start not in LOCATION_GRAPH:
        logger.warning(f"Start location '{start}' not found in location graph")
        return None
    
    if end not in LOCATION_GRAPH:
        logger.warning(f"End location '{end}' not found in location graph")
        return None
    
    if start == end:
        return []  # Already at destination
    
    # BFS to find shortest path
    queue = deque([(start, [])])
    visited = {start}
    
    while queue:
        current_loc, path = queue.popleft()
        
        # Check all portals from current location
        portals = get_location_portals(current_loc)
        
        for next_loc, portal_info in portals.items():
            if next_loc in visited:
                continue
            
            # Check if we meet requirements for this portal
            if requirements and portal_info.get('requirements'):
                # TODO: Implement requirement checking
                # For now, assume all requirements are met
                pass
            
            # Add to path
            new_path = path + [(current_loc, next_loc, portal_info)]
            
            # Check if we reached destination
            if next_loc == end:
                return new_path
            
            # Continue searching
            visited.add(next_loc)
            queue.append((next_loc, new_path))
    
    # No path found
    logger.warning(f"No path found from '{start}' to '{end}'")
    return None


def get_location_display_name(location_name: str) -> str:
    """Get the display name for a location"""
    location_data = LOCATION_GRAPH.get(location_name.upper(), {})
    return location_data.get('display_name', location_name)


def get_location_description(location_name: str) -> str:
    """Get the description for a location"""
    location_data = LOCATION_GRAPH.get(location_name.upper(), {})
    return location_data.get('description', '')


def get_poi_at_location(location_name: str) -> Dict[str, Any]:
    """Get points of interest (trainers, NPCs, items) at a location"""
    location_data = LOCATION_GRAPH.get(location_name.upper(), {})
    return location_data.get('poi', {})


def get_trainers_at_location(location_name: str) -> List[Dict[str, Any]]:
    """Get list of trainers at a location"""
    location_data = LOCATION_GRAPH.get(location_name.upper(), {})
    return location_data.get('trainers', [])


def get_entrance_coords(city: str, building: str) -> Optional[Tuple[int, int]]:
    """
    Get the overworld coordinates of a building entrance (the tile you walk onto
    to enter a building from the city).

    Looks up ``city → building`` portal ``exit_coords`` — the tile in the *city*
    map that triggers the warp into *building*.

    >>> get_entrance_coords("RUSTBORO_CITY", "RUSTBORO_CITY_GYM")
    (27, 19)
    """
    portal = get_portal_info(city, building)
    if portal:
        return portal.get('exit_coords')
    return None


def get_interior_exit_coords(building: str) -> Optional[Tuple[int, int]]:
    """
    Get the warp-tile coordinates *inside* a building that lead back to the
    overworld.  Returns the ``entry_coords`` of the first portal whose type
    is ``warp_tile`` and whose destination is NOT an interior sublocation.

    For most buildings this is the single door tile:
    >>> get_interior_exit_coords("PROFESSOR_BIRCHS_LAB")
    (6, 13)
    >>> get_interior_exit_coords("RUSTBORO_CITY_POKEMON_CENTER_1F")
    (7, 9)
    """
    portals = get_location_portals(building)
    for dest, info in portals.items():
        if info.get('type') == 'warp_tile':
            return info.get('entry_coords')
    # Fallback: return entry_coords of first portal
    for dest, info in portals.items():
        return info.get('entry_coords')
    return None


def get_poi_coords(location_name: str, poi_key: str) -> Optional[Tuple[int, int]]:
    """
    Get the coordinates of a named point-of-interest at a location.

    >>> get_poi_coords("ROUTE_103", "rival_may")
    (9, 3)
    >>> get_poi_coords("PETALBURG_CITY_GYM", "norman")
    (4, 107)
    """
    poi = get_poi_at_location(location_name)
    entry = poi.get(poi_key)
    if entry:
        return entry.get('coords')
    return None


# === DEBUGGING / VALIDATION ===

def validate_location_graph() -> List[str]:
    """
    Validate the location graph for common issues.
    
    Returns:
        List of validation warnings/errors
    """
    issues = []
    
    for loc_name, loc_data in LOCATION_GRAPH.items():
        portals = loc_data.get('portals', {})
        
        for dest_name, portal_info in portals.items():
            # Check if destination exists
            if dest_name not in LOCATION_GRAPH:
                issues.append(f"{loc_name} -> {dest_name}: Destination not in graph")
                continue
            
            # Check for reciprocal portal (except one-way ledges)
            if portal_info.get('type') != 'ledge':
                dest_portals = LOCATION_GRAPH[dest_name].get('portals', {})
                if loc_name not in dest_portals:
                    issues.append(f"{loc_name} <-> {dest_name}: Missing reciprocal portal")
            
            # Check required fields
            required_fields = ['type', 'direction', 'entry_coords', 'exit_coords', 'description']
            for field in required_fields:
                if field not in portal_info:
                    issues.append(f"{loc_name} -> {dest_name}: Missing field '{field}'")
    
    return issues


if __name__ == "__main__":
    # Run validation
    print("Validating location graph...")
    issues = validate_location_graph()
    
    if issues:
        print(f"\n⚠️ Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Location graph validation passed!")
    
    # Test pathfinding
    print("\n" + "="*60)
    print("Testing pathfinding...")
    test_paths = [
        ("LITTLEROOT_TOWN", "ROUTE_103"),
        ("LITTLEROOT_TOWN", "RUSTBORO_CITY"),
        ("OLDALE_TOWN", "PETALBURG_CITY"),
    ]
    
    for start, end in test_paths:
        path = find_shortest_path(start, end)
        if path:
            print(f"\n{start} -> {end}:")
            for i, (from_loc, to_loc, portal) in enumerate(path, 1):
                print(f"  {i}. {from_loc} -> {to_loc} ({portal['type']}, {portal['direction']})")
        else:
            print(f"\n{start} -> {end}: No path found")
