"""
Objective Management Module

Lightweight objective management system extracted from SimpleAgent for use in the four-module architecture.
This module provides milestone-driven strategic planning without the complex state management overhead.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from agent.navigation_planner import NavigationPlanner
from agent.location_graph import (
    get_entrance_coords,
    get_interior_exit_coords,
    get_poi_coords,
    get_portal_info,
)

logger = logging.getLogger(__name__)

# Keywords indicating the player cannot progress (ported from GoalManager)
BLOCKING_KEYWORDS = ["wait", "stop", "don't go", "dangerous"]

# ============================================================================
# SEQUENTIAL MILESTONE PROGRESSION SYSTEM
# ============================================================================
# Milestones are ordered by game progression. The agent always targets the
# NEXT uncompleted milestone after the highest completed one.
# This eliminates brittle if/elif chains and backward-checking logic.
# ============================================================================

MILESTONE_PROGRESSION = [
    # [0-2] SPLIT 01: Game start
    {"milestone": "GAME_RUNNING", "target_location": None, "description": "Game initialized"},
    {"milestone": "PLAYER_NAME_SET", "target_location": None, "description": "Player named"},
    {"milestone": "INTRO_CUTSCENE_COMPLETE", "target_location": None, "description": "Intro complete"},
    
    # [3-7] SPLIT 02: Tutorial sequence
    {"milestone": "LITTLEROOT_TOWN", "target_location": "LITTLEROOT_TOWN", "description": "Arrive in Littleroot"},
    {"milestone": "PLAYER_HOUSE_ENTERED", "target_location": None, "description": "Enter player house"},
    {"milestone": "PLAYER_BEDROOM", "target_location": None, "description": "Go upstairs to bedroom"},
    {"milestone": "RIVAL_HOUSE", "target_location": None, "description": "Visit rival's house"},
    {"milestone": "RIVAL_BEDROOM", "target_location": None, "description": "Go to rival's bedroom"},
    
    # [8-10] SPLIT 03: Getting starter
    {"milestone": "ROUTE_101", "target_location": "ROUTE_101", "description": "Find Prof. Birch on Route 101"},
    {"milestone": "STARTER_CHOSEN", "target_location": None, "description": "Choose starter Pokemon"},
    {"milestone": "BIRCH_LAB_VISITED", "target_location": "PROFESSOR_BIRCHS_LAB", "description": "Visit Birch's Lab"},
    
    # [11-14] SPLIT 03: Rival battle sequence & Return to lab for Pokedex
    {"milestone": "OLDALE_TOWN", "target_location": "OLDALE_TOWN", "description": "Travel to Oldale Town"},
    {"milestone": "ROUTE_103", "target_location": "ROUTE_103", "target_coords_fn": lambda: get_poi_coords("ROUTE_103", "rival_may"), "description": "Go to Route 103"},
    {"milestone": "RIVAL_BATTLE_1", "target_location": "ROUTE_103", "target_coords_fn": lambda: get_poi_coords("ROUTE_103", "rival_may"), "description": "Battle rival May", "special": "rival_battle"},
    {"milestone": "RECEIVED_POKEDEX", "target_location": "PROFESSOR_BIRCHS_LAB", "description": "Return to Birch for Pokedex"},
    
    # [15-18] SPLIT 04: Petalburg City sequence
    {"milestone": "ROUTE_102", "target_location": "ROUTE_102", "description": "Travel through Route 102"},
    {"milestone": "PETALBURG_CITY", "target_location": "PETALBURG_CITY", "description": "Arrive at Petalburg City"},
    {"milestone": "DAD_FIRST_MEETING", "target_location": "PETALBURG_CITY_GYM", "target_coords_fn": lambda: get_entrance_coords("PETALBURG_CITY", "PETALBURG_CITY_GYM"), "description": "Enter gym to meet Dad", "special": "gym_dialogue"},
    {"milestone": "GYM_EXPLANATION", "target_location": None, "description": "Watch Wally tutorial", "special": "gym_dialogue"},
    
    # [19-22] SPLIT 05: Road to Rustboro
    {"milestone": "ROUTE_104_SOUTH", "target_location": "ROUTE_104_SOUTH", "description": "Travel to Route 104 South"},
    {"milestone": "PETALBURG_WOODS", "target_location": "PETALBURG_WOODS", "description": "Navigate Petalburg Woods"},
    {"milestone": "TEAM_AQUA_GRUNT_DEFEATED", "target_location": None, "description": "Defeat Team Aqua grunt"},
    {"milestone": "ROUTE_104_NORTH", "target_location": "ROUTE_104_NORTH", "description": "Exit woods to Route 104 North"},
    
    # [23-26] SPLIT 06: Rustboro Gym
    {"milestone": "RUSTBORO_CITY", "target_location": "RUSTBORO_CITY", "description": "Arrive at Rustboro City"},
    {"milestone": "RUSTBORO_GYM_ENTERED", "target_location": "RUSTBORO_CITY_GYM", "target_coords_fn": lambda: get_entrance_coords("RUSTBORO_CITY", "RUSTBORO_CITY_GYM"), "description": "Enter Rustboro Gym"},
    {"milestone": "ROXANNE_DEFEATED", "target_location": None, "description": "Defeat Roxanne"},
    {"milestone": "FIRST_GYM_COMPLETE", "target_location": None, "description": "First gym badge obtained"},
]

def get_highest_milestone_index(milestones: Dict[str, Any]) -> int:
    """
    Find the highest completed milestone index.
    Returns -1 if no milestones completed.
    """
    highest_index = -1
    
    for i, entry in enumerate(MILESTONE_PROGRESSION):
        milestone_id = entry["milestone"]
        milestone_data = milestones.get(milestone_id, {})
        is_complete = milestone_data.get("completed", False) if isinstance(milestone_data, dict) else False
        
        if is_complete:
            highest_index = i
    
    return highest_index

def get_next_milestone_target(milestones: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Get the next uncompleted milestone to target.
    Returns None if all milestones complete.
    """
    highest_index = get_highest_milestone_index(milestones)
    next_index = highest_index + 1
    
    if next_index >= len(MILESTONE_PROGRESSION):
        return None  # All milestones complete
    
    return {
        "index": next_index,
        **MILESTONE_PROGRESSION[next_index]
    }

@dataclass
class Objective:
    """Single objective/goal for the agent"""
    id: str
    description: str
    objective_type: str  # "location", "battle", "item", "dialogue", "custom", "system", "pokemon"
    target_value: Optional[Any] = None  # Specific target (coords, trainer name, item name, etc.)
    completed: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    progress_notes: str = ""
    storyline: bool = False  # True for main storyline objectives (auto-verified), False for agent sub-objectives
    milestone_id: Optional[str] = None  # Emulator milestone ID for storyline objectives


@dataclass
class Directive:
    """
    Structured directive returned by ObjectiveManager / NavigationPlanner.
    
    Supports dict-like access (``directive.get(key)``, ``key in directive``)
    so that action.py's existing consumption code works unchanged during
    the transition period.  New code should prefer attribute access.
    
    Use ``Directive.from_dict(d)`` to convert legacy ``dict`` directives.
    """
    # ── Core fields ──
    action: Optional[str] = None            # NAVIGATE, INTERACT, DIALOGUE, CROSS_BOUNDARY, etc.
    description: Optional[str] = None       # Human-readable description of the directive
    
    # ── Coordinate-based navigation ──
    goal_coords: Optional[tuple] = None     # (x, y, 'LOCATION') target
    goal_direction: Optional[str] = None    # 'north', 'south', 'east', 'west'
    should_interact: Optional[bool] = None  # Press A at destination
    npc_coords: Optional[tuple] = None      # (npc_x, npc_y) for facing direction
    avoid_grass: Optional[bool] = None      # A* avoids tall grass
    press_b_first: Optional[bool] = None    # Press B before navigating (warp settling)
    
    # ── Journey / milestone info ──
    milestone: Optional[str] = None
    journey_reason: Optional[str] = None
    journey_complete: Optional[bool] = None
    journey_progress: Optional[str] = None
    
    # ── Portal / boundary crossing ──
    direction: Optional[str] = None         # Direction for NAVIGATE_DIRECTION / CROSS_BOUNDARY
    target: Optional[Any] = None            # (x, y) tuple or location string
    target_location: Optional[str] = None   # Target location name
    location: Optional[str] = None          # Current location context
    portal_coords: Optional[tuple] = None   # (x, y) of portal
    portal_type: Optional[str] = None
    proximity_radius: Optional[int] = None  # Tiles within which portal activates
    from_location: Optional[str] = None
    to_location: Optional[str] = None
    
    # ── Stage progress ──
    stage_index: Optional[int] = None
    total_stages: Optional[int] = None
    
    # ── Transition waiting ──
    wait_for_transition: Optional[bool] = None
    expected_location: Optional[str] = None
    
    # ── Error / diagnostic ──
    error: Optional[bool] = None
    at_destination: Optional[bool] = None

    # ------------------------------------------------------------------
    # Dict-compatible accessors (backward compat with action.py)
    # ------------------------------------------------------------------

    def get(self, key: str, default=None):
        """``directive.get('goal_coords')`` → ``directive.goal_coords``"""
        val = getattr(self, key, _SENTINEL)
        if val is _SENTINEL:
            return default
        return val if val is not None else default

    def __contains__(self, key: str) -> bool:
        """``'goal_coords' in directive`` → True when the field is non-None."""
        val = getattr(self, key, _SENTINEL)
        return val is not _SENTINEL and val is not None

    def __getitem__(self, key: str):
        """``directive['goal_coords']`` → ``directive.goal_coords``"""
        val = getattr(self, key, _SENTINEL)
        if val is _SENTINEL:
            raise KeyError(key)
        return val

    def keys(self):
        """Return field names with non-None values (dict compatibility)."""
        from dataclasses import fields as dc_fields
        return [f.name for f in dc_fields(self) if getattr(self, f.name) is not None]

    def to_dict(self) -> Dict[str, Any]:
        """Export as plain dict (non-None fields only)."""
        from dataclasses import fields as dc_fields
        return {f.name: getattr(self, f.name) for f in dc_fields(self) if getattr(self, f.name) is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "Directive":
        """Build a Directive from a legacy dict, ignoring unknown keys."""
        from dataclasses import fields as dc_fields
        known = {f.name for f in dc_fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# Sentinel for .get() / __contains__ to distinguish "field exists but is None"
_SENTINEL = object()


class ObjectiveManager:
    """
    Lightweight objective management for strategic planning integration.
    
    Extracted from SimpleAgent to provide milestone-driven strategic planning
    without complex state management dependencies.
    """
    
    def __init__(self, strategic_planner=None, npc_registry=None):
        """Initialize with core storyline objectives.

        Args:
            strategic_planner: Optional ``StrategicPlanner`` instance.  When
                provided, enables **RAG-primary navigation** (Phase 4.3b):
                the walkthrough RAG planner determines the next target location,
                falling back to ``MILESTONE_PROGRESSION`` only when the RAG result
                does not resolve to a valid ``LOCATION_GRAPH`` key.
                Shadow-mode comparison logging (Phase 4.3a) also runs.
            npc_registry: Optional ``NpcRegistry`` instance (Phase 4.4d).
                When provided, ``_resolve_npc_coords`` uses semantic role
                lookups instead of hardcoded ``graphics_id`` constants.
        """
        self.objectives: List[Objective] = []
        self._initialize_storyline_objectives()
        
        # Track completed sub-goals to prevent repeating actions
        # This replaces individual flags like rival_battle_completed
        self.completed_goals = {
            # Example: 'ROUTE_103_RIVAL_BATTLE': True
        }
        
        # Track previous state for transition detection
        self._previous_state = {
            'in_battle': False,
            'location': None,
        }
        
        # Track if we've pressed B after entering gym
        self._pressed_b_after_gym_warp = False
        
        # NEW: Initialize NavigationPlanner for comparison testing
        self.navigation_planner = NavigationPlanner()
        self._last_planner_location = None
        self._last_planner_coords = None
        
        # ── Blocker / Recovery (ported from GoalManager) ──
        self.blocking_keywords = BLOCKING_KEYWORDS
        self._recovery_tasks: List[Dict[str, Any]] = []  # Stack of recovery sub-goals
        self._blocker_state: Optional[Dict[str, Any]] = None  # {reason, context} when blocked
        self._brain_prev_in_battle: bool = False
        self._last_logged_dialogue: Optional[str] = None
        self._consecutive_dialogue_steps: int = 0  # Track repeated dialogue presses to break NPC re-trigger loops
        
        # ── Phase 4.3a/b: RAG-based strategic planning ──
        self.strategic_planner = strategic_planner
        self.npc_registry = npc_registry  # Phase 4.4d: adaptive NPC discovery
        self._shadow_log_path = os.path.join("llm_logs", "shadow_comparison.jsonl")
        self._shadow_step_count: int = 0
        self._shadow_agree_count: int = 0
        self._shadow_total_count: int = 0
        self._last_rag_target: Optional[str] = None       # Cached RAG target for current step
        self._last_directive_source: str = "milestone"     # "rag" or "milestone"
        self._rag_override_count: int = 0                  # Times RAG drove navigation
        self._milestone_fallback_count: int = 0            # Times milestone fallback was used
        
        # RAG result cache — avoid re-querying LLM when location + milestone unchanged
        self._rag_cache_key: Optional[tuple] = None        # (location, last_milestone)
        self._rag_cache_result: Optional[Dict[str, Any]] = None
        
        logger.info(f"🏗️ [OBJECT LIFECYCLE] ObjectiveManager.__init__() called - created new instance with {len(self.objectives)} storyline objectives")
        print(f"🏗️ [OBJECT LIFECYCLE] ObjectiveManager.__init__() called - NEW INSTANCE CREATED")
        print(f"🗺️ [NAV PLANNER] NavigationPlanner initialized for comparison testing")
        if strategic_planner:
            print(f"🧠 [RAG PRIMARY] StrategicPlanner attached — RAG-primary navigation ACTIVE (Phase 4.3b)")
    
    def _initialize_storyline_objectives(self):
        """Initialize the main storyline objectives for Pokémon Emerald progression"""
        storyline_objectives = [
            {
                "id": "story_game_start",
                "description": "Complete title sequence and begin the game",
                "objective_type": "system",
                "target_value": "Game Running",
                "milestone_id": "GAME_RUNNING"
            },
            {
                "id": "story_littleroot_town",
                "description": "Arrive in Littleroot Town and explore the area",
                "objective_type": "location", 
                "target_value": "Littleroot Town",
                "milestone_id": "LITTLEROOT_TOWN"
            },
            {
                "id": "story_route_101",
                "description": "Travel north to Route 101 to find Professor Birch",
                "objective_type": "location",
                "target_value": "Route 101", 
                "milestone_id": "ROUTE_101"
            },
            {
                "id": "story_starter_chosen",
                "description": "Choose starter Pokémon and receive first party member",
                "objective_type": "pokemon",
                "target_value": "Starter Pokémon",
                "milestone_id": "STARTER_CHOSEN"
            },
            {
                "id": "story_oldale_town",
                "description": "Continue journey to Oldale Town",
                "objective_type": "location",
                "target_value": "Oldale Town",
                "milestone_id": "OLDALE_TOWN"
            },
            {
                "id": "story_route_103",
                "description": "Head to Route 103 for rival battle",
                "objective_type": "location",
                "target_value": "Route 103",
                "milestone_id": "ROUTE_103"
            },
            {
                "id": "story_rival_battle_1",
                "description": "Battle rival trainer for the first time",
                "objective_type": "battle",
                "target_value": "Rival Battle 1",
                "milestone_id": "FIRST_RIVAL_BATTLE"
            },
            {
                "id": "story_return_littleroot",
                "description": "Return to Littleroot Town after rival battle",
                "objective_type": "location",
                "target_value": "Littleroot Town Return",
                "milestone_id": "LITTLEROOT_RETURN"
            },
            {
                "id": "story_route_102",
                "description": "Travel west to Route 102 toward Petalburg City",
                "objective_type": "location",
                "target_value": "Route 102",
                "milestone_id": "ROUTE_102"
            },
            # === SPLIT 04: Petalburg City & Meeting Dad ===
            {
                "id": "story_petalburg_city",
                "description": "Arrive at Petalburg City",
                "objective_type": "location",
                "target_value": "Petalburg City",
                "milestone_id": "PETALBURG_CITY"
            },
            {
                "id": "story_dad_first_meeting",
                "description": "Enter Petalburg Gym and meet Dad (Norman)",
                "objective_type": "dialogue",
                "target_value": "Norman Meeting",
                "milestone_id": "DAD_FIRST_MEETING"
            },
            {
                "id": "story_gym_explanation",
                "description": "Receive gym explanation and watch Wally tutorial",
                "objective_type": "dialogue",
                "target_value": "Gym Tutorial",
                "milestone_id": "GYM_EXPLANATION"
            },
            
            # === SPLIT 05: Route 104 & Petalburg Woods ===
            {
                "id": "story_route_104_south",
                "description": "Travel north from Petalburg to Route 104 (southern section)",
                "objective_type": "location",
                "target_value": "Route 104 South",
                "milestone_id": "ROUTE_104_SOUTH"
            },
            {
                "id": "story_petalburg_woods",
                "description": "Navigate through Petalburg Woods",
                "objective_type": "location",
                "target_value": "Petalburg Woods",
                "milestone_id": "PETALBURG_WOODS"
            },
            {
                "id": "story_team_aqua_grunt",
                "description": "Defeat Team Aqua Grunt in Petalburg Woods",
                "objective_type": "battle",
                "target_value": "Team Aqua Grunt",
                "milestone_id": "TEAM_AQUA_GRUNT_DEFEATED"
            },
            {
                "id": "story_route_104_north",
                "description": "Reach northern section of Route 104",
                "objective_type": "location",
                "target_value": "Route 104 North",
                "milestone_id": "ROUTE_104_NORTH"
            },
            {
                "id": "story_rustboro_city",
                "description": "Arrive at Rustboro City",
                "objective_type": "location",
                "target_value": "Rustboro City",
                "milestone_id": "RUSTBORO_CITY"
            },
            
            # === SPLIT 06: Rustboro Gym & First Badge ===
            {
                "id": "story_rustboro_gym_entered",
                "description": "Enter Rustboro City Gym",
                "objective_type": "location",
                "target_value": "Rustboro Gym",
                "milestone_id": "RUSTBORO_GYM_ENTERED"
            },
            {
                "id": "story_roxanne_defeated",
                "description": "Challenge and defeat Gym Leader Roxanne",
                "objective_type": "battle",
                "target_value": "Gym Leader Roxanne",
                "milestone_id": "ROXANNE_DEFEATED"
            },
            {
                "id": "story_first_gym_complete",
                "description": "Complete first gym challenge",
                "objective_type": "system",
                "target_value": "First Gym Badge",
                "milestone_id": "FIRST_GYM_COMPLETE"
            },
            {
                "id": "story_stone_badge",
                "description": "Receive Stone Badge from Roxanne",
                "objective_type": "item",
                "target_value": "Stone Badge",
                "milestone_id": "STONE_BADGE"
            }
        ]
        
        # Convert to Objective instances
        for obj_data in storyline_objectives:
            objective = Objective(
                id=obj_data["id"],
                description=obj_data["description"],
                objective_type=obj_data["objective_type"],
                target_value=obj_data["target_value"],
                storyline=True,  # All these are storyline objectives
                milestone_id=obj_data["milestone_id"]
            )
            self.objectives.append(objective)
    
    def mark_goal_complete(self, goal_id: str, description: str = ""):
        """
        Mark a sub-goal as complete. This is persistent across calls.
        
        Args:
            goal_id: Unique identifier for the goal (e.g., 'ROUTE_103_RIVAL_BATTLE')
            description: Human-readable description for logging
        """
        if goal_id not in self.completed_goals:
            self.completed_goals[goal_id] = True
            logger.info(f"✅ [GOAL COMPLETE] {goal_id}: {description}")
            print(f"✅ [GOAL COMPLETE] {goal_id}" + (f": {description}" if description else ""))
    
    def is_goal_complete(self, goal_id: str) -> bool:
        """Check if a sub-goal has been completed"""
        return self.completed_goals.get(goal_id, False)
    
    def get_active_objectives(self) -> List[Objective]:
        """Get list of uncompleted objectives"""
        return [obj for obj in self.objectives if not obj.completed]
    
    def get_completed_objectives(self) -> List[Objective]:
        """Get list of completed objectives"""
        return [obj for obj in self.objectives if obj.completed]
    
    def check_storyline_milestones(self, state_data: Dict[str, Any]) -> List[str]:
        """
        Check emulator milestones and auto-complete corresponding storyline objectives.
        Also tracks state transitions for manual goal completion detection.
        """
        completed_ids = []
        
        # Get milestones from the game state (if available)
        milestones = state_data.get("milestones", {})
        
        if not milestones:
            # No milestone data available, skip checking
            logger.debug("No milestone data available in state_data")
            logger.debug(f"State data keys: {list(state_data.keys())}")
            return completed_ids
        
        logger.debug(f"Checking {len(milestones)} milestones: {list(milestones.keys())}")
            
        for obj in self.get_active_objectives():
            # Only check storyline objectives with milestone IDs
            if obj.storyline and obj.milestone_id and not obj.completed:
                # Check if the corresponding emulator milestone is completed
                milestone_data = milestones.get(obj.milestone_id, {})
                milestone_completed = milestone_data.get("completed", False) if isinstance(milestone_data, dict) else False
                
                logger.debug(f"Objective '{obj.id}' checking milestone '{obj.milestone_id}': {milestone_data}")
                
                if milestone_completed:
                    # Auto-complete the storyline objective
                    obj.completed = True
                    obj.completed_at = datetime.now()
                    obj.progress_notes = f"Auto-completed by emulator milestone: {obj.milestone_id}"
                    completed_ids.append(obj.id)
                    logger.info(f"✅ Auto-completed storyline objective via milestone {obj.milestone_id}: {obj.description}")
        
        # LOCATION-BASED MILESTONE DETECTION
        # Some locations don't have emulator milestones, so we detect them by location
        player_data = state_data.get('player', {})
        position = player_data.get('position', {})
        current_x = position.get('x', 0)
        current_y = position.get('y', 0)
        current_location = player_data.get('location', '').upper()
        
        # Detect Route 104 North arrival (no emulator milestone for this)
        # Use same Y-coordinate logic as location mapping: Y < 30 = North section
        if 'ROUTE 104' in current_location and current_y < 30:
            route_104_north_milestone = milestones.get('ROUTE_104_NORTH', {})
            if not route_104_north_milestone.get('completed', False):
                logger.info(f"✅ [LOCATION DETECTION] Route 104 North detected (Y={current_y} < 30)")
                print(f"✅ [LOCATION DETECTION] Route 104 North detected at ({current_x}, {current_y})")
                # Mark milestone as complete in the state data
                milestones['ROUTE_104_NORTH'] = {
                    'completed': True,
                    'timestamp': datetime.now().timestamp(),
                    'detected_by': 'location_check'
                }
        
        # CRITICAL FIX: Track state transitions for manual goal detection
        # This must happen here because check_storyline_milestones() is called every step
        # via planning → get_current_strategic_objective. get_next_action_directive() is
        # only called when battle_bot releases control, so it misses the transition.
        game_data = state_data.get('game', {})
        in_battle = game_data.get('in_battle', False)
        was_in_battle = self._previous_state.get('in_battle', False)
        
        # DEBUG: Log every state check
        logger.debug(f"[STATE TRACKING] in_battle={in_battle}, was_in_battle={was_in_battle}")
        
        # Detect rival battle completion: was in battle at rival position → now not in battle
        if was_in_battle and not in_battle:
            player_data = state_data.get('player', {})
            position = player_data.get('position', {})
            current_x = position.get('x', 0)
            current_y = position.get('y', 0)
            current_location = player_data.get('location', '').upper()
            _rival_coords = get_poi_coords("ROUTE_103", "rival_may") or (9, 3)
            at_rival_position = (current_x == _rival_coords[0] and current_y == _rival_coords[1] and 'ROUTE 103' in current_location)
            
            logger.info(f"🔍 [TRANSITION DETECTED] Battle ended! at_rival_position={at_rival_position}, x={current_x}, y={current_y}, loc={current_location}")
            print(f"🔍 [TRANSITION DETECTED] Battle ended! at_rival_position={at_rival_position}")
            
            if at_rival_position and not self.is_goal_complete('ROUTE_103_RIVAL_BATTLE'):
                self.mark_goal_complete('ROUTE_103_RIVAL_BATTLE', 'Defeated rival May on Route 103')
                logger.info(f"✅ [BATTLE COMPLETION] Detected rival battle completion via state transition")
                print(f"✅ [GOAL COMPLETE] ROUTE_103_RIVAL_BATTLE")
        
        # Detect Dad dialogue completion: Track 'A' button press when adjacent to Dad in Petalburg Gym
        player_data = state_data.get('player', {})
        position = player_data.get('position', {})
        current_x = position.get('x', 0)
        current_y = position.get('y', 0)
        current_location = player_data.get('location', '').upper()
        
        # Check if in Petalburg Gym
        in_petalburg_gym = 'PETALBURG CITY GYM' in current_location or 'PETALBURG_CITY_GYM' in current_location
        
        # Check if adjacent to Dad's position (looked up from location graph)
        _norman_coords = get_poi_coords("PETALBURG_CITY_GYM", "norman") or (4, 107)
        adjacent_to_dad = (
            in_petalburg_gym and
            abs(current_x - _norman_coords[0]) <= 1 and 
            abs(current_y - _norman_coords[1]) <= 1 and
            not (current_x == _norman_coords[0] and current_y == _norman_coords[1])  # Not on same tile
        )
        
        # Check if 'A' was pressed in recent actions
        recent_actions = state_data.get('recent_actions', [])
        pressed_a = 'A' in recent_actions or 'a' in recent_actions
        
        # Mark complete if we pressed A while adjacent to Dad
        if adjacent_to_dad and pressed_a and not self.is_goal_complete('PETALBURG_GYM_DAD_DIALOGUE'):
            self.mark_goal_complete('PETALBURG_GYM_DAD_DIALOGUE', 'Initiated dialogue with Norman at Petalburg Gym')
            logger.info(f"✅ [DAD DIALOGUE] Detected 'A' press at position ({current_x}, {current_y}) adjacent to Dad {_norman_coords}")
            print(f"✅ [GOAL COMPLETE] PETALBURG_GYM_DAD_DIALOGUE - Pressed A at ({current_x}, {current_y})")
        
        # Update previous state for next iteration
        old_in_battle = self._previous_state.get('in_battle', False)
        self._previous_state['in_battle'] = in_battle
        self._previous_state['location'] = state_data.get('player', {}).get('location', '').upper()
        
        if old_in_battle != in_battle:
            logger.info(f"🔄 [STATE CHANGE] Battle state changed: {old_in_battle} -> {in_battle}")
            print(f"🔄 [STATE CHANGE] Battle state: {old_in_battle} -> {in_battle}")
        
        logger.debug(f"🔍 [STATE UPDATE] Updated _previous_state: in_battle={in_battle}")
        
        return completed_ids
    
    def get_current_strategic_objective(self, state_data: Dict[str, Any]) -> Optional[Objective]:
        """Get the current strategic objective based on milestones and game state"""
        # First, update objectives based on milestones
        self.check_storyline_milestones(state_data)
        
        # Get the first uncompleted objective (they are in story order)
        active_objectives = self.get_active_objectives()
        if active_objectives:
            return active_objectives[0]
        
        # No objectives remaining
        return None
    
    def get_strategic_plan_description(self, state_data: Dict[str, Any]) -> Optional[str]:
        """Generate a strategic plan description for the current state"""
        # CRITICAL FIX: Use the current directive's description if available
        # This ensures the VLM gets the RIGHT goal (e.g., "Walk south to Oldale Town")
        # instead of the outdated milestone goal (e.g., "Battle rival trainer")
        current_directive = self.get_next_action_directive(state_data)
        if current_directive and current_directive.get('description'):
            directive_desc = current_directive['description']
            directive_action = current_directive.get('action', 'UNKNOWN')
            return f"CURRENT GOAL: {directive_desc} (Action: {directive_action})"
        
        # Fallback to milestone-based objective
        current_objective = self.get_current_strategic_objective(state_data)
        
        if current_objective:
            # Add tactical context based on objective type
            base_description = current_objective.description
            
            # Add context based on objective type
            if current_objective.objective_type == "location":
                return f"STRATEGIC GOAL: {base_description}. Navigate carefully and interact with NPCs for guidance."
            elif current_objective.objective_type == "battle":
                return f"STRATEGIC GOAL: {base_description}. Prepare for battle - heal Pokemon if needed first."
            elif current_objective.objective_type == "dialogue":
                return f"STRATEGIC GOAL: {base_description}. Look for the right NPC to interact with."
            elif current_objective.objective_type == "pokemon":
                return f"STRATEGIC GOAL: {base_description}. Follow story progression to obtain Pokemon."
            elif current_objective.objective_type == "system":
                return f"STRATEGIC GOAL: {base_description}. Complete basic game setup."
            else:
                return f"STRATEGIC GOAL: {base_description}"
        
        return None
    
    # =====================================================================
    # DYNAMIC NPC COORDINATE RESOLUTION (Phase 4.4b)
    # =====================================================================
    # Replaces hardcoded NPC coordinates with runtime memory reads.
    # Uses gObjectEvents data from read_active_npcs(), passed through
    # state_data['active_npcs'] from the memory reader pipeline.
    # =====================================================================

    def _resolve_npc_coords(
        self,
        state_data: Dict[str, Any],
        *,
        npc_role: Optional[str] = None,
        graphics_id: Optional[int] = None,
        local_id: Optional[int] = None,
        fallback: Optional[tuple] = None,
    ) -> Optional[tuple]:
        """
        Find an NPC's current (x, y) from the runtime gObjectEvents array.

        Resolution order:
        1. **Registry lookup** — if ``npc_role`` is given and ``self.npc_registry``
           exists, look up the learned ``graphics_id`` / ``local_id`` for that role.
        2. **Explicit criteria** — use caller-supplied ``graphics_id`` / ``local_id``
           (kept as optional safety net but no longer required).
        3. **Cold-start inference** — if neither registry nor explicit criteria
           matched, pick the nearest visible non-player NPC.

        Skips the player object and invisible / off-screen NPCs.

        Returns:
            (x, y) tuple or ``fallback`` if not found.
        """
        # ── Phase 4.4d: Registry-first resolution ──────────────────
        if npc_role and self.npc_registry:
            location = state_data.get('player', {}).get('location', '')
            entry = self.npc_registry.lookup_by_role(npc_role, location=location)
            if entry:
                # Registry hit — use learned identifiers
                reg_gfx = entry.get('graphics_id')
                reg_lid = entry.get('local_id')
                if reg_gfx is not None:
                    graphics_id = reg_gfx
                if reg_lid is not None:
                    local_id = reg_lid
                logger.info(f"[NPC RESOLVE] Registry hit for role={npc_role}: "
                            f"gfx={graphics_id}, local={local_id}")
            else:
                logger.debug(f"[NPC RESOLVE] Registry miss for role={npc_role}, "
                             f"trying explicit criteria or cold-start")

        active_npcs = state_data.get('active_npcs', [])
        if not active_npcs:
            if fallback:
                logger.debug(f"[NPC RESOLVE] No active_npcs in state — using fallback {fallback}")
            return fallback

        # ── Criteria-based scan (when graphics_id or local_id available) ──
        has_criteria = graphics_id is not None or local_id is not None
        if has_criteria:
            for npc in active_npcs:
                if npc.get('is_player'):
                    continue
                if npc.get('invisible') or npc.get('off_screen'):
                    continue
                if graphics_id is not None and npc.get('graphics_id') != graphics_id:
                    continue
                if local_id is not None and npc.get('local_id') != local_id:
                    continue

                coords = (npc['current_x'], npc['current_y'])
                logger.info(f"[NPC RESOLVE] Found NPC gfx={npc.get('graphics_id')} "
                            f"local={npc.get('local_id')} at {coords}")
                return coords

        # ── Cold-start: no criteria or criteria didn't match ──────
        from agent.brain.npc_registry import NpcRegistry
        player = state_data.get('player', {}).get('position', {})
        px = player.get('x', 0)
        py = player.get('y', 0)
        nearest = NpcRegistry.infer_nearest_npc(active_npcs, px, py)
        if nearest:
            coords = (nearest['current_x'], nearest['current_y'])
            logger.info(f"[NPC RESOLVE] Cold-start: nearest NPC at {coords} "
                        f"(role={npc_role})")
            return coords

        # Nothing found at all
        if fallback:
            logger.info(f"[NPC RESOLVE] NPC not found (gfx={graphics_id}, local={local_id}) — "
                        f"using fallback {fallback}")
        else:
            logger.info(f"[NPC RESOLVE] NPC not found (gfx={graphics_id}, local={local_id}) — no fallback")
        return fallback

    def get_next_action_directive(self, state_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get specific action directive based on current milestone state.
        
        When a ``StrategicPlanner`` is attached (Phase 4.3b), the RAG planner
        determines navigation targets.  Falls back to milestones when the RAG
        result is invalid.  Shadow-mode comparison logging (Phase 4.3a) still
        runs for observability.
        """
        directive = self._get_next_action_directive_inner(state_data)

        # Phase 4.3a: shadow comparison logging (never changes directive)
        if self.strategic_planner:
            self._run_shadow_comparison(state_data, directive)

        return directive

    def _get_next_action_directive_inner(self, state_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get specific action directive based on current milestone state.
        
        NOW USES NAVIGATIONPLANNER for multi-hop journeys while preserving
        all milestone tracking logic and special cases.
        
        Returns:
            {
                'action': 'NAVIGATE',  # or 'INTERACT', 'DIALOGUE', etc.
                'target': (x, y),      # Coordinate target (from planner)
                'description': 'Navigate to north exit in LITTLEROOT_TOWN',
                'milestone': 'OLDALE_TOWN'  # Expected milestone after completion
            }
        """
        logger.debug("[OBJECTIVE_MANAGER] get_next_action_directive() called")
        
        # =====================================================================
        # PRIORITY 0: EXECUTE RECOVERY TASKS FROM SLOW BRAIN (RAG + LLM)
        # =====================================================================
        # If the RecoveryPlanner generated a recovery task (e.g., after a
        # navigation failure or dialogue blocker), execute it BEFORE resuming
        # normal milestone progression.  This is the critical link between
        # the Slow Brain's output and the agent's "hands".
        # =====================================================================
        if self._recovery_tasks:
            task = self._recovery_tasks[0]
            task_text = task.get('task', '').lower()
            task_reason = task.get('reason', '')
            logger.info(f"🧠 [RECOVERY] Executing recovery task: '{task_text}' (reason: {task_reason})")
            print(f"🧠 [RECOVERY] Brain directive: '{task_text}' (reason: {task_reason})")

            # Heuristic translation of free-text recovery task → structured Directive
            if any(kw in task_text for kw in ['talk', 'speak', 'interact', 'npc', 'old man']):
                return {
                    'action': 'DIALOGUE',
                    'target': None,
                    'description': f'RECOVERY: {task["task"]}',
                    'milestone': None,
                    'recovery': True,
                }
            elif any(kw in task_text for kw in ['battle', 'fight', 'defeat']):
                # Battles are handled by the BattleBot priority (higher than us)
                # so auto-complete this recovery task and let the next step proceed.
                logger.info("🧠 [RECOVERY] Battle recovery — deferring to BattleBot")
                self.complete_recovery_task()
                self.clear_blocker()
                # Fall through to normal milestone logic
            elif any(kw in task_text for kw in ['explore', 'look', 'search', 'find']):
                return {
                    'goal_direction': 'south',
                    'description': f'RECOVERY: {task["task"]}',
                    'journey_reason': f'Recovery: {task_reason}',
                    'recovery': True,
                }
            elif any(kw in task_text for kw in ['heal', 'pokemon center', 'restore']):
                return {
                    'goal_direction': 'south',
                    'description': f'RECOVERY: {task["task"]} — head toward Pokemon Center',
                    'journey_reason': f'Recovery: {task_reason}',
                    'recovery': True,
                }
            else:
                # Default: treat unknown recovery tasks as "press A to progress"
                return {
                    'action': 'DIALOGUE',
                    'target': None,
                    'description': f'RECOVERY: {task["task"]}',
                    'milestone': None,
                    'recovery': True,
                }

        # First update objectives based on milestones
        self.check_storyline_milestones(state_data)
        
        # Get current position
        player_data = state_data.get('player', {})
        position = player_data.get('position', {})
        current_x = position.get('x', 0)
        current_y = position.get('y', 0)
        current_location = player_data.get('location', '').upper()
        
        # Convert location name to graph format for special case checks
        # CRITICAL: Check longer/more specific names FIRST to avoid substring matches
        location_mapping = {
            'PETALBURG CITY GYM': 'PETALBURG_CITY_GYM',
            'PETALBURG GYM': 'PETALBURG_CITY_GYM',
            'RUSTBORO CITY POKEMON CENTER': 'RUSTBORO_CITY_POKEMON_CENTER_1F',
            'RUSTBORO CITY GYM': 'RUSTBORO_CITY_GYM',
            'RUSTBORO GYM': 'RUSTBORO_CITY_GYM',
            'BIRCHS LAB': 'PROFESSOR_BIRCHS_LAB',
            'BIRCH LAB': 'PROFESSOR_BIRCHS_LAB',
            'LITTLEROOT TOWN': 'LITTLEROOT_TOWN',
            'OLDALE TOWN': 'OLDALE_TOWN',
            'RUSTBORO CITY': 'RUSTBORO_CITY',
            'PETALBURG CITY': 'PETALBURG_CITY',
            'ROUTE 101': 'ROUTE_101',
            'ROUTE 103': 'ROUTE_103',
            'ROUTE 102': 'ROUTE_102',
            'PETALBURG WOODS': 'PETALBURG_WOODS',
            'MAP_18_0B': 'PETALBURG_WOODS',
        }
        
        graph_location = None
        if 'ROUTE 104' in current_location:
            graph_location = 'ROUTE_104_SOUTH' if current_y >= 30 else 'ROUTE_104_NORTH'
        else:
            for loc_key, loc_value in location_mapping.items():
                if loc_key in current_location:
                    graph_location = loc_value
                    break
        
        # Get milestone states
        milestones = state_data.get('milestones', {})
        
        # Helper to check if milestone is complete
        def is_milestone_complete(milestone_id: str) -> bool:
            milestone_data = milestones.get(milestone_id, {})
            return milestone_data.get('completed', False) if isinstance(milestone_data, dict) else False
        
        # Helper to check if dialogue is active (prevents navigation during dialogue)
        def is_dialogue_active() -> bool:
            """Check if dialogue is currently active"""
            screen_context = state_data.get('screen_context', '')
            text_box_visible = state_data.get('visual_dialogue_active', False)
            
            # Check both screen_context and text_box visibility
            is_active = (screen_context == 'dialogue' or text_box_visible)
            
            if is_active:
                logger.info(f"🔍 [DIALOGUE CHECK] Dialogue active - screen_context={screen_context}, text_box={text_box_visible}")
                print(f"💬 [DIALOGUE] Active - waiting for dialogue to finish")
            
            return is_active
        
        # === PRIORITY: HANDLE ACTIVE DIALOGUE ===
        # CRITICAL: Always check for dialogue FIRST before any navigation
        # If dialogue is active, we must complete it before doing anything else
        if is_dialogue_active():
            self._consecutive_dialogue_steps += 1
            
            # After 6+ consecutive dialogue presses with no progress, force movement
            # to break free from NPC re-trigger loops (e.g., talking to defeated trainer)
            if self._consecutive_dialogue_steps > 6:
                logger.warning(f"🚨 [DIALOGUE LOOP] {self._consecutive_dialogue_steps} consecutive dialogue steps — forcing movement to break free")
                print(f"🚨 [DIALOGUE LOOP] Stuck in dialogue loop — moving away")
                self._consecutive_dialogue_steps = 0
                return {
                    'goal_direction': 'south',
                    'description': 'Break free from dialogue loop — move away from NPC',
                    'journey_reason': 'Dialogue loop escape'
                }
            
            # After 3+ consecutive dialogue presses, use B instead of A.
            # B advances dialogue text but won't re-initiate NPC conversation.
            if self._consecutive_dialogue_steps > 3:
                logger.info(f"⚠️ [DIALOGUE] {self._consecutive_dialogue_steps} consecutive presses — switching to B (won't re-trigger NPC)")
                print(f"⚠️ [DIALOGUE] Pressing B to avoid NPC re-trigger ({self._consecutive_dialogue_steps} consecutive)")
                return {
                    'action': 'DIALOGUE_B',
                    'target': None,
                    'description': 'Press B to advance dialogue (avoid NPC re-trigger)',
                    'milestone': None
                }
            
            return {
                'action': 'DIALOGUE',
                'target': None,
                'description': 'Press A to advance dialogue',
                'milestone': None
            }
        
        # =====================================================================
        # CRITICAL FIX: Exit unwanted buildings first
        # =====================================================================
        # If we're in a building that's NOT our target (e.g., entered a house door
        # while trying to navigate to the gym), exit it first before continuing.
        # This prevents A* from trying to navigate to outdoor goals while stuck indoors.
        # =====================================================================
        
        # List of building keywords that indicate we're indoors but shouldn't be
        unwanted_buildings = ['HOUSE', 'MART', 'SHOP']
        in_unwanted_building = any(keyword in current_location for keyword in unwanted_buildings)
        
        # Reset consecutive dialogue counter since we're past the dialogue check
        self._consecutive_dialogue_steps = 0
        
        if in_unwanted_building:
            logger.info(f"🏠 [EXIT BUILDING] Detected unwanted building: '{current_location}'")
            print(f"🏠 [EXIT BUILDING] Inside '{current_location}' - exiting before continuing to target")
            
            # Use directional movement to exit (typically DOWN for houses/marts)
            return {
                'goal_direction': 'south',
                'description': f'Exit {current_location} before continuing to target',
                'journey_reason': 'Leave unwanted building'
            }
        
        # =====================================================================
        # NEW: USE NAVIGATION PLANNER FOR MULTI-HOP JOURNEYS
        # =====================================================================
        # The planner handles complex navigation automatically while we still
        # control milestone-based objectives and special interactions
        # =====================================================================
        
        # === ROUTE 103: RIVAL BATTLE SEQUENCE (SPECIAL CASE) ===
        # The ROUTE_103 milestone completes when entering Route 103, not after battle
        # We use FIRST_RIVAL_BATTLE milestone to track actual battle completion
        
        # Get current battle state
        _rival_pos = get_poi_coords("ROUTE_103", "rival_may") or (9, 3)
        at_rival_position = (current_x == _rival_pos[0] and current_y == _rival_pos[1] and 'ROUTE 103' in current_location)
        game_data = state_data.get('game', {})
        in_battle = game_data.get('in_battle', False)
        was_in_battle = self._previous_state.get('in_battle', False)  # For logging only
        
        # Check if battle is complete (either via our detection or milestone)
        rival_battle_complete = self.is_goal_complete('ROUTE_103_RIVAL_BATTLE') or \
                               is_milestone_complete('FIRST_RIVAL_BATTLE')
        
        # logger.info(f"🔍 [RIVAL BATTLE] at (9,3)={at_rival_position}, in_battle={in_battle}, was_in_battle={was_in_battle}, complete={rival_battle_complete}")
        # print(f"🔍 [RIVAL BATTLE] Check: at (9,3)={at_rival_position}, in_battle={in_battle}, was_in_battle={was_in_battle}, complete={rival_battle_complete}")
        
        # === SPECIAL CASE: INSIDE BIRCH LAB ===
        # Wait for dialogue to auto-trigger (bypass planner)
        if 'BIRCHS LAB' in current_location or 'BIRCH LAB' in current_location:
            if rival_battle_complete and not is_milestone_complete('RECEIVED_POKEDEX'):
                return {
                    'action': 'WAIT_FOR_DIALOGUE',
                    'target': None,
                    'description': 'Wait for Birch to give Pokedex (auto-dialogue)',
                    'milestone': 'RECEIVED_POKEDEX'
                }
            elif is_milestone_complete('RECEIVED_POKEDEX') and not is_milestone_complete('ROUTE_102'):
                # Exit lab via warp tile (derived from location graph)
                _lab_exit = get_interior_exit_coords("PROFESSOR_BIRCHS_LAB") or (6, 13)
                return {
                    'action': 'NAVIGATE',
                    'target': (_lab_exit[0], _lab_exit[1], current_location),
                    'description': 'Exit Birch Lab',
                    'milestone': None
                }
        
        # === PETALBURG CITY → Talk to Dad in gym (HP-BASED SPLIT DETECTION) ===
        # =====================================================================
        # PETALBURG CITY: HP-BASED DAD DIALOGUE DETECTION
        # =====================================================================
        # Simple HP-based logic (ignore milestones):
        # - HP < 100% in Petalburg City/Gym → Go to Dad
        # - HP = 100% in Petalburg City/Gym → Head west to Route 104 South
        # =====================================================================
        
        in_petalburg_city = 'PETALBURG CITY' in current_location or 'PETALBURG_CITY' in current_location.replace(' ', '_')
        in_gym = 'PETALBURG CITY GYM' in current_location or 'PETALBURG_CITY_GYM' in current_location
        
        if in_petalburg_city or in_gym:
            # Check party HP to determine if Dad dialogue is complete
            party = state_data.get('player', {}).get('party', [])
            needs_dad_dialogue = False
            
            if party:
                for pokemon in party:
                    current_hp = pokemon.get('current_hp', 0)
                    max_hp = pokemon.get('max_hp', 1)
                    if max_hp > 0 and current_hp < max_hp:
                        needs_dad_dialogue = True
                        logger.info(f"🎯 [DAD HP CHECK] {pokemon.get('species_name', 'UNKNOWN')}: {current_hp}/{max_hp} HP - needs healing!")
                        print(f"🎯 [DAD HP CHECK] {pokemon.get('species_name', 'UNKNOWN')}: {current_hp}/{max_hp} HP - needs healing!")
                        break
            
            logger.info(f"🎯 [DAD HP CHECK] HP < 100%: {needs_dad_dialogue}, in_city: {in_petalburg_city}, in_gym: {in_gym}")
            print(f"🎯 [DAD HP CHECK] HP < 100%: {needs_dad_dialogue}, in_city: {in_petalburg_city}, in_gym: {in_gym}")
            
            if needs_dad_dialogue:
                # HP < 100% = need to talk to Dad
                if in_gym:
                    # Dynamically resolve Norman's position in Petalburg Gym
                    # Phase 4.4d: registry-first lookup, cold-start nearest NPC fallback
                    NORMAN_FALLBACK = get_poi_coords("PETALBURG_CITY_GYM", "norman") or (4, 107)

                    norman_coords = self._resolve_npc_coords(
                        state_data,
                        npc_role="gym_leader_norman",
                        fallback=NORMAN_FALLBACK,
                    )
                    # Stand one tile south of Norman to face UP
                    goal_x = norman_coords[0]
                    goal_y = norman_coords[1] + 1

                    logger.info(f"💚 [DAD HP] HP < 100%, in gym, Norman at {norman_coords}")
                    print(f"💚 [DAD HP] HP < 100%, in gym, navigating to Norman [{norman_coords}]")
                    
                    return {
                        'goal_coords': (goal_x, goal_y, 'PETALBURG_CITY_GYM'),
                        'npc_coords': norman_coords,
                        'should_interact': True,
                        'description': f'Navigate to Norman at {norman_coords} [HP: {current_hp}/{max_hp}]'
                    }
                else:
                    # In Petalburg City - navigate to gym entrance
                    logger.info(f"💚 [DAD HP] HP < 100%, in city, navigating to gym")
                    print(f"💚 [DAD HP] HP < 100%, in city, navigating to gym")
                    
                    # Use navigation planner to get to gym
                    _gym_entrance = get_entrance_coords("PETALBURG_CITY", "PETALBURG_CITY_GYM") or (15, 8)
                    success = self.navigation_planner.plan_journey(
                        start_location=graph_location,
                        end_location='PETALBURG_CITY_GYM',
                        final_coords=_gym_entrance  # Gym entrance warp tile
                    )
                    
                    if success:
                        raw = self.navigation_planner.get_current_directive(
                            current_location=graph_location,
                            current_coords=(current_x, current_y)
                        )
                        return self._translate_planner_directive(raw, graph_location)
                    else:
                        logger.error(f"❌ [NAV PLANNER] Failed to plan journey to gym")
                        return None
            else:
                # HP = 100% = Dad dialogue complete, head west to Route 104 South
                logger.info(f"✅ [DAD HP] HP = 100%, heading west to Route 104 South")
                print(f"✅ [DAD HP] HP = 100%, heading west to Route 104 South")
                
                # Use navigation planner to head west
                success = self.navigation_planner.plan_journey(
                    start_location=graph_location,
                    end_location='ROUTE_104_SOUTH'
                )
                
                if success:
                    raw = self.navigation_planner.get_current_directive(
                        current_location=graph_location,
                        current_coords=(current_x, current_y)
                    )
                    return self._translate_planner_directive(raw, graph_location)
                else:
                    logger.error(f"❌ [NAV PLANNER] Failed to plan journey to Route 104 South")
                    return None
        
        # =====================================================================
        # SUB-GOAL: RUSTBORO CITY POKEMON CENTER HEALING
        # =====================================================================
        # CRITICAL: This must run BEFORE the sequential milestone system
        # because we need to heal before challenging the gym
        # =====================================================================
        # If we've reached Rustboro City but don't have the gym badge yet,
        # and our Pokemon need healing (HP or PP), go to Pokemon Center first
        # =====================================================================
        
        # DEBUG: Always log this check
        logger.debug(f"[POKECENTER] graph_location='{graph_location}', current_location='{current_location}'")
        
        # CRITICAL: Check both outside Pokemon Center (RUSTBORO_CITY) AND inside (RUSTBORO_CITY_POKEMON_CENTER_1F)
        in_rustboro_city = graph_location == 'RUSTBORO_CITY'
        in_pokecenter = graph_location == 'RUSTBORO_CITY_POKEMON_CENTER_1F'
        
        logger.debug(f"[POKECENTER] in_rustboro_city={in_rustboro_city}, in_pokecenter={in_pokecenter}")
        
        if in_rustboro_city or in_pokecenter:
            rustboro_complete = is_milestone_complete('RUSTBORO_CITY')
            has_stone_badge = is_milestone_complete('STONE_BADGE')
            
            logger.info(f"🏥 [POKECENTER CHECK] In Rustboro: milestone={rustboro_complete}, badge={has_stone_badge}")
            print(f"🏥 [POKECENTER CHECK] In Rustboro: milestone={rustboro_complete}, badge={has_stone_badge}")
            
            if rustboro_complete and not has_stone_badge:
                # =====================================================================
                # COMPREHENSIVE PARTY DATA DEBUGGING
                # =====================================================================
                # Trace the entire path: state_data → game → party → pokemon objects
                # =====================================================================
                
                # =====================================================================
                # FIX: Party data is in state_data['player']['party'], NOT state_data['game']['party']
                # The game memory reader stores party in the player section, not game section
                # Fallback: Also check state_data['party'] if player['party'] is empty
                # =====================================================================
                party = state_data.get('player', {}).get('party', [])
                
                # Fallback to top-level party if player party is empty
                if not party:
                    party = state_data.get('party', [])
                    if party:
                        print(f"🔍 [POKECENTER] Using fallback: state_data['party']")
                
                needs_healing = False
                healing_reasons = []
                
                logger.debug(f"[POKECENTER] Party length: {len(party) if party else 0}")
                
                if party:
                    for i, pokemon in enumerate(party):
                        # Pokemon is a dictionary, not an object
                        species = pokemon.get('species_name', 'UNKNOWN')
                        current_hp = pokemon.get('current_hp', 0)
                        max_hp = pokemon.get('max_hp', 1)
                        hp_percent = (current_hp / max_hp * 100) if max_hp > 0 else 0
                        
                        # Get move PP data
                        move_pp = pokemon.get('move_pp', [])  # List of current PP values
                        moves = pokemon.get('moves', [])  # Move names
                        
                        # DEBUG: Show raw data
                        logger.debug(f"[POKEMON {i+1}] {species} HP={current_hp}/{max_hp}, moves={moves}, pp={move_pp}")
                        
                        # Check HP
                        if current_hp < max_hp:
                            needs_healing = True
                            reason = f"{species} HP: {current_hp}/{max_hp} ({hp_percent:.1f}%)"
                            healing_reasons.append(reason)
                        
                        # Check PP for all moves
                        # Note: We don't have max PP in the data, so we can't check PP percentage
                        # We'll assume any move with 0 PP needs healing
                        for move_idx, (move_name, current_pp) in enumerate(zip(moves, move_pp)):
                            if move_name and move_name != 'NONE':
                                if current_pp == 0:
                                    needs_healing = True
                                    reason = f"{species} {move_name}: PP depleted (0 PP)"
                                    healing_reasons.append(reason)
                    
                logger.info(f"🏥 [POKECENTER] needs_healing={needs_healing}, reasons: {healing_reasons}")
                if needs_healing:
                    print(f"🏥 [POKECENTER] Healing needed: {healing_reasons}")
                
                # CRITICAL: Exit Pokemon Center after healing is complete
                # Healing complete + inside Pokemon Center = time to leave
                if not needs_healing and in_pokecenter:
                    logger.info(f"🏥 [POKECENTER EXIT] Healing complete, exiting Pokemon Center")
                    print(f"🏥 [POKECENTER EXIT] Pokemon healed - leaving Pokemon Center")
                    
                    # Derive exit warp tile from location graph
                    _exit_warp = get_interior_exit_coords(graph_location) or (7, 9)
                    # Warp tile itself may not be walkable; approach from one tile above
                    _pre_warp = (_exit_warp[0], _exit_warp[1] - 1)
                    
                    if current_x == _pre_warp[0] and current_y == _pre_warp[1]:
                        logger.info(f"🏥 [POKECENTER EXIT] At {_pre_warp}, pushing DOWN through warp")
                        return {
                            'goal_direction': 'south',
                            'description': f'Push DOWN from {_pre_warp} to exit Pokemon Center via warp',
                            'journey_reason': 'Exit Pokemon Center after healing'
                        }
                    else:
                        logger.info(f"🏥 [POKECENTER EXIT] Navigating to {_pre_warp} before exit warp")
                        return {
                            'goal_coords': (_pre_warp[0], _pre_warp[1], current_location),
                            'description': f'Navigate to {_pre_warp} to prepare for Pokemon Center exit',
                            'journey_reason': 'Position for Pokemon Center exit warp'
                        }
                
                # SPECIAL CASE: After exiting Pokemon Center, navigate NORTH to stable area before gym
                # Pokemon Center exit drops us at (16, 38-39) in southern Rustboro
                # # Gym is at (27, 19) in northern Rustboro - navigate UP first to avoid pathfinding issues
                # if not needs_healing and in_rustboro_city and current_y > 35:
                #     logger.info(f"🏥 [POST-HEAL NAV] After healing, navigating NORTH from Y={current_y} toward gym area")
                #     print(f"🏥 [POST-HEAL] Healed! Moving NORTH toward gym (currently at Y={current_y})")
                    
                #     return {
                #         'goal_direction': 'north',
                #         'description': f'Navigate NORTH from Pokemon Center area (Y={current_y}) toward gym region',
                #         'journey_reason': 'Move north after healing before navigating to gym'
                #     }
                
                if needs_healing:
                    # Check if we're already in the Pokemon Center
                    in_pokecenter = 'POKEMON CENTER' in current_location or 'POKECENTER' in current_location
                    
                    if in_pokecenter:
                        # Dynamically resolve Nurse Joy position
                        # Phase 4.4d: registry-first lookup, cold-start nearest NPC fallback
                        NURSE_FALLBACK = get_poi_coords(graph_location, "nurse_joy") or (7, 3)

                        nurse_coords = self._resolve_npc_coords(
                            state_data,
                            npc_role="nurse",
                            fallback=NURSE_FALLBACK,
                        )
                        # Stand one tile south of nurse to face UP
                        goal_x = nurse_coords[0]
                        goal_y = nurse_coords[1] + 1

                        logger.info(f"🏥 [POKECENTER] In Pokemon Center, nurse at {nurse_coords}")
                        print(f"🏥 [POKECENTER] In Pokemon Center - talking to nurse [{nurse_coords}]")
                        
                        return {
                            'goal_coords': (goal_x, goal_y, current_location),
                            'npc_coords': nurse_coords,
                            'should_interact': True,
                            'description': f'Navigate to ({goal_x},{goal_y}) and interact with nurse at {nurse_coords} to heal Pokemon'
                        }
                    else:
                        # Not in Pokemon Center yet - navigate there using the navigation planner
                        # This ensures proper pathfinding through the location graph
                        logger.info(f"🏥 [POKECENTER] Need healing, planning journey to Pokemon Center")
                        print(f"🏥 [POKECENTER] Pokemon need healing - planning route to Pokemon Center")
                        print(f"🏥 [POKECENTER] Current: {graph_location}, Target: RUSTBORO_CITY_POKEMON_CENTER_1F")
                        
                        # SPECIAL CASE: Rustboro City boundary navigation
                        # If agent is in lower Rustboro (Y > 55), navigate UP to safer bounds first
                        # This prevents map stitcher edge case issues
                        if graph_location == 'RUSTBORO_CITY' and current_y > 48:
                            logger.info(f"🏙️ [RUSTBORO BOUNDARY] Agent at edge (Y={current_y}), navigating UP to stable region")
                            print(f"🏙️ [RUSTBORO BOUNDARY] At Y={current_y} (edge area), moving UP to stable zone")
                            
                            # Use simple upward navigation until we're in stable bounds (Y <= 55)
                            return {
                                'goal_direction': 'north',
                                'description': f'Navigate UP from Rustboro edge (Y={current_y}) to stable region',
                                'journey_reason': 'Move to stable map region before Pokemon Center navigation'
                            }
                        
                        # RUSTBORO CITY WAYPOINT: Navigate to (23, 29) when in trigger zone
                        # Trigger zone: X between 12-35 AND Y between 28-38
                        if graph_location == 'RUSTBORO_CITY':
                            in_waypoint_zone = (12 <= current_x <= 35) and (28 <= current_y <= 38)
                            
                            if in_waypoint_zone:
                                WAYPOINT = (23, 29)
                                current_pos = (current_x, current_y)
                                
                                if current_pos != WAYPOINT:
                                    logger.info(f"🏙️ [RUSTBORO WAYPOINT] In zone at ({current_x}, {current_y}), routing to {WAYPOINT}")
                                    print(f"🏙️ [RUSTBORO WAYPOINT] Detected at ({current_x}, {current_y}) - navigating to {WAYPOINT}")
                                    
                                    return {
                                        'goal_coords': (23, 29, 'RUSTBORO_CITY'),
                                        'should_interact': False,
                                        'description': 'Navigate to (23, 29) waypoint in Rustboro City',
                                        'journey_reason': 'Rustboro City navigation waypoint'
                                    }
                                else:
                                    logger.info(f"✅ [RUSTBORO WAYPOINT] At waypoint {WAYPOINT} - continuing")
                                    print(f"✅ [RUSTBORO WAYPOINT] Waypoint reached - resuming navigation")
                        
                        # Use navigation planner to create journey
                        _pc_entrance = get_entrance_coords("RUSTBORO_CITY", "RUSTBORO_CITY_POKEMON_CENTER_1F") or (16, 38)
                        success = self.navigation_planner.plan_journey(
                            start_location=graph_location,
                            end_location="RUSTBORO_CITY_POKEMON_CENTER_1F",
                            final_coords=_pc_entrance  # Pokemon Center entrance in Rustboro
                        )
                        
                        if success:
                            # Get directive from planner
                            current_pos = (current_x, current_y)
                            planner_directive = self.navigation_planner.get_current_directive("RUSTBORO_CITY_POKEMON_CENTER_1F", current_pos)
                            
                            if planner_directive:
                                logger.info(f"🏥 [POKECENTER] Planner directive: {planner_directive.get('description', 'Unknown')}")
                                print(f"🗺️ [POKECENTER] Navigation Planner active: {planner_directive.get('description', 'Unknown')}")
                                
                                translated = self._translate_planner_directive(planner_directive, "RUSTBORO_CITY")
                                if translated:
                                    translated['journey_reason'] = 'Pokemon Center healing (via planner)'
                                    return translated
                        
                        # Fallback: direct coordinate navigation
                        _pc_fallback = get_entrance_coords("RUSTBORO_CITY", "RUSTBORO_CITY_POKEMON_CENTER_1F") or (16, 38)
                        logger.warning(f"🏥 [POKECENTER] Planner failed, using direct navigation")
                        print(f"🏥 [POKECENTER] Using direct navigation to {_pc_fallback}")
                        
                        return {
                            'goal_coords': (_pc_fallback[0], _pc_fallback[1], 'RUSTBORO_CITY'),
                            'should_interact': True,
                            'description': f'Navigate to Pokemon Center at {_pc_fallback} to heal Pokemon',
                            'journey_reason': 'Heal Pokemon before challenging gym'
                        }
                else:
                    print(f"✅ [POKECENTER] All Pokemon at 100% HP/PP - no healing needed!")
        
        # =====================================================================
        # SEQUENTIAL MILESTONE SYSTEM
        # =====================================================================
        # Find the next uncompleted milestone and target it
        # No more brittle if/elif chains or backward-checking logic
        # =====================================================================
        
        next_milestone = get_next_milestone_target(milestones)
        
        if not next_milestone:
            logger.info("🎉 [MILESTONE] All milestones complete!")
            return None
        
        milestone_id = next_milestone["milestone"]
        target_location = next_milestone.get("target_location")
        target_coords = next_milestone.get("target_coords") or (next_milestone["target_coords_fn"]() if "target_coords_fn" in next_milestone else None)
        special_handling = next_milestone.get("special")
        description = next_milestone.get("description", "Continue progression")
        
        logger.info(f"📍 [MILESTONE {next_milestone['index']}] Next target: {milestone_id} - {description}")
        print(f"📍 [MILESTONE {next_milestone['index']}] Targeting: {milestone_id}")
        
        # =====================================================================
        # SPECIAL CASE HANDLING
        # =====================================================================
        
        # RIVAL BATTLE: Navigate to specific coordinates and interact
        # Logic: ROUTE_103 complete AND RECEIVED_POKEDEX not complete AND rival battle goal not complete
        if special_handling == "rival_battle":
            # Check using GOALS (not milestones - RIVAL_BATTLE_1 milestone doesn't exist in game)
            rival_battle_complete = self.is_goal_complete('ROUTE_103_RIVAL_BATTLE')
            has_pokedex = is_milestone_complete('RECEIVED_POKEDEX')
            
            # If we already battled rival OR already have Pokedex, skip to NEXT milestone
            if rival_battle_complete or has_pokedex:
                logger.info(f"✅ [RIVAL BATTLE] Already complete - goal_complete={rival_battle_complete}, pokedex={has_pokedex}")
                logger.info(f"✅ [RIVAL BATTLE] Skipping to next milestone after RIVAL_BATTLE_1")
                print(f"✅ [RIVAL BATTLE] Already complete, moving to next milestone")
                
                # Get the NEXT milestone after RIVAL_BATTLE_1 (index 13 → 14: RECEIVED_POKEDEX)
                next_milestone_index = next_milestone['index'] + 1
                if next_milestone_index >= len(MILESTONE_PROGRESSION):
                    logger.info("🎉 [MILESTONE] All milestones complete!")
                    return None
                
                # Get milestone at index 14 (RECEIVED_POKEDEX)
                next_after_rival = {
                    "index": next_milestone_index,
                    **MILESTONE_PROGRESSION[next_milestone_index]
                }
                
                # Update ALL milestone variables for the new milestone
                next_milestone = next_after_rival  # Update the main milestone object
                milestone_id = next_after_rival["milestone"]
                target_location = next_after_rival.get("target_location")
                target_coords = next_after_rival.get("target_coords") or (next_after_rival["target_coords_fn"]() if "target_coords_fn" in next_after_rival else None)
                special_handling = next_after_rival.get("special")  # Already done below, but be explicit
                description = next_after_rival.get("description", "Continue progression")
                
                logger.info(f"📍 [MILESTONE {next_after_rival['index']}] Next target: {milestone_id} - {description}")
                print(f"📍 [MILESTONE {next_after_rival['index']}] Targeting: {milestone_id}")
                
                # Fall through to navigation handling below with the NEW milestone
                next_milestone = next_after_rival
                special_handling = next_after_rival.get("special")  # Update special handling for new milestone
            else:
                # Dynamically resolve rival position from gObjectEvents
                # Phase 4.4d: registry-first lookup, cold-start nearest NPC fallback
                _rival_poi = get_poi_coords("ROUTE_103", "rival_may") or (9, 3)
                RIVAL_FALLBACK = (_rival_poi[0] + 1, _rival_poi[1])  # one tile east as safety net

                rival_coords = self._resolve_npc_coords(
                    state_data,
                    npc_role="rival",
                    fallback=RIVAL_FALLBACK,
                )
                # Stand one tile west of the rival to face RIGHT
                goal_x = rival_coords[0] - 1
                goal_y = rival_coords[1]

                logger.info(f"🎯 [RIVAL BATTLE] ROUTE_103 complete, RECEIVED_POKEDEX not complete, rival not battled")
                logger.info(f"🎯 [RIVAL BATTLE] Rival at {rival_coords} → goal ({goal_x},{goal_y})")
                print(f"🎯 [RIVAL BATTLE] Navigate to rival at Route 103 [{rival_coords}]")
                
                return {
                    'goal_coords': (goal_x, goal_y, 'ROUTE_103'),
                    'npc_coords': rival_coords,
                    'should_interact': True,
                    'description': f'Navigate to ({goal_x},{goal_y}) and face RIGHT to interact with rival at {rival_coords}'
                }
        
        # =====================================================================
        # PETALBURG WOODS: Navigate around obstacle zone
        # =====================================================================
        # In the eastern section of Petalburg Woods (x=11-25, y=31-38), there's
        # a difficult area with NPCs/obstacles. Route to (9,34) to avoid them.
        # =====================================================================
        in_petalburg_woods = graph_location == 'PETALBURG_WOODS' or 'MAP_18_0B' in current_location
        
        if in_petalburg_woods:
            position = player_data.get('position', {})
            current_x = position.get('x', 0)
            current_y = position.get('y', 0)
            
            # Trigger zone: X between 11-25 AND Y between 31-38
            in_obstacle_zone = (11 <= current_x <= 25) and (31 <= current_y <= 38)
            
            if in_obstacle_zone:
                WAYPOINT = (9, 34)
                current_pos = (current_x, current_y)
                
                logger.info(f"🌲 [PETALBURG WOODS] In obstacle zone at ({current_x}, {current_y})")
                print(f"🌲 [PETALBURG WOODS] Obstacle zone detected - navigating to waypoint {WAYPOINT}")
                
                # Navigate to waypoint unless already there
                if current_pos != WAYPOINT:
                    return {
                        'goal_coords': (9, 34, 'PETALBURG_WOODS'),
                        'description': 'Navigate to (9,34) to avoid Petalburg Woods obstacle zone',
                        'avoid_grass': True
                    }
                else:
                    logger.info(f"✅ [PETALBURG WOODS] Reached waypoint {WAYPOINT} - continuing")
                    print(f"✅ [PETALBURG WOODS] Waypoint reached - resuming normal navigation")
        
        # ROUTE 104 SOUTH: Navigate around NPC using waypoint system
        # The NPC at (11, 44) blocks the direct path with a large dialogue zone
        # Blocked tiles: (11,44), (12,44), (13,44), (14,44), (15,44), (11,43), (11,42), (11,41), (11,40)
        # Strategy: Route through grass on south side via waypoints 16,45 -> 10,45
        in_route_104_south = graph_location is not None and 'ROUTE_104' in graph_location and 'SOUTH' in graph_location
        
        if in_route_104_south:
            position = player_data.get('position', {})
            current_x = position.get('x', 0)
            current_y = position.get('y', 0)
            
            # Check if we're in the trigger zone that requires special routing
            # Trigger: X between 16-25 AND Y between 41-53
            in_trigger_zone = (16 <= current_x <= 25) and (41 <= current_y <= 53)
            
            # SPECIAL: If the player is on the horizontal grass path (y=45, x=10-17)
            # between waypoints, route them to (10,45) regardless of trigger zone.
            # This handles battle interruptions along the path from (16,45) to (10,45).
            on_grass_path = (current_y == 45) and (10 <= current_x <= 17)
            
            if in_trigger_zone or on_grass_path:
                if on_grass_path and not in_trigger_zone:
                    logger.info(f"🌿 [ROUTE 104 SOUTH] On grass path at ({current_x}, {current_y}), routing to (10,45)")
                    print(f"🌿 [ROUTE 104 SOUTH] On grass path y=45, continue west to (10,45)")
                else:
                    logger.info(f"🌿 [ROUTE 104 SOUTH] In trigger zone at ({current_x}, {current_y})")
                    print(f"🌿 [ROUTE 104 SOUTH] Navigating around NPC via grass waypoints")
                
                # Waypoint sequence: current position -> (16,45) -> (10,45) -> continue
                WAYPOINT_SEQUENCE = [
                    (16, 45),  # East waypoint (approach grass from east)
                    (10, 45),  # West waypoint (exit grass to west side)
                ]
                
                current_pos = (current_x, current_y)

                # CRITICAL: If on the horizontal grass path (y=45, x=10-17), always route to (10,45)
                # This is the direct path from waypoint 1 (16,45) to waypoint 2 (10,45)
                # Catches battle interruptions at positions like (11,45), (12,45), etc.
                if (current_y == 45) and (10 <= current_x <= 17):
                    final_wp = WAYPOINT_SEQUENCE[-1]  # (10, 45)
                    if current_pos != final_wp:
                        logger.info(f"🌿 [ROUTE 104 SOUTH] On horizontal path at {current_pos}, routing to {final_wp}")
                        print(f"🌿 [ROUTE 104 SOUTH] On y=45 path, heading to {final_wp}")
                        return {
                            'goal_coords': (final_wp[0], final_wp[1], 'ROUTE_104_SOUTH'),
                            'description': f'Continue west on grass path to {final_wp}',
                            'avoid_grass': False
                        }

                # FALLBACK: Check broader grass corridor for other interruptions
                # x in [10..16], y in [44..46]
                GRASS_PATH_MIN_X, GRASS_PATH_MAX_X = 10, 16
                GRASS_PATH_MIN_Y, GRASS_PATH_MAX_Y = 44, 46

                if (GRASS_PATH_MIN_X <= current_x <= GRASS_PATH_MAX_X) and (GRASS_PATH_MIN_Y <= current_y <= GRASS_PATH_MAX_Y):
                    # If we're already at the final west waypoint, fall through normally
                    final_wp = WAYPOINT_SEQUENCE[-1]
                    if current_pos != final_wp:
                        logger.info(f"🌿 [ROUTE 104 SOUTH] In grass corridor at {current_pos}, directing to final waypoint {final_wp}")
                        print(f"🌿 [ROUTE 104 SOUTH] In grass corridor, continue to {final_wp}")
                        return {
                            'goal_coords': (final_wp[0], final_wp[1], 'ROUTE_104_SOUTH'),
                            'description': f'Continue through grass to {final_wp} (resume interrupted path)',
                            'avoid_grass': False
                        }

                # Check if we're at or past a waypoint
                for i, waypoint_pos in enumerate(WAYPOINT_SEQUENCE):
                    if current_pos == waypoint_pos:
                        # At this waypoint - move to next
                        if i < len(WAYPOINT_SEQUENCE) - 1:
                            next_pos = WAYPOINT_SEQUENCE[i + 1]
                            next_x, next_y = next_pos
                            
                            logger.info(f"🎯 [ROUTE 104 SOUTH] At waypoint {i+1}/{len(WAYPOINT_SEQUENCE)}: {current_pos} -> {next_pos}")
                            print(f"🎯 [ROUTE 104 SOUTH] Waypoint {i+1}/{len(WAYPOINT_SEQUENCE)}: going through grass")
                            
                            return {
                                'goal_coords': (next_x, next_y, 'ROUTE_104_SOUTH'),
                                'description': f'Navigate to waypoint {next_pos} (grass route around NPC)',
                                'avoid_grass': False  # CRITICAL: Allow grass pathfinding for this route
                            }
                        else:
                            # At final waypoint (10, 45) - release restrictions, continue normally
                            logger.info(f"✅ [ROUTE 104 SOUTH] Completed waypoint sequence at {current_pos}")
                            print(f"✅ [ROUTE 104 SOUTH] Past NPC zone, resuming normal navigation")
                            # Fall through to normal navigation
                            break
                
                # Not at any waypoint yet - navigate to first waypoint (16, 45)
                if current_pos not in WAYPOINT_SEQUENCE:
                    first_waypoint = WAYPOINT_SEQUENCE[0]
                    logger.info(f"🎯 [ROUTE 104 SOUTH] At ({current_x},{current_y}), navigating to first waypoint {first_waypoint}")
                    print(f"🎯 [ROUTE 104 SOUTH] Heading to waypoint 1: {first_waypoint}")
                    
                    return {
                        'goal_coords': (first_waypoint[0], first_waypoint[1], 'ROUTE_104_SOUTH'),
                        'description': f'Navigate to waypoint {first_waypoint} (avoid NPC dialogue zone)',
                        'avoid_grass': True  # Avoid grass before reaching waypoint
                    }
        
        # ROXANNE BATTLE: Navigate through gym trainers then to gym leader
        # Logic: RUSTBORO_GYM_ENTERED complete AND ROXANNE_DEFEATED not complete
        # CRITICAL: Must visit waypoints to trigger trainer battles before Roxanne
        # Simple position-based logic: Check current position, return next waypoint in sequence
        if milestone_id == "ROXANNE_DEFEATED":
            gym_entered = is_milestone_complete('RUSTBORO_GYM_ENTERED')
            roxanne_defeated = is_milestone_complete('ROXANNE_DEFEATED')
            in_gym = 'RUSTBORO CITY GYM' in current_location or 'RUSTBORO_CITY_GYM' in current_location
            
            logger.info(f"🎯 [ROXANNE] gym_entered={gym_entered}, defeated={roxanne_defeated}, in_gym={in_gym}")
            print(f"🎯 [ROXANNE] Gym: {gym_entered}, Defeated: {roxanne_defeated}, Inside: {in_gym}")
            
            if not roxanne_defeated and in_gym:
                position = player_data.get('position', {})
                current_x = position.get('x', 0)
                current_y = position.get('y', 0)
                
                logger.info(f"🎯 [ROXANNE] Current position: ({current_x}, {current_y})")
                print(f"🎯 [ROXANNE] Current position: ({current_x}, {current_y})")
                
                # Waypoint sequence: entrance -> trainers -> Roxanne
                # Just list the positions in order - we'll navigate from [i] to [i+1]
                WAYPOINT_SEQUENCE = [
                    (5, 19),   # 0: Gym entrance
                    (8, 15),   
                    (5, 14),   # Trainer 1
                    (4, 14),   
                    (2, 10),   
                    (2, 9),    # Trainer 2
                    (2, 7),
                    (1, 7), 
                    (2, 8),    # Trainer 3
                    (5, 3),    # Roxanne position
                ]
                
                # Roxanne is at (5, 2), so when at (5, 3) we need to interact
                ROXANNE_POSITION = (5, 3)
                ROXANNE_FALLBACK = (5, 2)
                # Dynamically resolve Roxanne's position
                # Roxanne: graphics_id=89 (GYM_LEADER sprite), local_id=1
                # Phase 4.4d: registry-first lookup, cold-start nearest NPC fallback
                roxanne_npc_coords = self._resolve_npc_coords(
                    state_data,
                    npc_role="gym_leader_roxanne",
                    fallback=ROXANNE_FALLBACK,
                )
                
                current_pos = (current_x, current_y)
                
                # Find current position in sequence and navigate to next
                for i, waypoint_pos in enumerate(WAYPOINT_SEQUENCE):
                    if current_pos == waypoint_pos:
                        # At this waypoint - determine next target
                        if i < len(WAYPOINT_SEQUENCE) - 1:
                            next_pos = WAYPOINT_SEQUENCE[i + 1]
                        else:
                            # At last waypoint (Roxanne position) - stay here and interact
                            next_pos = waypoint_pos
                        
                        next_x, next_y = next_pos
                        
                        # Check if we're at Roxanne position
                        should_interact = (current_pos == ROXANNE_POSITION)
                        
                        logger.info(f"🎯 [ROXANNE] At waypoint {i}/{len(WAYPOINT_SEQUENCE)-1}: {current_pos} -> {next_pos}")
                        print(f"🎯 [ROXANNE] Waypoint {i+1}/{len(WAYPOINT_SEQUENCE)}: {current_pos} -> {next_pos}")
                        
                        result = {
                            'goal_coords': (next_x, next_y, 'RUSTBORO_CITY_GYM'),
                            'description': f'Navigate to {next_pos}' + (' and interact with Roxanne' if should_interact else '')
                        }
                        
                        if should_interact:
                            result['should_interact'] = True
                            result['npc_coords'] = roxanne_npc_coords
                        
                        return result
                
                # Default: Not at any recognized waypoint, go to first waypoint
                # SPECIAL: If we're at the gym entrance coords, we just entered via warp
                # The warp takes us to (5, 19) which is waypoint 0
                # After warp, we might be briefly showing entrance coords before update
                GYM_ENTRANCE_OUTSIDE = get_entrance_coords("RUSTBORO_CITY", "RUSTBORO_CITY_GYM") or (27, 19)
                if current_pos == GYM_ENTRANCE_OUTSIDE:
                    # Just entered gym, but position hasn't updated yet
                    # Return directive for waypoint 1 (already at waypoint 0 after warp)
                    second_waypoint = WAYPOINT_SEQUENCE[1]
                    logger.info(f"🎯 [ROXANNE] Just entered gym at entrance tile, navigating to waypoint 2: {second_waypoint}")
                    print(f"🎯 [ROXANNE] Gym entrance detected, navigating to waypoint 2: {second_waypoint}")
                    return {
                        'goal_coords': (second_waypoint[0], second_waypoint[1], 'RUSTBORO_CITY_GYM'),
                        'description': f'Navigate to {second_waypoint} - second waypoint (after entrance warp)'
                    }
                
                first_waypoint = WAYPOINT_SEQUENCE[0]
                logger.info(f"🎯 [ROXANNE] At ({current_x},{current_y}), navigating to first waypoint {first_waypoint}")
                print(f"🎯 [ROXANNE] Starting gym, navigating to waypoint 1: {first_waypoint}")
                return {
                    'goal_coords': (first_waypoint[0], first_waypoint[1], 'RUSTBORO_CITY_GYM'),
                    'description': f'Navigate to {first_waypoint} - first trainer waypoint'
                }
            
            elif roxanne_defeated and in_gym:
                # VICTORY! Exit the gym after defeating Roxanne
                logger.info(f"� [ROXANNE DEFEATED] Victory! Exiting gym")
                print(f"� [ROXANNE DEFEATED] Stone Badge obtained! Leaving gym")
                
                return {
                    'goal_direction': 'south',
                    'description': 'Exit Rustboro Gym after defeating Roxanne',
                    'journey_reason': 'Victory! First gym badge obtained - exiting gym'
                }
            
            elif not in_gym and not roxanne_defeated:
                # Not in gym yet, need to enter - fall through to sequential system
                logger.info(f"🎯 [ROXANNE] Not in gym yet, need to enter")
                print(f"🎯 [ROXANNE] Need to enter gym first")
                # Fall through to target RUSTBORO_GYM_ENTERED
        
        # =====================================================================
        # NAVIGATION HANDLING
        # =====================================================================
        
        if not target_location:
            # No navigation needed - milestone will auto-complete (dialogue, events, etc.)
            logger.info(f"📍 [MILESTONE] {milestone_id} - waiting for auto-complete")
            return None
        
        # =====================================================================
        # PHASE 4.3b: RAG-PRIMARY NAVIGATION TARGET
        # =====================================================================
        # When a StrategicPlanner is attached, ask the walkthrough RAG for the
        # next navigation target.  If it resolves to a valid LOCATION_GRAPH
        # key, use it.  Otherwise fall back to MILESTONE_PROGRESSION target.
        #
        # Special-case handling above (gym, Pokémon Center, rival, waypoints)
        # fires BEFORE this block — those return early and are never overridden.
        # =====================================================================
        final_target = target_location
        final_coords = target_coords
        final_description = description
        self._last_directive_source = "milestone"  # Default
        
        logger.debug(f"[NAV] Before RAG: milestone_target={final_target}")
        rag_result = self._query_rag_target(state_data)
        logger.debug(f"[NAV] RAG returned: {rag_result}")
        if rag_result and rag_result.get("target_location"):
            rag_loc = rag_result["target_location"]
            logger.debug(f"[NAV] RAG={rag_loc}, milestone={target_location}, match={rag_loc == target_location}")
            # Only override if RAG target differs from milestone target
            # (if they agree, milestone coords are usually more precise)
            if rag_loc != target_location:
                logger.debug(f"[NAV] RAG OVERRIDE: {target_location} → {rag_loc}")
                final_target = rag_loc
                final_coords = rag_result.get("target_coords")
                final_description = rag_result.get("description", description)
                self._last_directive_source = "rag"
                self._rag_override_count += 1
                logger.info(
                    f"🔮 [RAG PRIMARY] Overriding milestone target "
                    f"{target_location} → {rag_loc} ({rag_result.get('display_name')})"
                )
                print(
                    f"🔮 [RAG PRIMARY] Using RAG target: {rag_loc} "
                    f"(milestone was {target_location})"
                )
            else:
                self._last_directive_source = "rag"  # RAG agreed
                self._rag_override_count += 1
                logger.info(f"🔮 [RAG] agrees with milestone: {target_location}")
        else:
            self._milestone_fallback_count += 1
            logger.info(f"🔮 [RAG FALLBACK] Using milestone target: {target_location}")
        
        # =====================================================================
        # LOOK-AHEAD: Plan through pass-through locations to final destination
        # =====================================================================
        # Some milestones are just waypoints (e.g., PETALBURG_WOODS) - we should
        # plan to the final destination (e.g., ROUTE_104_NORTH) instead
        # NOTE: Only applies when using milestone target (RAG naturally resolves
        # to the final destination).
        # =====================================================================
        if self._last_directive_source == "milestone" and milestone_id == "PETALBURG_WOODS":
            # PETALBURG_WOODS is milestone 20, look ahead to 22 (ROUTE_104_NORTH)
            # Skip 21 (TEAM_AQUA_GRUNT_DEFEATED) since it's a battle, not navigation
            logger.info(f"🔍 [LOOK-AHEAD] PETALBURG_WOODS is pass-through, checking next navigation milestone...")
            print(f"🔍 [LOOK-AHEAD] Checking for final destination beyond PETALBURG_WOODS...")
            
            # Look ahead 2 milestones (20 -> 21 -> 22)
            lookahead_index = next_milestone['index'] + 2
            if lookahead_index < len(MILESTONE_PROGRESSION):
                lookahead_milestone = MILESTONE_PROGRESSION[lookahead_index]
                lookahead_location = lookahead_milestone.get("target_location")
                
                if lookahead_location and lookahead_milestone["milestone"] == "ROUTE_104_NORTH":
                    # Found the final destination - plan to Route 104 North instead
                    final_target = lookahead_location
                    final_coords = lookahead_milestone.get("target_coords") or (lookahead_milestone["target_coords_fn"]() if "target_coords_fn" in lookahead_milestone else None)
                    final_description = f"Navigate through Petalburg Woods to {lookahead_location}"
                    
                    logger.info(f"✅ [LOOK-AHEAD] Planning to final destination: {final_target}")
                    print(f"✅ [LOOK-AHEAD] Planning through PETALBURG_WOODS to {final_target}")
        
        # Get directive from navigation planner with final destination
        planner_directive = self._get_navigation_planner_directive(state_data, final_target, final_coords, final_description)
        
        if planner_directive and not planner_directive.get('error'):
            # Planner successfully provided a directive
            planner_directive['milestone'] = milestone_id
            planner_directive['journey_reason'] = description
            
            logger.info(f"🗺️ [PLANNER] Using NavigationPlanner directive: {planner_directive.get('description')}")
            print(f"🗺️ [PLANNER] Directive: {planner_directive.get('description')}")
            
            return planner_directive
        else:
            # Planner failed - log error but don't crash
            error_msg = planner_directive.get('description', 'Unknown error') if planner_directive else 'Planner returned None'
            logger.warning(f"⚠️ [PLANNER] Failed to get directive: {error_msg}")
            print(f"⚠️ [PLANNER] Failed: {error_msg}")
            # Fall through to return None (VLM will handle it)
        
        # No specific directive - return None to let VLM handle it
        logger.debug(f"📍 [DIRECTIVE] No specific directive for current state")
        return None
    
    def get_objectives_summary(self) -> Dict[str, Any]:
        """Get a summary of objectives for debugging/monitoring"""
        active = self.get_active_objectives()
        completed = self.get_completed_objectives()
        
        current_objective = None
        if active:
            current_objective = {
                "id": active[0].id,
                "description": active[0].description,
                "type": active[0].objective_type,
                "milestone_id": active[0].milestone_id
            }
        
        return {
            "current_objective": current_objective,
            "active_count": len(active),
            "completed_count": len(completed),
            "total_count": len(self.objectives),
            "completion_rate": len(completed) / len(self.objectives) if self.objectives else 0
        }
    
    # ====================================================================
    # PHASE 4.3b: RAG-PRIMARY NAVIGATION
    # ====================================================================
    # When a StrategicPlanner is attached, the RAG planner determines the
    # next navigation target.  If the RAG result doesn't resolve to a valid
    # LOCATION_GRAPH key, fall back to MILESTONE_PROGRESSION.
    # ====================================================================

    def _query_rag_target(self, state_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Query the StrategicPlanner for a RAG-based navigation target.

        Returns a dict with ``target_location``, ``target_coords``,
        ``description``, ``display_name``, ``priority_actions`` if the RAG
        result resolves to a valid LOCATION_GRAPH key.  Returns ``None``
        if the planner is unavailable, or the result doesn't resolve.
        """
        if not self.strategic_planner:
            return None

        try:
            player_data = state_data.get("player", {})
            current_location = player_data.get("location", "Unknown").upper()
            game_data = state_data.get("game", {})

            # Badge count
            badge_count = (
                bin(game_data.get("badges", 0)).count("1")
                if game_data.get("badges")
                else 0
            )

            # Pokemon summary
            party = player_data.get("party", [])
            if party:
                pokemon_summary = ", ".join(
                    f"{p.get('species_name', '?')} Lv{p.get('level', '?')}"
                    for p in party[:3]
                )
            else:
                pokemon_summary = "Unknown"

            # Last milestone
            milestones = state_data.get("milestones", {})
            last_milestone = "None"
            highest_idx = get_highest_milestone_index(milestones)
            if highest_idx >= 0:
                last_milestone = MILESTONE_PROGRESSION[highest_idx]["milestone"]

            # ── Cache check: skip LLM if location + milestone unchanged ──
            cache_key = (current_location, last_milestone)
            if cache_key == self._rag_cache_key and self._rag_cache_result is not None:
                print(f"🔮 [RAG CACHE HIT] Reusing cached result for ({current_location}, {last_milestone})")
                return self._rag_cache_result

            logger.debug(f"[RAG] Querying: loc={current_location}, badges={badge_count}, milestone={last_milestone}")

            rag_result = self.strategic_planner.get_next_directive(
                current_location=current_location,
                badge_count=badge_count,
                pokemon_summary=pokemon_summary,
                last_milestone=last_milestone,
                state_data=state_data,
            )

            logger.debug(f"[RAG] Full result: {rag_result}")

            target_location = rag_result.get("target_location")
            if not target_location:
                logger.info("🔮 [RAG] No target_location in RAG result — falling back to milestone")
                return None

            # Build result with coords if available
            result = {
                "target_location": target_location,
                "target_coords": None,
                "description": rag_result.get("description", "Continue exploring."),
                "display_name": rag_result.get("target_display_name", target_location),
                "priority_actions": rag_result.get("priority_actions", []),
            }

            # Extract goal_coords → target_coords (just x,y without location key)
            gc = rag_result.get("goal_coords")
            if gc and len(gc) >= 2:
                result["target_coords"] = (gc[0], gc[1])

            logger.info(
                f"🔮 [RAG] target={target_location} "
                f"({result['display_name']}), coords={result['target_coords']}"
            )

            # ── Store in cache ──
            self._rag_cache_key = cache_key
            self._rag_cache_result = result

            return result

        except Exception as e:
            logger.warning(f"🔮 [RAG] Query failed: {e}")
            return None

    # ====================================================================
    # PHASE 4.3a: SHADOW-MODE RAG COMPARISON (logging)
    # ====================================================================
    # Shadow logging still runs for observability.  Now also records
    # whether RAG or milestone was the actual directive source.
    # ====================================================================

    def _run_shadow_comparison(
        self,
        state_data: Dict[str, Any],
        milestone_directive: Optional[Dict[str, Any]],
    ) -> None:
        """Fire the RAG planner and log a comparison with the milestone directive.

        This method is called at the end of ``get_next_action_directive()``
        (only when ``self.strategic_planner`` is set).  It never modifies the
        returned directive — it only logs.
        """
        if not self.strategic_planner:
            return

        self._shadow_step_count += 1

        # Throttle: only compare once every 20 steps to avoid LLM spam
        if self._shadow_step_count % 20 != 1:
            return

        try:
            # Extract game state for the RAG query
            player_data = state_data.get("player", {})
            current_location = player_data.get("location", "Unknown").upper()
            position = player_data.get("position", {})
            current_x = position.get("x", 0)
            current_y = position.get("y", 0)
            game_data = state_data.get("game", {})
            in_battle = game_data.get("in_battle", False)

            # Don't shadow-compare during battles or dialogue
            if in_battle:
                return
            screen_context = state_data.get("screen_context", "")
            if screen_context == "dialogue":
                return

            # Badge count
            badge_count = bin(game_data.get("badges", 0)).count("1") if game_data.get("badges") else 0

            # Pokemon summary
            party = player_data.get("party", [])
            if party:
                pokemon_summary = ", ".join(
                    f"{p.get('species_name', '?')} Lv{p.get('level', '?')}"
                    for p in party[:3]
                )
            else:
                pokemon_summary = "Unknown"

            # Last milestone
            milestones = state_data.get("milestones", {})
            last_milestone = "None"
            highest_idx = get_highest_milestone_index(milestones)
            if highest_idx >= 0:
                last_milestone = MILESTONE_PROGRESSION[highest_idx]["milestone"]

            # Extract milestone target from the directive
            # Priority: milestone (final destination) > target_location > goal_coords[2]
            # NOTE: goal_coords[2] is the *intermediate waypoint* the nav planner
            # is routing through, NOT the final destination.  The 'milestone' key
            # holds the actual target (e.g. ROUTE_103) while goal_coords might
            # say ROUTE_102 because the agent is still traversing that route.
            milestone_target = None
            if milestone_directive:
                milestone_target = (
                    milestone_directive.get("milestone")
                    or milestone_directive.get("target_location")
                )
                # Only fall back to goal_coords location if nothing better
                if not milestone_target:
                    gc = milestone_directive.get("goal_coords")
                    if gc and len(gc) >= 3 and isinstance(gc[2], str):
                        milestone_target = gc[2]
                # Last resort: description text
                if not milestone_target:
                    milestone_target = milestone_directive.get("description", "")

            # Fire RAG planner
            rag_result = self.strategic_planner.get_next_directive(
                current_location=current_location,
                badge_count=badge_count,
                pokemon_summary=pokemon_summary,
                last_milestone=last_milestone,
                state_data=state_data,
            )

            rag_target = rag_result.get("target_location")

            # Compare
            comparison = self.strategic_planner.shadow_compare(milestone_target, rag_target)

            # Update counters
            self._shadow_total_count += 1
            if comparison["agree"]:
                self._shadow_agree_count += 1

            agreement_rate = (
                self._shadow_agree_count / self._shadow_total_count * 100
                if self._shadow_total_count > 0
                else 0.0
            )

            # Build log entry
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "step": self._shadow_step_count,
                "location": current_location,
                "coords": [current_x, current_y],
                "badges": badge_count,
                "last_milestone": last_milestone,
                "milestone_target": milestone_target,
                "rag_target": rag_target,
                "rag_display_name": rag_result.get("target_display_name"),
                "rag_description": rag_result.get("description"),
                "rag_priority_actions": rag_result.get("priority_actions", []),
                "rag_goal_coords": (
                    list(rag_result["goal_coords"]) if rag_result.get("goal_coords") else None
                ),
                "agree": comparison["agree"],
                "agreement_rate": round(agreement_rate, 1),
                "comparisons_so_far": self._shadow_total_count,
                "directive_source": self._last_directive_source,
            }

            # Log to console
            agree_symbol = "✅" if comparison["agree"] else "❌"
            logger.info(
                f"🔮 [SHADOW] {agree_symbol} milestone={milestone_target} vs "
                f"rag={rag_target} | rate={agreement_rate:.0f}% "
                f"({self._shadow_agree_count}/{self._shadow_total_count})"
            )
            print(
                f"🔮 [SHADOW] {agree_symbol} Milestone: {milestone_target} | "
                f"RAG: {rag_target} ({rag_result.get('target_display_name', '?')}) | "
                f"Agreement: {agreement_rate:.0f}% "
                f"({self._shadow_agree_count}/{self._shadow_total_count})"
            )

            # Log to JSONL file
            self._write_shadow_log(log_entry)

        except Exception as exc:
            logger.warning(f"🔮 [SHADOW] Comparison failed: {exc}", exc_info=True)

    def _write_shadow_log(self, entry: Dict[str, Any]) -> None:
        """Append a shadow-comparison entry to the JSONL log file."""
        try:
            os.makedirs(os.path.dirname(self._shadow_log_path), exist_ok=True)
            with open(self._shadow_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as exc:
            logger.warning(f"[SHADOW] Failed to write log: {exc}")

    def get_shadow_stats(self) -> Dict[str, Any]:
        """Return current shadow-mode agreement and RAG-primary statistics."""
        rate = (
            self._shadow_agree_count / self._shadow_total_count * 100
            if self._shadow_total_count > 0
            else 0.0
        )
        return {
            "total_comparisons": self._shadow_total_count,
            "agreements": self._shadow_agree_count,
            "disagreements": self._shadow_total_count - self._shadow_agree_count,
            "agreement_rate": round(rate, 1),
            "steps_processed": self._shadow_step_count,
            "rag_overrides": self._rag_override_count,
            "milestone_fallbacks": self._milestone_fallback_count,
        }

    # ====================================================================
    # BLOCKER / RECOVERY SYSTEM (ported from GoalManager)
    # ====================================================================
    # Detects dialogue-based blockers (keyword matching) and battle
    # transitions, then uses RAG + LLM to generate recovery plans.
    # ====================================================================

    @property
    def is_blocked(self) -> bool:
        """True when a non-battle blocker is active."""
        return self._blocker_state is not None

    @property
    def current_brain_directive(self) -> str:
        """High-level plan string for logging / LLM context."""
        if self._recovery_tasks:
            rt = self._recovery_tasks[0]
            return f"RECOVERY: {rt['task']} (reason: {rt.get('reason', '?')})"
        # Use active objectives as a lightweight plan summary
        active = self.get_active_objectives()
        if active:
            return f"Current Goal: {active[0].description}"
        return "Current Goal: All objectives complete"

    def signal_blocker(self, reason: str, context: str):
        """External trigger to enter BLOCKED state (e.g., battle transition)."""
        self._handle_blocker(reason=reason, context=context)

    def _handle_blocker(self, reason: str, context: str):
        """Transition into BLOCKED state (idempotent)."""
        if self._blocker_state is not None:
            return  # already blocked
        print(f"⚠️ [ObjectiveManager] BLOCKER DETECTED: {reason}")
        print(f"   Context: '{context}'")
        self._blocker_state = {"reason": reason, "context": context}

    def clear_blocker(self):
        """Exit BLOCKED state (called after battle end or recovery complete)."""
        if self._blocker_state:
            print(f"✅ [ObjectiveManager] Blocker cleared: {self._blocker_state['reason']}")
        self._blocker_state = None

    def add_recovery_task(self, task_description: str, reason: str = ""):
        """Push a recovery sub-goal onto the stack."""
        self._recovery_tasks.insert(0, {
            "task": task_description,
            "status": "IN_PROGRESS",
            "type": "RECOVERY",
            "reason": reason,
        })
        print(f"📋 [ObjectiveManager] Recovery task added: {task_description}")

    def complete_recovery_task(self):
        """Pop the top recovery task."""
        if self._recovery_tasks:
            completed = self._recovery_tasks.pop(0)
            print(f"✅ [ObjectiveManager] Recovery task done: {completed['task']}")

    def _scan_dialogue_for_blockers(self, perception_output: dict):
        """Scan VLM perception output for blocking keywords (ported from GoalManager.update)."""
        visual_data = perception_output.get("visual_data", {})
        on_screen_text = visual_data.get("on_screen_text", {})

        if isinstance(on_screen_text, str):
            dialogue = on_screen_text
        elif isinstance(on_screen_text, dict):
            dialogue = on_screen_text.get("dialogue") or ""
        else:
            dialogue = ""

        if dialogue and isinstance(dialogue, str):
            dialogue_lower = dialogue.lower()
            for keyword in self.blocking_keywords:
                if keyword in dialogue_lower:
                    self._handle_blocker(
                        reason=f"NPC Dialogue Keyword: '{keyword}'",
                        context=dialogue,
                    )
                    break

    def update_brain(
        self,
        perception_output: dict,
        state_data: Dict[str, Any],
        episodic_memory=None,
        recovery_planner=None,
    ) -> Optional[list]:
        """
        Unified brain update — replaces the scattered logic in Agent.__init__.py.

        Performs:
          A. Log new dialogue to episodic memory.
          B. Detect battle start/end transitions → signal blocker → recovery plan.
          C. Keyword-based blocker detection in dialogue.
          D. If blocked (non-battle), fire RAG recovery and short-circuit.

        Returns:
            A short-circuit action list (e.g. ``['A']``) when the brain wants
            to override the normal pipeline, or *None* to let the pipeline
            continue.
        """
        # ── A. Dialogue → episodic memory ──
        brain_visual = perception_output.get("visual_data", {})
        brain_on_screen = brain_visual.get("on_screen_text", {})
        brain_dialogue = None
        if isinstance(brain_on_screen, str):
            brain_dialogue = brain_on_screen
        elif isinstance(brain_on_screen, dict):
            brain_dialogue = brain_on_screen.get("dialogue")

        if brain_dialogue and brain_dialogue != self._last_logged_dialogue:
            if episodic_memory is not None:
                episodic_memory.log_event(
                    f"Heard dialogue: '{brain_dialogue}'",
                    {"type": "dialogue"},
                    state_data=state_data,
                )
            self._last_logged_dialogue = brain_dialogue

        # ── B. Battle transitions ──
        brain_in_battle = state_data.get("game", {}).get("in_battle", False)
        brain_location = state_data.get("player", {}).get("location", "Unknown")

        if brain_in_battle and not self._brain_prev_in_battle:
            # ── BATTLE START ──
            battle_context = brain_dialogue or f"Entered battle on {brain_location}"
            if episodic_memory is not None:
                episodic_memory.log_event(
                    f"Battle started on {brain_location}: {battle_context}",
                    {"type": "battle_start", "location": brain_location},
                    state_data=state_data,
                )
            self.signal_blocker(reason="Trainer Battle", context=battle_context)
            if recovery_planner is not None and not self._recovery_tasks:
                plan = recovery_planner.generate_recovery_plan(
                    current_goal=self.get_strategic_plan_description(state_data),
                    blocker_reason="Trainer Battle",
                    blocker_context=battle_context,
                )
                print(f"💡 [ObjectiveManager] Recovery Plan: {plan['recovery_task']}")
                self.add_recovery_task(plan["recovery_task"], reason="Trainer Battle")

        elif not brain_in_battle and self._brain_prev_in_battle:
            # ── BATTLE END ──
            if episodic_memory is not None:
                episodic_memory.log_event(
                    f"Battle ended on {brain_location}. Resumed navigation.",
                    {"type": "battle_end", "location": brain_location},
                    state_data=state_data,
                )
            self.complete_recovery_task()
            self.clear_blocker()
            print(f"✅ [ObjectiveManager] Battle complete. Resuming navigation.")

        self._brain_prev_in_battle = brain_in_battle

        # ── C. Keyword-based blocker detection (non-battle) ──
        if not brain_in_battle:
            self._scan_dialogue_for_blockers(perception_output)

        # ── D. Non-battle blocker: generate recovery plan then let
        #      get_next_action_directive() execute it on the NEXT step. ──
        if not brain_in_battle and self.is_blocked:
            if recovery_planner is not None and not self._recovery_tasks:
                print("🤔 [ObjectiveManager] Thinking... Querying Memory & LLM...")
                plan = recovery_planner.generate_recovery_plan(
                    current_goal=self.get_strategic_plan_description(state_data),
                    blocker_reason="Obstacle Detected",
                    blocker_context=self._blocker_state.get("context", ""),
                )
                print(f"💡 [ObjectiveManager] Recovery Plan: {plan['recovery_task']}")
                self.add_recovery_task(plan["recovery_task"], reason="Obstacle Detected")
            # Don't short-circuit here — let the recovery task flow through
            # get_next_action_directive() so the Slow Brain's plan drives actions.
            return None

        return None  # let normal pipeline continue

    def _translate_planner_directive(self, planner_directive: Dict[str, Any], graph_location: str) -> Optional[Dict[str, Any]]:
        """
        Translate a raw NavigationPlanner directive into the goal_coords/goal_direction
        format that directive_nav.py understands.
        """
        if not planner_directive:
            return None

        action_type = planner_directive.get('action')

        if action_type == 'NAVIGATE_AND_INTERACT':
            target_coords = planner_directive['target']
            location = planner_directive.get('location', graph_location)
            should_interact = planner_directive.get('should_interact', False)
            avoid_grass = planner_directive.get('avoid_grass', True)
            return {
                'goal_coords': (*target_coords, location),
                'should_interact': should_interact,
                'avoid_grass': avoid_grass,
                'description': planner_directive.get('description', 'Navigate to coordinates'),
                'journey_progress': self.navigation_planner.get_progress_summary()
            }

        elif action_type == 'NAVIGATE_DIRECTION':
            direction = planner_directive.get('direction')
            portal_coords = planner_directive.get('portal_coords')
            return {
                'goal_direction': direction,
                'portal_coords': portal_coords,
                'description': planner_directive.get('description', f'Move {direction}'),
                'journey_progress': self.navigation_planner.get_progress_summary()
            }

        elif action_type == 'INTERACT_WARP':
            target_coords = planner_directive['target']
            location = planner_directive.get('location', graph_location)
            return {
                'goal_coords': (*target_coords, location),
                'should_interact': True,
                'description': planner_directive.get('description', 'Interact with warp'),
                'journey_progress': self.navigation_planner.get_progress_summary()
            }

        elif action_type == 'COMPLETE':
            return {
                'journey_complete': True,
                'description': 'Navigation journey complete',
                'journey_progress': self.navigation_planner.get_progress_summary()
            }

        elif action_type == 'CROSS_BOUNDARY':
            direction = planner_directive.get('direction', 'north')
            to_location = planner_directive.get('to_location', '')
            print(f"\U0001f6aa [CROSS_BOUNDARY] Crossing from {planner_directive.get('from_location')} to {to_location}")
            return {
                'goal_direction': direction,
                'description': planner_directive.get('description', f'Cross boundary {direction} to {to_location}'),
                'journey_progress': self.navigation_planner.get_progress_summary()
            }

        elif action_type == 'WAIT':
            expected_location = planner_directive.get('expected_location', '')
            print(f"\u23f3 [WAIT] Waiting for warp to {expected_location}")
            return {
                'wait_for_transition': True,
                'expected_location': expected_location,
                'description': planner_directive.get('description', f'Wait for transition to {expected_location}'),
                'journey_progress': self.navigation_planner.get_progress_summary()
            }

        elif action_type == 'UNKNOWN':
            print(f"\u2753 [UNKNOWN ACTION] NavigationPlanner returned UNKNOWN action")
            return None

        else:
            logger.warning(f"Unhandled NavigationPlanner action type: {action_type}")
            return None

    def _get_navigation_planner_directive(self, state_data: Dict[str, Any], target_location: Optional[str] = None, target_coords: Optional[tuple] = None, journey_reason: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get directive from NavigationPlanner for comparison testing.
        This runs in parallel with the existing navigation logic.
        
        Args:
            state_data: Current game state
            target_location: Target location from sequential milestone system (if provided)
            target_coords: Target coordinates from sequential milestone system (if provided)
            journey_reason: Description of the journey from sequential milestone system
        """
        # Get current position
        player_data = state_data.get('player', {})
        position = player_data.get('position', {})
        current_x = position.get('x', 0)
        current_y = position.get('y', 0)
        current_location = player_data.get('location', '').upper()
        
        # Convert location name to graph format
        # CRITICAL: Check longer/more specific names FIRST to avoid substring matches
        # e.g., "PETALBURG CITY GYM" must be checked before "PETALBURG CITY"
        location_mapping = {
            'PETALBURG CITY GYM': 'PETALBURG_CITY_GYM',  # Specific first
            'PETALBURG GYM': 'PETALBURG_CITY_GYM',
            'RUSTBORO CITY POKEMON CENTER': 'RUSTBORO_CITY_POKEMON_CENTER_1F',  # Specific first
            'RUSTBORO CITY GYM': 'RUSTBORO_CITY_GYM',
            'RUSTBORO GYM': 'RUSTBORO_CITY_GYM',
            'BIRCHS LAB': 'PROFESSOR_BIRCHS_LAB',  # Note: BIRCHS with S to match location_graph
            'BIRCH LAB': 'PROFESSOR_BIRCHS_LAB',
            'LITTLEROOT TOWN': 'LITTLEROOT_TOWN',
            'OLDALE TOWN': 'OLDALE_TOWN',
            'RUSTBORO CITY': 'RUSTBORO_CITY',
            'PETALBURG CITY': 'PETALBURG_CITY',  # General after specific
            'ROUTE 101': 'ROUTE_101',
            'ROUTE 103': 'ROUTE_103',
            'ROUTE 102': 'ROUTE_102',
            # ROUTE 104 handled specially below (needs Y coord to distinguish north/south)
            'PETALBURG WOODS': 'PETALBURG_WOODS',
            'MAP_18_0B': 'PETALBURG_WOODS',  # Raw map ID for Petalburg Woods
        }
        
        # Find matching location
        graph_location = None
        
        # Special case: Route 104 uses Y coordinate to distinguish north/south
        # South section: Y > 30 (portal to Petalburg Woods at y=38)
        # North section: Y < 30 (after exiting Petalburg Woods at y=0-29)
        if 'ROUTE 104' in current_location:
            if current_y >= 30:
                graph_location = 'ROUTE_104_SOUTH'
                logger.info(f"🗺️ [ROUTE 104] Y={current_y} >= 30 → SOUTH section")
                print(f"🗺️ [ROUTE 104] Y={current_y} >= 30 → SOUTH section")
            else:
                graph_location = 'ROUTE_104_NORTH'
                logger.info(f"🗺️ [ROUTE 104] Y={current_y} < 30 → NORTH section")
                print(f"🗺️ [ROUTE 104] Y={current_y} < 30 → NORTH section")
        else:
            # Standard location mapping
            for loc_key, loc_value in location_mapping.items():
                if loc_key in current_location:
                    graph_location = loc_value
                    break
        
        # DEBUG: Log location matching attempt
        if not graph_location:
            logger.warning(f"⚠️ [LOCATION MAPPING] Failed to map location '{current_location}' to graph")
            print(f"⚠️ [LOCATION MAPPING] Unknown location: '{current_location}'")
            print(f"   Available mappings: {list(location_mapping.keys())}")
        else:
            logger.info(f"✅ [LOCATION MAPPING] '{current_location}' → '{graph_location}'")
            print(f"✅ [LOCATION MAPPING] '{current_location}' → '{graph_location}'")
        
        if not graph_location:
            # Unknown location - can't use planner
            return {
                'action': 'UNKNOWN_LOCATION',
                'description': f'Location "{current_location}" not in navigation graph',
                'error': True
            }
        
        # Detect if we changed location (planner might auto-advance)
        location_changed = (graph_location != self._last_planner_location)
        coords_changed = ((current_x, current_y) != self._last_planner_coords)
        
        self._last_planner_location = graph_location
        self._last_planner_coords = (current_x, current_y)
        
        # Get milestones for determining target
        milestones = state_data.get('milestones', {})
        
        def is_milestone_complete(milestone_id: str) -> bool:
            milestone_data = milestones.get(milestone_id, {})
            return milestone_data.get('completed', False) if isinstance(milestone_data, dict) else False
        
        # =====================================================================
        # SIMPLIFIED PLANNER LOGIC
        # =====================================================================
        # The sequential MILESTONE_PROGRESSION system provides target_location
        # and target_coords. We just use them directly - no fallback needed!
        # =====================================================================
        
        if target_location is None:
            # No target means milestone will auto-complete (dialogue, events, etc.)
            logger.info(f"📍 [PLANNER] No target_location - milestone will auto-complete")
            return None
        
        # DEBUG: Log what target was provided by sequential system
        logger.info(f"🔍 [TARGET DEBUG] target_location={target_location}, target_coords={target_coords}, graph_location={graph_location}")
        print(f"🔍 [TARGET DEBUG] target_location={target_location}, target_coords={target_coords}, graph_location={graph_location}")
        
        # =====================================================================
        # SUB-GOAL: ROUTE 104 NORTH BRIDGE WAYPOINT (Avoid NPCs at 27,15 and 28,15)
        # =====================================================================
        # ROUTE 104 NORTH: Navigate around NPCs on bridge using waypoint system
        # NPCs at (27,15) and (28,15) block the direct path north
        # Similar to Route 104 South NPC avoidance, use waypoint to route around them
        # 
        # Trigger zone: x=27-33, y=15-26 (bridge area with NPCs)
        # Waypoint: (26,16) - west of bridge to avoid NPC dialogue zones
        # =====================================================================
        in_route_104_north = graph_location == 'ROUTE_104_NORTH'
        
        if in_route_104_north:
            # Get current position
            position = player_data.get('position', {})
            current_x, current_y = position.get('x', 0), position.get('y', 0)
            
            # Trigger zone: X between 27-33 AND Y between 15-26 (bridge area)
            in_bridge_zone = (27 <= current_x <= 33) and (15 <= current_y <= 26)
            
            if in_bridge_zone:
                BRIDGE_WAYPOINT = (26, 16)
                
                logger.info(f"🌉 [ROUTE 104 BRIDGE] Player at ({current_x}, {current_y}) in bridge zone")
                logger.info(f"🌉 [ROUTE 104 BRIDGE] NPCs at (27,15) and (28,15) - routing to waypoint {BRIDGE_WAYPOINT}")
                print(f"🌉 [ROUTE 104 BRIDGE] Avoiding NPCs - navigating to waypoint {BRIDGE_WAYPOINT}")
                
                current_pos = (current_x, current_y)
                
                # If not at waypoint, navigate to it
                if current_pos != BRIDGE_WAYPOINT:
                    return {
                        'goal_coords': (26, 16, 'ROUTE_104_NORTH'),
                        'description': 'Navigate to (26,16) to avoid bridge NPCs at (27,15) and (28,15)',
                        'avoid_grass': True  # Standard pathfinding to waypoint
                    }
                else:
                    # At waypoint - continue to Rustboro entrance
                    logger.info(f"✅ [ROUTE 104 BRIDGE] Reached waypoint {BRIDGE_WAYPOINT} - continuing to Rustboro")
                    print(f"✅ [ROUTE 104 BRIDGE] Waypoint reached - continuing north to Rustboro City")
            
            # OLD WAYPOINT SYSTEM (kept for reference, disabled)
            # Waypoint 1: Lower-left area - guide east to avoid dead-end
            # in_waypoint1_zone = (2 <= current_x <= 18) and (19 <= current_y <= 29)
            # if in_waypoint1_zone:
            #     return {'goal_coords': (19, 22, 'ROUTE_104_NORTH'), ...}
        
        # If we have a target, plan/update journey
        if target_location:
            # Check if target is same location (intra-location navigation to coords)
            if target_location == graph_location and target_coords:
                # Same location - just navigate to coordinates
                # Don't use planner, return simple goal_coords directive
                logger.info(f"🎯 [OBJECTIVE] Intra-location navigation to {target_coords} in {graph_location}")
                print(f"🎯 [OBJECTIVE] Navigating to {target_coords} in current location")
                
                return {
                    'goal_coords': (*target_coords, graph_location),
                    'should_interact': True,  # Interact with rival/NPC
                    'description': journey_reason or f"Navigate to {target_coords}"
                }
            
            # Different location - use planner for multi-hop journey
            elif target_location != graph_location:
                # Check if we need to create a new plan
                if not self.navigation_planner.has_active_plan():
                    success = self.navigation_planner.plan_journey(
                        start_location=graph_location,
                        end_location=target_location,
                        final_coords=target_coords
                    )
                    if success:
                        print(f"\n{'=' * 80}")
                        print(f"🗺️ [NAV PLANNER] NEW JOURNEY PLANNED")
                        print(f"{'=' * 80}")
                        print(f"Reason: {journey_reason}")
                        print(f"Start: {graph_location}")
                        print(f"End: {target_location}")
                        if target_coords:
                            print(f"Final Target: {target_coords}")
                        print(f"Total Stages: {len(self.navigation_planner.stages)}")
                        print(f"{'=' * 80}\n")
                    else:
                        return {
                            'action': 'PLAN_FAILED',
                            'description': f'Failed to plan journey from {graph_location} to {target_location}',
                            'error': True
                        }
                elif self.navigation_planner.journey_start != graph_location:
                    # Location changed unexpectedly (agent wandered or warped) - replan from current location
                    print(f"\n⚠️ [NAV PLANNER] Location changed unexpectedly: planned start was {self.navigation_planner.journey_start}, now at {graph_location}")
                    print(f"Replanning journey from current location...\n")
                    self.navigation_planner.clear_plan()
                    # Recursive call with same parameters
                    return self._get_navigation_planner_directive(state_data, target_location, target_coords, journey_reason)
                elif self.navigation_planner.journey_end != target_location:
                    # Journey target changed - replan
                    print(f"\n⚠️ [NAV PLANNER] Target changed from {self.navigation_planner.journey_end} to {target_location}")
                    print(f"Replanning journey...\n")
                    self.navigation_planner.clear_plan()
                    # Recursive call with same parameters
                    return self._get_navigation_planner_directive(state_data, target_location, target_coords, journey_reason)
        
        # Get current directive from planner
        if self.navigation_planner.has_active_plan():
            # Get the raw planner directive (with action types like NAVIGATE, CROSS_BOUNDARY, etc.)
            planner_directive = self.navigation_planner.get_current_directive(
                graph_location,
                (current_x, current_y)
            )
            
            if not planner_directive:
                return None
            
            # TRANSLATION LAYER: Convert planner's stage-based directives into simple GOAL COORDINATES
            # The planner tells us WHERE to go, action.py decides HOW to get there
            return self._translate_planner_directive(planner_directive, graph_location)
        else:
            # No active plan - agent is at destination or unknown state
            return None
    
    def compare_navigation_systems(self, state_data: Dict[str, Any]):
        """
        Compare old directive system with new NavigationPlanner.
        Prints detailed comparison for analysis.
        """
        print(f"\n{'█' * 80}")
        print(f"{'█' * 80}")
        print(f"🔍 NAVIGATION COMPARISON")
        print(f"{'█' * 80}")
        print(f"{'█' * 80}\n")
        
        # Get current position
        player_data = state_data.get('player', {})
        position = player_data.get('position', {})
        current_x = position.get('x', 0)
        current_y = position.get('y', 0)
        current_location = player_data.get('location', '').upper()
        
        print(f"📍 Current Position: ({current_x}, {current_y}) in {current_location}")
        
        # Get current milestone status
        milestones = state_data.get('milestones', {})
        active_milestones = [k for k, v in milestones.items() if isinstance(v, dict) and v.get('completed', False)]
        print(f"✅ Active Milestones: {', '.join(active_milestones) if active_milestones else 'None'}")
        
        print(f"\n{'-' * 80}")
        print(f"OLD SYSTEM (get_next_action_directive)")
        print(f"{'-' * 80}\n")
        
        old_directive = self.get_next_action_directive(state_data)
        if old_directive:
            print(f"Action: {old_directive.get('action')}")
            print(f"Target: {old_directive.get('target')}")
            print(f"Description: {old_directive.get('description')}")
            print(f"Milestone: {old_directive.get('milestone')}")
            if 'direction' in old_directive:
                print(f"Direction: {old_directive.get('direction')}")
                print(f"Target Location: {old_directive.get('target_location')}")
                print(f"Portal Coords: {old_directive.get('portal_coords')}")
        else:
            print("No directive (None)")
        
        print(f"\n{'-' * 80}")
        print(f"NEW SYSTEM (NavigationPlanner)")
        print(f"{'-' * 80}\n")
        
        new_directive = self._get_navigation_planner_directive(state_data)
        if new_directive:
            is_error = new_directive.get('error', False)
            is_at_dest = new_directive.get('at_destination', False)
            
            if is_error:
                print(f"❌ ERROR: {new_directive.get('description')}")
            elif is_at_dest:
                print(f"🎯 {new_directive.get('description')}")
            else:
                print(f"Action: {new_directive.get('action')}")
                if 'target' in new_directive and new_directive['target']:
                    print(f"Target: {new_directive.get('target')}")
                print(f"Description: {new_directive.get('description')}")
                
                # Show stage progress
                stage_idx = new_directive.get('stage_index', 0)
                total_stages = new_directive.get('total_stages', 0)
                if total_stages > 0:
                    print(f"Progress: Stage {stage_idx + 1}/{total_stages}")
                    
                # Show journey progress
                journey_progress = new_directive.get('journey_progress')
                if journey_progress:
                    print(f"Journey: {journey_progress}")
        else:
            print("No directive (None)")
        
        print(f"\n{'-' * 80}")
        print(f"COMPARISON ANALYSIS")
        print(f"{'-' * 80}\n")
        
        # Compare actions
        old_action = old_directive.get('action') if old_directive else None
        new_action = new_directive.get('action') if new_directive else None
        
        if old_action == new_action:
            print(f"✅ Actions MATCH: Both systems suggest '{old_action}'")
        else:
            print(f"⚠️ Actions DIFFER:")
            print(f"   Old: {old_action}")
            print(f"   New: {new_action}")
        
        # Compare targets
        old_target = old_directive.get('target') if old_directive else None
        new_target = new_directive.get('target') if new_directive else None
        
        if old_target and new_target:
            if old_target == new_target:
                print(f"✅ Targets MATCH: Both point to {old_target}")
            else:
                print(f"⚠️ Targets DIFFER:")
                print(f"   Old: {old_target}")
                print(f"   New: {new_target}")
        elif old_target or new_target:
            print(f"⚠️ One system has target, other doesn't:")
            print(f"   Old: {old_target}")
            print(f"   New: {new_target}")
        
        # Compare descriptions
        old_desc = old_directive.get('description') if old_directive else None
        new_desc = new_directive.get('description') if new_directive else None
        
        if old_desc and new_desc:
            print(f"\n📝 Description Comparison:")
            print(f"   Old: {old_desc}")
            print(f"   New: {new_desc}")
        
        print(f"\n{'█' * 80}")
        print(f"{'█' * 80}\n")