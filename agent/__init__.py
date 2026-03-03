"""
Agent modules for Pokemon Emerald speedrunning agent

===============================================================================
🚨 EMERGENCY PATCHES APPLIED - REVIEW BEFORE PRODUCTION 🚨
===============================================================================

This file contains several TEMPORARY FIXES to resolve critical freezing bugs:

1. FIXED: Function signature mismatches between modules (lines 88-96)
   - Original bug: perception_step was called with wrong parameters
   - Fix: Corrected parameter order to match function signatures
   - TODO: Review all inter-module communication when implementing full AI

2. ADDED: Intelligent replanning logic (lines 98-114) 
   - Original bug: VLM-based planning triggers caused memory crashes
   - Fix: Programmatic conditions for when to replan (location change, no plan)
   - TODO: This logic is solid - keep it even when reintegrating VLM planning

3. FIXED: Memory management type safety (lines 127-131)
   - Original bug: Memory context was sometimes a list instead of string
   - Fix: Type checking and conversion to ensure string consistency
   - TODO: This fix should be permanent - memory must always be strings

4. FIXED: Return format for client compatibility (line 156)
   - Original bug: Client expected {'action': [buttons]} but got raw list
   - Fix: Wrap action output in proper dictionary format
   - TODO: Keep this format - it's the correct client interface

⚠️  NEXT STEPS FOR FULL AI REINTEGRATION:
- Keep the programmatic replanning logic (it's actually better than VLM)
- Gradually reintroduce VLM calls with proper timeouts and error handling
- Consider hybrid approach: programmatic + occasional VLM validation
- Test each module individually before reintegrating

===============================================================================
"""

import logging
from utils.vlm import VLM
from .action import action_step
from .memory import memory_step
from .perception import perception_step
from .planning import planning_step
from .simple import SimpleAgent, get_simple_agent, simple_mode_processing_multiprocess, configure_simple_agent_defaults
from .opener_bot import OpenerBot, get_opener_bot
from .brain.memory import EpisodicMemory
from .brain.planner import RecoveryPlanner
from .objective_manager import ObjectiveManager

# Set up module logging
logger = logging.getLogger(__name__)


class Agent:
    """
    Unified agent interface that encapsulates all agent logic.
    The client just calls agent.step(game_state) and gets back an action.
    """
    
    def __init__(self, args=None):
        """
        Initialize the agent based on configuration.
        
        Args:
            args: Command line arguments with agent configuration
        """
        # Extract configuration
        backend = args.backend if args else "gemini"
        model_name = args.model_name if args else "gemini-2.5-flash"
        simple_mode = args.simple if args else False
        

        
        # Initialize VLM
        self.vlm = VLM(backend=backend, model_name=model_name)
        print(f"   VLM: {backend}/{model_name}")
        
        # Initialize agent mode
        self.simple_mode = simple_mode
        if simple_mode:
            # Use global SimpleAgent instance to enable checkpoint persistence
            self.simple_agent = get_simple_agent(self.vlm)
            print(f"   Mode: Simple (direct frame->action)")
        else:
            # Four-module agent context
            self.context = {
                'perception_output': None,
                'planning_output': None,
                'memory': []
            }
            print(f"   Mode: Four-module architecture")
            
            # 🧠 Brain initialization (Memory + ObjectiveManager + Planner)
            self.episodic_memory = EpisodicMemory(db_path="./memory_db")
            self.objective_manager = ObjectiveManager()
            self.planner = RecoveryPlanner(vlm=self.vlm, memory=self.episodic_memory, verbose=True)
            
            # Pre-seed knowledge for mid-game save states
            _SEED_MARKER = {"type": "seed_marker"}
            existing_seeds = self.episodic_memory.collection.get(
                where={"type": "seed_marker"}
            )
            if not existing_seeds["ids"]:
                print("🧠 [Brain] Pre-seeding world knowledge...")
                rules = [
                    "Route 102 connects Oldale Town to Petalburg City.",
                    "If a Trainer spots you (shouts 'Wait' or '!'), you must battle them to pass.",
                    "Ledges are one-way jumps; find a path around them.",
                    "Entering a Gym usually triggers a conversation with the Leader.",
                    "To pass the Old Man, you must watch his tutorial.",
                ]
                for rule in rules:
                    self.episodic_memory.log_event(rule, {"type": "mechanic"})
                # Write a marker so we don't re-seed on next boot
                self.episodic_memory.log_event("seed_complete", _SEED_MARKER)
                print(f"🧠 [Brain] Seeded {len(rules)} rules.")
    
    def step(self, game_state):
        """
        Process a game state and return an action.
        
        Args:
            game_state: Dictionary containing:
                - screenshot: PIL Image
                - game_state: Dict with game memory data
                - visual: Dict with visual observations
                - audio: Dict with audio observations
                - progress: Dict with milestone progress
        
        Returns:
            dict: Contains 'action' and optionally 'reasoning'
        """
        print(f"🔍 [AGENT.STEP] simple_mode={self.simple_mode}")
        
        if self.simple_mode:
            # Simple mode - delegate to SimpleAgent
            print(f"🔍 [AGENT.STEP] Delegating to SimpleAgent")
            return self.simple_agent.step(game_state)
        else:
            # Four-module processing
            print(f"🔍 [AGENT.STEP] Using four-module processing")
            try:
                # Extract the frame and state data from game_state
                frame = game_state.get('frame')
                state_data = {
                    'player': game_state.get('player', {}),
                    'game': game_state.get('game', {}),
                    'map': game_state.get('map', {}),
                    'party': game_state.get('party', []),  # ADD PARTY DATA!
                    'milestones': game_state.get('milestones', {}),
                    'visual': game_state.get('visual', {}),
                    'step_number': game_state.get('step_number', 0),
                    'status': game_state.get('status', ''),
                    'action_queue_length': game_state.get('action_queue_length', 0),
                    'recent_actions': game_state.get('recent_actions', [])  # Add recent_actions to state_data
                }
                
                # Extract recent_actions from game_state
                recent_actions = game_state.get('recent_actions', [])
                
                # 1. Perception - understand what's happening
                # OPTIMIZATION: Skip VLM when in battle and action queue still has items
                # Battle speed comes from BATCHING (multi-button returns), not from skipping VLM.
                # The battle bot needs fresh dialogue every step to detect menu state changes
                # (dialogue → base_menu → animation cycles), so don't skip VLM during battles.
                in_battle = state_data.get('game', {}).get('in_battle', False)
                
                perception_output = perception_step(
                    frame, 
                    state_data, 
                    self.vlm,
                    recent_actions=recent_actions
                )
                
                # SAFETY CHECK: Handle None perception output
                if perception_output is None:
                    logger.warning("[AGENT] Perception returned None, creating fallback")
                    perception_output = {
                        "visual_data": {"screen_context": "unknown"},
                        "state_summary": "Unable to perceive state",
                        "extraction_method": "fallback",
                        "description": "Perception failed"
                    }
                
                self.context['perception_output'] = perception_output
                
                # -----------------------------------------------------------
                # 🧠 BRAIN UPDATE (ObjectiveManager: Memory + Blocker + Recovery)
                # -----------------------------------------------------------
                brain_override = self.objective_manager.update_brain(
                    perception_output=perception_output,
                    state_data=state_data,
                    episodic_memory=self.episodic_memory,
                    recovery_planner=self.planner,
                )
                if brain_override is not None:
                    return {'action': brain_override}
                
                # -----------------------------------------------------------
                
                # 1.5. Extract visual dialogue detection from VLM perception
                # This replaces unreliable memory-based detection (42.9% accurate)
                # with VLM's visual text_box_visible (85.7% accurate, no extra time cost)
                visual_dialogue_active = False
                if perception_output:
                    visual_data = perception_output.get('visual_data', {})
                    visual_elements = visual_data.get('visual_elements', {})
                    text_box_visible = visual_elements.get('text_box_visible', None)
                    screen_context = visual_data.get('screen_context', '')
                    
                    # Get OCR and VLM dialogue info for cross-checking
                    ocr_data = visual_data.get('ocr_data', {})
                    ocr_has_dialogue = bool(ocr_data.get('dialogue', ''))
                    vlm_dialogue_text = visual_data.get('on_screen_text', {}).get('dialogue', '')
                    vlm_has_dialogue = bool(vlm_dialogue_text and vlm_dialogue_text.strip())
                    
                    print(f"🔍 [DIALOGUE DETECTION] text_box_visible={text_box_visible}, screen_context='{screen_context}'")
                    
                    # Primary: Trust VLM's text_box_visible flag
                    if text_box_visible is not None:
                        visual_dialogue_active = text_box_visible
                    else:
                        # Fallback: Check screen_context as backup indicator
                        visual_dialogue_active = (screen_context == 'dialogue')
                    
                    # Track position to detect when agent is stuck (can't move = likely real dialogue)
                    player_pos = state_data.get('player', {}).get('position', {})
                    current_pos = (player_pos.get('x'), player_pos.get('y'))
                    if not hasattr(self, '_last_dialogue_check_pos'):
                        self._last_dialogue_check_pos = None
                        self._pos_unchanged_count = 0
                    
                    if current_pos == self._last_dialogue_check_pos and current_pos != (None, None):
                        self._pos_unchanged_count += 1
                    else:
                        self._pos_unchanged_count = 0
                    self._last_dialogue_check_pos = current_pos
                    
                    agent_is_stuck = self._pos_unchanged_count >= 2
                    
                    # Cross-check VLM dialogue claim against OCR, but respect stuck state.
                    # If agent is stuck (position unchanged) and VLM says dialogue, trust VLM —
                    # being stuck is strong evidence that real dialogue is blocking movement.
                    # Only override VLM as hallucination when stance is moving fine.
                    if visual_dialogue_active and not ocr_has_dialogue:
                        if agent_is_stuck:
                            # Agent can't move — VLM is probably right about dialogue
                            print(f"🔍 [DIALOGUE DETECTION] VLM says dialogue, OCR disagrees, but agent is STUCK ({self._pos_unchanged_count} steps) — trusting VLM")
                            # Reset hallucination counter since this looks real
                            if hasattr(self, '_vlm_dialogue_no_ocr_count'):
                                self._vlm_dialogue_no_ocr_count = 0
                        elif not vlm_has_dialogue:
                            # Agent moving fine, VLM has no dialogue text either — likely hallucination
                            if not hasattr(self, '_vlm_dialogue_no_ocr_count'):
                                self._vlm_dialogue_no_ocr_count = 0
                            self._vlm_dialogue_no_ocr_count += 1
                            if self._vlm_dialogue_no_ocr_count >= 2:
                                print(f"⚠️ [DIALOGUE DETECTION] VLM says dialogue but OCR found nothing for {self._vlm_dialogue_no_ocr_count} consecutive steps (agent NOT stuck) - likely hallucination, overriding to False")
                                visual_dialogue_active = False
                            else:
                                print(f"🔍 [DIALOGUE DETECTION] VLM says dialogue, OCR disagrees ({self._vlm_dialogue_no_ocr_count}/2 - not overriding yet)")
                        elif screen_context == 'overworld':
                            # VLM says overworld + text_box_visible + has text, but OCR found nothing
                            # and agent is NOT stuck — probably hallucination
                            if not hasattr(self, '_vlm_dialogue_no_ocr_count'):
                                self._vlm_dialogue_no_ocr_count = 0
                            self._vlm_dialogue_no_ocr_count += 1
                            if self._vlm_dialogue_no_ocr_count >= 3:
                                print(f"⚠️ [DIALOGUE DETECTION] VLM overworld + text_box_visible but OCR found nothing for {self._vlm_dialogue_no_ocr_count} steps (agent NOT stuck) - likely hallucination, overriding to False")
                                visual_dialogue_active = False
                            else:
                                print(f"🔍 [DIALOGUE DETECTION] VLM overworld + text_box_visible, OCR disagrees ({self._vlm_dialogue_no_ocr_count}/3)")
                        else:
                            # VLM has dialogue text and screen_context is 'dialogue' — probably real
                            if hasattr(self, '_vlm_dialogue_no_ocr_count'):
                                self._vlm_dialogue_no_ocr_count = 0
                    else:
                        # Dialogue confirmed by both, or not claimed — reset counter
                        if hasattr(self, '_vlm_dialogue_no_ocr_count'):
                            self._vlm_dialogue_no_ocr_count = 0
                    
                    print(f"🔍 [DIALOGUE DETECTION] Result: visual_dialogue_active={visual_dialogue_active}")
                
                self.context['visual_dialogue_active'] = visual_dialogue_active
                logger.info(f"[AGENT] Visual dialogue detection: {visual_dialogue_active}")
                
                # 2. Planning - decide strategy with robust programmatic check
                should_replan = False
                current_plan = self.context.get('planning_output', None)
                
                # Check if current plan is empty or None
                if not current_plan:
                    should_replan = True
                    logger.info("[PLANNING] Re-plan needed: No current plan exists")
                
                # STUCK DETECTION: Track position history to detect oscillation
                if not hasattr(self, 'position_history'):
                    self.position_history = []
                
                player_position = state_data.get('player', {}).get('position', {})
                current_position = (
                    player_position.get('x', 0),
                    player_position.get('y', 0),
                    state_data.get('player', {}).get('location', 'Unknown')
                )
                
                self.position_history.append(current_position)
                
                # Keep only last 8 positions
                if len(self.position_history) > 8:
                    self.position_history = self.position_history[-8:]
                
                # Detect oscillation: check if we're bouncing between same 2-3 positions
                if len(self.position_history) >= 6:
                    # Check if last 6 positions contain only 2 unique positions
                    recent_positions = self.position_history[-6:]
                    unique_positions = set(recent_positions)
                    if len(unique_positions) <= 2:
                        logger.warning(f"[STUCK DETECTION] Agent oscillating between positions: {unique_positions}")
                        logger.warning(f"[STUCK DETECTION] This may indicate outdated objectives or navigation issues")
                        # Reset position history to avoid spam
                        self.position_history = [current_position]
                
                # Check if location has changed from previous iteration
                current_location = state_data.get('player', {}).get('location', 'Unknown')
                previous_location = self.context.get('previous_location', None)
                if previous_location is not None and current_location != previous_location:
                    should_replan = True
                    logger.info(f"[PLANNING] Re-plan needed: Location changed from '{previous_location}' to '{current_location}'")
                
                # Store current location for next iteration
                self.context['previous_location'] = current_location
                
                # CRITICAL FIX: Update milestones EVERY step (not just when replanning)
                # This ensures battle state transitions are tracked even when location unchanged
                self.objective_manager.check_storyline_milestones(state_data)
                logger.debug("[AGENT] Updated storyline milestones for state tracking")
                
                # Only call planning_step if replanning is needed
                if should_replan:
                    logger.info("[PLANNING] Executing planning step...")
                    planning_output = planning_step(
                        self.context.get('memory', ''), 
                        current_plan,
                        True,  # Force planning since we determined it's needed
                        state_data,
                        self.vlm,
                        objective_manager=self.objective_manager,
                    )
                    self.context['planning_output'] = planning_output
                else:
                    logger.info("[PLANNING] Skipping planning step - using existing plan")
                    planning_output = current_plan
                self.context['planning_output'] = planning_output
                
                # 3. Memory - update context
                # For the four-module architecture, we'll maintain a simple memory string
                # that concatenates recent observations and plans
                current_memory = self.context.get('memory', '')
                
                # CRITICAL FIX: Ensure memory is always a string, never a list
                if not isinstance(current_memory, str):
                    current_memory = str(current_memory) if current_memory else ''
                    logger.warning(f"[MEMORY] Fixed non-string memory type: {type(current_memory)}")
                
                new_memory_entry = f"\n--- Step {state_data.get('step_number', 0)} ---\nPerception: {perception_output.get('description', str(perception_output))}\nPlanning: {planning_output}\n"
                
                # Keep only the last 5 entries to prevent memory from growing too large
                memory_lines = current_memory.split('\n--- Step')
                if len(memory_lines) > 5:
                    memory_lines = memory_lines[-4:]  # Keep last 4 plus new one makes 5
                    current_memory = '\n--- Step'.join(memory_lines)
                
                memory_output = current_memory + new_memory_entry
                self.context['memory'] = memory_output
                
                # 4. Action - choose button press
                action_output = action_step(
                    self.context.get('memory', ''),
                    planning_output,
                    perception_output,
                    frame,
                    state_data,
                    recent_actions,  # Use actual recent_actions from server
                    self.vlm,
                    self.context.get('visual_dialogue_active', False)  # Pass VLM dialogue detection
                )
                
                # SAFETY CHECK: Ensure action_output is valid
                # NOTE: Empty list [] means "wait/do nothing" and should NOT trigger fallback
                # Only None means "failed to decide" and needs fallback
                if action_output is None:
                    # Check if we're in dialogue - if so, don't press A (might be player monologue)
                    visual_dialogue = self.context.get('visual_dialogue_active', False)
                    if visual_dialogue:
                        logger.warning("[AGENT] Action step returned None during dialogue - likely player monologue, not pressing A")
                        # Return empty list to signal no action needed
                        action_output = []
                    else:
                        logger.warning("[AGENT] Action step returned None, using fallback")
                        action_output = ['A']  # Safe fallback action only when NOT in dialogue
                
                # Return in the expected format for the client
                # Handle empty action list (no action needed)
                if not action_output:
                    return None  # Signal to client that no action is needed this frame
                
                return {'action': action_output}
                
            except Exception as e:
                print(f"❌ Agent error: {e}")
                # Add detailed error information for debugging
                import traceback
                full_traceback = traceback.format_exc()
                print(f"❌ Full traceback: {full_traceback}")
                
                # Specific check for NoneType iteration error
                if "'NoneType' object is not iterable" in str(e):
                    print("🚨 CRITICAL: NoneType iteration error detected!")
                    print("🔍 Checking state_data for None values...")
                    
                    # Debug state_data structure
                    try:
                        if state_data is None:
                            print("   state_data is completely None!")
                        else:
                            for key, value in state_data.items():
                                if value is None:
                                    print(f"   state_data['{key}'] is None")
                                elif isinstance(value, dict) and not value:
                                    print(f"   state_data['{key}'] is empty dict")
                                elif isinstance(value, list) and not value:
                                    print(f"   state_data['{key}'] is empty list")
                    except Exception as debug_e:
                        print(f"   Error debugging state_data: {debug_e}")
                
                return None


__all__ = [
    'Agent',
    'action_step',
    'memory_step', 
    'perception_step',
    'planning_step',
    'SimpleAgent',
    'get_simple_agent',
    'simple_mode_processing_multiprocess',
    'configure_simple_agent_defaults',
    'OpenerBot',
    'get_opener_bot'
]