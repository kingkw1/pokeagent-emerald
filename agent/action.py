import logging
from agent.system_prompt import system_prompt
from agent.opener_bot import get_opener_bot
from agent.battle_bot import get_battle_bot

# === NEW MODULE IMPORTS (Phase 2 refactor) ===
# Stuck detection, warp handling, and post-dialogue limiting
from agent import stuck_handler

# Set up module logging
logger = logging.getLogger(__name__)

def format_observation_for_action(observation):
    """Format observation data for use in action prompts"""
    if isinstance(observation, dict) and 'visual_data' in observation:
        # Structured format - provide a clean summary for action decision
        visual_data = observation['visual_data']
        summary = f"Screen: {visual_data.get('screen_context', 'unknown')}"
        
        # Add key text information
        on_screen_text = visual_data.get('on_screen_text', {})
        if on_screen_text.get('dialogue'):
            summary += f" | Dialogue: \"{on_screen_text['dialogue']}\""
        if on_screen_text.get('menu_title'):
            summary += f" | Menu: {on_screen_text['menu_title']}"
            
        # Add entity information - handle various entity formats
        entities = visual_data.get('visible_entities', [])
        if entities:
            try:
                entity_names = []
                if isinstance(entities, list):
                    for e in entities[:3]:  # Limit to first 3
                        if isinstance(e, dict):
                            entity_names.append(e.get('name', 'unnamed'))
                        elif isinstance(e, str):
                            entity_names.append(e)
                        else:
                            entity_names.append(str(e))
                elif isinstance(entities, str):
                    entity_names = [entities]
                elif isinstance(entities, dict):
                    # Handle case where entities is a dict with keys like NPC, Pokemon
                    for key, value in entities.items():
                        if value and value != "none" and value != "null":
                            entity_names.append(f"{key}: {value}")
                
                if entity_names:
                    summary += f" | Entities: {', '.join(entity_names[:3])}"  # Limit display
            except Exception as e:
                # Fallback if entity processing fails
                summary += f" | Entities: {str(entities)[:50]}"
            
        return summary
    else:
        # Original text format or non-structured data
        return str(observation)


def action_step(memory_context, current_plan, latest_observation, frame, state_data, recent_actions, vlm, visual_dialogue_active=False, objective_manager=None):
    """
    Decide and perform the next action button(s) based on memory, plan, observation, and comprehensive state.
    Returns a list of action buttons as strings.
    
    Args:
        memory_context: Recent memory/history
        current_plan: Current plan from planning module
        latest_observation: Perception output with visual data
        frame: Current screenshot
        state_data: Game state data
        recent_actions: List of recent actions taken
        vlm: VLM instance for action decisions
        visual_dialogue_active: VLM's visual detection of dialogue box (85.7% accurate, no time cost)
        objective_manager: ObjectiveManager instance (passed directly from Agent)
    """
    logger.debug("[ACTION_STEP] Called")
    
    # Decrement TTLs for dynamically blocked tiles
    stuck_handler.decrement_blocked_tile_ttls()

    # Extract position for use throughout action_step
    try:
        player_data = state_data.get('player', {})
        position = player_data.get('position', {})
        current_x = position.get('x')
        current_y = position.get('y')
        location = player_data.get('location', '')

        # Track position and detect warps
        stuck_handler.update_position_tracking(current_x, current_y, location)
    except Exception as e:
        print(f"⚠️ [POSITION TRACKING] Error tracking position: {e}")
        current_x = None
        current_y = None
        location = ''

    # Stuck detection - check if agent is stuck and get recovery action
    stuck_result = stuck_handler.check_stuck(state_data, recent_actions, visual_dialogue_active, current_x, current_y, location)
    if stuck_result is not None:
        return stuck_result

    # ⚔️ PRIORITY 0A: BATTLE BOT - Combat State Machine (HIGHEST PRIORITY)
    # This MUST be checked BEFORE opener bot to prevent navigation commands during battles.
    # The battle bot handles all combat encounters with rule-based move selection.
    # Returns symbolic decisions (e.g., "BATTLE_FIGHT") which are mapped to button presses.
    try:
        battle_bot = get_battle_bot()
        
        if battle_bot.should_handle(state_data):
            print(f"⚔️ [PRIORITY] BATTLE_BOT is handling (objective_manager will NOT be consulted)")
            logger.info(f"⚔️ [PRIORITY] BATTLE_BOT is handling")
            
            # Add latest_observation to state_data so battle_bot can access dialogue text
            state_data['latest_observation'] = latest_observation
            
            battle_decision = battle_bot.get_action(state_data)
            
            if battle_decision is not None:
                logger.info(f"⚔️ [BATTLE BOT] Battle active, decision: {battle_decision}")
                
                # Map symbolic decision to button press
                # Multi-step sequences (like menu navigation) happen across multiple frames
                button_recommendation = None
                decision_explanation = ""
                
                if battle_decision == "ADVANCE_BATTLE_DIALOGUE":
                    # Battle dialogue - send two B presses to advance through dialogue faster
                    # Extra B during battle is harmless (B on base_menu does nothing)
                    decision_explanation = "Advance battle dialogue (batched)"
                    logger.info("💬 [BATTLE BOT] Recommending B×2 to advance dialogue")
                    return ['B', 'B']
                
                elif battle_decision == "RECOVER_FROM_RUN_FAILURE":
                    # We tried to run from a trainer battle - press B to dismiss message
                    button_recommendation = "B"
                    decision_explanation = "Dismiss 'no running from trainer' message"
                    logger.info("⚠️ [BATTLE BOT ERROR RECOVERY] Recommending B to dismiss message")
                    print("⚠️ [BATTLE BOT] RECOVERY: Recommending B")
                
                elif battle_decision == "SELECT_RUN":
                    # SELECT_RUN means: press A to confirm RUN selection
                    # (cursor should already be on RUN from previous RIGHT presses)
                    button_recommendation = "A"
                    decision_explanation = "Press A to confirm RUN and flee from wild battle"
                    logger.info("🏃 [BATTLE BOT] Recommending A to confirm RUN")
                
                elif battle_decision == "VLM_SELECT_RUN":
                    # Navigate to RUN option deterministically (batched)
                    # Pokemon battle menu is a 2x2 grid:
                    #   FIGHT  | BAG
                    #   POKEMON| RUN
                    # Cursor starts on FIGHT. Navigate: DOWN → RIGHT → A
                    decision_explanation = "Navigating to RUN: DOWN→RIGHT→A (batched)"
                    logger.info("🏃 [BATTLE BOT] RUN nav: DOWN→RIGHT→A (batched)")
                    print("🏃 [BATTLE BOT] RUN nav: DOWN→RIGHT→A (batched)")
                    return ['DOWN', 'RIGHT', 'A']
                
                elif battle_decision == "PRESS_RIGHT":
                    # Move cursor toward RUN option
                    button_recommendation = "RIGHT"
                    decision_explanation = "Navigate to RUN option in battle menu"
                    logger.info("🏃 [BATTLE BOT] Recommending RIGHT to navigate to RUN")
                
                elif battle_decision == "SELECT_FIGHT":
                    # Select FIGHT from base menu (just press A, it's the default)
                    button_recommendation = "A"
                    decision_explanation = "Select FIGHT from battle menu (default option)"
                    logger.info("⚔️ [BATTLE BOT] Recommending A to select FIGHT")
                
                elif battle_decision == "USE_MOVE_ABSORB":
                    # Battle menu navigation: A (select FIGHT) → DOWN (select ABSORB) → A (confirm)
                    # Full sequence batched for efficiency
                    decision_explanation = "Select FIGHT → ABSORB: A→DOWN→A (batched)"
                    logger.info("🌿 [BATTLE BOT] ABSORB: A→DOWN→A (batched)")
                    print("🌿 [BATTLE BOT] ABSORB: A→DOWN→A (batched)")
                    return ['A', 'DOWN', 'A']
                
                elif battle_decision == "USE_MOVE_POUND":
                    # Battle menu navigation: A (select FIGHT) → UP (select POUND) → A (confirm)
                    # UP ensures cursor on POUND even if it was on ABSORB from previous turn
                    # Full sequence batched for efficiency
                    decision_explanation = "Select FIGHT → POUND: A→UP→A (batched)"
                    logger.info("🥊 [BATTLE BOT] POUND: A→UP→A (batched)")
                    print("🥊 [BATTLE BOT] POUND: A→UP→A (batched)")
                    return ['A', 'UP', 'A']
                
                elif battle_decision == "PRESS_B":
                    # Exit submenu (fight menu or bag menu)
                    button_recommendation = "B"
                    decision_explanation = "Exit submenu (back to main battle menu)"
                    recommended_sequence = ["B"]
                    logger.info("🔙 [BATTLE BOT] Exiting submenu with B")
                
                elif battle_decision == "PRESS_A_ONLY":
                    # Press A only (no B-A-B sequence) - for battle animations or unknown states
                    button_recommendation = "A"
                    decision_explanation = "Advance battle animation or unknown state"
                    recommended_sequence = ["A"]
                    logger.info("⏩ [BATTLE BOT] Pressing A only (animation/unknown state)")
                
                elif battle_decision == "BATTLE_FIGHT" or battle_decision == "USE_MOVE_1":
                    button_recommendation = "A"
                    decision_explanation = "Select FIGHT to use first available move"
                    
                elif battle_decision.startswith("NAV_RUN_STEP_"):
                    # Hard-coded RUN navigation sequence
                    # Format: NAV_RUN_STEP_<BUTTON>
                    extracted_button = battle_decision.split("_")[-1]  # Extract button name
                    button_recommendation = extracted_button
                    decision_explanation = f"Navigating to RUN option (step: {extracted_button})"
                    recommended_sequence = [extracted_button]
                    logger.info(f"🏃 [BATTLE BOT] RUN navigation: pressing {extracted_button}")
                    
                elif battle_decision == "RUN_FROM_WILD":
                    # Fallback if we somehow get the old decision
                    button_recommendation = "A"
                    decision_explanation = "Navigate to RUN option and select it (WILD BATTLE)"
                    logger.warning("🏃 [BATTLE BOT] Got old RUN_FROM_WILD decision - this shouldn't happen")
                    
                elif battle_decision == "RUN_FROM_BATTLE":
                    button_recommendation = "RUN"  # Will need to navigate to RUN option
                    decision_explanation = "Select RUN to flee from battle"
                else:
                    # Unknown decision - let VLM handle
                    logger.warning(f"⚔️ [BATTLE BOT] Unknown decision '{battle_decision}', falling back to VLM")
                    button_recommendation = "A"
                    decision_explanation = "Default to A button"
                
                # Return the button directly
                logger.info(f"⚔️ [BATTLE BOT] {battle_decision} → {button_recommendation}")
                return [button_recommendation]
            else:
                # Battle bot returned None - fallback to VLM
                logger.debug(f"⚔️ [BATTLE BOT] Battle bot uncertain, falling back to VLM")
        
    except Exception as e:
        logger.error(f"⚔️ [BATTLE BOT] Error: {e}", exc_info=True)
        # Continue to next priority level on error
    
    # 🤖 PRIORITY 0B: OPENER BOT - Programmatic State Machine (Splits 0-4)
    # Handles deterministic early game states with high reliability using memory state
    # and milestone tracking as primary signals. Returns None to fallback to VLM.
    try:
        from agent.opener_bot import NavigationGoal, ForceDialogueGoal
        opener_bot = get_opener_bot()
        visual_data = latest_observation.get('visual_data', {}) if isinstance(latest_observation, dict) else {}
        
        should_handle = opener_bot.should_handle(state_data, visual_data)
        
        if should_handle:
            opener_action = opener_bot.get_action(state_data, visual_data, current_plan)
            
            if opener_action is not None:
                bot_state = opener_bot.get_state_summary()
                
                # Check if it's a ForceDialogueGoal (misclassified dialogue)
                if isinstance(opener_action, ForceDialogueGoal):
                    logger.info(f"🚨 [FORCE DIALOGUE] Detected misclassified dialogue: {opener_action.reason}")
                    print(f"🚨 [FORCE DIALOGUE] Pressing A to clear misclassified dialogue")
                    return ['A']
                
                # Check if it's a NavigationGoal
                if isinstance(opener_action, NavigationGoal):
                    # Convert navigation goal to simple direction command
                    current_x = state_data.get('player', {}).get('position', {}).get('x', 0)
                    current_y = state_data.get('player', {}).get('position', {}).get('y', 0)
                    goal_x = opener_action.x
                    goal_y = opener_action.y
                    
                    logger.info(f"🤖 [OPENER BOT] Navigation Goal: {opener_action.description}")
                    logger.info(f"🤖 [OPENER BOT] Current: ({current_x}, {current_y}) -> Goal: ({goal_x}, {goal_y})")
                    
                    # Determine the action based on navigation logic
                    nav_action = None
                    nav_reasoning = ""
                    
                    # At exact goal position - interact
                    if current_x == goal_x and current_y == goal_y:
                        logger.info(f"🤖 [OPENER BOT] At exact goal - interacting with A")
                        nav_action = ['A']
                        nav_reasoning = f"At exact goal position ({goal_x}, {goal_y}), need to interact"
                    else:
                        # Calculate which direction we need to move/face to reach goal
                        required_direction = None
                        if current_x < goal_x:
                            required_direction = 'RIGHT'
                        elif current_x > goal_x:
                            required_direction = 'LEFT'
                        elif current_y < goal_y:
                            required_direction = 'DOWN'
                        elif current_y > goal_y:
                            required_direction = 'UP'
                        
                        # Determine player's current orientation from last directional command
                        current_orientation = None
                        if recent_actions:
                            for action in reversed(recent_actions):
                                if isinstance(action, list):
                                    action = action[0] if action else None
                                if action in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                                    current_orientation = action
                                    break
                        
                        # Check if we're adjacent to goal (distance = 1)
                        distance = abs(current_x - goal_x) + abs(current_y - goal_y)
                        if distance == 1:
                            # Adjacent to goal - but for stairs/warp tiles, we need to WALK ON, not interact
                            if opener_action.should_interact is not None:
                                should_interact = opener_action.should_interact
                            else:
                                goal_desc_lower = (opener_action.description or "").lower()
                                should_interact = any(keyword in goal_desc_lower for keyword in ['interact', 'talk', 'speak', 'check'])
                            
                            print(f"🔍 [NAV] Adjacent to goal. Description: '{opener_action.description}'")
                            print(f"🔍 [NAV] Should interact: {should_interact}")
                            
                            if should_interact:
                                # This is an interact-with-A goal (like NPC or sign)
                                if current_orientation == required_direction:
                                    # Already facing the goal - interact!
                                    logger.info(f"🤖 [OPENER BOT] Facing goal correctly - pressing A")
                                    nav_action = ['A']
                                    nav_reasoning = f"Adjacent to goal, facing {required_direction}, ready to interact"
                                else:
                                    # Need to turn toward goal first
                                    logger.info(f"🤖 [OPENER BOT] Turning to face goal: {required_direction}")
                                    nav_action = [required_direction]
                                    nav_reasoning = f"Adjacent to goal, need to turn {required_direction} before interacting"
                            else:
                                # This is a walk-to goal (stairs, warp tile, position) - keep moving
                                logger.info(f"🤖 [OPENER BOT] Adjacent to walk-to goal ({opener_action.description}) - continuing")
                                nav_action = [required_direction] if required_direction else ['A']
                                nav_reasoning = f"Adjacent to walk-to goal, continuing {required_direction}"
                        else:
                            # Not at exact goal yet - move toward goal
                            if required_direction:
                                nav_action = [required_direction]
                                nav_reasoning = f"Moving {required_direction} toward goal at ({goal_x}, {goal_y})"
                            else:
                                # Shouldn't reach here, but fallback to interact
                                nav_action = ['A']
                                nav_reasoning = "At goal position, defaulting to interact"
                    
                    # Return nav_action directly
                    if nav_action:
                        opener_action = nav_action
                
                # Return the opener bot's action directly
                action_list = opener_action if isinstance(opener_action, list) else [opener_action]
                logger.info(f"🤖 [OPENER BOT] State: {bot_state['current_state']} | Action: {' → '.join(str(a) for a in action_list)}")
                return action_list
            else:
                # Opener bot returned None - fallback to VLM
                logger.debug(f"🤖 [OPENER BOT] Fallback to VLM in state: {opener_bot.current_state_name}")
        else:
            logger.info(f"[ACTION] 🤖 Opener bot should NOT handle - continuing to VLM/dialogue detection")
        
    except Exception as e:
        logger.error(f"🤖 [OPENER BOT] Error: {e}", exc_info=True)
        # Continue to VLM logic on error
    
    # 📍 PRIORITY 0C: OBJECTIVE MANAGER DIRECTIVES (PATH 1 QUICK WIN)
    # This is the "directive system" that provides specific tactical instructions
    # based on milestone progression. Acts as a bridge between high-level objectives
    # and low-level navigation/interaction commands.
    #
    # WHY THIS EXISTS:
    # - Problem: Agent reaches Route 103 but doesn't know to interact with rival at (9,3)
    # - Root cause: ObjectiveManager only provides high-level goals, not specific actions
    # - Solution: get_next_action_directive() returns precise instructions like "walk to (9,3) and press A"
    #
    # ARCHITECTURE:
    # - Uses existing ObjectiveManager from planning module
    # - Returns directives like {"action": "NAVIGATE_AND_INTERACT", "target": (x, y, map)}
    # - Converts directives to NavigationGoal objects (reuses Opener Bot pathfinding)
    # - Returns button presses directly
    try:
        from agent.opener_bot import NavigationGoal
        
        # ObjectiveManager is now passed directly; fall back to legacy attribute
        obj_manager = objective_manager
        if obj_manager is None:
            from agent.planning import planning_step
            obj_manager = getattr(planning_step, 'objective_manager', None)
        
        logger.debug(f"[DIRECTIVE] objective_manager resolved: {obj_manager is not None}")
        
        if obj_manager is not None:
            
            # Log state data for debugging battle detection
            player_data = state_data.get('player', {})
            money = player_data.get('money', 0)
            in_battle = state_data.get('in_battle', False)
            screen_context = state_data.get('screen_context', '')
            player_loc = player_data.get('location', '')
            
            # Add visual_dialogue_active to state_data so objective_manager can access it
            state_data['visual_dialogue_active'] = visual_dialogue_active
            
            logger.debug(f"[DIRECTIVE] loc={player_loc} money={money} battle={in_battle} screen={screen_context} dialogue={visual_dialogue_active}")
            
            directive = obj_manager.get_next_action_directive(state_data)
            
            # Convert legacy dict directives to Directive dataclass
            if isinstance(directive, dict):
                from agent.objective_manager import Directive
                directive = Directive.from_dict(directive)
            
            logger.debug(f"[DIRECTIVE] returned: {directive}")
            
            # PRIORITY CHECK: If we detected a warp jump, press B to settle position first
            # This prevents navigation from executing before the game position stabilizes
            if stuck_handler.should_settle_warp():
                return ['B']
            
            if directive:
                description = directive.get('description', '')
                logger.info(f"📍 [DIRECTIVE] Active directive: {description}")
                print(f"📍 [DIRECTIVE] {description}")

                from agent.directive_nav import execute_directive
                result = execute_directive(directive, state_data, recent_actions, description=description)
                if result is not None:
                    if result:
                        return result
                    else:
                        return []  # journey_complete or at-goal
                # result is None  →  unknown directive type, fall through to VLM
        else:
            logger.debug(f"📍 [DIRECTIVE] ObjectiveManager not initialized yet")
            
    except ImportError as e:
        logger.debug(f"📍 [DIRECTIVE] Planning module not available: {e}")
        print(f"⚠️ [DIRECTIVE] ImportError: {e}")
    except Exception as e:
        logger.error(f"📍 [DIRECTIVE] Error processing directive: {e}", exc_info=True)
        print(f"❌ [DIRECTIVE] Exception caught! {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        # Fall through to other priorities
    
    # 🎯 PRIORITY 1: VLM VISUAL DIALOGUE DETECTION (HIGHEST PRIORITY - BUT ONLY IF OPENER BOT NOT ACTIVE)
    # NEW: Check for continue_prompt_visible (red triangle indicator) - MOST RELIABLE
    # The red triangle ❤️ at end of dialogue is a perfect signal for "press A"
    # 
    # DIALOGUE FALSE POSITIVE PROTECTION:
    # Some locations (e.g., MOVING_VAN) have visual elements that VLM mistakes for dialogue boxes.
    # Blacklist these specific locations to prevent the agent from spamming A button.
    DIALOGUE_DETECTION_BLACKLIST = [
        'MOVING_VAN',  # Has cardboard boxes that VLM mistakes for dialogue boxes
        # Add other problematic locations here if discovered
    ]
    
    # 🔺 PRIORITY 1A: RED TRIANGLE INDICATOR (MOST RELIABLE)
    # The red triangle ❤️ at end of dialogue is the PERFECT signal for "press A"
    # This is much more reliable than text_box_visible alone
    # BUT: Must check for player monologue to avoid false positives!
    if isinstance(latest_observation, dict) and 'visual_data' in latest_observation:
        visual_elements = latest_observation.get('visual_data', {}).get('visual_elements', {})
        on_screen_text = latest_observation.get('visual_data', {}).get('on_screen_text', {})
        continue_prompt_visible = visual_elements.get('continue_prompt_visible', False)
        
        if continue_prompt_visible:
            current_location = state_data.get('player', {}).get('location', '')
            
            # CRITICAL: Check if this is player monologue before pressing A
            dialogue_text = on_screen_text.get('dialogue', '')
            speaker = on_screen_text.get('speaker', '')
            
            # Player monologue ONLY detects "Player:" prefix in dialogue text
            # Do NOT use speaker field - it's unreliable (Mom talking TO Casey shows speaker="CASEY")
            is_player_monologue = (dialogue_text and dialogue_text.strip().upper().startswith('PLAYER:'))
            
            if is_player_monologue:
                # Player monologues are likely VLM hallucinations - ignore them
                logger.info(f"🔺 [CONTINUE PROMPT] Player monologue detected - ignoring (likely hallucination)")
                print(f"⚠️ [DIALOGUE] Player monologue detected - ignoring (likely VLM hallucination)")
                # Don't return anything, fall through to check other priorities
            elif current_location in DIALOGUE_DETECTION_BLACKLIST:
                logger.warning(f"🔺 [CONTINUE PROMPT] Red triangle detected but location '{current_location}' is blacklisted - ignoring")
                print(f"⚠️ [DIALOGUE] Ignoring continue prompt in {current_location} (known false positive)")
            else:
                # Red triangle detected - press A to advance dialogue
                logger.info(f"🔺 [CONTINUE PROMPT] Red triangle indicator detected - pressing A")
                print(f"🔺 [DIALOGUE] Red triangle ❤️ visible - pressing A")
                return ['A']
    
    # 🔺 PRIORITY 1B: FALLBACK - TEXT BOX DETECTION
    # Simple rule: If we see dialogue (text box visible), press A to advance
    # UNLESS it's player monologue (which we must ignore completely)
    # The VLM sees the full rendered text, so we can trust it's ready for input
    if visual_dialogue_active:
        current_location = state_data.get('player', {}).get('location', '')
        
        if current_location in DIALOGUE_DETECTION_BLACKLIST:
            logger.warning(f"💬 [DIALOGUE] VLM detected dialogue but location '{current_location}' is blacklisted - ignoring false positive")
            print(f"⚠️ [DIALOGUE] Ignoring VLM dialogue in {current_location} (known false positive)")
        else:
            # CRITICAL: Check for player monologue BEFORE pressing A
            visual_data = latest_observation.get('visual_data', {})
            on_screen_text = visual_data.get('on_screen_text', {})
            dialogue_text = on_screen_text.get('dialogue', '')
            
            # Player monologue ONLY detects "Player:" prefix in dialogue text
            # Do NOT use speaker field - it's unreliable (Mom talking TO Casey shows speaker="CASEY")
            is_player_monologue = (dialogue_text and dialogue_text.strip().upper().startswith('PLAYER:'))
            
            if is_player_monologue:
                # Player monologues are likely VLM hallucinations - ignore them
                logger.info(f"💬 [DIALOGUE 1B] Player monologue detected - ignoring (likely hallucination)")
                print(f"💬 [DIALOGUE 1B] Player monologue detected - ignoring (likely VLM hallucination)")
                # Don't return anything, fall through to check other priorities
            else:
                # Dialogue detected - press A to advance
                logger.info(f"💬 [DIALOGUE 1B] Dialogue visible - pressing A")
                print(f"💬 [DIALOGUE] Text box visible - pressing A")
                return ['A']
    
    # 🚨 PRIORITY 2: NEW GAME MENU DETECTION
    # Must happen before ANY other logic to prevent override conflicts
    if isinstance(latest_observation, dict) and 'visual_data' in latest_observation:
        visual_data = latest_observation.get('visual_data', {})
        on_screen_text = visual_data.get('on_screen_text', {})
        dialogue_text = (on_screen_text.get('dialogue') or '').upper()
        menu_title = (on_screen_text.get('menu_title') or '').upper()
        
        # Check milestones to ensure we haven't progressed past this screen
        milestones = state_data.get('milestones', {})
        player_name_milestone = milestones.get('PLAYER_NAME_SET', {})
        player_name_set = player_name_milestone.get('completed', False) if isinstance(player_name_milestone, dict) else bool(player_name_milestone)
        
        if ('NEW GAME' in dialogue_text or 'NEW GAME' in menu_title) and not player_name_set:
            logger.info(f"🎯 [NEW GAME FIX] NEW GAME menu detected - pressing A")
            logger.info(f"🎯 [NEW GAME FIX] - dialogue: '{dialogue_text}'")
            logger.info(f"🎯 [NEW GAME FIX] - menu_title: '{menu_title}'")
            print(f"🎯 [NEW GAME FIX] NEW GAME menu detected - pressing A")
            return ['A']
    
    """
    ===============================================================================
    🚨 EMERGENCY PATCH APPLIED - REVIEW BEFORE PRODUCTION 🚨
    ===============================================================================
    
    PATCH: Title Screen Bypass (lines 18-22)
    - Original issue: Agent would freeze on title screen due to complex VLM processing
    - Emergency fix: Hard-coded "A" button press for title screen state
    - TODO: Replace with smarter detection that handles:
      * Multiple title screen states (main menu, options, etc.)
      * Character creation screens  
      * Save/load dialogs
      * Any other menu-like states that need simple navigation
    
    INTEGRATION NOTES:
    - This bypass should be expanded to handle more menu states programmatically
    - Consider creating a "simple_navigation_mode" for all menu/UI interactions
    - The main VLM action logic below this patch is intact and working
    - When reintegrating full AI, keep this as a fallback for known simple states
    
    ===============================================================================
    """
    # 📍 PRIORITY 2: TITLE / INTRO / NAME-SELECTION SCREENS
    from agent.intro_handler import check_intro_screens
    intro_result = check_intro_screens(state_data, latest_observation, recent_actions)
    if intro_result is not None:
        return intro_result
    
    # ============================================================================
    # VLM NAVIGATION LOGIC — delegated to vlm_prompt module
    # ============================================================================
    from agent.vlm_prompt import build_vlm_prompt_and_act
    return build_vlm_prompt_and_act(
        state_data, latest_observation, current_plan,
        recent_actions, vlm, system_prompt,
    )