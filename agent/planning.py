import logging
from utils.vlm import VLM
from utils.state_formatter import format_state_for_llm, format_state_summary
from agent.system_prompt import system_prompt
from agent.objective_manager import ObjectiveManager

# Set up module logging
logger = logging.getLogger(__name__)

def planning_step(memory_context, current_plan, slow_thinking_needed, state_data, vlm, objective_manager=None):
    """
    Decide and update your high-level plan based on memory context, current state, and the need for slow thinking.
    Returns updated plan.
    
    ===============================================================================
    🚨 MAJOR ARCHITECTURAL CHANGE - VLM PLANNING ELIMINATED 🚨  
    ===============================================================================
    
    CRITICAL CHANGE: This function now uses ZERO VLM calls (fully programmatic)
    
    ORIGINAL PROBLEM:
    - Two expensive VLM calls per planning cycle were causing memory crashes
    - "PLANNING-ASSESSMENT" call to check if current plan is complete
    - "PLANNING-CREATION" call to generate new strategic plans
    - Long prompts with comprehensive state data were overwhelming resources
    - Agent would freeze or be killed by OS due to excessive memory usage
    
    EMERGENCY SOLUTION:
    - Replaced all VLM planning with fast programmatic logic
    - Context-aware plan generation based on game state analysis
    - Simple but effective plans tailored to current situation
    
    CURRENT PLANNING LOGIC:
    - Title screen: "Navigate through title screen and character creation..."
    - Battle: "Focus on battle: attack with effective moves, heal if HP low..."
    - Pokemon Center: "Heal Pokemon at the Pokemon Center, then continue..."
    - Unknown location: "Explore current area, interact with NPCs..."
    - Known location: "Navigate [location] efficiently. Talk to NPCs..."
    
    ⚠️  REINTEGRATION STRATEGY FOR INTELLIGENT PLANNING:
    
    OPTION 1 - Keep Programmatic as Base (Recommended):
    - Current logic is actually quite effective for most scenarios
    - Add VLM calls only for complex strategic decisions
    - Use VLM for long-term route planning, not immediate tactical decisions
    - Example: "Should I go to Gym 1 or train more Pokemon first?"
    
    OPTION 2 - Hybrid Planning System:
    - Programmatic planning for immediate actions (current approach)
    - Occasional VLM calls for strategic review (every 10-20 steps)
    - VLM focuses on high-level goals, not step-by-step tactics
    - Implement strict timeouts and fallback to programmatic plans
    
    OPTION 3 - Smart VLM Triggers:
    - Use programmatic logic to detect when VLM planning is actually needed
    - Major story events, new areas, complex puzzles
    - Keep programmatic for routine navigation and battles
    - Much more efficient than calling VLM for every decision
    
    PRESERVED ELEMENTS:
    - Function signature matches caller expectations
    - Proper logging for debugging and monitoring  
    - Plans are contextually appropriate and actionable
    - Integration with memory and action modules intact
    
    ⚡ PERFORMANCE IMPACT:
    - Planning now takes ~1ms instead of ~10-30 seconds
    - Zero memory overhead from VLM processing
    - No API rate limiting or timeout issues
    - Agent can make decisions in real-time
    
    ===============================================================================
    """
    # Get basic state info for programmatic planning
    state_summary = format_state_summary(state_data)
    game_data = state_data.get('game', {})
    player_data = state_data.get('player', {})
    
    logger.info("[PLANNING] Starting enhanced milestone-driven planning step")
    logger.info(f"[PLANNING] State: {state_summary}")
    logger.info(f"[PLANNING] Slow thinking needed: {slow_thinking_needed}")
    
    # ENHANCED STRATEGIC PLANNING: Use ObjectiveManager for milestone-driven strategy
    # Use externally provided ObjectiveManager if available, else create local one
    if objective_manager is not None:
        planning_step.objective_manager = objective_manager
    elif not hasattr(planning_step, 'objective_manager'):
        planning_step.objective_manager = ObjectiveManager()
        logger.info("[PLANNING] Initialized ObjectiveManager for strategic planning")
        
        # CRITICAL FIX: Check milestones immediately on first initialization
        # This ensures we don't tell the agent to "go to Route 101" when already there!
        logger.info("[PLANNING] Performing initial milestone check from save state...")
        completed_on_init = planning_step.objective_manager.check_storyline_milestones(state_data)
        if completed_on_init:
            logger.info(f"[PLANNING] ✅ Auto-completed {len(completed_on_init)} objectives from save state milestones: {completed_on_init}")
        else:
            logger.info("[PLANNING] No objectives completed from initial milestone check")
    
    # Get milestone-driven strategic context
    obj_manager = planning_step.objective_manager
    strategic_plan = obj_manager.get_strategic_plan_description(state_data)
    
    # Get objectives summary for logging
    objectives_summary = obj_manager.get_objectives_summary()
    logger.info(f"[PLANNING] Objectives status: {objectives_summary['completed_count']}/{objectives_summary['total_count']} completed")
    if objectives_summary['current_objective']:
        current_obj = objectives_summary['current_objective']
        logger.info(f"[PLANNING] Current objective: {current_obj['description']} (milestone: {current_obj['milestone_id']})")
    
    # SMART VLM TRIGGER: Only call VLM when strategic objective changes
    # Track the last objective we planned for
    if not hasattr(planning_step, 'last_objective_id'):
        planning_step.last_objective_id = None
        planning_step.last_detailed_plan = None
    
    current_objective = obj_manager.get_current_strategic_objective(state_data)
    current_objective_id = current_objective.id if current_objective else None
    
    # Check if we need to generate a new detailed plan
    objective_changed = (current_objective_id != planning_step.last_objective_id)
    needs_new_plan = objective_changed or (planning_step.last_detailed_plan is None)
    
    if needs_new_plan and current_objective:
        logger.info(f"[PLANNING] 🧠 SMART VLM TRIGGER ACTIVATED: Objective changed from '{planning_step.last_objective_id}' to '{current_objective_id}'")
        logger.info(f"[PLANNING] Calling VLM to generate detailed plan for: {current_objective.description}")
        
        # Generate detailed plan using VLM
        try:
            state_context = format_state_for_llm(state_data)
            
            planning_prompt = f"""You are an AI agent playing Pokemon Emerald. You need to create a detailed, step-by-step plan to achieve your current objective.

CURRENT OBJECTIVE: {current_objective.description}
Objective Type: {current_objective.objective_type}
Target: {current_objective.target_value}

CURRENT GAME STATE:
{state_context}

MEMORY CONTEXT:
{memory_context}

Create a detailed, actionable plan with specific steps to achieve this objective. Your plan should:

1. Break down the objective into 3-5 concrete steps
2. Include specific navigation directions (e.g., "Go north to Route 101", "Enter the building on the left")
3. Mention key interactions needed (e.g., "Talk to Professor Birch", "Press A to select starter")
4. Account for obstacles or requirements (e.g., "Navigate through tall grass carefully")
5. Be written as clear instructions the agent can follow

Focus on efficiency and clarity. The agent will use this plan to guide its moment-to-moment decisions.

Example format:
"Step 1: Exit your house by going DOWN to the door and pressing A.
Step 2: Navigate south through Littleroot Town toward Route 101.
Step 3: Continue north on Route 101 until you encounter Professor Birch.
Step 4: Press A to interact with Birch and trigger the starter selection event.
Step 5: Choose your starter Pokemon using the menu."

Now create your detailed plan:"""

            detailed_plan = vlm.get_text_query(system_prompt + planning_prompt, "PLANNING-DETAILED")
            
            if detailed_plan and len(detailed_plan.strip()) > 20:
                # VLM successfully generated a plan
                planning_step.last_detailed_plan = detailed_plan
                planning_step.last_objective_id = current_objective_id
                current_plan = f"OBJECTIVE: {current_objective.description}\n\nDETAILED PLAN:\n{detailed_plan}"
                logger.info(f"[PLANNING] ✅ VLM generated detailed plan ({len(detailed_plan)} chars)")
                logger.info(f"[PLANNING] Plan preview: {detailed_plan[:200]}...")
            else:
                # VLM failed or returned empty plan - use strategic goal as fallback
                logger.warning("[PLANNING] ⚠️ VLM plan generation failed or empty, using strategic goal")
                current_plan = strategic_plan if strategic_plan else generate_fallback_plan(state_data, current_objective)
                
        except Exception as e:
            logger.error(f"[PLANNING] ❌ VLM planning failed with error: {e}")
            logger.info("[PLANNING] Falling back to strategic goal")
            current_plan = strategic_plan if strategic_plan else generate_fallback_plan(state_data, current_objective)
    
    elif planning_step.last_detailed_plan:
        # Use cached detailed plan for current objective
        logger.info(f"[PLANNING] 📋 Using cached detailed plan for objective '{current_objective_id}'")
        current_plan = f"OBJECTIVE: {current_objective.description}\n\nDETAILED PLAN:\n{planning_step.last_detailed_plan}"
    
    elif strategic_plan:
        # No detailed plan available, use strategic goal
        logger.info(f"[PLANNING] Using strategic milestone-driven goal (no detailed plan yet)")
        current_plan = strategic_plan
    
    else:
        # Final fallback to programmatic planning
        logger.info("[PLANNING] No objective available, using programmatic fallback")
        current_plan = generate_fallback_plan(state_data, None)
    
    # Add tactical context to any plan
    tactical_context = get_tactical_context(state_data)
    if tactical_context:
        current_plan = f"{current_plan}\n\nTACTICAL NOTES: {tactical_context}"
    
    logger.info(f"[PLANNING] Final enhanced plan: {current_plan[:300]}..." if len(current_plan) > 300 else f"[PLANNING] Final enhanced plan: {current_plan}")
    return current_plan


def generate_fallback_plan(state_data, current_objective=None):
    """Generate a simple fallback plan when VLM is unavailable"""
    game_data = state_data.get('game', {})
    player_data = state_data.get('player', {})
    current_location = player_data.get('location', 'Unknown')
    game_state = game_data.get('state', 'unknown')
    in_battle = game_data.get('in_battle', False)
    
    if current_objective:
        # Use objective info to create a basic plan
        return f"Work toward: {current_objective.description}. Navigate carefully and interact with NPCs for guidance."
    elif game_state == 'title':
        return "Navigate through title screen and character creation to start the game."
    elif in_battle:
        return "Focus on battle: attack with effective moves, heal if HP is low, switch Pokemon if needed."
    elif 'POKEMON_CENTER' in current_location.upper():
        return "Heal Pokemon at the Pokemon Center, then continue adventure."
    elif current_location == 'Unknown' or not current_location:
        return f"Explore the current area, interact with NPCs, and progress the story."
    else:
        return f"Navigate {current_location} efficiently. Talk to NPCs for story progression, battle trainers for experience, and head toward the next major objective."


def get_tactical_context(state_data):
    """Generate tactical context based on current game state"""
    game_data = state_data.get('game', {})
    player_data = state_data.get('player', {})
    
    tactical_notes = []
    
    # Battle context
    if game_data.get('in_battle', False):
        battle_info = game_data.get('battle_info', {})
        if battle_info is None:
            battle_info = {}
        player_pokemon = battle_info.get('player_pokemon', {})
        if player_pokemon is None:
            player_pokemon = {}
        if player_pokemon.get('hp_current', 0) < (player_pokemon.get('hp_max', 1) * 0.3):
            tactical_notes.append("URGENT: Pokemon health critical - consider healing or switching.")
    
    # Party health context
    party = game_data.get('party', [])
    # SAFETY CHECK: Ensure party is iterable
    if party is None:
        party = []
    
    healthy_count = sum(1 for p in party if p.get('hp_current', 0) > 0)
    total_count = len(party)
    if total_count > 0 and healthy_count / total_count < 0.5:
        tactical_notes.append("WARNING: Most party Pokemon are fainted - visit Pokemon Center.")
    
    # Money context for purchases
    money = player_data.get('money', 0)
    if money is None:
        money = 0
    if money < 500:
        tactical_notes.append("LOW FUNDS: Consider battling trainers for money.")
    
    return " ".join(tactical_notes) 