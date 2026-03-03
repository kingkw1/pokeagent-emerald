import logging
from utils.state_formatter import format_state_summary
from agent.objective_manager import ObjectiveManager

# Set up module logging
logger = logging.getLogger(__name__)

def planning_step(memory_context, current_plan, slow_thinking_needed, state_data, vlm, objective_manager=None):
    """
    Fully programmatic planning step — ZERO VLM calls.

    Generates a context-aware strategic plan string using the ObjectiveManager's
    milestone-driven logic and simple game-state heuristics.  The output is passed
    as ``current_plan`` context to the VLM fallback prompt in ``vlm_prompt.py``
    and to ``opener_bot.get_action()``.

    The Slow Brain (RAG + LLM) is now handled exclusively by ``RecoveryPlanner``
    which fires on blocker/failure events inside ``ObjectiveManager.update_brain()``.
    Recovery tasks are consumed at the top of ``get_next_action_directive()``.

    ⚡ Performance: ~1 ms per call, zero API tokens.
    """
    state_summary = format_state_summary(state_data)
    logger.info(f"[PLANNING] Programmatic planning step — {state_summary}")

    # ── Attach / create ObjectiveManager (legacy wiring for action.py) ──
    if objective_manager is not None:
        planning_step.objective_manager = objective_manager
    elif not hasattr(planning_step, 'objective_manager'):
        planning_step.objective_manager = ObjectiveManager()
        logger.info("[PLANNING] Initialized ObjectiveManager (first call)")
        completed_on_init = planning_step.objective_manager.check_storyline_milestones(state_data)
        if completed_on_init:
            logger.info(f"[PLANNING] ✅ Auto-completed {len(completed_on_init)} milestones from save state")

    obj_manager = planning_step.objective_manager

    # ── Build plan string from ObjectiveManager ──
    strategic_plan = obj_manager.get_strategic_plan_description(state_data)

    # Log objective progress
    objectives_summary = obj_manager.get_objectives_summary()
    logger.info(f"[PLANNING] Objectives: {objectives_summary['completed_count']}/{objectives_summary['total_count']} completed")
    if objectives_summary['current_objective']:
        current_obj = objectives_summary['current_objective']
        logger.info(f"[PLANNING] Current: {current_obj['description']} (milestone: {current_obj['milestone_id']})")

    current_objective = obj_manager.get_current_strategic_objective(state_data)

    if strategic_plan:
        current_plan = strategic_plan
    elif current_objective:
        current_plan = generate_fallback_plan(state_data, current_objective)
    else:
        current_plan = generate_fallback_plan(state_data, None)

    # Append tactical notes (low HP warnings, etc.)
    tactical_context = get_tactical_context(state_data)
    if tactical_context:
        current_plan = f"{current_plan}\n\nTACTICAL NOTES: {tactical_context}"

    logger.info(f"[PLANNING] Plan: {current_plan[:300]}{'...' if len(current_plan) > 300 else ''}")
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