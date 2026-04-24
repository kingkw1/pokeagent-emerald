"""
Title / Intro Screen Handler for Pokemon Emerald Agent.

Detects and handles title screens, name-selection screens, post-name
override, NEW GAME menus, and gates the transition into VLM navigation mode.

Extracted from the PRIORITY 2 block of action.py.
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def check_intro_screens(state_data, latest_observation, recent_actions, milestones=None):
    """
    Check if the game is on a title / name-selection / early-intro screen
    and return button presses to advance past it automatically.

    Args:
        state_data: Full game state dictionary
        latest_observation: Perception output with visual data
        recent_actions: List of recent actions taken
        milestones: Milestones dict (if None, extracted from state_data)

    Returns:
        List[str] of button presses if handling an intro screen,
        or None to indicate "fall through to VLM navigation".
    """
    game_data = state_data.get('game', {})
    player_data = state_data.get('player', {})

    player_location = player_data.get("location", "")
    game_state_value = game_data.get("game_state", "").lower()
    player_name = player_data.get("name", "").strip()

    if milestones is None:
        milestones = state_data.get('milestones', {})

    # ------------------------------------------------------------------
    # TITLE SCREEN DETECTION (ultra-conservative)
    # ------------------------------------------------------------------
    is_title_screen = (
        player_location == "TITLE_SEQUENCE" or
        game_state_value == "title" or
        ((not player_name or player_name == "????????") and
         (player_data.get('position', {}).get('x', -1) == 0 and
          player_data.get('position', {}).get('y', -1) == 0) and
         player_location.lower() in ['', 'unknown', 'title_sequence'])
    )

    # Milestone override — if player name is set we're past title
    if milestones.get('PLAYER_NAME_SET', False) or milestones.get('INTRO_CUTSCENE_COMPLETE', False):
        is_title_screen = False

    if is_title_screen:
        logger.info(f"[INTRO] Title screen detected! loc='{player_location}' state='{game_state_value}' name='{player_name}'")
        return ["A"]

    # ------------------------------------------------------------------
    # NAME SELECTION SCREEN
    # ------------------------------------------------------------------
    visual_data = latest_observation.get('visual_data', {}) if isinstance(latest_observation, dict) else {}
    on_screen_text = visual_data.get('on_screen_text', {})

    dialogue_text = (on_screen_text.get('dialogue') or '').upper()
    menu_title = (on_screen_text.get('menu_title') or '').upper()

    current_step = state_data.get('step_number', len(recent_actions or []))

    # Debug logging at critical steps
    if current_step in [33, 34, 35, 40, 45, 50, 51]:
        intro_complete = milestones.get('INTRO_CUTSCENE_COMPLETE', False)
        print(f"🚨 [CRITICAL] Step {current_step}: PLAYER_NAME_SET={milestones.get('PLAYER_NAME_SET', False)}, INTRO_COMPLETE={intro_complete}")

    if current_step >= 30:
        logger.info(f"[INTRO] Step {current_step} — dialogue='{dialogue_text}', menu='{menu_title}', PLAYER_NAME_SET={milestones.get('PLAYER_NAME_SET', False)}")

    name_text_detected = ('YOUR NAME' in dialogue_text or 'NAME?' in dialogue_text or
                          'YOUR NAME' in menu_title or 'NAME?' in menu_title or
                          'SELECT NAME' in menu_title or 'SELECT YOUR NAME' in menu_title)

    vlm_context_name_selection = False
    if isinstance(latest_observation, dict) and 'visual_data' in latest_observation:
        vd = latest_observation['visual_data']
        vlm_dialogue = vd.get('on_screen_text', {}).get('dialogue', '')
        vlm_menu = vd.get('on_screen_text', {}).get('menu_title', '')
        if vlm_dialogue and ('YOUR NAME' in vlm_dialogue or 'NAME?' in vlm_dialogue):
            vlm_context_name_selection = True
        if vlm_menu and ('SELECT' in vlm_menu and 'NAME' in vlm_menu):
            vlm_context_name_selection = True

    in_name_step_range = (25 <= current_step <= 50 and not milestones.get('PLAYER_NAME_SET', False))

    if ((name_text_detected or vlm_context_name_selection or in_name_step_range)
            and not milestones.get('PLAYER_NAME_SET', False)):
        logger.info(f"[INTRO] Name selection detected! text={name_text_detected}, vlm={vlm_context_name_selection}, range={in_name_step_range}")
        return ["A"]

    # ------------------------------------------------------------------
    # NEW GAME MENU
    # ------------------------------------------------------------------
    if ('NEW GAME' in dialogue_text or 'NEW GAME' in menu_title) and not milestones.get('PLAYER_NAME_SET', False):
        logger.info(f"[INTRO] NEW GAME menu detected — selecting with A")
        return ["A"]

    # ------------------------------------------------------------------
    # POST-NAME OVERRIDE: press A until intro cutscene completes
    # ------------------------------------------------------------------
    intro_complete = milestones.get('INTRO_CUTSCENE_COMPLETE', False)
    is_in_moving_van = "MOVING_VAN" in str(player_location).upper()
    on_route_101 = 'ROUTE 101' in str(player_location).upper()
    has_pokemon = state_data.get('player', {}).get('party', [])
    advanced_location = on_route_101 or 'ROUTE' in str(player_location).upper()
    player_has_pokemon = len(has_pokemon) > 0 if has_pokemon else False

    # POST-NAME OVERRIDE: keep pressing A until location changes or intro completes.
    # No step-count limit — the other guards (intro_complete, advanced_location,
    # player_has_pokemon, is_in_moving_van) are sufficient to scope this correctly.
    if (milestones.get('PLAYER_NAME_SET', False) and
            not intro_complete and
            not advanced_location and
            not player_has_pokemon and
            not is_in_moving_van):
        logger.info(f"[INTRO] Post-name override active — step {current_step}, loc={player_location}")
        print(f"🔧 [OVERRIDE] Step {current_step} — post-name override: pressing A")
        return ["A"]

    # ------------------------------------------------------------------
    # VLM NAVIGATION MODE GATING
    # ------------------------------------------------------------------
    if intro_complete or is_in_moving_van or advanced_location:
        # VLM mode — fall through
        if current_step % 5 == 0 or advanced_location:
            print(f"🤖 [VLM MODE] Step {current_step} — VLM Navigation Active")
        return None  # fall through to VLM

    # Early-game legacy path (TITLE_SEQUENCE, no milestones yet)
    return None  # fall through to VLM
