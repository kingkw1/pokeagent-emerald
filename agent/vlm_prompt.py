"""
VLM Prompt Builder for Pokemon Emerald Agent.

Assembles the complete prompt sent to the Vision Language Model for
overworld navigation decisions.  Includes:
  - Extended map-view generation from MapStitcher
  - Action-context list building (battle info, overworld context, party, entities, etc.)
  - Movement preview extraction + walkable-option parsing + warp-avoidance filtering
  - Anti-stuck direction override
  - Multiple-choice prompt builder (goal parser + A* map-stitcher nav suggestion)
  - Free-form fallback prompt template

Pure data → string transformation.  No side effects except printing debug info.
Extracted from the VLM navigation block of action.py.
"""

import logging
import re
import random
from typing import Dict, Any, Optional, List, Tuple

from utils.state_formatter import (
    format_state_for_llm, format_state_summary,
    get_movement_options, get_party_health_summary,
    format_movement_preview_for_llm,
)
from agent.pathfinding import (
    _local_pathfind_from_tiles,
    _astar_pathfind_with_grid_data,
    _recent_positions,
)
from agent import stuck_handler
from agent.vlm_action import parse_vlm_response

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state for anti-oscillation tracking
# (previously kept on action_step function attributes)
# ---------------------------------------------------------------------------
_position_history: list = []
_last_position: tuple = (None, None, None)


def _reset_position_tracking():
    """Clear position history (useful for testing)."""
    global _position_history, _last_position
    _position_history = []
    _last_position = (None, None, None)


# ===================================================================
#  INTERNAL HELPERS
# ===================================================================

def _generate_extended_map_view(state_data, current_x, current_y):
    """Build extended map text + exploration status from MapStitcher."""
    extended_map_view = None
    exploration_status = None
    try:
        map_stitcher = state_data.get('map', {}).get('_map_stitcher_instance')
        if map_stitcher:
            location_name = state_data.get('player', {}).get('location', 'Unknown')
            player_pos = (current_x, current_y) if isinstance(current_x, int) and isinstance(current_y, int) else None

            if location_name and location_name != 'Unknown' and player_pos:
                try:
                    npcs = state_data.get('npcs', [])
                    connections = map_stitcher.get_location_connections(location_name)
                    extended_map_lines = map_stitcher.generate_location_map_display(
                        location_name, player_pos, npcs=npcs, connections=connections,
                    )
                    if extended_map_lines:
                        extended_map_view = '\n'.join(extended_map_lines)
                        print(f"🗺️ [EXTENDED MAP] Generated {len(extended_map_lines)} line extended view for {location_name}")

                        map_id = map_stitcher.get_map_id(
                            state_data.get('map', {}).get('bank', 0),
                            state_data.get('map', {}).get('number', 0),
                        )
                        if map_id in map_stitcher.map_areas:
                            area = map_stitcher.map_areas[map_id]
                            bounds = getattr(area, 'explored_bounds', {})
                            if bounds:
                                width = bounds.get('max_x', 0) - bounds.get('min_x', 0) + 1
                                height = bounds.get('max_y', 0) - bounds.get('min_y', 0) + 1
                                visited_count = getattr(area, 'visited_count', 1)
                                exploration_status = f"Explored area: {width}x{height} tiles | Visited: {visited_count}x"
                                print(f"🗺️ [EXPLORATION] {exploration_status}")
                except Exception as e:
                    logger.warning(f"[EXTENDED MAP] Error generating extended map view: {e}")
                    print(f"⚠️ [EXTENDED MAP] Failed to generate extended view: {e}")
    except Exception as e:
        logger.warning(f"[EXTENDED MAP] Error accessing map stitcher: {e}")

    return extended_map_view, exploration_status


def _build_action_context(state_data, latest_observation, extended_map_view, exploration_status,
                          movement_options, state_context, party_health, recent_actions):
    """Assemble the action_context lines list."""
    action_context: list[str] = []
    game_data = state_data.get('game', {})
    player_data = state_data.get('player', {})

    # ---- Battle vs Overworld ----
    if game_data.get('in_battle', False):
        action_context.append("=== BATTLE MODE ===")
        battle_info = game_data.get('battle_info', {})
        if battle_info:
            if 'player_pokemon' in battle_info and battle_info['player_pokemon']:
                pp = battle_info['player_pokemon']
                action_context.append(
                    f"Your Pokemon: {pp.get('species_name', pp.get('species', 'Unknown'))} "
                    f"(Lv.{pp.get('level', '?')}) HP: {pp.get('current_hp', '?')}/{pp.get('max_hp', '?')}"
                )
            if 'opponent_pokemon' in battle_info and battle_info['opponent_pokemon']:
                op = battle_info['opponent_pokemon']
                action_context.append(
                    f"Opponent: {op.get('species_name', op.get('species', 'Unknown'))} "
                    f"(Lv.{op.get('level', '?')}) HP: {op.get('current_hp', '?')}/{op.get('max_hp', '?')}"
                )
    else:
        action_context.append("=== OVERWORLD MODE ===")

        if extended_map_view:
            action_context.append("")
            action_context.append("=== EXTENDED MAP VIEW (from exploration memory) ===")
            if exploration_status:
                action_context.append(exploration_status)
            action_context.append("")
            action_context.append(extended_map_view)
            action_context.append("")
            action_context.append("Legend: P=You, N=NPC, .=walkable, #=blocked, ~=grass, ≈=water, S=stairs/warp")
            action_context.append("This is your COMPLETE explored map - use it to plan paths and avoid dead ends!")
            action_context.append("")

        if movement_options:
            action_context.append("=== IMMEDIATE MOVEMENT OPTIONS ===")
            for direction, description in movement_options.items():
                action_context.append(f"  {direction}: {description}")

    # State context
    if state_context and state_context.strip():
        action_context.append("=== GAME STATE CONTEXT ===")
        action_context.append(state_context.strip())
        print(f"🗺️ [MAP DEBUG] Added state context to VLM prompt ({len(state_context)} chars)")
    else:
        print(f"🗺️ [MAP DEBUG] No state context available for VLM prompt")

    # Party health
    if party_health['total_count'] > 0:
        action_context.append("=== PARTY STATUS ===")
        action_context.append(f"Healthy Pokemon: {party_health['healthy_count']}/{party_health['total_count']}")
        if party_health['critical_pokemon']:
            action_context.append("Critical Pokemon:")
            for critical in party_health['critical_pokemon']:
                action_context.append(f"  {critical}")

    # Recent actions
    if recent_actions:
        try:
            recent_list = list(recent_actions) if recent_actions else []
            if recent_list:
                action_context.append(f"Recent Actions: {', '.join(recent_list[-5:])}")
        except Exception as e:
            logger.warning(f"[ACTION] Error processing recent_actions: {e}")

    # ---- Visual perception context ----
    if isinstance(latest_observation, dict) and 'visual_data' in latest_observation:
        visual_data = latest_observation['visual_data']
        action_context.append("=== VISUAL PERCEPTION ===")
        action_context.append(f"Screen Context: {visual_data.get('screen_context', 'unknown')}")

        on_screen_text = visual_data.get('on_screen_text', {})
        visual_elements = visual_data.get('visual_elements', {})
        text_box_visible = visual_elements.get('text_box_visible', False)

        if on_screen_text.get('menu_title'):
            action_context.append(f"Menu: {on_screen_text['menu_title']}")
        if on_screen_text.get('button_prompts'):
            button_prompts = on_screen_text['button_prompts']
            if isinstance(button_prompts, list):
                prompt_strs = []
                for prompt in button_prompts:
                    if isinstance(prompt, dict):
                        prompt_strs.append(prompt.get('text', str(prompt)))
                    else:
                        prompt_strs.append(str(prompt))
                action_context.append(f"Button Prompts: {', '.join(prompt_strs)}")
            else:
                action_context.append(f"Button Prompts: {str(button_prompts)}")

        action_context.append(f"Dialogue Box Status: {'VISIBLE' if text_box_visible else 'NOT VISIBLE'}")

        # Dialogue debug
        if on_screen_text.get('dialogue'):
            print(f"🗨️ [DIALOGUE DEBUG] Dialogue detected: '{on_screen_text.get('dialogue')}'")
            print(f"🗨️ [DIALOGUE DEBUG] - text_box_visible: {text_box_visible}")
        else:
            print(f"🗨️ [DIALOGUE DEBUG] NO dialogue detected in on_screen_text: {on_screen_text}")

        # Fake-dialogue filter
        screen_context = visual_data.get('screen_context', 'unknown')
        dialogue_text = on_screen_text.get('dialogue', '')
        is_fake_dialogue = dialogue_text and any(marker in dialogue_text for marker in [
            'Location:', 'Pos:', 'Money:', 'HP:', 'Pokedex:',
            'ONLY text from dialogue boxes',
            'DO NOT include HUD',
            'If no dialogue box visible',
        ])

        if screen_context == 'overworld' and dialogue_text and not is_fake_dialogue:
            print(f"🚨 [CRITICAL ERROR] VLM reports 'overworld' but REAL dialogue exists!")
        elif is_fake_dialogue:
            print(f"✅ [DIALOGUE FILTER] Ignoring fake 'dialogue': '{dialogue_text[:70]}...')")

        dialogue_text_raw = on_screen_text.get('dialogue', '')
        if dialogue_text_raw and not is_fake_dialogue:
            dialogue_text_lower = dialogue_text_raw.lower()
            if 'pokémon' in dialogue_text_lower and ('box' in dialogue_text_lower or 'logo' in dialogue_text_lower):
                print(f"🎁 [BOX INTERACTION] Detected box/sign dialogue: '{dialogue_text_raw}'")
                action_context.append(f"📦 ACTIVE BOX DIALOGUE: \"{dialogue_text_raw}\" - MUST PRESS A TO CLOSE")
            elif text_box_visible:
                action_context.append(f"ACTIVE Dialogue: \"{dialogue_text_raw}\" - {on_screen_text.get('speaker', 'Unknown')}")
            elif not text_box_visible:
                action_context.append(f"Residual Text (NO dialogue box): \"{dialogue_text_raw}\" - IGNORE THIS")

        # Visible entities
        entities = visual_data.get('visible_entities', [])
        if entities:
            action_context.append("Visible Entities:")
            try:
                if isinstance(entities, list) and len(entities) > 0:
                    for entity in entities[:5]:
                        if isinstance(entity, dict):
                            action_context.append(f"  - {entity.get('type', 'unknown')}: {entity.get('name', 'unnamed')} at {entity.get('position', 'unknown position')}")
                        elif isinstance(entity, str):
                            action_context.append(f"  - {entity}")
                        else:
                            action_context.append(f"  - {str(entity)}")
                elif isinstance(entities, str):
                    action_context.append(f"  - {entities}")
            except Exception as e:
                action_context.append(f"  - Entities: {str(entities)[:100]}")

        # Active visual elements
        active_elements = [k.replace('_', ' ').title() for k, v in visual_elements.items() if v]
        if active_elements:
            action_context.append(f"Active Visual Elements: {', '.join(active_elements)}")

        # Navigation info
        navigation_info = visual_data.get('navigation_info', {})
        if navigation_info:
            action_context.append("=== NAVIGATION ANALYSIS ===")
            exits = navigation_info.get('exits_visible', [])
            if exits and any(exit for exit in exits if exit):
                action_context.append(f"Exits Visible: {', '.join(str(e) for e in exits if e)}")
            interactables = navigation_info.get('interactable_objects', [])
            if interactables and any(obj for obj in interactables if obj):
                action_context.append(f"Interactable Objects: {', '.join(str(o) for o in interactables if o)}")
            barriers = navigation_info.get('movement_barriers', [])
            if barriers and any(b for b in barriers if b):
                action_context.append(f"Movement Barriers: {', '.join(str(b) for b in barriers if b)}")
            open_paths = navigation_info.get('open_paths', [])
            if open_paths and any(p for p in open_paths if p):
                action_context.append(f"Open Paths: {', '.join(str(p) for p in open_paths if p)}")

        # Spatial layout
        spatial_layout = visual_data.get('spatial_layout', {})
        if spatial_layout:
            room_type = spatial_layout.get('room_type')
            player_pos = spatial_layout.get('player_position')
            features = spatial_layout.get('notable_features', [])
            if room_type:
                action_context.append(f"Room Type: {room_type}")
            if player_pos:
                action_context.append(f"Player Position: {player_pos}")
            if features and any(f for f in features if f):
                action_context.append(f"Notable Features: {', '.join(str(f) for f in features if f)}")

    return action_context


def _extract_movement_preview(state_data, player_data, game_data, recent_actions):
    """
    Get movement preview text + walkable_options list.
    Also applies warp-avoidance filtering.
    Returns (movement_preview_text, walkable_options, movement_preview_dict).
    """
    movement_preview_text = ""
    movement_preview = {}
    walkable_options: list = []

    if game_data.get('in_battle', False):
        return movement_preview_text, walkable_options, movement_preview

    try:
        from utils.state_formatter import get_movement_preview
        movement_preview = get_movement_preview(state_data)
        movement_preview_text = format_movement_preview_for_llm(state_data)
        print(f"🗺️ [MOVEMENT DEBUG] Raw movement preview result: '{movement_preview_text}'")

        if movement_preview_text and movement_preview_text != "Movement preview: Not available":
            for line in movement_preview_text.split('\n'):
                if 'WALKABLE' in line:
                    for direction in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                        if line.strip().startswith(direction):
                            parts = line.split(':')
                            if len(parts) >= 2:
                                coords_and_desc = parts[1].strip()
                                walkable_options.append({
                                    'direction': direction,
                                    'details': coords_and_desc,
                                })
                            break

            print(f"🗺️ [MOVEMENT DEBUG] Extracted {len(walkable_options)} walkable options: {[o['direction'] for o in walkable_options]}")

            # Warp-avoidance filter
            current_location_key = player_data.get('location', '')
            current_x = player_data.get('position', {}).get('x')

            if _recent_positions and len(_recent_positions) >= 2 and current_x is not None:
                prev_location = None
                for px, py, ploc in reversed(list(_recent_positions)):
                    if ploc != current_location_key:
                        prev_location = ploc
                        break

                if prev_location:
                    filtered = []
                    for opt in walkable_options:
                        details = opt['details']
                        coord_match = re.search(r'\(\s*(\d+),\s*(\d+)\)', details)
                        if coord_match:
                            tx, ty = int(coord_match.group(1)), int(coord_match.group(2))
                            leads_back = False
                            for rx, ry, rloc in _recent_positions:
                                if rx == tx and ry == ty and rloc != current_location_key:
                                    leads_back = True
                                    print(f"🚫 [WARP AVOID] Filtering {opt['direction']} at ({tx}, {ty}) — recent warp target")
                                    break
                            if not leads_back:
                                filtered.append(opt)
                        else:
                            filtered.append(opt)

                    if len(filtered) < len(walkable_options) and filtered:
                        print(f"✅ [WARP AVOID] Filtered {len(walkable_options) - len(filtered)} warp-back options")
                        walkable_options = filtered

            movement_preview_text = f"\n{movement_preview_text}\n"
        else:
            print(f"🗺️ [MOVEMENT DEBUG] Movement preview empty or not available")
            movement_preview_text = ""
    except Exception as e:
        print(f"🗺️ [MOVEMENT DEBUG] Error getting movement preview: {e}")
        logger.warning(f"[ACTION] Error getting movement preview: {e}")
        movement_preview_text = ""
        walkable_options = []

    return movement_preview_text, walkable_options, movement_preview


def _check_anti_stuck(recent_actions, movement_preview_text):
    """
    If agent pressed the same direction 5+ times and that direction is BLOCKED,
    return an alternative WALKABLE direction.  Returns list[str] or None.
    """
    if not recent_actions or len(recent_actions) < 5:
        return None

    last_5 = recent_actions[-5:]
    if not (all(a == last_5[0] for a in last_5) and last_5[0] in ['UP', 'DOWN', 'LEFT', 'RIGHT']):
        return None

    stuck_direction = last_5[0]
    print(f"🚨 [ANTI-STUCK] Agent pressed {stuck_direction} 5+ times in a row!")

    if movement_preview_text and "BLOCKED" in movement_preview_text:
        if stuck_direction in movement_preview_text.split("BLOCKED")[0][-20:]:
            print(f"🚨 [ANTI-STUCK] {stuck_direction} is BLOCKED!")
            walkable_dirs = []
            for d in ['UP', 'DOWN', 'LEFT', 'RIGHT']:
                if d in movement_preview_text:
                    start = movement_preview_text.find(d)
                    end = movement_preview_text.find('\n', start)
                    line = movement_preview_text[start:end]
                    if 'WALKABLE' in line:
                        walkable_dirs.append(d)
            if walkable_dirs:
                alt = next((d for d in walkable_dirs if d != stuck_direction), walkable_dirs[0])
                print(f"💡 [ANTI-STUCK] Switching to {alt}")
                logger.info(f"💡 [ANTI-STUCK] Stuck on {stuck_direction}, switching to {alt}")
                return [alt]
    return None


# ===================================================================
#  NAVIGATION SUGGESTION (goal parser + A* + safety)
# ===================================================================

def _compute_nav_suggestion(state_data, player_data, current_plan, current_position,
                            latest_observation, recent_actions, all_options, location):
    """
    Return (suggested_option_num, suggestion_reason, navigation_goal) using
    the 3-layer navigation architecture: plan → goal parser → A*/direction map.
    """
    suggested_option_num = None
    suggestion_reason = ""
    navigation_goal = None

    try:
        from utils.goal_parser import get_goal_parser
        goal_parser = get_goal_parser()

        navigation_goal = goal_parser.extract_goal_from_plan(
            plan=current_plan if current_plan else "",
            current_location=location,
            current_objective=None,
        )

        if navigation_goal and navigation_goal.get('confidence', 0) >= 0.6:
            print(f"🎯 [GOAL PARSER] Extracted navigation goal: {navigation_goal}")
            direction_hint = navigation_goal.get('direction_hint', '').lower()

            hint_to_direction = {
                'north': 'UP', 'south': 'DOWN', 'east': 'RIGHT', 'west': 'LEFT',
                'up': 'UP', 'down': 'DOWN', 'left': 'LEFT', 'right': 'RIGHT',
            }

            # A* pathfinding from map-stitcher data
            astar_direction = None
            stitched_map_info = state_data.get('map', {}).get('stitched_map_info')

            print(f"🗺️ [A* DEBUG] stitched_map_info present: {stitched_map_info is not None}")
            if stitched_map_info:
                print(f"🗺️ [A* DEBUG] available: {stitched_map_info.get('available')}")

            if stitched_map_info and stitched_map_info.get('available') and direction_hint:
                current_area = stitched_map_info.get('current_area', {})
                grid_serializable = current_area.get('grid')
                bounds = current_area.get('bounds')

                if grid_serializable and bounds:
                    print(f"🗺️ [A* MAP] Map stitcher data available, attempting pathfinding to '{direction_hint}'")
                    player_x = current_position.get('x', 0)
                    player_y = current_position.get('y', 0)

                    origin_offset = current_area.get('origin_offset')
                    player_grid_pos = current_area.get('player_grid_pos')

                    if origin_offset and player_grid_pos:
                        player_grid_x, player_grid_y = player_grid_pos
                        print(f"🗺️ [A* MAP] Translated coords: local=({player_x},{player_y}), grid=({player_grid_x},{player_grid_y})")
                        coords_ok = (bounds['min_x'] <= player_grid_x <= bounds['max_x'] and
                                     bounds['min_y'] <= player_grid_y <= bounds['max_y'])
                    else:
                        player_grid_x, player_grid_y = player_x, player_y
                        coords_ok = (bounds['min_x'] <= player_x <= bounds['max_x'] and
                                     bounds['min_y'] <= player_y <= bounds['max_y'])

                    if not coords_ok:
                        print(f"⚠️ [A* MAP] Coordinate mismatch — falling back to local pathfinding")
                        astar_direction = _local_pathfind_from_tiles(state_data, direction_hint, recent_actions)
                    else:
                        location_grid = {}
                        for key, value in grid_serializable.items():
                            x, y = map(int, key.split(','))
                            location_grid[(x, y)] = value

                        astar_direction = _astar_pathfind_with_grid_data(
                            location_grid=location_grid, bounds=bounds,
                            current_pos=(player_grid_x, player_grid_y),
                            location=location, goal_direction=direction_hint,
                            recent_positions=_recent_positions,
                        )
                        if astar_direction:
                            print(f"✅ [A* MAP] Pathfinding succeeded: {astar_direction}")
                        else:
                            print(f"⚠️ [A* MAP] Pathfinding failed — simple direction fallback")

            preferred_direction = astar_direction if astar_direction else hint_to_direction.get(direction_hint)

            if preferred_direction:
                milestones = state_data.get('milestones', {})
                current_location_str = player_data.get('location', '')
                for i, opt in enumerate(all_options, 1):
                    if opt['direction'] != preferred_direction:
                        continue

                    is_safe = True
                    details = opt.get('details', '')

                    # Safety: just-got-starter should not re-enter buildings
                    if milestones.get('STARTER_CHOSEN', False) and 'TOWN' in current_location_str and ('[D]' in details or 'Door' in details):
                        is_safe = False
                        print(f"⚠️ [NAV SUGGESTION] Option {i} ({preferred_direction}) is a door — skipping (post-starter)")

                    elif '[D]' in details or 'Door' in details:
                        coord_match = re.search(r'\(\s*(\d+),\s*(\d+)\)', details)
                        if coord_match and _recent_positions and len(_recent_positions) >= 2:
                            tx, ty = int(coord_match.group(1)), int(coord_match.group(2))
                            for rx, ry, rloc in _recent_positions:
                                if rx == tx and ry == ty and rloc != current_location_str:
                                    is_safe = False
                                    print(f"⚠️ [NAV SUGGESTION] Option {i} ({preferred_direction}) door back to '{rloc}' — skipping")
                                    break

                    if is_safe:
                        suggested_option_num = i
                        suggestion_reason = f"Goal is {navigation_goal.get('target', 'unknown')} to the {direction_hint}"
                        print(f"💡 [NAV SUGGESTION] Recommending option {i} ({preferred_direction}) — {suggestion_reason}")
                    break

                # Smart fallback (lateral movement around obstacle)
                if not suggested_option_num and preferred_direction and 'TOWN' in player_data.get('location', ''):
                    fallback_priority = (['RIGHT', 'LEFT', 'DOWN', 'UP'] if preferred_direction in ['UP', 'DOWN']
                                         else ['UP', 'DOWN', 'RIGHT', 'LEFT'])
                    for fb_dir in fallback_priority:
                        for i, opt in enumerate(all_options, 1):
                            if opt['direction'] == fb_dir:
                                details = opt.get('details', '')
                                if '[D]' not in details and 'Door' not in details:
                                    suggested_option_num = i
                                    suggestion_reason = f"Navigate around obstacle by going {fb_dir}"
                                    print(f"💡 [NAV SUGGESTION FALLBACK] suggesting {fb_dir} (option {i})")
                                    break
                        if suggested_option_num:
                            break

    except Exception as e:
        logger.warning(f"[NAV SUGGESTION] Error: {e}")
        import traceback
        traceback.print_exc()

    return suggested_option_num, suggestion_reason, navigation_goal


# ===================================================================
#  MULTIPLE-CHOICE PROMPT
# ===================================================================

def _build_multiple_choice_prompt(state_data, player_data, latest_observation, current_plan,
                                  recent_actions, walkable_options, navigation_goal,
                                  suggested_option_num, suggestion_reason,
                                  current_x, current_y):
    """Build numbered-option action prompt for VLM."""
    global _position_history

    # Anti-oscillation tracking
    current_pos_tuple = (current_x, current_y)
    _position_history.append(current_pos_tuple)
    if len(_position_history) > 10:
        _position_history.pop(0)

    oscillation_warning = ""
    if len(_position_history) >= 6:
        recent_pos = _position_history[-6:]
        unique = set(recent_pos)
        if len(unique) <= 2:
            oscillation_warning = "⚠️ WARNING: You've been oscillating between the same positions! TRY A DIFFERENT DIRECTION."
            print(f"🔄 [OSCILLATION DETECTED] Bouncing between {unique}")
        elif len(unique) == 3 and len(_position_history) >= 8:
            recent_8 = _position_history[-8:]
            if len(set(recent_8)) <= 3:
                oscillation_warning = "⚠️ You're moving in a small loop. Explore in a NEW direction."
                print(f"🔄 [SMALL LOOP] Stuck in small area: {set(recent_8)}")

    # Interactables check
    has_interactables = False
    interactable_description = ""
    if isinstance(latest_observation, dict) and 'visual_data' in latest_observation:
        visual_data = latest_observation['visual_data']
        entities = visual_data.get('visible_entities', [])
        if entities and any(e for e in entities if e and e not in ['none', 'null', '']):
            has_interactables = True
            elist = []
            for e in (entities if isinstance(entities, list) else []):
                if e and e not in ['none', 'null', '']:
                    elist.append(e.get('name', 'NPC') if isinstance(e, dict) else str(e))
            interactable_description = f"NPCs: {', '.join(elist[:3])}" if elist else ""

        nav_info = visual_data.get('navigation_info', {})
        interactables = nav_info.get('interactable_objects', [])
        if interactables and any(o for o in interactables if o and o not in ['none', 'null', '']):
            has_interactables = True
            olist = [str(o) for o in interactables if o and o not in ['none', 'null', '']]
            if interactable_description:
                interactable_description += f", Objects: {', '.join(olist[:3])}"
            else:
                interactable_description = f"Objects: {', '.join(olist[:3])}"

    all_options = walkable_options.copy()
    if has_interactables:
        all_options.append({'direction': 'INTERACT', 'details': f'Press A to interact ({interactable_description})', 'is_interact': True})
        print(f"🎯 [INTERACT MODE] Added INTERACT option — {interactable_description}")

    # Goal context
    location = state_data.get('player', {}).get('location', '')
    goal_context = ""
    if navigation_goal and navigation_goal.get('target'):
        target = navigation_goal['target']
        direction = navigation_goal.get('direction_hint', '')
        goal_context = f"Goal: {target} ({direction.upper()})" if direction else f"Goal: {target}"
    elif 'MOVING_VAN' in location.upper():
        goal_context = "Goal: Exit van"
    elif 'HOUSE' in location.upper() or 'ROOM' in location.upper():
        goal_context = "Goal: Exit building"
    elif 'LAB' in location.upper():
        goal_context = "Goal: Exit lab"
    else:
        goal_context = "Goal: Explore"

    # Instruction
    if oscillation_warning:
        instruction = f"{goal_context}\n⚠️ Stuck in loop — try NEW direction\nPick option:"
    elif suggested_option_num:
        instruction = f"{goal_context}\n**PATHFINDING RECOMMENDATION: Choose option {suggested_option_num}** ({suggestion_reason})\nPick option:"
    else:
        instruction = f"{goal_context}\nPick option:"

    action_prompt = f"""{instruction}

"""

    # Smart reordering: doors last when no suggestion
    display_options = all_options.copy()
    if not suggested_option_num:
        doors, non_doors = [], []
        for opt in display_options:
            (doors if ('[D]' in opt.get('details', '') or 'Door' in opt.get('details', '')) else non_doors).append(opt)
        if doors and non_doors:
            display_options = non_doors + doors
            print(f"🔄 [SMART REORDER] Moved {len(doors)} door(s) to end")

    # Numbered list
    for i, option in enumerate(display_options, 1):
        details = option.get('details', '')
        tile_sym = ''
        m = re.search(r'\[(.)\]', details)
        if m:
            tile_sym = f" [{m.group(1)}]"
        action_prompt += f"{i}. {option['direction']}{tile_sym}\n"

    action_prompt += f"\nAnswer: "

    return action_prompt, display_options


# ===================================================================
#  FREE-FORM PROMPT
# ===================================================================

def _build_freeform_prompt(visual_context, strategic_goal, movement_preview_text):
    """Fallback prompt when no walkable options are available."""
    return f"""Playing Pokemon Emerald. Screen: {visual_context}

{strategic_goal}=== NAVIGATION TASK ===

**CRITICAL: You have access to your COMPLETE explored map (shown above in EXTENDED MAP VIEW if available).**

**Step 1: Check the EXTENDED MAP VIEW (if shown above)**
- This shows the ENTIRE area you've explored, not just your immediate 15x15 view
- You are marked as 'P' on the map
- Use this to see paths, dead ends, and unexplored areas
- Plan your route to avoid getting stuck in cul-de-sacs

**Step 2: Check the MOVEMENT PREVIEW** below for immediate options:
{movement_preview_text}

**Step 3: Choose ONE WALKABLE direction** that:
- Avoids dead ends visible on the extended map
- Moves toward your strategic goal
- Is marked WALKABLE in the movement preview

**PATHFINDING RULES:**
- If the extended map shows a dead end ahead, DON'T GO THERE - backtrack
- If you're stuck (no forward progress), check the extended map for alternate routes
- NEVER repeatedly move into blocked tiles

=== DECISION RULES ===

🚨 **IF DIALOGUE BOX IS VISIBLE** (you see text at bottom of screen):
   → Press A to advance/close the dialogue

🎯 **IF IN OVERWORLD** (no dialogue, no menu):
   → First: Check EXTENDED MAP VIEW (above) to plan your route and avoid dead ends
   → Second: Choose a WALKABLE direction from MOVEMENT PREVIEW
   → Third: Move toward your goal while avoiding obstacles visible on the extended map

📋 **IF IN MENU**:
   → Use UP/DOWN to navigate options
   → Press A to select

⚔️ **IF IN BATTLE**:
   → Press A for moves/attacks

=== OUTPUT FORMAT - CRITICAL ===
You MUST respond with this EXACT format:

Line 1-2: Brief reasoning about THIS SPECIFIC frame (what you actually see, your current goal, your chosen direction)
Line 3: ONLY the button name - ONE of these exact words: A, B, UP, DOWN, LEFT, RIGHT, START

⚠️ CRITICAL INSTRUCTIONS:
1. ANALYZE THIS SPECIFIC FRAME
2. Look at the MOVEMENT PREVIEW to see which directions are WALKABLE
3. Choose a direction that matches your strategic goal
4. DO NOT hallucinate doors or features not visible in the movement data
5. If you're navigating to a location, pick the direction that gets you closer

Example 1 - Movement:
I'm on Route 101. My goal is north. The movement preview shows UP is walkable.
UP

Example 2 - Dialogue:
I see a dialogue box at the bottom with text. I need to close it.
A

Example 3 - Navigation:
I need to go to Littleroot Town which is south. DOWN is walkable according to preview.
DOWN

Now analyze THIS frame and respond with your reasoning and button:
"""


# ===================================================================
#  PUBLIC ENTRY POINT
# ===================================================================

def build_vlm_prompt_and_act(state_data, latest_observation, current_plan,
                             recent_actions, vlm, system_prompt_text):
    """
    Assemble full context, build prompt, call VLM, parse response.

    Returns:
        List[str] — button presses chosen by the VLM (or intelligent default).
    """
    global _last_position

    player_data = state_data.get('player', {})
    game_data = state_data.get('game', {})
    current_position = player_data.get('position', {})
    current_x = current_position.get('x', 'unknown')
    current_y = current_position.get('y', 'unknown')
    current_map = current_position.get('map', 'unknown')

    current_step = state_data.get('step_number', len(recent_actions or []))

    print(f"🎯 [POSITION DEBUG] Step {len(recent_actions) if recent_actions else 0}: Player at ({current_x}, {current_y}) on map {current_map}")

    # Position-change tracking
    if _last_position == (None, None, None):
        _last_position = (current_x, current_y, current_map)
    lx, ly, lm = _last_position
    if (current_x, current_y, current_map) != (lx, ly, lm):
        print(f"✅ [POSITION CHANGE] Moved from ({lx}, {ly}) to ({current_x}, {current_y})")
        _last_position = (current_x, current_y, current_map)
    else:
        print(f"⚠️ [POSITION STUCK] NO MOVEMENT — Still at ({current_x}, {current_y})")

    # --- Gather context pieces ---
    state_context = format_state_for_llm(state_data)
    state_summary = format_state_summary(state_data)
    movement_options = get_movement_options(state_data)
    party_health = get_party_health_summary(state_data)

    extended_map_view, exploration_status = _generate_extended_map_view(state_data, current_x, current_y)

    logger.info("[ACTION] Starting action decision")
    logger.info(f"[ACTION] State: {state_summary}")
    logger.info(f"[ACTION] Party health: {party_health['healthy_count']}/{party_health['total_count']} healthy")

    action_context = _build_action_context(
        state_data, latest_observation, extended_map_view, exploration_status,
        movement_options, state_context, party_health, recent_actions,
    )
    context_str = "\n".join(action_context)

    # Visual context
    visual_context = "unknown"
    if isinstance(latest_observation, dict) and 'visual_data' in latest_observation:
        visual_context = latest_observation['visual_data'].get('screen_context', 'unknown') or "unknown"

    # Strategic goal
    strategic_goal = ""
    if current_plan and current_plan.strip():
        strategic_goal = f"\n=== YOUR STRATEGIC GOAL ===\n{current_plan.strip()}\n\n"

    # Navigation guidance (exit/path/room hints)
    navigation_guidance = ""
    if isinstance(latest_observation, dict) and 'visual_data' in latest_observation:
        vd = latest_observation['visual_data']
        nav = vd.get('navigation_info', {})
        exits = nav.get('exits_visible', [])
        open_paths = nav.get('open_paths', [])
        features = vd.get('spatial_layout', {}).get('notable_features', [])
        room_type = vd.get('spatial_layout', {}).get('room_type', '')

        if any(e for e in exits if e and "door" in str(e).lower()):
            navigation_guidance += "\n🚪 EXITS detected — PRIORITIZE movement toward them.\n"
        if any(f for f in features if f and "door" in str(f).lower()):
            navigation_guidance += f"\n🎯 NOTABLE FEATURES: {features}\n"
        if any(p for p in open_paths if p and p != "none"):
            navigation_guidance += f"\n🛤️ OPEN PATHS: {open_paths}\n"
        if 'interior' in str(room_type).lower() or 'house' in str(room_type).lower():
            navigation_guidance += "\n🏠 ROOM EXIT: Try all directions to find the way out.\n"

    # --- Movement preview + walkable options ---
    movement_preview_text, walkable_options, movement_preview = _extract_movement_preview(
        state_data, player_data, game_data, recent_actions,
    )

    # --- Anti-stuck override ---
    anti_stuck = _check_anti_stuck(recent_actions, movement_preview_text)
    if anti_stuck:
        return anti_stuck

    # --- Build prompt ---
    if walkable_options and len(walkable_options) > 0:
        location = player_data.get('location', '')
        suggested_option_num, suggestion_reason, navigation_goal = _compute_nav_suggestion(
            state_data, player_data, current_plan, current_position,
            latest_observation, recent_actions, walkable_options, location,
        )

        action_prompt, display_options = _build_multiple_choice_prompt(
            state_data, player_data, latest_observation, current_plan,
            recent_actions, walkable_options, navigation_goal,
            suggested_option_num, suggestion_reason, current_x, current_y,
        )
        walkable_options = display_options  # reordered
    else:
        action_prompt = _build_freeform_prompt(visual_context, strategic_goal, movement_preview_text)

    complete_prompt = system_prompt_text + action_prompt

    # --- Debug logging ---
    actual_step = len(recent_actions) if recent_actions else 0
    print(f"📞 [VLM CALL] Step {actual_step} — About to call VLM")

    # Perception debug
    print(f"👁️ [PERCEPTION DEBUG] Latest observation type: {type(latest_observation)}")
    if isinstance(latest_observation, dict) and 'visual_data' in latest_observation:
        vd = latest_observation['visual_data']
        print(f"👁️ [PERCEPTION DEBUG] Screen context: '{vd.get('screen_context', 'missing')}' | Method: {latest_observation.get('extraction_method', 'unknown')}")
        dialogue_data = vd.get('on_screen_text', {}).get('dialogue', '')
        if dialogue_data and vd.get('screen_context') == 'overworld':
            print(f"🚨 [MISCLASSIFICATION] VLM says 'overworld' but dialogue exists!")
        if dialogue_data and 'pokémon' in dialogue_data.lower() and ('box' in dialogue_data.lower() or 'logo' in dialogue_data.lower()):
            print(f"🎁 [BOX DETECTED] Box/sign dialogue — player must press A!")

    if recent_actions is None:
        print(f"⚠️ [CRITICAL] recent_actions is None!")
    elif len(recent_actions) == 0:
        print(f"⚠️ [CRITICAL] recent_actions is empty list!")
    else:
        print(f"✅ [DEBUG] recent_actions has {len(recent_actions)} items: {recent_actions[-5:]}")

    if walkable_options and len(walkable_options) > 0:
        print(f"🎯 [MULTIPLE-CHOICE MODE] {len(walkable_options)} options:")
        for i, opt in enumerate(walkable_options, 1):
            print(f"   {i}. {opt['direction']} — {opt['details']}")
    else:
        print(f"📝 [FREE-FORM MODE] Using traditional free-form action selection")

    # --- VLM call ---
    action_response = vlm.get_text_query(complete_prompt, "ACTION")

    print(f"🔍 [VLM RESPONSE] Step {actual_step} — FULL Response:")
    print("=" * 80)
    print(action_response)
    print("=" * 80)

    actions = parse_vlm_response(action_response, walkable_options, recent_actions, actual_step)

    # Intelligent default
    if not actions:
        if game_data.get('in_battle', False):
            actions = ['A']
        elif party_health['total_count'] == 0:
            actions = ['A', 'A', 'A']
        else:
            actions = [random.choice(['A', 'RIGHT', 'UP', 'DOWN', 'LEFT'])]

    logger.info(f"[ACTION] Actions decided: {', '.join(actions)}")

    # Post-dialogue movement limiting
    override = stuck_handler.check_post_dialogue_limit(actions, latest_observation)
    if override is not None:
        return override

    return actions
