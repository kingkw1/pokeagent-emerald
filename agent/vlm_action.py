"""
VLM Action Decision Module for Pokemon Emerald Agent.

Contains VLM response parsing and validation for action decisions.
Extracts the core parsing logic from the monolithic action_step() function
so it can be tested and maintained independently.
"""

import re
import random
import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Valid GBA buttons the agent can press
VALID_BUTTONS = ['A', 'B', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT']


def sanitize_vlm_response(action_response: Optional[str]) -> str:
    """
    Pre-process VLM response: handle None, truncate runaway responses,
    detect repetitive hallucinations.

    Returns:
        Cleaned response string (never None)
    """
    if action_response is None:
        logger.warning("[VLM] VLM returned None response, using fallback")
        return "A"

    action_response = action_response.strip()

    # Truncate suspiciously long responses (possible hallucination)
    if len(action_response) > 500:
        print(f"⚠️ [VLM WARNING] Response is suspiciously long ({len(action_response)} chars)")
        action_response = action_response[:200]

    # Detect repetitive hallucination patterns
    if len(action_response) > 50:
        first_50 = action_response[:50].lower()
        if "you are in battle mode" in first_50 and action_response.lower().count("you are in battle mode") > 3:
            print(f"🚨 [VLM ERROR] Detected repetitive hallucination - forcing 'A'")
            return "A"

    return action_response


def extract_action_line(action_response: str):
    """
    Extract reasoning and action line from VLM response.

    VLM is expected to return reasoning followed by a button on the last line.

    Returns:
        (reasoning: str or None, action_line: str)
    """
    lines = [line.strip() for line in action_response.split('\n') if line.strip()]

    if len(lines) >= 2:
        reasoning = '\n'.join(lines[:-1])
        action_line = lines[-1].upper()
        print(f"🧠 [VLM REASONING] {reasoning}")
        print(f"🎮 [VLM ACTION LINE] {action_line}")
        return reasoning, action_line
    elif len(lines) == 1:
        action_line = lines[0].upper()
        print(f"🎮 [VLM ACTION LINE] {action_line}")
        return None, action_line
    else:
        print(f"❌ [EMPTY RESPONSE] VLM returned empty response")
        return None, "A"


def parse_multiple_choice(response_str: str, walkable_options: List[dict]) -> Optional[List[str]]:
    """
    Parse VLM response as a numbered multiple-choice selection.

    Args:
        response_str: Uppercased VLM action line
        walkable_options: List of option dicts with 'direction' and 'details'

    Returns:
        Action list if a valid choice was found, None otherwise
    """
    if not walkable_options:
        return None

    print(f"🎯 [MULTIPLE-CHOICE] Parsing response for {len(walkable_options)} options")

    number_match = re.search(r'\b([1-9])\b', response_str)

    if number_match:
        choice_num = int(number_match.group(1))
        print(f"✅ [CHOICE DETECTED] VLM selected option {choice_num}")

        if 1 <= choice_num <= len(walkable_options):
            selected_option = walkable_options[choice_num - 1]
            selected_direction = selected_option['direction']
            print(f"✅ [MULTIPLE-CHOICE] Option {choice_num} maps to: {selected_direction}")
            print(f"   Details: {selected_option['details']}")

            if selected_direction == 'INTERACT':
                print(f"🎯 [INTERACT] Converting INTERACT to A button press")
                return ['A']
            return [selected_direction]
        else:
            print(f"❌ [INVALID CHOICE] Option {choice_num} out of range (1-{len(walkable_options)})")
            return [walkable_options[0]['direction']]
    else:
        print(f"⚠️ [NO NUMBER] VLM didn't provide a number, checking for button names...")
        return None


def parse_button_from_response(response_str: str, valid_buttons: Optional[List[str]] = None) -> List[str]:
    """
    Extract a valid button press from free-form VLM response text.

    Tries multiple parsing strategies in priority order:
    1. Exact match of last line
    2. Clean VLM artifacts and match
    3. "BUTTON (explanation)" format
    4. Comma-separated multi-action
    5. First valid button anywhere in text
    6. Direction/action word patterns (north → UP, etc.)

    Args:
        response_str: Uppercased VLM action line
        valid_buttons: List of valid button names (defaults to VALID_BUTTONS)

    Returns:
        Action list (may be empty if nothing found)
    """
    if valid_buttons is None:
        valid_buttons = VALID_BUTTONS

    actions = []

    first_line = response_str.split('\n')[0].strip().upper()

    # PRIORITY 1: Clean VLM artifacts and check exact match
    cleaned_first_line = first_line
    for artifact in ['</OUTPUT>', '</output>', '<|END|>', '<|end|>',
                     '<|ASSISTANT|>', '<|assistant|>', '|user|']:
        cleaned_first_line = cleaned_first_line.replace(artifact, '').strip()

    if cleaned_first_line in valid_buttons:
        return [cleaned_first_line]
    if first_line in valid_buttons:
        return [first_line]

    # PRIORITY 1.5: "A (explanation)" format
    if '(' in first_line:
        button_part = first_line.split('(')[0].strip().upper()
        if button_part in valid_buttons:
            return [button_part]

    # PRIORITY 2: Comma-separated multi-action
    if ',' in response_str:
        raw_actions = [btn.strip().upper() for btn in response_str.split(',')]
        actions = [btn for btn in raw_actions if btn in valid_buttons][:3]
        if actions:
            return actions

    # PRIORITY 3: Exact match of whole response
    if response_str.upper() in valid_buttons:
        return [response_str.upper()]

    # PRIORITY 3.5: Cleaned whole response
    cleaned_response = response_str.upper()
    for artifact in ['</OUTPUT>', '</output>', '<|END|>', '<|end|>',
                     '<|ASSISTANT|>', '<|assistant|>', '|user|', '|assistant|']:
        cleaned_response = cleaned_response.replace(artifact, '').strip()
    if cleaned_response in valid_buttons:
        return [cleaned_response]

    # PRIORITY 4: First valid button anywhere in text
    for button in valid_buttons:
        if button.lower() in response_str.lower():
            return [button]

    # PRIORITY 5: Direction/action word patterns
    response_lower = response_str.lower()
    if 'up' in response_lower or 'north' in response_lower:
        return ['UP']
    elif 'down' in response_lower or 'south' in response_lower:
        return ['DOWN']
    elif 'left' in response_lower or 'west' in response_lower:
        return ['LEFT']
    elif 'right' in response_lower or 'east' in response_lower:
        return ['RIGHT']
    elif 'interact' in response_lower or 'confirm' in response_lower or 'select' in response_lower:
        return ['A']
    elif 'back' in response_lower or 'cancel' in response_lower or 'menu' in response_lower:
        return ['B']

    return []


def parse_vlm_response(action_response: Optional[str],
                       walkable_options: List[dict],
                       recent_actions: Optional[List] = None,
                       current_step: int = 0) -> List[str]:
    """
    Full VLM response parsing pipeline.

    Sanitizes the response, extracts the action line, tries multiple-choice
    parsing (if options provided), then falls back to button parsing.
    Also applies anti-loop and anti-hallucination guards.

    Args:
        action_response: Raw VLM response text
        walkable_options: Movement options presented to VLM (may be empty)
        recent_actions: Recent action history for anti-loop detection
        current_step: Current step number for logging

    Returns:
        List of action strings (never empty - has fallback defaults)
    """
    # Step 1: Sanitize
    action_response = sanitize_vlm_response(action_response)

    # Step 2: Extract action line
    reasoning, action_line = extract_action_line(action_response)
    response_str = action_line

    # Step 3: Try multiple-choice parsing
    actions = []
    if walkable_options:
        result = parse_multiple_choice(response_str, walkable_options)
        if result:
            actions = result

    # Step 4: Try button parsing if no multiple-choice match
    if not actions:
        actions = parse_button_from_response(response_str)

    # Step 5: Log results
    print(f"✅ Parsed actions: {actions}")
    if not actions:
        print(f"❌ No valid actions parsed from: '{action_response}'")
        print(f"   Valid buttons are: {VALID_BUTTONS}")
        if action_response and len(action_response) > 200:
            print(f"🚨 [ANTI-HALLUCINATION] VLM response too long - forcing simple navigation")
            actions = [random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'A'])]
    else:
        print(f"✅ Successfully parsed {len(actions)} action(s): {actions}")

    print("-" * 80 + "\n")

    # Step 6: Anti-loop - detect repeated A presses
    if actions == ['A'] and recent_actions:
        recent_a_count = sum(1 for a in recent_actions[-10:] if a == 'A')
        if recent_a_count >= 8 and len(recent_actions) >= 10:
            print(f"🔄 [ANTI-LOOP] Step {current_step} - Detected A-loop ({recent_a_count}/10). Forcing exploration.")
            actions = [random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT'])]
            print(f"   Forcing exploration with: {actions}")

    # Step 7: Final fallback
    if not actions:
        actions = [random.choice(['A', 'RIGHT', 'UP', 'DOWN', 'LEFT'])]

    return actions
