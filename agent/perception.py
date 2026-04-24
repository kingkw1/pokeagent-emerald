import time
import logging
import json
import re
import signal
from utils.vlm import VLM
from utils.state_formatter import format_state_for_llm, format_state_summary

# Set up module logging
logger = logging.getLogger(__name__)

# Template phrases that indicate VLM returned instructions instead of actual game content
TEMPLATE_PHRASES = [
    "ONLY text from dialogue boxes",
    "DO NOT include HUD/status text",
    "character name if shown in dialogue box",
    "exact menu header text or null",
    "any button instructions like",
    "describe NPCs, Pokemon, or characters",
]

def is_template_text(text):
    """
    Check if text is template instructions rather than actual game dialogue.
    
    The VLM sometimes returns the JSON template/instructions verbatim instead
    of analyzing the actual image. This function detects that case.
    """
    if not text or not isinstance(text, str):
        return False
    text_lower = text.lower()
    return any(phrase.lower() in text_lower for phrase in TEMPLATE_PHRASES)


def perception_step(frame, state_data, vlm, recent_actions=None):
    """
    Extract structured visual information from the game frame using VLM-based analysis.
    Returns observation dictionary with structured visual data.
    
    Args:
        frame: Current game screenshot
        state_data: Game state dictionary
        vlm: VLM instance for visual queries
        recent_actions: List of recent actions (optional, for dialogue detection optimization)
    
    ===============================================================================
    🔍 HYBRID APPROACH - STRUCTURED VLM EXTRACTION 🔍
    ===============================================================================
    
    NEW STRATEGY: Image-to-Structure Extraction
    - Ask VLM to extract specific information into structured JSON format
    - Much more focused and efficient than long text descriptions
    - Machine-readable output that downstream modules can easily use
    - Fallback to programmatic analysis if VLM fails
    
    ADVANTAGES:
    - ✅ Efficient: Focused prompts, faster responses
    - ✅ Structured: JSON output, no text parsing needed
    - ✅ Reliable: Built-in fallback to programmatic analysis
    - ✅ Useful: Extracts visual information not available in game state
    
    EXTRACTED INFORMATION:
    - Screen context (overworld, battle, menu, dialogue)
    - On-screen text and dialogue
    - Visible entities (NPCs, items, trainers)
    - Menu states and UI elements
    
    ===============================================================================
    """
    import time
    perception_start = time.time()
    
    # Get basic state info for analysis
    state_summary = format_state_summary(state_data)
    game_data = state_data.get('game', {})
    player_data = state_data.get('player', {})
    
    logger.info("[PERCEPTION] Starting hybrid VLM-based structured extraction")
    logger.info(f"[PERCEPTION] State: {state_summary}")
    
    setup_time = time.time() - perception_start
    
    # Determine if we should use VLM or fallback to programmatic analysis
    current_location = player_data.get('location', 'Unknown')
    game_state = game_data.get('state', 'unknown')
    in_battle = game_data.get('in_battle', False)

    # During title sequence use a much shorter VLM timeout so timeouts fail fast.
    # (Position is always (0,0) and there is no walkable map, but dialogue text IS
    # meaningful — the VLM is the most reliable way to detect the Prof Birch textbox.)
    _is_title_seq = current_location == 'TITLE_SEQUENCE'
    _vlm_timeout = 5 if _is_title_seq else 15

    # Try VLM-based structured extraction first
    visual_data = None
    
    # Import json and re at module level to avoid scoping issues
    import json
    import re
    import signal
    
    try:
        if frame is not None:
            logger.info("[PERCEPTION] Attempting VLM-based structured extraction...")
            
            # Get movement options for VLM context
            from utils.state_formatter import get_movement_options
            movement_options = get_movement_options(state_data)
            
            # Format movement options for VLM
            movement_context = ""
            if movement_options:
                movement_context = "\n\nMovement Analysis Context:"
                for direction, description in movement_options.items():
                    movement_context += f"\n- {direction}: {description}"
            
            # Create SIMPLIFIED JSON extraction prompt for Pokemon Emerald
            # CRITICAL: Keep prompt short to prevent VLM hallucinations/template echoing
            extraction_prompt = f"""
Analyze this Pokemon Emerald screenshot. Current state: {state_summary}

Look at the image and extract:

1. DIALOGUE BOX CHECK (CRITICAL):
   - Is there a white/black text box at the BOTTOM of the screen?
   - If YES: Extract the dialogue text you see
   - Look for a small RED TRIANGLE ❤️ or arrow at the END of the text
   - The red triangle means "waiting for A button press"
   - IGNORE the player name display at TOP of screen (e.g., "Player: JOHNNY")
   - ONLY extract text from dialogue boxes at BOTTOM of screen

2. SCREEN TYPE:
   - "overworld" = top-down map view with player character
   - "dialogue" = dialogue box is the main focus
   - "battle" = Pokemon battle interface
   - "menu" = menu is open

3. NAVIGATION (for overworld only):
   - Describe what you see in each direction (paths, obstacles, exits)

Return JSON:

{{
    "screen_context": "overworld or dialogue or battle or menu",
    "on_screen_text": {{
        "dialogue": "exact text from dialogue box or null if none",
        "speaker": "character name or null",
        "menu_title": "menu title or null"
    }},
    "visible_entities": ["NPCs or Pokemon you see"],
    "navigation_info": {{
        "open_paths": ["UP: what you see", "DOWN: what you see", "LEFT: what you see", "RIGHT: what you see"]
    }},
    "visual_elements": {{
        "text_box_visible": true_or_false,
        "continue_prompt_visible": true_or_false
    }}
}}

CRITICAL RULES:
- If you see a red triangle ❤️ at end of dialogue, set continue_prompt_visible = true
- DO NOT copy prompt text into the JSON - analyze the actual image
- If no dialogue box, set dialogue = null and text_box_visible = false
- IMPORTANT: Player's internal thoughts (e.g., "Player: What should I do?") are NOT dialogue boxes - set text_box_visible = false for these
- Only set text_box_visible = true for NPC dialogue boxes (when an NPC is speaking to you)
- Return only real observations from THIS image

Return ONLY the JSON object, nothing else:"""
            
            # Make VLM call with timeout protection
            def timeout_handler(signum, frame):
                raise TimeoutError("VLM call timed out")
            
            # Set up timeout (15 seconds for cloud API; local GPU models use a separate config)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(_vlm_timeout)  # 5s for title sequence, 15s otherwise
            
            try:
                logger.debug("[PERCEPTION] Calling VLM for visual analysis")
                # print(f"🖼️ [PERCEPTION] Frame type: {type(frame)}")
                # print(f"📝 [PERCEPTION] Extraction prompt length: {len(extraction_prompt)} chars")
                
                vlm_response = vlm.get_query(frame, extraction_prompt, "PERCEPTION-EXTRACT")
                signal.alarm(0)  # Cancel timeout
                
                # print(f"🔍 [PERCEPTION] VLM Raw Response:")
                # print(f"=== START VLM RESPONSE ===")
                # print(vlm_response)
                # print(f"=== END VLM RESPONSE ===")
                
                # Extract JSON from response (handle cases where VLM adds extra text or markdown)
                # First try to find JSON in markdown code blocks
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', vlm_response, re.DOTALL)
                if not json_match:
                    # Fallback to finding any JSON-like structure
                    json_match = re.search(r'\{.*\}', vlm_response, re.DOTALL)
                
                if not json_match:
                    # RETRY: VLM didn't return JSON at all - try again with stricter prompt
                    signal.alarm(0)  # Cancel any lingering alarm before retry
                    logger.debug("[PERCEPTION] No JSON in VLM response, retrying with strict prompt")
                    logger.warning(f"[PERCEPTION] VLM failed to return JSON, retrying")
                    
                    retry_prompt = f"""Return ONLY a JSON object analyzing this Pokemon Emerald screenshot. No explanations.

Current state: {state_summary}

JSON format (fill in with observations from the image):
{{
    "screen_context": "overworld",
    "on_screen_text": {{
        "dialogue": null,
        "speaker": null,
        "menu_title": null
    }},
    "visible_entities": [],
    "navigation_info": {{
        "open_paths": []
    }},
    "visual_elements": {{
        "text_box_visible": false,
        "continue_prompt_visible": false
    }}
}}

Return ONLY the JSON, nothing else:"""
                    
                    signal.alarm(10)  # Shorter timeout for retry
                    try:
                        vlm_response = vlm.get_query(frame, retry_prompt, "PERCEPTION-RETRY")
                    finally:
                        signal.alarm(0)  # Always cancel alarm after retry attempt
                    
                    # Try to extract JSON again
                    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', vlm_response, re.DOTALL)
                    if not json_match:
                        json_match = re.search(r'\{.*\}', vlm_response, re.DOTALL)
                    
                    if not json_match:
                        print(f"❌ [PERCEPTION] Still no JSON after retry!")
                        logger.error(f"[PERCEPTION] VLM retry also failed to return JSON")
                
                if json_match:
                    # Get the JSON text (use group 1 if we matched markdown, otherwise group 0)
                    json_text = json_match.group(1) if json_match.lastindex else json_match.group(0)
                    logger.debug(f"[PERCEPTION] Extracted JSON: {json_text[:200]}")
                    
                    # Fix Python tuple syntax to JSON array syntax before parsing
                    json_text = re.sub(r'\((\d+),\s*(\d+)\)', r'[\1, \2]', json_text)
                    
                    # Try to fix common JSON malformation issues
                    # Add missing closing brace if needed
                    brace_count = json_text.count('{') - json_text.count('}')
                    if brace_count > 0:
                        json_text += '}' * brace_count
                        print(f"🔧 [PERCEPTION] Fixed missing closing braces: added {brace_count}")
                    
                    visual_data = json.loads(json_text)
                    logger.debug(f"[PERCEPTION] VLM extraction successful: {visual_data.get('screen_context', 'unknown')}")
                    logger.info("[PERCEPTION] VLM extraction successful")
                    
                    # ============================================================
                    # FIX MALFORMED on_screen_text - Sometimes VLM returns string instead of dict
                    # ============================================================
                    on_screen_text = visual_data.get('on_screen_text', {})
                    if isinstance(on_screen_text, str):
                        # VLM returned a string instead of a dict - convert it
                        logger.warning(f"[PERCEPTION] VLM returned on_screen_text as string: '{on_screen_text[:60]}'")
                        print(f"⚠️ [PERCEPTION] Fixing malformed on_screen_text (was string, expected dict)")
                        visual_data['on_screen_text'] = {
                            'dialogue': on_screen_text,
                            'speaker': None,
                            'menu_title': None
                        }
                    
                    # ============================================================
                    # SAVE RAW DIALOGUE - Before hallucination filtering
                    # ============================================================
                    # Battle bot needs access to unfiltered dialogue to detect "What will POKEMON do?"
                    # Save it before the hallucination filter clears it
                    raw_dialogue = visual_data.get('on_screen_text', {}).get('dialogue', '')
                    if raw_dialogue:
                        visual_data['on_screen_text']['raw_dialogue'] = raw_dialogue
                    
                    # ============================================================
                    # HALLUCINATION FILTERING - Known VLM hallucinations
                    # ============================================================
                    # The VLM consistently hallucinates specific phrases that are NOT in the game
                    # Filter these out immediately before any other processing
                    KNOWN_HALLUCINATIONS = [
                        "Player: What should I do?",
                        "Player: What should I do",
                        "What should I do?",
                        "What should I do",
                    ]
                    
                    dialogue = visual_data.get('on_screen_text', {}).get('dialogue', '')
                    if dialogue and isinstance(dialogue, str):
                        dialogue_cleaned = dialogue.strip()
                        for hallucination in KNOWN_HALLUCINATIONS:
                            if dialogue_cleaned.lower() == hallucination.lower() or \
                               dialogue_cleaned.lower().startswith(hallucination.lower()):
                                # print(f"🚫 [HALLUCINATION FILTER] Caught VLM hallucination: '{dialogue_cleaned[:60]}'")
                                # print(f"     This is a known false positive - clearing dialogue")
                                # logger.warning(f"[PERCEPTION] Filtered known VLM hallucination: {dialogue_cleaned}")
                                visual_data['on_screen_text']['dialogue'] = None
                                visual_data['on_screen_text']['speaker'] = None
                                # Don't mark as dialogue context if we filtered it out
                                if visual_data.get('screen_context') == 'dialogue':
                                    visual_data['screen_context'] = 'overworld'
                                if 'visual_elements' not in visual_data:
                                    visual_data['visual_elements'] = {}
                                visual_data['visual_elements']['text_box_visible'] = False
                                visual_data['visual_elements']['continue_prompt_visible'] = False
                                break
                    
                    # ============================================================
                    # FALSE POSITIVE FILTERING - HUD/Status text detection
                    # ============================================================
                    # The VLM sometimes mistakes HUD elements (player name, status bars) for dialogue
                    # Filter out HUD patterns that are not real dialogue boxes
                    def is_hud_text(data):
                        """Check if dialogue is actually just HUD/status text, not real dialogue"""
                        if not isinstance(data, dict):
                            return False
                        
                        dialogue = data.get('on_screen_text', {}).get('dialogue', '')
                        if not dialogue:
                            return False
                        
                        dialogue_str = str(dialogue).strip()
                        
                        import re
                        
                        # Pattern 1: Debug/Status HUD with pipe-separated fields
                        # Example: "Player: JOHNNY | Location: TOWN | Pos: (8, 7) | State: dialog | Money: $3000..."
                        if '|' in dialogue_str and any(keyword in dialogue_str for keyword in ['Location:', 'Pos:', 'State:', 'Money:', 'Pokedex:', 'Time:']):
                            return True
                        
                        # Pattern 2: Simple player name HUD
                        # Examples: "Player: JOHNNY", "PLAYER: JOHNNY"
                        player_hud_patterns = [
                            r'^Player:\s*\w+$',  # "Player: JOHNNY"
                            r'^PLAYER:\s*\w+$',  # "PLAYER: JOHNNY"
                            r'^\w+:\s*JOHNNY$',  # "Johnny: JOHNNY" (confused VLM)
                        ]
                        
                        for pattern in player_hud_patterns:
                            if re.match(pattern, dialogue_str, re.IGNORECASE):
                                return True
                        
                        # Pattern 3: VLM hallucination - describing the scene instead of reading dialogue
                        # Real dialogue is short character speech. Hallucinations are narrative descriptions.
                        # Examples: "You are in the middle of a forest. You can see a path leading..."
                        #           "You are now on Route 101. You have 3000 coins..."
                        #           "You are currently at position (7, 7). You have $3000 in your bank..."
                        hallucination_indicators = [
                            'you can see',
                            'you are in',
                            'you are on',
                            'you are now',
                            'you are currently',
                            'you have',
                            'there is a',
                            'there are',
                            'standing in the middle',
                            'group of trees',
                            'middle of a forest',
                            'small black and white pokemon',
                            'in the forest',
                            'path leading',
                            'in your bank',
                            'in your party',
                        ]
                        
                        dialogue_lower = dialogue_str.lower()
                        hallucination_count = sum(1 for indicator in hallucination_indicators if indicator in dialogue_lower)
                        
                        # If 2+ hallucination indicators, it's likely a scene description, not real dialogue
                        if hallucination_count >= 2:
                            return True
                        
                        # ENHANCED: Check for game state narration patterns
                        # These are dead giveaways that the VLM is describing the game state, not reading dialogue
                        game_state_patterns = [
                            'you have $',           # "You have $3000"
                            'you have 1 pokemon',   # "You have 1 Pokemon in your party"
                            'currently at position', # "You are currently at position (7, 7)"
                            'in your bank',         # "You have $3000 in your bank"
                            'in your party',        # "You have 1 Pokemon in your party"
                            'in the overworld',     # "You are currently in the overworld"
                        ]
                        
                        for pattern in game_state_patterns:
                            if pattern in dialogue_lower:
                                return True
                        
                        # Also check length - real dialogue boxes are usually concise (under 200 chars)
                        # Scene descriptions tend to be longer and more detailed
                        if len(dialogue_str) > 150 and any(indicator in dialogue_lower for indicator in ['you can see', 'you are in', 'there is', 'there are']):
                            return True
                        
                        return False
                    
                    if is_hud_text(visual_data):
                        dialogue_preview = str(visual_data.get('on_screen_text', {}).get('dialogue', ''))[:100]
                        print(f"🚫 [FALSE POSITIVE] VLM detected HUD/status text as dialogue!")
                        print(f"     Text: '{dialogue_preview}...'")
                        print(f"     This is NOT a dialogue box, clearing...")
                        logger.warning("[PERCEPTION] Filtered out HUD/status text mistaken for dialogue")
                        
                        # Clear the false positive dialogue
                        visual_data['on_screen_text']['dialogue'] = None
                        visual_data['on_screen_text']['speaker'] = None
                        visual_data['screen_context'] = 'overworld'  # Correct the context
                        if 'visual_elements' not in visual_data:
                            visual_data['visual_elements'] = {}
                        visual_data['visual_elements']['text_box_visible'] = False
                        visual_data['visual_elements']['continue_prompt_visible'] = False
                    
                    # ============================================================
                    # TEMPLATE TEXT DETECTION - VLM hallucination check
                    # ============================================================
                    # Sometimes VLM returns the instruction text instead of analyzing the image
                    # Check for common template phrases and flag as invalid
                    def contains_template_phrases(data):
                        """Check if data contains instruction text instead of real observations"""
                        if not isinstance(data, dict):
                            return False
                        
                        # Check dialogue field for instruction text
                        dialogue = data.get('on_screen_text', {}).get('dialogue', '')
                        if dialogue:
                            template_phrases = [
                                'ONLY text from dialogue boxes',
                                'DO NOT include HUD',
                                'If no dialogue box visible',
                                'exact text from dialogue',
                                'or null if none'
                            ]
                            dialogue_upper = str(dialogue).upper()
                            if any(phrase.upper() in dialogue_upper for phrase in template_phrases):
                                return True
                        
                        # Check if speaker field has instruction text
                        speaker = data.get('on_screen_text', {}).get('speaker', '')
                        if speaker and ('character name' in str(speaker).lower() or 'or null' in str(speaker).lower()):
                            return True
                        
                        return False
                    
                    if contains_template_phrases(visual_data):
                        print(f"⚠️ [TEMPLATE DETECTED] VLM returned instruction text instead of analyzing image!")
                        print(f"     This indicates VLM hallucination. Doing simple dialogue check...")
                        logger.warning("[PERCEPTION] VLM returned template text - doing simple dialogue/triangle check")
                        
                        # Try a much simpler VLM call to detect dialogue and red triangle
                        try:
                            signal.signal(signal.SIGALRM, timeout_handler)
                            signal.alarm(10)
                            simple_prompt = """Look at this Pokemon game screenshot. Answer these 2 questions:

1. Is there a white text box at the bottom of the screen? (YES/NO)
2. Do you see a small red triangle ❤️ or arrow at the end of the text? (YES/NO)

Answer in format: "1: YES/NO, 2: YES/NO" """
                            
                            simple_response = vlm.get_query(frame, simple_prompt, "SIMPLE_CHECK")
                            signal.alarm(0)
                            
                            print(f"🔍 [SIMPLE CHECK] Response: '{simple_response}'")
                            
                            # Parse response
                            response_upper = simple_response.upper()
                            has_text_box = '1' in response_upper and 'YES' in response_upper.split('2')[0]
                            has_triangle = '2' in response_upper and 'YES' in response_upper.split('2')[-1]
                            
                            # Update visual_data with simple check results
                            visual_data['on_screen_text']['dialogue'] = None
                            visual_data['on_screen_text']['speaker'] = None
                            if 'visual_elements' not in visual_data:
                                visual_data['visual_elements'] = {}
                            visual_data['visual_elements']['text_box_visible'] = has_text_box
                            visual_data['visual_elements']['continue_prompt_visible'] = has_triangle
                            
                            print(f"✅ [SIMPLE CHECK] text_box={has_text_box}, red_triangle={has_triangle}")
                            
                        except Exception as e:
                            signal.alarm(0)
                            print(f"⚠️ [SIMPLE CHECK] Failed: {e}")
                            # Clear and mark as no dialogue
                            visual_data['on_screen_text']['dialogue'] = None
                            visual_data['on_screen_text']['speaker'] = None
                            if 'visual_elements' not in visual_data:
                                visual_data['visual_elements'] = {}
                            visual_data['visual_elements']['text_box_visible'] = False
                            visual_data['visual_elements']['continue_prompt_visible'] = False
                    
                    # ============================================================
                    # QWEN-2B DIALOGUE FIX: Simple Yes/No dialogue check
                    # ============================================================
                    # Qwen-2B often copies template text instead of extracting dialogue.
                    # Add a second, very simple VLM call to directly ask about dialogue visibility.
                    is_qwen_2b = vlm and hasattr(vlm, 'model_name') and 'Qwen2-VL-2B-Instruct' in vlm.model_name
                    
                    if is_qwen_2b:
                        # Check if dialogue status is uncertain
                        visual_elements = visual_data.get('visual_elements', {})
                        text_box_visible = visual_elements.get('text_box_visible')
                        
                        # Only do secondary check if:
                        # 1. Last action was dialogue-related (pressing A to advance dialogue)
                        # 2. AND (text_box_visible is missing/None OR dialogue seems like template)
                        last_action_was_dialogue = (
                            recent_actions and 
                            len(recent_actions) > 0 and 
                            recent_actions[-1] == 'A'
                        )
                        
                        dialogue_text = visual_data.get('on_screen_text', {}).get('dialogue', '')
                        needs_secondary_check = (
                            last_action_was_dialogue and
                            (text_box_visible is None or 
                             (dialogue_text and is_template_text(dialogue_text)))
                        )
                        
                        if needs_secondary_check:
                            print(f"🔍 [QWEN-2B FIX] Running secondary dialogue check (last_action=A, text_box_visible={text_box_visible}, has_template={is_template_text(dialogue_text) if dialogue_text else False})")
                            logger.info("[PERCEPTION] Qwen-2B: Performing secondary dialogue visibility check (last action was A)")
                            
                            try:
                                # More specific prompt to avoid false positives (e.g., cardboard boxes in moving van)
                                # Explicitly mention "white text box" and "character dialogue" to be clear
                                simple_dialogue_prompt = "Is there a white text box at the bottom of the screen showing character dialogue or speech? Answer YES or NO."
                                
                                # Make second VLM call with timeout
                                signal.signal(signal.SIGALRM, timeout_handler)
                                signal.alarm(10)  # Shorter timeout for simple query
                                
                                dialogue_check_response = vlm.get_query(frame, simple_dialogue_prompt, "DIALOGUE_CHECK")
                                signal.alarm(0)  # Cancel timeout
                                
                                print(f"🔍 [QWEN-2B FIX] Dialogue check response: '{dialogue_check_response}'")
                                
                                # Parse YES/NO response
                                response_upper = dialogue_check_response.strip().upper()
                                has_dialogue_box = 'YES' in response_upper and 'NO' not in response_upper[:10]  # YES should come before NO
                                
                                # Update visual_data
                                if 'visual_elements' not in visual_data:
                                    visual_data['visual_elements'] = {}
                                visual_data['visual_elements']['text_box_visible'] = has_dialogue_box
                                
                                print(f"✅ [QWEN-2B FIX] Secondary check complete: text_box_visible={has_dialogue_box}")
                                logger.info(f"[PERCEPTION] Qwen-2B dialogue check: text_box_visible={has_dialogue_box}")
                                
                                # OVERRIDE LOGIC: If VLM extracted real dialogue text AND classified as dialogue screen,
                                # but secondary check says NO, trust the primary extraction
                                if not has_dialogue_box and dialogue_text and not is_template_text(dialogue_text):
                                    screen_context = visual_data.get('screen_context', '')
                                    if screen_context == 'dialogue' and len(dialogue_text) > 10:
                                        print(f"🔧 [QWEN-2B FIX] Secondary check said NO but primary extracted real dialogue:")
                                        print(f"     Dialogue: '{dialogue_text[:50]}...'")
                                        print(f"     Screen context: '{screen_context}'")
                                        print(f"     Overriding: text_box_visible = True")
                                        visual_data['visual_elements']['text_box_visible'] = True
                                        logger.info("[PERCEPTION] Overriding secondary check - primary extraction has strong dialogue evidence")
                                
                                # If dialogue box is visible but text was template, clear it
                                if has_dialogue_box and dialogue_text and is_template_text(dialogue_text):
                                    visual_data['on_screen_text']['dialogue'] = None
                                    visual_data['screen_context'] = 'dialogue'
                                    print(f"🔧 [QWEN-2B FIX] Cleared template text, set screen_context=dialogue")
                                
                            except (TimeoutError, Exception) as secondary_error:
                                signal.alarm(0)  # Cancel timeout
                                print(f"⚠️ [QWEN-2B FIX] Secondary dialogue check failed: {secondary_error}")
                                logger.warning(f"[PERCEPTION] Qwen-2B dialogue check failed: {secondary_error}")
                                # Default to False if check fails
                                if 'visual_elements' not in visual_data:
                                    visual_data['visual_elements'] = {}
                                visual_data['visual_elements']['text_box_visible'] = False
                        else:
                            # Skip secondary check - explain why
                            if not last_action_was_dialogue:
                                last_action = recent_actions[-1] if recent_actions else 'None'
                                print(f"⏭️ [QWEN-2B FIX] Skipping secondary dialogue check (last_action={last_action}, not dialogue-related)")
                                logger.info(f"[PERCEPTION] Skipping secondary dialogue check - last action was '{last_action}', not 'A'")
                    
                    # ============================================================
                    # END QWEN-2B FIX
                    # ============================================================
                    
                    # CRITICAL FIX: Check if VLM returned template text instead of real dialogue
                    # If so, use OCR as fallback to detect actual dialogue
                    dialogue_text = visual_data.get('on_screen_text', {}).get('dialogue', '')
                    if dialogue_text and is_template_text(dialogue_text):
                        print(f"⚠️ [PERCEPTION] VLM returned template text, using OCR fallback for dialogue detection")
                        try:
                            from utils.ocr_dialogue import create_ocr_detector
                            detector = create_ocr_detector()
                            ocr_dialogue = detector.detect_dialogue_from_screenshot(frame)
                            
                            if ocr_dialogue and ocr_dialogue.strip():
                                logger.debug(f"[OCR FALLBACK] Detected dialogue: '{ocr_dialogue}'")
                                visual_data['on_screen_text']['dialogue'] = ocr_dialogue
                                visual_data['screen_context'] = 'dialogue'
                                visual_data['visual_elements']['text_box_visible'] = True
                            else:
                                logger.debug("[OCR FALLBACK] No dialogue detected")
                                visual_data['on_screen_text']['dialogue'] = None
                        except Exception as ocr_error:
                            print(f"❌ [OCR FALLBACK] Failed: {ocr_error}")
                            # Clear template text even if OCR fails
                            visual_data['on_screen_text']['dialogue'] = None
                    
                else:
                    print(f"❌ [PERCEPTION] No JSON found in VLM response!")
                    logger.warning("[PERCEPTION] VLM response not in JSON format, using fallback")
                    
            except (TimeoutError, json.JSONDecodeError, Exception) as e:
                signal.alarm(0)  # Cancel timeout
                print(f"❌ [PERCEPTION] VLM extraction failed: {e}")
                logger.warning(f"[PERCEPTION] VLM extraction failed: {e}, using fallback")
                visual_data = None
                
    except Exception as e:
        logger.warning(f"[PERCEPTION] VLM setup failed: {e}, using fallback")
        visual_data = None
    
    # Fallback to programmatic analysis if VLM failed
    if visual_data is None:
        print(f"🔧 [PERCEPTION] VLM failed - using programmatic fallback")
        logger.info("[PERCEPTION] Using programmatic fallback analysis")
        visual_data = create_programmatic_visual_data(game_state, in_battle, current_location, game_data)
        print(f"🔧 [PERCEPTION] Fallback result: {visual_data.get('screen_context', 'unknown')}")
        
        # CRITICAL FIX: During title sequence, use OCR to detect dialogue
        # VLM often fails during intro cutscene, but OCR is reliable for text detection
        # BYPASS dialogue box detection since title sequence may have different borders
        if current_location == 'TITLE_SEQUENCE' and frame is not None:
            print(f"🔍 [TITLE SEQUENCE] Using OCR fallback for dialogue detection")
            logger.info("[PERCEPTION] Title sequence - using OCR for dialogue detection")
            try:
                from utils.ocr_dialogue import create_ocr_detector
                detector = create_ocr_detector()
                
                # CRITICAL: Temporarily bypass dialogue box detection for title sequence
                # Title sequence dialogue may have different border colors than normal gameplay
                # Also enable full frame scan to catch all text
                original_skip_setting = detector.skip_dialogue_box_detection
                original_full_frame = detector.use_full_frame_scan
                detector.skip_dialogue_box_detection = True
                detector.use_full_frame_scan = True  # Enable full-frame scanning for title sequence
                
                ocr_dialogue = detector.detect_dialogue_from_screenshot(frame)
                
                # Restore original settings
                detector.skip_dialogue_box_detection = original_skip_setting
                detector.use_full_frame_scan = original_full_frame
                
                if ocr_dialogue and ocr_dialogue.strip():
                    print(f"✅ [OCR TITLE] Detected dialogue: '{ocr_dialogue[:100]}...'")
                    visual_data['on_screen_text']['dialogue'] = ocr_dialogue
                    visual_data['screen_context'] = 'dialogue'
                    visual_data['visual_elements']['text_box_visible'] = True
                    logger.info(f"[PERCEPTION] OCR detected dialogue during title sequence: {ocr_dialogue[:50]}...")
                else:
                    print(f"ℹ️  [OCR TITLE] No dialogue detected")
                    logger.info("[PERCEPTION] OCR found no dialogue during title sequence")
            except Exception as ocr_error:
                print(f"❌ [OCR TITLE] Failed: {ocr_error}")
                logger.warning(f"[PERCEPTION] OCR failed during title sequence: {ocr_error}")
    else:
        logger.debug(f"[PERCEPTION] VLM success - screen_context={visual_data.get('screen_context', 'unknown')}")
    
    # ============================================================================
    # 🔍 OCR AUGMENTATION - Run OCR on EVERY step to support dialogue detection
    # ============================================================================
    # OCR provides reliable dialogue detection that doesn't depend on VLM success.
    # This prevents loops where VLM fails and battle bot can't detect menu states.
    # 
    # Strategy: Run OCR always, merge results with VLM data
    # - If VLM has dialogue: Keep VLM, store OCR as backup
    # - If VLM has no dialogue: Use OCR dialogue if available
    # - Always store OCR result for debugging/fallback
    if frame is not None:
        try:
            from utils.ocr_dialogue import create_ocr_detector
            detector = create_ocr_detector()
            
            # CRITICAL: If VLM detected text_box_visible, bypass dialogue box detection
            # VLM is better at detecting dialogue boxes than OCR color matching
            vlm_text_box_visible = visual_data.get('visual_elements', {}).get('text_box_visible', False)
            if vlm_text_box_visible:
                # VLM says dialogue box is visible - force OCR to extract text
                original_skip = detector.skip_dialogue_box_detection
                detector.skip_dialogue_box_detection = True
                ocr_dialogue = detector.detect_dialogue_from_screenshot(frame)
                detector.skip_dialogue_box_detection = original_skip
                if ocr_dialogue:
                    print(f"✅ [OCR FORCED] VLM detected text box, OCR extracted: '{ocr_dialogue[:60]}...'")
            else:
                # Normal OCR with dialogue box detection
                ocr_dialogue = detector.detect_dialogue_from_screenshot(frame)
            
            # Also try to extract all visible text for menu detection
            # This helps detect battle menu states like "What will X do?"
            try:
                ocr_all_text = detector.detect_all_text_regions(frame)
            except:
                ocr_all_text = []
            
            # Store OCR results
            if 'ocr_data' not in visual_data:
                visual_data['ocr_data'] = {}
            
            if ocr_dialogue and ocr_dialogue.strip():
                visual_data['ocr_data']['dialogue'] = ocr_dialogue
                
                # If VLM has no dialogue, use OCR
                vlm_dialogue = visual_data.get('on_screen_text', {}).get('dialogue', '')
                if not vlm_dialogue or vlm_dialogue.strip() == '':
                    print(f"🔍 [OCR] VLM dialogue empty, using OCR: '{ocr_dialogue[:60]}...'")
                    logger.info(f"[OCR] Using OCR dialogue (VLM empty): {ocr_dialogue[:50]}...")
                    visual_data['on_screen_text']['dialogue'] = ocr_dialogue
                    visual_data['on_screen_text']['raw_dialogue'] = ocr_dialogue
                    # Update context if we detected dialogue
                    if visual_data.get('screen_context') == 'overworld':
                        visual_data['screen_context'] = 'dialogue'
                    if 'visual_elements' not in visual_data:
                        visual_data['visual_elements'] = {}
                    visual_data['visual_elements']['text_box_visible'] = True
                else:
                    # VLM has dialogue - keep it but log OCR for comparison
                    logger.debug(f"[OCR] VLM dialogue present, keeping VLM (OCR backup: '{ocr_dialogue[:40]}...')")
                    logger.debug(f"[OCR] VLM dialogue: '{vlm_dialogue[:40]}', OCR: '{ocr_dialogue[:40]}'")
            else:
                logger.debug("[OCR] No dialogue detected")
            
            # Store all text regions for debugging (helps with menu detection)
            if ocr_all_text:
                visual_data['ocr_data']['all_text_regions'] = ocr_all_text
                logger.debug(f"[OCR] Found {len(ocr_all_text)} text regions")
                
        except Exception as ocr_error:
            print(f"⚠️ [OCR] Failed: {ocr_error}")
            logger.warning(f"[OCR] OCR extraction failed: {ocr_error}")
    
    # Create structured observation
    observation = {
        "visual_data": visual_data,
        "state_summary": state_summary,
        "extraction_method": "vlm" if visual_data.get("_source") != "programmatic" else "programmatic",
        # Backwards compatibility for memory logging
        "description": visual_data.get("scene_description", f"Screen: {visual_data.get('screen_context', 'unknown')}")
    }
    
    # Format state context for memory purposes
    state_context = format_state_for_llm(state_data)
    observation["state_data"] = state_context
    
    # Final timing
    total_time = time.time() - perception_start
    
    logger.info(f"[PERCEPTION] Extraction completed via {observation['extraction_method']} method")
    logger.info(f"[PERCEPTION] ⏱️  TOTAL TIME: {total_time:.3f}s (setup: {setup_time:.3f}s)")
    return observation


def create_programmatic_visual_data(game_state, in_battle, current_location, game_data):
    """
    Create structured visual data using programmatic analysis as fallback.
    Returns the same JSON structure as VLM extraction.
    """
    if game_state == 'title':
        return {
            "_source": "programmatic",
            "screen_context": "title",
            "on_screen_text": {
                "dialogue": None,
                "speaker": None,
                "menu_title": "Pokemon Emerald",
                "button_prompts": ["Press A to continue"]
            },
            "visible_entities": [],
            "navigation_info": {
                "exits_visible": [],
                "interactable_objects": ["Title screen menu"],
                "movement_barriers": [],
                "open_paths": []
            },
            "spatial_layout": {
                "player_position": "title screen",
                "room_type": "menu",
                "notable_features": ["Title screen interface"]
            },
            "menu_state": "title_screen",
            "visual_elements": {
                "health_bars_visible": False,
                "pokemon_sprites_visible": False,
                "overworld_map_visible": False,
                "text_box_visible": False
            }
        }
    elif in_battle:
        battle_info = game_data.get('battle_info', {})
        player_pokemon = battle_info.get('player_pokemon') or {}
        opponent_pokemon = battle_info.get('opponent_pokemon') or {}
        
        return {
            "_source": "programmatic",
            "screen_context": "battle",
            "on_screen_text": {
                "dialogue": None,
                "speaker": None,
                "menu_title": None,
                "button_prompts": []
            },
            "visible_entities": [
                {
                    "type": "player_pokemon",
                    "name": player_pokemon.get('species', 'Unknown') if player_pokemon else 'Unknown',
                    "position": "bottom_left"
                },
                {
                    "type": "opponent_pokemon", 
                    "name": opponent_pokemon.get('species', 'Unknown') if opponent_pokemon else 'Unknown',
                    "position": "top_right"
                }
            ],
            "navigation_info": {
                "exits_visible": [],
                "interactable_objects": ["Battle menu options"],
                "movement_barriers": [],
                "open_paths": []
            },
            "spatial_layout": {
                "player_position": "battle screen",
                "room_type": "battle",
                "notable_features": ["Pokemon battle interface"]
            },
            "menu_state": "battle_menu",
            "visual_elements": {
                "health_bars_visible": True,
                "pokemon_sprites_visible": True,
                "overworld_map_visible": False,
                "text_box_visible": False
            }
        }
    elif current_location and current_location != 'Unknown':
        return {
            "_source": "programmatic",
            "screen_context": "overworld",
            "on_screen_text": {
                "dialogue": None,
                "speaker": None,
                "menu_title": None,
                "button_prompts": []
            },
            "visible_entities": [],
            "navigation_info": {
                "exits_visible": ["Check for exits in all directions"],
                "interactable_objects": ["Look for NPCs, items, or interactive elements"],
                "movement_barriers": ["Unknown - check programmatically"],
                "open_paths": ["Movement depends on traversability data"]
            },
            "spatial_layout": {
                "player_position": f"In {current_location}",
                "room_type": "overworld",
                "notable_features": ["Check for doors, exits, or stairs"]
            },
            "menu_state": "closed",
            "visual_elements": {
                "health_bars_visible": False,
                "pokemon_sprites_visible": False,
                "overworld_map_visible": True,
                "text_box_visible": False
            }
        }
    else:
        return {
            "_source": "programmatic",
            "screen_context": "unknown",
            "on_screen_text": {
                "dialogue": None,
                "speaker": None,
                "menu_title": None,
                "button_prompts": []
            },
            "visible_entities": [],
            "navigation_info": {
                "exits_visible": [],
                "interactable_objects": [],
                "movement_barriers": [],
                "open_paths": []
            },
            "spatial_layout": {
                "player_position": "unknown",
                "room_type": "unknown",
                "notable_features": []
            },
            "menu_state": "unknown",
            "visual_elements": {
                "health_bars_visible": False,
                "pokemon_sprites_visible": False,
                "overworld_map_visible": False,
                "text_box_visible": False
            }
        } 