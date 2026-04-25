"""
Opener Bot - STATEFUL State Machine for Game Opening (Splits 0-4)

This module implements a STATEFUL rule-based controller that reliably handles
the deterministic opening sequence of Pokemon Emerald, from title screen through
starter selection.

===============================================================================
ARCHITECTURE OVERVIEW
===============================================================================

**State Machine Design:**
- STATEFUL: Bot remembers current state (self.current_state_name)
- Each frame, only checks transition conditions from current state
- Prevents state oscillation (e.g., S6_NAV_TO_CLOCK ↔ S8_NAV_OUT_OF_HOUSE)
- States use explicit next_state_fn to define completion criteria

**Signal Hierarchy:**
1. PRIMARY: Memory state (milestones, game_state) - 100% reliable
2. SECONDARY: Visual elements (text_box_visible) - 85% reliable  
3. TERTIARY: VLM text parsing - 60% reliable (hint only)

**State Lifecycle:**
```
Title Screen → Name Selection → House Interior → Exit House → 
Choose Starter → Exit Lab → COMPLETED
```

**COMPLETED State Behavior:**
Once STARTER_CHOSEN milestone is achieved AND player exits Birch's Lab,
the bot transitions to COMPLETED state and PERMANENTLY hands off to VLM.

The bot will NOT re-activate even if player returns to lab later. This is
intentional to prevent interference with normal gameplay after the opening.

===============================================================================
USAGE
===============================================================================

The opener_bot is called from action.py as Priority 0 (before all other logic):

```python
# In action_step():
opener_bot = get_opener_bot()
if opener_bot.should_handle(state_data, visual_data):
    action = opener_bot.get_action(state_data, visual_data)
    if action:
        return action  # Bot handles this step
# Otherwise, fall through to VLM navigation
```

Once COMPLETED, should_handle() always returns False, permanently handing off
control to the VLM-based agent.

===============================================================================
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Script-idle polling helper
# ---------------------------------------------------------------------------
# During TITLE_SEQUENCE the emulator runs at ~60 FPS.  Prof Birch's dialogue
# boxes take ~25-35 frames to fully scroll.  If an A-press lands while text
# is still printing it only speeds up the scroll; a *second* A is needed to
# actually advance.  This makes the step counter unreliable.
#
# wait_for_script_idle() polls /script_state (added to server/app.py) which
# reads sGlobalScriptContext.mode directly from GBA RAM.  Mode 0 means the
# script engine is stopped and the game is waiting for the A button.  We poll
# until idle or the timeout expires, then the caller can safely send A.

def wait_for_script_idle(
    server_url: str = "http://localhost:8000",
    timeout: float = 2.0,
    poll_interval: float = 0.08,
) -> bool:
    """Block until the GBA script context is idle (safe to press A) or timeout.

    Args:
        server_url:    Base URL of the game server.
        timeout:       Maximum seconds to wait before giving up.
        poll_interval: Seconds between poll requests.

    Returns:
        True if the script became idle within the timeout, False if we gave up.
    """
    try:
        import requests as _req
    except ImportError:
        return True  # Can't poll — caller proceeds optimistically

    t_start = time.monotonic()
    deadline = t_start + timeout
    polls = 0
    while time.monotonic() < deadline:
        try:
            resp = _req.get(f"{server_url}/script_state", timeout=0.5)
            if resp.status_code == 200:
                data = resp.json()
                idle        = data.get("idle", True)
                script_mode = data.get("script_mode", "?")
                imm_mode    = data.get("immediate_mode", "?")
                reason      = data.get("reason", "")
                polls += 1
                if polls == 1:
                    # Always print the first read so we can see what the endpoint returns
                    tag = "idle" if idle else f"BUSY (mode={script_mode} imm={imm_mode})"
                    extra = f" [{reason}]" if reason else ""
                    print(f"🔬 [SCRIPT_IDLE] first read → {tag}{extra}")
                if idle:
                    waited_ms = int((time.monotonic() - t_start) * 1000)
                    if waited_ms > 20:
                        print(f"🔬 [SCRIPT_IDLE] became idle after {polls} poll(s) / {waited_ms}ms")
                    return True
                print(f"🔬 [SCRIPT_IDLE] poll {polls}: still busy (mode={script_mode} imm={imm_mode}) — waiting…")
            else:
                print(f"🔬 [SCRIPT_IDLE] HTTP {resp.status_code} — proceeding")
                return True
        except Exception as exc:
            # Server busy / transient error — treat as idle to avoid blocking
            print(f"🔬 [SCRIPT_IDLE] request error ({exc}) — proceeding")
            return True
        time.sleep(poll_interval)

    elapsed_ms = int((time.monotonic() - t_start) * 1000)
    print(f"🔬 [SCRIPT_IDLE] timeout after {elapsed_ms}ms / {polls} poll(s) — proceeding anyway")
    return False

MAX_MOVEMENT_BATCH_SIZE = 10  # ~1.3 seconds of movement at 60 FPS

# Global tracker for dismissed player monologues (shared across all functions)
# This prevents VLM hallucinations from causing infinite loops
_dismissed_monologues = set()

@dataclass
class NavigationGoal:
    """A data class to represent a navigation sub-goal for the A* Navigator."""
    x: int
    y: int
    map_location: str
    description: str
    should_interact: bool = None  # If None, infer from description keywords

@dataclass
class ForceDialogueGoal:
    """
    Special goal type to force pressing 'A' button for misclassified dialogue.
    Used when we detect dialogue blocking movement that was misclassified by the visual system
    (e.g., "................................" thinking dialogue).
    """
    reason: str  # Why we're forcing A (for logging)

@dataclass
class BotState:
    """Represents a state in the stateful opener bot state machine"""
    name: str
    description: str
    
    # Action or Goal to execute while in this state
    action_fn: Callable[[Dict[str, Any], Dict[str, Any]], Union[List[str], NavigationGoal, None]]
    
    # Transition logic - returns NEXT STATE NAME when complete, None to stay
    next_state_fn: Callable[[Dict[str, Any], Dict[str, Any]], Optional[str]]
    
    # Safety limits
    max_attempts: int = 60
    timeout_seconds: float = 180.0


class OpenerBot:
    """
    A STATEFUL state machine for the Pokemon Emerald opening sequence.
    It remembers its current state and only checks transition conditions from that state.
    """
    
    def __init__(self):
        """Initialize the stateful opener bot"""
        self.states = self._build_state_machine()
        self.current_state_name: str = 'S0_TITLE_SCREEN'  # Start at first state
        self.state_entry_time: float = time.time()
        self.state_attempt_count: int = 0
        self.last_action: Any = None
        self.state_history: List[tuple] = []
        self.initialized_state: bool = False  # Track if we've auto-detected state
        
        logger.info("[OPENER BOT] Initialized STATEFUL at state: S0_TITLE_SCREEN")
    
    def get_current_state(self) -> str:
        """Get the current state name"""
        return self.current_state_name

    def _transition_to_state(self, new_state: str):
        """Transition to a new state, resetting counters"""
        if new_state not in self.states:
            logger.error(f"[OPENER BOT] Unknown state transition: {new_state}")
            new_state = 'COMPLETED'
            
        old_state = self.current_state_name
        self.current_state_name = new_state
        self.state_entry_time = time.time()
        self.state_attempt_count = 0
        self.last_action = None
        
        logger.info(f"[OPENER BOT] 🔄 State Transition: {old_state} -> {new_state}")
        self.state_history.append((self.current_state_name, time.time(), "TRANSITION"))

    def should_handle(self, state_data: Dict[str, Any], visual_data: Dict[str, Any]) -> bool:
        """
        Determines if the opener bot should be active.

        Once the state machine reaches COMPLETED (after nickname/dialogue
        handling in S23/S24), it permanently hands off to ObjectiveManager.
        ObjectiveManager then handles lab exit and all subsequent navigation.
        """
        # Once COMPLETED, permanently hand off to ObjectiveManager
        if self.current_state_name == 'COMPLETED':
            return False

        milestones = state_data.get('milestones', {})
        starter_chosen = milestones.get('STARTER_CHOSEN', {}).get('completed', False)
        player_loc = state_data.get('player', {}).get('location', '')

        # Safety net: if somehow still active after starter and outside lab
        if starter_chosen and 'PROFESSOR BIRCHS LAB' not in player_loc:
            self._transition_to_state('COMPLETED')
            return False

        print(f"🤖 [OPENER BOT SHOULD_HANDLE] Bot is ACTIVE, will handle action")
        return True  # Bot is active

    def get_action(self, state_data: Dict[str, Any], visual_data: Dict[str, Any], 
                   current_plan: str = "") -> Union[List[str], NavigationGoal, None]:
        """
        Main stateful logic loop:
        1. Check for dialogue override (yield to dialogue system)
        2. Get current state
        3. Check safety fallbacks
        4. Check if state's transition condition is met (if yes, transition)
        5. Execute current state's action
        """
        # CRITICAL: Check for active NPC dialogue FIRST
        # If dialogue is active, YIELD to dialogue detection system (Priority 1)
        # This prevents the opener bot from spamming A during actual NPC conversations
        visual_elements = visual_data.get('visual_elements', {})
        on_screen_text = visual_data.get('on_screen_text', {})
        
        # Check for REAL dialogue indicators (not player's internal monologue)
        continue_prompt_visible = visual_elements.get('continue_prompt_visible', False)
        text_box_visible = visual_elements.get('text_box_visible', False)
        dialogue_text = on_screen_text.get('dialogue') or ''  # Handle None from hallucination filter
        speaker = on_screen_text.get('speaker') or ''
        
        # Player monologue detection - ONLY check dialogue text prefix
        # Do NOT use speaker field - unreliable (Mom talking TO Casey shows speaker="CASEY")
        is_player_monologue = (dialogue_text and dialogue_text.strip().upper().startswith('PLAYER:'))
        
        # DOT DIALOGUE DETECTION: VLM often can't read "................................" dialogue
        # and hallucinates "Player: What should I do?" instead
        # If dialogue is mostly dots/periods, treat as REAL NPC dialogue (thinking/pause)
        dialogue_stripped = dialogue_text.strip() if dialogue_text else ""
        is_dot_dialogue = False
        if dialogue_stripped and not is_player_monologue:
            # Count dots vs total characters (excluding spaces)
            non_space_chars = dialogue_stripped.replace(' ', '')
            if len(non_space_chars) > 3:  # At least 4 chars to avoid false positives
                dot_count = non_space_chars.count('.')
                dot_ratio = dot_count / len(non_space_chars)
                if dot_ratio > 0.7:  # More than 70% dots = thinking dialogue
                    is_dot_dialogue = True
                    print(f"🤖 [OPENER BOT] Dot dialogue detected ({dot_count}/{len(non_space_chars)} dots)")
        
        # CLOCK DIALOGUE DETECTION: Special case - don't yield on clock dialogue
        # The clock needs special handling (UP then A for Yes/No menu)
        # Let state machine transition to S7_SET_CLOCK which uses action_special_clock
        dialogue_upper = dialogue_text.upper()
        is_clock_dialogue = (
            "THE CLOCK" in dialogue_upper or
            "SET IT AND START IT" in dialogue_upper or
            "IS THIS" in dialogue_upper and "CORRECT TIME" in dialogue_upper
        )
        
        # NICKNAME DIALOGUE DETECTION: Special case - don't yield on nickname dialogue
        # The nickname screen needs special handling (B → START → A sequence)
        # Let state machine transition to S24_NICKNAME which uses action_special_nickname
        menu_title = (on_screen_text.get('menu_title', '') or '').upper()
        is_nickname_dialogue = (
            "NICKNAME" in dialogue_upper or
            "NICKNAME" in menu_title
        )
        
        # NAMING SCREEN DETECTION: Special case - don't yield on naming screen
        # The naming screen needs special handling (B → START → A sequence)
        # Indicators: "YOUR NAME?" in dialogue or menu_title, screen_context='menu'
        screen_context = visual_data.get('screen_context', '').lower()
        is_naming_screen = (
            ("YOUR NAME" in dialogue_upper or "YOUR NAME" in menu_title) and
            (screen_context == 'menu' or 'NAME' in menu_title)
        )
        
        # STATE-BASED EXCEPTION: Check if current state is designed to handle dialogue
        # States like S4_MOM_DIALOG_1F, S1_PROF_DIALOG, etc. should NOT yield even when dialogue is present
        # They use action_clear_dialogue or action_clear_dialogue_persistent to handle it
        # 
        # CRITICAL DECISION: The opener bot is designed to handle the ENTIRE intro sequence
        # deterministically. It should NEVER yield to the dialogue system during the opening.
        # All dialogue in the intro (Mom, TV cutscene, etc.) is part of the scripted sequence.
        # Therefore, we treat ALL states as dialogue-handling states during the opener.
        is_dialogue_handling_state = True  # Opener bot ALWAYS handles its own dialogue
        
        # DIALOGUE DETECTION: Yield to dialogue system if we see dialogue (and it's NOT player monologue)
        # EXCEPTION 1: Special screens (clock, nickname, naming) - state machine handles them
        # EXCEPTION 2: Current state is designed to handle dialogue - don't yield
        # EXCEPTION 3: OPENER BOT NEVER YIELDS - it handles the entire intro deterministically
        # SPECIAL CASE: Dot dialogue is always treated as real dialogue
        is_real_dialogue = (continue_prompt_visible or text_box_visible) and (is_dot_dialogue or (not is_player_monologue and not is_clock_dialogue and not is_nickname_dialogue and not is_naming_screen and not is_dialogue_handling_state))
        
        # DEBUG: Always log dialogue detection
        print(f"🤖 [OPENER BOT DIALOGUE CHECK] text_box={text_box_visible}, continue_prompt={continue_prompt_visible}, player_mono={is_player_monologue}, clock={is_clock_dialogue}, nickname={is_nickname_dialogue}, naming={is_naming_screen}, dots={is_dot_dialogue}, is_real={is_real_dialogue}")
        if dialogue_text:
            print(f"🤖 [OPENER BOT DIALOGUE CHECK] Dialogue: '{dialogue_text[:60]}'...")
        
        if is_real_dialogue:
            print(f"🤖 [OPENER BOT] YIELDING to dialogue system (speaker: {speaker}, continue_prompt: {continue_prompt_visible})")
            logger.info(f"[OPENER BOT] Dialogue active, yielding to Priority 1 dialogue detection")
            return None  # Let dialogue detection (Priority 1) handle this
        elif is_player_monologue:
            print(f"🤖 [OPENER BOT] Player monologue detected - ignoring and continuing with state logic")
        elif is_clock_dialogue:
            print(f"🤖 [OPENER BOT] Clock dialogue detected - letting state machine handle it")
            logger.info(f"[OPENER BOT] Clock dialogue detected - state machine will handle it")
        elif is_nickname_dialogue:
            print(f"🤖 [OPENER BOT] Nickname dialogue detected - letting state machine handle it")
            logger.info(f"[OPENER BOT] Nickname dialogue detected - state machine will handle it")
        elif is_naming_screen:
            print(f"🤖 [OPENER BOT] Naming screen detected - letting state machine handle it")
            logger.info(f"[OPENER BOT] Naming screen detected - state machine will handle it")
        
        # FAILED MOVEMENT DETECTION: Safety net for misclassified dialogue
        # If agent tries to move but position doesn't change for 3+ frames while game_state='dialog',
        # it's probably real dialogue that VLM misclassified (e.g., "................................")
        # Track position history to detect failed movements
        if not hasattr(self, '_movement_history'):
            self._movement_history = []  # List of position tuples
        if not hasattr(self, '_last_position'):
            self._last_position = None
        
        current_pos_tuple = (
            state_data.get('player', {}).get('position', {}).get('x'),
            state_data.get('player', {}).get('position', {}).get('y'),
            state_data.get('player', {}).get('location', '')
        )
        
        # Check if position changed since last frame
        if self._last_position is not None and self._last_position == current_pos_tuple:
            # Position didn't change - possible failed movement
            self._movement_history.append(current_pos_tuple)
        else:
            # Position changed or first frame - reset history
            self._movement_history = [current_pos_tuple]
        
        # Update last position for next frame
        self._last_position = current_pos_tuple
        
        # Check for stuck pattern: 3+ failed movements while in dialog state
        # BUT: Don't trigger for clock dialogue (which has special handling)
        game_state = state_data.get('game', {}).get('game_state', '').lower()
        if len(self._movement_history) >= 3 and game_state == 'dialog' and not is_clock_dialogue:
            # Agent has tried to move 3+ times without position changing, and game is in dialog state
            # This is likely real dialogue blocking movement that VLM misclassified
            print(f"⚠️ [FAILED MOVEMENT] Position stuck for {len(self._movement_history)} frames during dialog state")
            print(f"⚠️ [FAILED MOVEMENT] Likely misclassified dialogue - will force VLM to choose A")
            self._movement_history = []  # Reset history
            # Return ForceDialogueGoal to force A press
            # Dialogue is blocking movement but was misclassified by visual system
            return ForceDialogueGoal(reason="Position stuck during dialog state - likely misclassified dialogue")
        
        # Debug: Show what data we received
        player_pos = state_data.get('player', {}).get('position', {})
        player_loc = state_data.get('player', {}).get('location', '')
        print(f"🤖 [OPENER BOT GET_ACTION] ========================================")
        print(f"🤖 [OPENER BOT GET_ACTION] Current state: {self.current_state_name}")
        print(f"🤖 [OPENER BOT GET_ACTION] Player position: ({player_pos.get('x', '?')}, {player_pos.get('y', '?')})")
        print(f"🤖 [OPENER BOT GET_ACTION] Player location: {player_loc}")
        print(f"🤖 [OPENER BOT GET_ACTION] Attempt: {self.state_attempt_count + 1}")
        
        # AUTO-DETECT STARTING STATE on first call OR if party data just became available OR significant location change
        party = state_data.get('party', [])
        has_party = len(party) > 0 if party else False
        player_loc = state_data.get('player', {}).get('location', '')
        
        print(f"🤖 [OPENER BOT DEBUG] Party check: party={party}, has_party={has_party}, current_state={self.current_state_name}")
        
        # Track last location to detect significant changes
        if not hasattr(self, '_last_detected_location'):
            self._last_detected_location = None
        
        # Force re-detection if:
        # 1. Party appeared after being in S20_INTERACT_BAG
        # 2. Location changed significantly (e.g., warp to MOVING_VAN from TITLE_SEQUENCE)
        force_redetect_party = (has_party and self.current_state_name == 'S20_INTERACT_BAG')
        force_redetect_location = (
            self._last_detected_location is not None and 
            player_loc != self._last_detected_location and
            player_loc in ['MOVING_VAN', 'PLAYERS_HOUSE_1F', 'PLAYERS_HOUSE_2F', 'ROUTE_101', 'PROFESSOR BIRCHS LAB']
        )
        
        print(f"🤖 [OPENER BOT DEBUG] Force redetect: party={force_redetect_party}, location={force_redetect_location}")
        
        if not self.initialized_state or force_redetect_party or force_redetect_location:
            if force_redetect_party:
                print(f"🤖 [OPENER BOT REDETECT] Party data now available ({len(party)} Pokemon), re-detecting state from S20")
            if force_redetect_location:
                print(f"🤖 [OPENER BOT REDETECT] Significant location change: {self._last_detected_location} -> {player_loc}")
            detected_state = self._detect_starting_state(state_data)
            if detected_state and detected_state != self.current_state_name:
                print(f"🤖 [OPENER BOT INIT] Auto-detected starting state: {detected_state}")
                self._transition_to_state(detected_state)
            self.initialized_state = True
        
        # Update last detected location
        self._last_detected_location = player_loc
        
        state = self.states.get(self.current_state_name)
        if not state:
            logger.error(f"[OPENER BOT] In unknown state {self.current_state_name}, completing.")
            self._transition_to_state('COMPLETED')
            return None

        logger.info(f"[OPENER BOT GET_ACTION] State description: {state.description}")

        # 1. Check Safety Fallbacks
        self.state_attempt_count += 1
        elapsed = time.time() - self.state_entry_time
        logger.info(f"[OPENER BOT GET_ACTION] Safety check: {self.state_attempt_count}/{state.max_attempts} attempts, {elapsed:.1f}/{state.timeout_seconds}s elapsed")
        
        if self.state_attempt_count > state.max_attempts or elapsed > state.timeout_seconds:
            logger.warning(f"[OPENER BOT] ⚠️ SAFETY FALLBACK: State {state.name} timed out! Handing off to VLM.")
            self._transition_to_state('COMPLETED')
            return None

        # 2. Check for State Transition
        if state.next_state_fn:
            try:
                print(f"🔍 [OPENER BOT] Checking transition for {state.name}...")
                next_state_name = state.next_state_fn(state_data, visual_data)
                print(f"🔍 [OPENER BOT] Transition result: {next_state_name}")
                if next_state_name:
                    logger.info(f"[OPENER BOT] ✅ State {state.name} transition condition MET. Moving to {next_state_name}.")
                    print(f"✅ [OPENER BOT] Transitioning from {state.name} to {next_state_name}")
                    self._transition_to_state(next_state_name)
                    # Get the NEW state after transition
                    state = self.states[self.current_state_name]
                    logger.info(f"[OPENER BOT GET_ACTION] Now in new state: {state.name} - {state.description}")
                else:
                    logger.info(f"[OPENER BOT GET_ACTION] Transition condition NOT met, staying in {state.name}")
            except Exception as e:
                logger.error(f"[OPENER BOT] Error in next_state_fn for {state.name}: {e}")
                import traceback
                traceback.print_exc()
                self._transition_to_state('COMPLETED')
                return None
        else:
            print(f"🔍 [OPENER BOT] No transition function defined for {state.name}")

        # 3. Execute Current State's Action
        action_or_goal = None
        if state.action_fn:
            try:
                logger.info(f"[OPENER BOT GET_ACTION] Executing action_fn for {state.name}...")
                action_or_goal = state.action_fn(state_data, visual_data)
                self.last_action = action_or_goal
                if isinstance(action_or_goal, NavigationGoal):
                    logger.info(f"[OPENER BOT] State: {state.name} | Goal: {action_or_goal.description} to ({action_or_goal.x}, {action_or_goal.y})")
                else:
                    logger.info(f"[OPENER BOT] State: {state.name} | Action: {action_or_goal} | Attempt: {self.state_attempt_count}")
            except Exception as e:
                logger.error(f"[OPENER BOT] Error in action_fn for {state.name}: {e}")
                import traceback
                traceback.print_exc()
        
        logger.info(f"[OPENER BOT GET_ACTION] Returning: {action_or_goal}")
        logger.info(f"[OPENER BOT GET_ACTION] ========================================")
        return action_or_goal
    
    def _detect_starting_state(self, state_data: Dict[str, Any]) -> Optional[str]:
        """
        Auto-detect which state we should start in based on game progress.
        Used when loading from save states.
        
        IMPORTANT: Check LOCATION first (ground truth), then milestones.
        Location tells us WHERE we are, milestones tell us WHAT we've done.
        """
        milestones = state_data.get('milestones', {})
        player_loc = state_data.get('player', {}).get('location', '')
        player_pos = state_data.get('player', {}).get('position', {})
        x, y = player_pos.get('x', 0), player_pos.get('y', 0)
        
        print(f"🔍 [STATE DETECTION] Location: {player_loc}, Position: ({x}, {y})")
        print(f"🔍 [STATE DETECTION] Milestones: {list(milestones.keys())}")
        
        # ========================================
        # LOCATION CHECKS (Ground Truth - Check First!)
        # ========================================
        
        # Route 101 - heading to Birch or already there
        if 'ROUTE 101' in player_loc or 'ROUTE_101' in player_loc or 'ROUTE101' in player_loc:
            print(f"🔍 [STATE DETECTION] On Route 101!")
            # Check if starter already chosen (battle or post-battle)
            if milestones.get('STARTER_CHOSEN', {}).get('completed', False):
                print(f"🔍 [STATE DETECTION] Starter chosen - in battle or post-battle")
                return 'S22_FIRST_BATTLE'  # Battle state or after
            # On Route 101 but no starter yet - heading to Birch/bag
            if y > 10:  # Near bag location (y=14)
                print(f"🔍 [STATE DETECTION] Near bag location (y={y})")
                return 'S19_NAV_TO_BAG'
            else:  # Just entered from town
                print(f"🔍 [STATE DETECTION] Just entered Route 101 (y={y})")
                return 'S18_BIRCH_DIALOG'
        
        # Birch's Lab
        if 'PROFESSOR BIRCHS LAB' in player_loc or 'BIRCH' in player_loc:
            print(f"🔍 [STATE DETECTION] In Birch's Lab!")
            # Check if we have a starter Pokemon in party
            party = state_data.get('party', [])
            has_starter = len(party) > 0 if party else False
            print(f"🔍 [STATE DETECTION] Party data: {party}")
            print(f"🔍 [STATE DETECTION] Party count: {len(party) if party else 0}, has_starter: {has_starter}")
            
            if has_starter:
                print(f"🔍 [STATE DETECTION] Has starter - returning S23_BIRCH_DIALOG_2")
                return 'S23_BIRCH_DIALOG_2'  # After getting starter
            else:
                print(f"🔍 [STATE DETECTION] No starter yet - returning S20_INTERACT_BAG")
                return 'S20_INTERACT_BAG'  # Interacting with bag
        
        # Check for May's house BEFORE Littleroot Town check (both contain "LITTLEROOT TOWN")
        if 'MAYS HOUSE' in player_loc or ('MAY' in player_loc and 'HOUSE' in player_loc):
            print(f"🔍 [STATE DETECTION] In May's house!")
            if '2F' in player_loc:
                return 'S11B_NAV_TO_POKEBALL'  # On 2F
            else:
                return 'S10_MAYS_MOTHER_DIALOG'  # On 1F
        
        # Check if we're in player's house (Brendan's house) BEFORE Littleroot Town check
        if ('PLAYERS_HOUSE' in player_loc or 'BRENDANS_HOUSE' in player_loc or 
            ('BRENDAN' in player_loc and 'HOUSE' in player_loc)):
            print(f"🔍 [STATE DETECTION] In player's house!")
            if '2F' in player_loc:
                # On 2nd floor - setting clock or leaving
                return 'S6_NAV_TO_CLOCK'
            else:
                # On 1st floor - check if we've already been here (save state loaded after Mom dialogue)
                player_house_entered = milestones.get('PLAYER_HOUSE_ENTERED', {}).get('completed', False)
                if player_house_entered:
                    print(f"🔍 [STATE DETECTION] PLAYER_HOUSE_ENTERED already done - skipping S4, going to S5")
                    return 'S5_NAV_TO_STAIRS_1F'
                else:
                    # First time entering - Mom dialogue
                    return 'S4_MOM_DIALOG_1F'
        
        # Littleroot Town (OVERWORLD) - only matches if NOT in a specific building
        # This check must come AFTER house checks since location names include "LITTLEROOT TOWN"
        if 'LITTLEROOT' in player_loc and 'TOWN' in player_loc:
            print(f"🔍 [STATE DETECTION] In Littleroot Town overworld!")
            
            # Check what we've completed to determine next step
            if milestones.get('RIVAL_BEDROOM', {}).get('completed', False):
                # We've been to May's room - now heading to NPC
                print(f"🔍 [STATE DETECTION] Visited rival's bedroom, heading to NPC")
                if y < 8:  # Near north edge of town
                    print(f"🔍 [STATE DETECTION] Near north edge (y={y}), at NPC or heading to Route 101")
                    return 'S16_NPC_DIALOG'  # At NPC or just passed
                else:
                    print(f"🔍 [STATE DETECTION] South part of town (y={y}), navigating to NPC")
                    return 'S15_NAV_TO_NPC_NORTH'
            elif milestones.get('RIVAL_HOUSE', {}).get('completed', False):
                # We've been to rival house but not bedroom - probably navigating to May's room
                print(f"🔍 [STATE DETECTION] Visited rival's house, heading to May's room or already there")
                return 'S9_NAV_TO_MAYS_HOUSE'
            elif milestones.get('PLAYER_BEDROOM', {}).get('completed', False):
                # Been to our bedroom - heading to or inside rival's house
                print(f"🔍 [STATE DETECTION] Been to our bedroom, heading to rival's house")
                return 'S9_NAV_TO_MAYS_HOUSE'
            else:
                # In Littleroot but haven't done much - probably just started
                print(f"🔍 [STATE DETECTION] Early in Littleroot Town")
                return 'S4_MOM_DIALOG_1F'

        
        # Moving Van / Truck
        if 'MOVING_VAN' in player_loc or 'VAN' in player_loc or 'TRUCK' in player_loc:
            print(f"🔍 [STATE DETECTION] In moving van/truck!")
            return 'S3_TRUCK_RIDE'
        
        # ========================================
        # MILESTONE-ONLY CHECKS (Fallback if location unclear)
        # ========================================
        
        if milestones.get('PLAYER_NAME_SET', {}).get('completed', False):
            print(f"🔍 [STATE DETECTION] Name set but unclear location - defaulting to house")
            return 'S4_MOM_DIALOG_1F'
        
        # Default to title screen if no milestones
        print(f"🔍 [STATE DETECTION] No clear state detected - defaulting to title screen")
        return 'S0_TITLE_SCREEN'
        
    def _build_state_machine(self) -> Dict[str, BotState]:
        """
        Builds the STATEFUL state machine.
        Each state has an action_fn (what to do) and next_state_fn (when to transition).
        """
        
        # --- Helper Action Functions ---

        def action_press_a(s, v):
            return ['A']

        def action_press_start(s, v):
            return ['START']
            
        def action_press_b(s, v):
            return ['B']

        def action_clear_dialogue(s, v):
            """
            Press A if dialogue is detected, OR handle naming keyboard with B→START→A sequence.
            This handles the intro naming screen that appears during Prof Birch dialogue.
            
            Naming keyboard sequence (like nicknaming):
            1. Press B to clear any existing text
            2. Press START to jump cursor to OK button
            3. Press A to confirm and exit
            
            IMPORTANT: Does NOT rely solely on game_state='dialog' because it can get stuck.
            Requires visual confirmation (text_box or screen_context) to keep pressing A.
            """
            screen_context = v.get('screen_context', '').lower()
            text_box_visible = v.get('visual_elements', {}).get('text_box_visible', False)
            continue_prompt_visible = v.get('visual_elements', {}).get('continue_prompt_visible', False)
            game_state = s.get('game', {}).get('game_state', '').lower()
            dialogue = (v.get('on_screen_text', {}).get('dialogue', '') or '').upper()
            menu_title = (v.get('on_screen_text', {}).get('menu_title', '') or '').upper()
            
            # CRITICAL: Check if this is player monologue BEFORE pressing A
            # Player's internal thoughts have "Player:" prefix in dialogue text
            # Do NOT use speaker field - it's unreliable (Mom talking TO Casey shows speaker="CASEY")
            dialogue_text = v.get('on_screen_text', {}).get('dialogue') or ''  # Handle None from hallucination filter
            is_player_monologue = (dialogue_text and dialogue_text.strip().upper().startswith('PLAYER:'))
            
            if is_player_monologue:
                # Player monologues - don't press A, likely VLM hallucination
                # Let stuck detection in action_nav handle it if navigation is blocked
                print("💬 [CLEAR_DIALOGUE] Player monologue detected - ignoring (likely VLM hallucination)")
                return None
            
            # Initialize step counter for naming keyboard sequence
            if not hasattr(action_clear_dialogue, '_naming_step'):
                action_clear_dialogue._naming_step = 0
            
            # SPECIAL CASE: Detect naming keyboard screen
            # Indicators: "YOUR NAME?" in dialogue OR menu_title + still in TITLE_SEQUENCE
            # Return full B→START→A sequence to skip naming
            location = s.get('player', {}).get('location', '')
            # Check both dialogue and menu_title since VLM may put it in either field
            has_your_name = 'YOUR NAME' in dialogue or 'YOUR NAME' in menu_title
            if has_your_name and 'TITLE_SEQUENCE' in location:
                print(f"🎮 [S1_NAMING] Naming keyboard detected - sending B→START→A")
                return ['B', 'START', 'A']
            else:
                # Reset step counter when not in naming keyboard
                action_clear_dialogue._naming_step = 0
                if hasattr(action_clear_dialogue, '_naming_shortcut_step'):
                    action_clear_dialogue._naming_shortcut_step = 0
            
            # Press A if we have VISUAL confirmation of dialogue
            # NOTE: Don't press A based on game_state alone - it can get stuck
            # Require at least one visual indicator (text_box, screen_context, or continue_prompt)
            has_visual_dialogue = text_box_visible or screen_context == 'dialogue' or continue_prompt_visible
            
            if has_visual_dialogue:
                return ['A']
            elif location == 'TITLE_SEQUENCE':
                # Track total calls in TITLE_SEQUENCE so we can time the naming shortcut.
                # S0 contributes 5 presses (fixed counter), S1 has ~6 Birch dialogue boxes
                # before the naming keyboard appears.  Step 12+ = naming keyboard territory.
                if not hasattr(action_clear_dialogue, '_title_steps'):
                    action_clear_dialogue._title_steps = 0
                action_clear_dialogue._title_steps += 1
                steps = action_clear_dialogue._title_steps

                name_done = s.get('milestones', {}).get('PLAYER_NAME_SET', {}).get('completed', False)

                # Once PLAYER_NAME_SET fires we're on the naming keyboard.
                # Stop pressing A here — S1 will immediately transition to S2_GENDER_NAME_SELECT
                # which uses action_special_naming to handle the keyboard properly.
                if name_done:
                    return ['A']  # safe no-op while transition fires

                # Before name is set, keep pressing A to clear Birch's dialogue boxes.
                # Steps 1-16 are all dialogue (14 to boy/girl + 1 for boy + 1 more).
                # The naming keyboard opens at step 17, at which point PLAYER_NAME_SET
                # fires and the branch above takes over.
                return ['A']
                # Note: wait_for_script_idle() is NOT called here.
                # The GBA title intro uses a callback-based scene manager, not
                # sGlobalScriptContext, so global_mode is always 0 throughout
                # TITLE_SEQUENCE and the script-idle signal is meaningless here.
                return ['A']
            elif game_state == 'dialog':
                # Visual indicators say no dialogue, but game_state stuck as 'dialog'
                # This is a known issue - don't press A, let transition logic handle it
                print(f"⚠️ [CLEAR_DIALOGUE] game_state='dialog' but no visual indicators - not pressing A (likely stuck state)")
                return None
            
            return None
        
        def action_clear_dialogue_persistent(s, v):
            """
            Press A to clear dialogue, including multi-page dialogues.
            
            Unlike action_clear_dialogue, this function will KEEP PRESSING A even when
            visual indicators are clear but game_state='dialog' is stuck. This is needed
            for NPCs with multi-page dialogue (like "..." continuation pages) where:
            1. First page shows real dialogue
            2. Press A → VLM hallucinates on next page
            3. Visual indicators clear but game_state still 'dialog'
            4. Need to press A again to advance past "..." page
            
            This prevents premature transitions while dialogue is still active.
            
            CRITICAL: Must check for player monologue to avoid spamming A on internal thoughts!
            """
            screen_context = v.get('screen_context', '').lower()
            text_box_visible = v.get('visual_elements', {}).get('text_box_visible', False)
            continue_prompt_visible = v.get('visual_elements', {}).get('continue_prompt_visible', False)
            game_state = s.get('game', {}).get('game_state', '').lower()
            
            # Check for player monologue BEFORE pressing A
            on_screen_text = v.get('on_screen_text', {})
            dialogue_text = on_screen_text.get('dialogue') or ''  # Handle None from hallucination filter
            
            # Player monologue ONLY detects "Player:" prefix in dialogue text
            # Do NOT use speaker field - it's unreliable (Mom talking TO Casey shows speaker="CASEY")
            is_player_monologue = (dialogue_text and dialogue_text.strip().upper().startswith('PLAYER:'))
            
            if is_player_monologue:
                # Player monologues - don't press A, likely VLM hallucination
                # Let the state machine transition when dialogue actually clears
                print(f"💬 [PERSISTENT_DIALOGUE] Player monologue detected - ignoring (likely VLM hallucination)")
                return None
            
            # Press A if we have visual confirmation OR if game_state says dialogue is active
            # This handles both visible dialogue and hidden pages (like "...")
            has_visual_dialogue = text_box_visible or screen_context == 'dialogue' or continue_prompt_visible
            
            if has_visual_dialogue:
                print(f"💬 [PERSISTENT_DIALOGUE] Visual dialogue detected, pressing A")
                return ['A']
            elif game_state == 'dialog':
                # Game memory says dialogue active - keep pressing A even without visuals
                # This catches "..." pages that VLM hallucinates as HUD text
                print(f"💬 [PERSISTENT_DIALOGUE] game_state='dialog', pressing A (may be hidden page)")
                return ['A']
            
            # Only return None if BOTH visual AND memory say dialogue is done
            return None
        
        def action_wander_until_dialogue(s, v):
            """
            Wander around until dialogue triggers, then clear it.
            Used when dialogue is triggered by position (e.g., May appearing downstairs).
            
            Strategy:
            - If dialogue present: press A to clear it
            - If no dialogue: move RIGHT to explore and trigger events
            - Returns None only if both dialogue cleared AND position changing
            """
            screen_context = v.get('screen_context', '').lower()
            text_box_visible = v.get('visual_elements', {}).get('text_box_visible', False)
            continue_prompt_visible = v.get('visual_elements', {}).get('continue_prompt_visible', False)
            game_state = s.get('game', {}).get('game_state', '').lower()
            on_screen_text = v.get('on_screen_text', {})
            
            # Check for player monologue (hallucination)
            dialogue_text = on_screen_text.get('dialogue') or ''
            is_player_monologue = (dialogue_text and dialogue_text.strip().upper().startswith('PLAYER:'))
            
            if is_player_monologue:
                print("⚠️ [WANDER] Ignoring player monologue (hallucination)")
                return None  # Skip hallucinated player monologues
            
            # Check if dialogue is active
            has_visual_dialogue = text_box_visible or screen_context == 'dialogue' or continue_prompt_visible
            
            if has_visual_dialogue or game_state == 'dialog':
                print("📝 [WANDER] Dialogue detected, pressing A to clear")
                return ['A']
            else:
                # No dialogue yet - move RIGHT to explore and trigger May
                print("🚶 [WANDER] No dialogue yet, moving RIGHT to explore")
                return ['RIGHT']
        
        def action_clear_dialogue_then_move_away(direction: str) -> Callable:
            """
            Clear dialogue if present, otherwise move in the specified direction.
            Used to prevent re-triggering NPCs after dialogue ends.
            """
            def action_fn(s, v):
                screen_context = v.get('screen_context', '').lower()
                text_box_visible = v.get('visual_elements', {}).get('text_box_visible', False)
                continue_prompt_visible = v.get('visual_elements', {}).get('continue_prompt_visible', False)
                game_state = s.get('game', {}).get('game_state', '').lower()
                
                # If in dialogue, press A to clear
                if text_box_visible or screen_context == 'dialogue' or game_state == 'dialog' or continue_prompt_visible:
                    return ['A']
                # Otherwise move away from NPC
                return [direction]
            return action_fn

        def action_clear_dialogue_then_try_move(direction: str) -> Callable:
            """
            Smart dialogue clearing with movement validation:
            1. If strong dialogue indicators show active dialogue (text_box + prompt/context), press A
            2. If weak/uncertain indicators, try to move 
            3. If movement failed (position unchanged), dialogue must still be blocking - press A again
            
            This is robust against perception errors, looping dialogues, and stuck game_state flags.
            
            NOTE: We don't trust game_state='dialog' alone because it can stay stuck even after dialogue clears.
            We require at least TWO indicators to confirm dialogue is active.
            
            Args:
                direction: Direction to move after dialogue is cleared
            """
            def action_fn(s, v):
                # Check dialogue indicators
                screen_context = v.get('screen_context', '').lower()
                text_box_visible = v.get('visual_elements', {}).get('text_box_visible', False)
                continue_prompt_visible = v.get('visual_elements', {}).get('continue_prompt_visible', False)
                game_state = s.get('game', {}).get('game_state', '').lower()
                
                # Count how many strong indicators say dialogue is active
                # Strong indicators: text_box + continue_prompt, screen_context=dialogue
                strong_indicators = 0
                if text_box_visible and continue_prompt_visible:
                    strong_indicators += 1
                if screen_context == 'dialogue':
                    strong_indicators += 1
                
                # Dialogue is clearly active if we have strong visual indicators
                dialogue_clearly_active = strong_indicators >= 1
                
                # Get current position
                pos = s.get('player', {}).get('position', {})
                current_x, current_y = pos.get('x', -1), pos.get('y', -1)
                
                # Get last known position and action
                last_pos = getattr(action_fn, '_last_pos', (current_x, current_y))
                last_action = getattr(action_fn, '_last_action', None)
                
                # CASE 1: Dialogue is clearly active (strong visual evidence) - press A
                if dialogue_clearly_active:
                    action_fn._last_pos = (current_x, current_y)
                    action_fn._last_action = 'A'
                    print(f"🔍 [CLEAR_DIALOGUE_TRY_MOVE] Dialogue clearly active ({strong_indicators} indicators), pressing A (text_box={text_box_visible}, context={screen_context}, game_state={game_state}, prompt={continue_prompt_visible})")
                    return ['A']
                
                # CASE 2: We tried to move last time but position didn't change - dialogue is blocking
                if last_action == direction and last_pos == (current_x, current_y):
                    action_fn._last_pos = (current_x, current_y)
                    action_fn._last_action = 'A'
                    print(f"🔍 [CLEAR_DIALOGUE_TRY_MOVE] Movement BLOCKED! Position unchanged ({current_x},{current_y}). Dialogue still active - pressing A")
                    return ['A']
                
                # CASE 3: No strong dialogue indicators and (first attempt or movement succeeded) - try to move
                action_fn._last_pos = (current_x, current_y)
                action_fn._last_action = direction
                print(f"🔍 [CLEAR_DIALOGUE_TRY_MOVE] No strong dialogue indicators (game_state={game_state} alone not trusted), trying to move {direction} from ({current_x},{current_y})")
                return [direction]
            
            return action_fn

        def action_nav(goal: NavigationGoal):
            """
            Factory for navigation actions with stuck detection and movement batching.
            
            Strategy:
            1. Check for active dialogue and clear it FIRST (prevents navigation attempts while dialogue is blocking)
            2. Calculate sequence of movements toward goal
            3. Batch up to MAX_MOVEMENT_BATCH_SIZE movements to reduce VLM calls
            4. Track position history to detect being stuck
            5. Only clear dialogue when stuck AND dialogue is detected in game state
            6. This prevents reacting to VLM hallucinations while handling real dialogue blocks
            """
            def nav_fn(s, v):
                # CRITICAL SAFETY CHECK: Don't navigate if we're still on naming screen!
                # This prevents getting stuck trying to move when the keyboard is visible
                dialogue = (v.get('on_screen_text', {}).get('dialogue', '') or '').upper()
                menu_title = (v.get('on_screen_text', {}).get('menu_title', '') or '').upper()
                screen_context = v.get('screen_context', '').lower()
                
                has_your_name = "YOUR NAME" in dialogue or "YOUR NAME" in menu_title
                has_keyboard = any(letters in menu_title for letters in ['A B C', 'NAME INPUT'])
                is_naming_screen = has_your_name or has_keyboard or (screen_context == 'menu' and ('NAME' in dialogue or 'NAME' in menu_title))
                
                if is_naming_screen:
                    print(f"🚫 [NAV SAFETY] Naming screen detected - refusing to navigate!")
                    print(f"   dialogue='{dialogue[:40]}...', menu='{menu_title[:40]}...', context='{screen_context}'")
                    print(f"   Pressing A to advance instead")
                    return ['A']  # Press A to try to advance past naming screen
                
                # DIALOGUE BLOCKING CHECK: Before attempting navigation, check if dialogue is active
                # This prevents the bot from trying to navigate while dialogue is blocking movement
                # CRITICAL: Only check VISUAL indicators, not game_state='dialog' which can be misleading
                # (game may be in dialog state without actual visible text boxes)
                visual_elements = v.get('visual_elements', {})
                text_box_visible = visual_elements.get('text_box_visible', False)
                continue_prompt_visible = visual_elements.get('continue_prompt_visible', False)
                
                # If VISIBLE dialogue is active, clear it first before navigating
                if text_box_visible or continue_prompt_visible:
                    print(f"💬 [NAV DIALOGUE] Visible dialogue detected before navigation - clearing it first")
                    print(f"   text_box_visible={text_box_visible}, continue_prompt={continue_prompt_visible}")
                    return ['A']  # Press A to clear dialogue
                
                # Initialize position tracking
                if not hasattr(nav_fn, '_position_history'):
                    nav_fn._position_history = deque(maxlen=5)
                if not hasattr(nav_fn, '_stuck_frames'):
                    nav_fn._stuck_frames = 0
                if not hasattr(nav_fn, '_last_navigation_attempt'):
                    nav_fn._last_navigation_attempt = None
                
                # Get current position
                player_data = s.get('player', {})
                position = player_data.get('position', {})
                current_pos = (position.get('x'), position.get('y'), player_data.get('location'))
                current_x = position.get('x')
                current_y = position.get('y')
                
                # Track position history
                nav_fn._position_history.append(current_pos)
                
                # Check if stuck (same position for 3+ consecutive frames)
                is_stuck = False
                if len(nav_fn._position_history) >= 3:
                    recent_positions = list(nav_fn._position_history)[-3:]
                    if all(pos == current_pos for pos in recent_positions):
                        is_stuck = True
                        nav_fn._stuck_frames += 1
                    else:
                        nav_fn._stuck_frames = 0
                else:
                    nav_fn._stuck_frames = 0
                
                # If stuck for 3+ frames, diagnose and intervene
                if is_stuck and nav_fn._stuck_frames >= 3:
                    # Check if stuck due to dialogue (use game memory, not VLM which may be hallucinating)
                    game_state = s.get('game', {}).get('game_state', '').lower()
                    
                    # CRITICAL: When hallucination filter is active, VLM visuals are cleared
                    # but game memory still shows game_state='dialog' or 'overworld'
                    # We detect stuck-in-dialogue by:
                    # 1. Position unchanged for 3+ frames (is_stuck=True)
                    # 2. We're in a navigation state trying to move
                    # 3. Movements keep failing (position not changing)
                    # This pattern = dialogue blocking us (even if VLM can't see it due to hallucinations)
                    
                    # Strategy: Press A to try clearing any hidden dialogue
                    if nav_fn._stuck_frames >= 3:
                        print(f"🚫 [NAV STUCK] Stuck at {current_pos} for {nav_fn._stuck_frames} frames")
                        print(f"🚫 [NAV STUCK] Game state: {game_state}")
                        print(f"🚫 [NAV STUCK] Likely stuck in dialogue - pressing A to attempt clearance")
                        nav_fn._stuck_frames = 0  # Reset counter after intervention
                        return ['A']  # Press A to try clearing dialogue
                    
                    # Legacy code kept for safety (shouldn't reach here)
                    visual_elements = v.get('visual_elements', {})
                    text_box_visible = visual_elements.get('text_box_visible', False)
                    dialogue_blocking = (
                        game_state == 'dialog' or 
                        text_box_visible or
                        v.get('screen_context', '').lower() == 'dialogue'
                    )
                    
                    if dialogue_blocking:
                        print(f"🚫 [NAV STUCK] [LEGACY] Dialogue detected, returning None")
                        return None
                    else:
                        print(f"🚫 [NAV STUCK] No dialogue detected, continuing with goal")
                
                # Calculate path to goal - batch multiple steps for efficiency
                goal_x = goal.x
                goal_y = goal.y
                
                # Check if we've reached the goal
                if current_x == goal_x and current_y == goal_y:
                    # At goal - handle interaction if needed
                    if goal.should_interact:
                        print(f"🎯 [NAV] Reached goal ({goal_x}, {goal_y}) - interacting")
                        return ['A']
                    else:
                        print(f"🎯 [NAV] Reached goal ({goal_x}, {goal_y}) - no interaction needed")
                        return []
                
                # Calculate sequence of movements toward goal
                movements = []
                temp_x, temp_y = current_x, current_y
                
                # Calculate Manhattan distance
                while len(movements) < MAX_MOVEMENT_BATCH_SIZE:
                    dx = goal_x - temp_x
                    dy = goal_y - temp_y
                    
                    if dx == 0 and dy == 0:
                        break  # Reached goal
                    
                    # Prioritize axis with larger distance (same strategy as main agent)
                    if abs(dy) > abs(dx):
                        # Y-axis is primary
                        if dy < 0:
                            movements.append('UP')
                            temp_y -= 1
                        else:
                            movements.append('DOWN')
                            temp_y += 1
                    elif abs(dx) > abs(dy):
                        # X-axis is primary
                        if dx < 0:
                            movements.append('LEFT')
                            temp_x -= 1
                        else:
                            movements.append('RIGHT')
                            temp_x += 1
                    else:
                        # Equal distance - prefer Y-axis first (consistent with main agent)
                        if dy != 0:
                            if dy < 0:
                                movements.append('UP')
                                temp_y -= 1
                            else:
                                movements.append('DOWN')
                                temp_y += 1
                        elif dx != 0:
                            if dx < 0:
                                movements.append('LEFT')
                                temp_x -= 1
                            else:
                                movements.append('RIGHT')
                                temp_x += 1
                
                if movements:
                    print(f"🗺️ [NAV] From ({current_x},{current_y}) to ({goal_x},{goal_y}): {' → '.join(movements[:3])}{'...' if len(movements) > 3 else ''} ({len(movements)} steps)")
                    return movements
                else:
                    print(f"⚠️ [NAV] Already at goal position")
                    return []
            
            return nav_fn

        def action_simple(actions: List[str]):
            """Returns a simple action (like ['UP']) every time. No dialogue clearing."""
            def simple_fn(s, v):
                return actions
            return simple_fn

        def action_special_naming(s, v):
            """
            Handles gender and name selection screens.
            
            Sequence:
            1. "Are you a boy or girl?" - Press A (selects BOY)
            2. "What's your name?" - Press A ONCE to enter naming screen
            3. Naming screen appears with keyboard - Wait for it to fully load
            4. On keyboard screen - Press START to accept default name
            5. "So it's [NAME]?" - Press A (confirms choice)
            
            CRITICAL: We check screen_context to know if we're IN the naming keyboard.
            The naming keyboard shows screen_context='dialogue' with letters/grid visible.
            """
            dialogue = (v.get('on_screen_text', {}).get('dialogue', '') or '').upper()
            screen_context = v.get('screen_context', '').lower()
            menu_title = (v.get('on_screen_text', {}).get('menu_title', '') or '').upper()
            
            # Initialize state tracking
            if not hasattr(action_special_naming, '_entered_naming_screen'):
                action_special_naming._entered_naming_screen = False
            if not hasattr(action_special_naming, '_naming_step'):
                action_special_naming._naming_step = 0
            
            print(f"🎮 [NAMING DEBUG] dialogue='{dialogue[:50] if dialogue else ''}', screen_context={screen_context}, menu_title='{menu_title[:30] if menu_title else ''}', entered={action_special_naming._entered_naming_screen}")
            
            # Gender selection
            if "ARE YOU A BOY" in dialogue or "BOY OR" in dialogue:
                action_special_naming._entered_naming_screen = False
                action_special_naming._naming_step = 0
                print("🎮 [NAMING] Gender selection - pressing A")
                return ['A']
            
            # Name confirmation ("So it's..." or "Is this your name?")  
            if ("SO IT" in dialogue or "IS THIS" in dialogue or "THAT'S A NICE" in dialogue) and "NAME" in dialogue:
                action_special_naming._entered_naming_screen = False
                action_special_naming._naming_step = 0
                print("🎮 [NAMING] Name confirmation - pressing A")
                return ['A']
            
            # SIMPLIFIED: If we see "YOUR NAME?" anywhere (dialogue or menu_title), use the shortcut
            # Return full B→START→A sequence to skip naming
            has_your_name = "YOUR NAME" in dialogue or "YOUR NAME" in menu_title
            if has_your_name:
                print(f"🎮 [NAMING] 'YOUR NAME?' detected - sending B→START→A")
                action_special_naming._entered_naming_screen = False  # Reset
                return ['B', 'START', 'A']
            
            # General dialogue clearing (for other dialogue in this state)
            if v.get('visual_elements', {}).get('text_box_visible', False):
                print("🎮 [NAMING] Clearing general dialogue with A")
                return ['A']
            
            # --- Counter-based fallback (VLM is off during TITLE_SEQUENCE) ---
            # We enter S2 the moment the naming keyboard opens (PLAYER_NAME_SET fires).
            # VLM is skipped so all the visual checks above produce empty strings.
            #
            # Exactly ONE B→START→A is needed:
            #   B:     safe no-op when no chars typed; deletes last char if any
            #   START: jumps cursor to END/OK on the naming keyboard
            #   A:     confirms END → shows "Is this your name? YES/NO"
            #
            # After that, plain A confirms YES and clears all post-naming dialogue
            # until the location changes away from TITLE_SEQUENCE (S2 transition).
            # Sending a second B→START→A would press B on the YES/NO screen which
            # selects NO, cancelling the name and sending us back to the keyboard.
            action_special_naming._naming_step += 1
            step = action_special_naming._naming_step
            if step == 1:
                print(f"🎮 [NAMING] Counter step 1 → B→START→A (confirm name on keyboard)")
                return ['B', 'START', 'A']
            print(f"🎮 [NAMING] Counter step {step} → A (post-naming dialogue)")
            return ['A']

        def action_special_clock(s, v):
            """
            Handles the clock UI including the Yes/No confirmation.
            
            Clock sequence:
            1. "The clock..." - press A
            2. "Better set it and start it!" - press A to set time
            3. "Is this the correct time?" with Yes/No menu - Step 1: UP, Step 2: A
            
            Uses step-based approach for Yes/No menu (can't send two buttons in one frame).
            """
            dialogue = (v.get('on_screen_text', {}).get('dialogue') or '').upper()  # Handle None from hallucination filter
            
            # Initialize step counter for Yes/No menu
            if not hasattr(action_special_clock, '_yesno_step'):
                action_special_clock._yesno_step = 0
            
            # Check if we're at the "Is this the correct time?" dialogue
            if "IS THIS" in dialogue and "CORRECT TIME" in dialogue:
                # Yes/No menu - need to press UP then A (separate frames)
                print(f"🕐 [CLOCK] Yes/No menu detected (step {action_special_clock._yesno_step})")
                
                if action_special_clock._yesno_step == 0:
                    action_special_clock._yesno_step = 1
                    print("🕐 [CLOCK] Step 1: Pressing UP to select YES")
                    return ['UP']
                else:
                    action_special_clock._yesno_step = 0  # Reset for next time
                    print("🕐 [CLOCK] Step 2: Pressing A to confirm")
                    return ['A']
            else:
                # Reset step counter when not in Yes/No menu
                action_special_clock._yesno_step = 0
            
            # For all other clock-related dialogue, just press A
            if "SET THE CLOCK" in dialogue or "IS THIS TIME" in dialogue or "THE CLOCK" in dialogue:
                print(f"🕐 [CLOCK] Clock dialogue detected, pressing A")
                return ['A']
                
            # General dialogue clearing - prefer VLM, fall back to memory-based game_state
            if v.get('visual_elements', {}).get('text_box_visible', False):
                print(f"🕐 [CLOCK] Text box visible, pressing A")
                return ['A']

            game_state = s.get('game', {}).get('game_state', '')
            if game_state == 'dialog':
                print(f"🕐 [CLOCK] game_state='dialog' (memory fallback), pressing A")
                return ['A']

            return None

        def action_special_starter(s, v):
            """Handles selecting the starter."""
            dialogue = (v.get('on_screen_text', {}).get('dialogue') or '').upper()  # Handle None from hallucination filter
            menu_title = (v.get('on_screen_text', {}).get('menu_title') or '').upper()  # Handle None
            if "CHOOSE A POKéMON" in dialogue or "BAG" in menu_title:
                return ['A']
            if "DO YOU CHOOSE THIS" in dialogue:
                return ['A']
            if v.get('visual_elements', {}).get('text_box_visible', False):
                return ['A']
            return None

        def action_special_nickname(s, v):
            """
            Handles nickname screen - either decline before entering OR exit if already inside.
            
            Strategy: Due to VLM timing, we often enter the naming window before detecting it.
            If inside the naming window, use shortcut: B → START → A sequence to exit
            """
            dialogue = (v.get('on_screen_text', {}).get('dialogue', '') or '').upper()
            screen_context = v.get('screen_context', '').lower()
            menu_title = (v.get('on_screen_text', {}).get('menu_title', '') or '').upper()
            
            # Check if we're INSIDE the naming window (keyboard visible)
            # Indicators: screen_context='menu', "NICKNAME" in dialogue/menu_title
            inside_naming_window = (
                screen_context == 'menu' and 
                ("NICKNAME" in dialogue or "NICKNAME" in menu_title)
            )
            
            if inside_naming_window:
                # We're already inside - use B→START→A shortcut to exit
                print(f"🎮 [NICKNAME WINDOW] Inside naming screen - sending B→START→A")
                return ['B', 'START', 'A']
            
            # If asking about nickname BEFORE entering naming window (ideal case)
            # Return full DOWN→A sequence to select NO
            if "NICKNAME" in dialogue and not inside_naming_window:
                print(f"🎮 [NICKNAME PROMPT] At nickname prompt - sending DOWN→A (select NO)")
                return ['DOWN', 'A']
            
            # Clear any remaining dialogue
            if v.get('visual_elements', {}).get('text_box_visible', False):
                print("🎮 [NICKNAME] Clearing dialogue with A")
                return ['A']
            
            return None
            
        def action_pass_to_battle_bot(s, v):
            """Do nothing - lets main action.py call Battle Bot."""
            return None

        # --- Transition Check Functions (FIXED: Return state names, not True/None) ---
        
        def trans_game_state_not(state_value: str, next_state: str) -> Callable:
            """Transition when game_state is NOT the given value."""
            def check_fn(s, v):
                current_game_state = s.get('game', {}).get('game_state', '').lower()
                print(f"🔍 [TRANS_GAME_STATE_NOT] Checking if game_state '{current_game_state}' != '{state_value}': {current_game_state != state_value.lower()}")
                if current_game_state != state_value.lower():
                    print(f"🔍 [TRANS_GAME_STATE_NOT] Transitioning to {next_state}")
                    return next_state
                return None
            return check_fn

        def trans_dialogue_contains(text: str, next_state: str) -> Callable:
            """Transition when dialogue OR menu_title contains specific text."""
            def check_fn(s, v):
                dialogue = v.get('on_screen_text', {}).get('dialogue', '') or ''
                menu_title = v.get('on_screen_text', {}).get('menu_title', '') or ''
                contains_text = text.upper() in dialogue.upper() or text.upper() in menu_title.upper()
                print(f"🔍 [TRANS_DIALOGUE] Checking if '{text}' in '{dialogue[:50]}...' or menu_title: {contains_text}")
                if contains_text:
                    print(f"🔍 [TRANS_DIALOGUE] Transitioning to {next_state}")
                    return next_state
                return None
            return check_fn

        def trans_milestone_complete(milestone_id: str, next_state: str) -> Callable:
            """Transition when milestone is completed."""
            def check_fn(s, v):
                completed = s.get('milestones', {}).get(milestone_id, {}).get('completed', False)
                print(f"🔍 [TRANS_MILESTONE] Checking milestone '{milestone_id}': completed={completed}")
                if completed:
                    print(f"🔍 [TRANS_MILESTONE] Milestone '{milestone_id}' complete! Transitioning to {next_state}")
                    return next_state
                return None
            return check_fn

        def trans_location_contains(loc_name: str, next_state: str) -> Callable:
            """Transition when location contains specific text."""
            def check_fn(s, v):
                location = s.get('player', {}).get('location', '') or ''
                result = loc_name.upper() in location.upper()
                print(f"🔍 [TRANS_LOCATION] Checking if '{loc_name}' in '{location}': {result}")
                if result:
                    print(f"🔍 [TRANS_LOCATION] Transitioning to {next_state}")
                    return next_state
                return None
            return check_fn

        def trans_location_exact(loc_name: str, next_state: str) -> Callable:
            """Transition when location exactly matches (case-insensitive)."""
            def check_fn(s, v):
                location = s.get('player', {}).get('location', '') or ''
                result = loc_name.upper() == location.upper()
                print(f"🔍 [TRANS_LOCATION_EXACT] Checking if '{loc_name}' == '{location}': {result}")
                if result:
                    print(f"🔍 [TRANS_LOCATION_EXACT] Transitioning to {next_state}")
                    return next_state
                return None
            return check_fn

        def trans_no_dialogue(next_state: str, min_wait_steps: int = 2) -> Callable:
            """
            Transition when dialogue is FULLY cleared - both visually AND in game memory.
            
            CRITICAL: Must check BOTH conditions:
            1. Visual indicators (VLM can hallucinate, so check all indicators)
            2. Game memory state (player cannot move while game_state='dialog')
            
            PLAYER MONOLOGUE HANDLING: If VLM reports "Player: ..." dialogue, this is
            likely a hallucination. Treat it as NO dialogue for transition purposes.
            
            ESCAPE MECHANISM: If visuals are clear but game_state is stuck as 'dialog',
            force transition after waiting min_wait_steps to allow action to complete.
            This prevents premature transitions during multi-page dialogues while still
            escaping truly stuck states.
            
            Args:
                next_state: State to transition to
                min_wait_steps: Minimum steps to wait when visuals clear but game_state stuck
            """
            def check_fn(s, v):
                screen_context = v.get('screen_context', '').lower()
                text_box_visible = v.get('visual_elements', {}).get('text_box_visible', False)
                continue_prompt_visible = v.get('visual_elements', {}).get('continue_prompt_visible', False)
                game_state = s.get('game', {}).get('game_state', '')
                
                # CRITICAL: Check for player monologue (VLM hallucination)
                # Player monologue ONLY detects "Player:" prefix in dialogue text
                dialogue_text = v.get('on_screen_text', {}).get('dialogue') or ''  # Handle None from hallucination filter
                is_player_monologue = (dialogue_text and dialogue_text.strip().upper().startswith('PLAYER:'))
                
                # If player monologue detected, treat all visual indicators as false (likely hallucination)
                if is_player_monologue:
                    print(f"🔍 [TRANS_NO_DIALOGUE] Player monologue detected - treating as no dialogue (likely VLM hallucination)")
                    text_box_visible = False
                    continue_prompt_visible = False
                    if screen_context == 'dialogue':
                        screen_context = 'overworld'
                
                # Transition if dialogue is FULLY cleared:
                # 1. No visual indicators (text_box, dialogue context, continue prompt)
                # 2. Game memory state is NOT 'dialog' (player can actually move)
                visuals_clear = not text_box_visible and screen_context != 'dialogue' and not continue_prompt_visible
                memory_clear = game_state != 'dialog'
                
                # Initialize counter for stuck detection
                if not hasattr(check_fn, '_stuck_steps'):
                    check_fn._stuck_steps = 0
                
                # Track how many steps visuals have been clear while game_state stuck
                if visuals_clear and not memory_clear:
                    check_fn._stuck_steps += 1
                else:
                    check_fn._stuck_steps = 0
                
                # CASE 1: Both clear - normal transition
                if visuals_clear and memory_clear:
                    check_fn._stuck_steps = 0  # Reset counter
                    print(f"🔍 [TRANS_NO_DIALOGUE] Dialogue fully cleared! visual={visuals_clear}, game_state={game_state}")
                    return next_state
                
                # CASE 2: Visuals clear but game_state stuck - wait min_wait_steps before forcing
                # This gives action_clear_dialogue_persistent time to complete multi-page dialogues
                elif visuals_clear and not memory_clear:
                    if check_fn._stuck_steps >= min_wait_steps:
                        print(f"⚠️ [TRANS_NO_DIALOGUE] Stuck for {check_fn._stuck_steps} steps (game_state={game_state}) - forcing transition!")
                        check_fn._stuck_steps = 0
                        return next_state
                    else:
                        print(f"🔍 [TRANS_NO_DIALOGUE] Visuals clear but game_state={game_state} - waiting ({check_fn._stuck_steps}/{min_wait_steps})")
                        return None
                
                # CASE 3: Visuals still show dialogue - keep waiting
                else:
                    return None
            return check_fn
        
        def trans_has_dialogue(next_state: str) -> Callable:
            """Transition when dialogue IS active (inverse of trans_no_dialogue)."""
            def check_fn(s, v):
                location = s.get('player', {}).get('location', '')

                # TITLE_SEQUENCE: VLM is skipped so text_box_visible is always False.
                # Use a press counter instead — after pressing A a few times we are
                # guaranteed to be inside Prof Birch's intro dialogue.
                if location == 'TITLE_SEQUENCE':
                    if not hasattr(check_fn, '_title_seq_presses'):
                        check_fn._title_seq_presses = 0
                    check_fn._title_seq_presses += 1
                    if check_fn._title_seq_presses >= 5:
                        print(f"🔍 [TRANS_DIALOGUE] TITLE_SEQUENCE: auto-transitioning to {next_state} after {check_fn._title_seq_presses} A-presses")
                        check_fn._title_seq_presses = 0  # reset for potential future reuse
                        return next_state
                    return None

                screen_context = v.get('screen_context', '').lower()
                text_box_visible = v.get('visual_elements', {}).get('text_box_visible', False)
                continue_prompt_visible = v.get('visual_elements', {}).get('continue_prompt_visible', False)
                
                # Count strong indicators (don't trust game_state alone)
                strong_indicators = 0
                if text_box_visible and continue_prompt_visible:
                    strong_indicators += 1
                if screen_context == 'dialogue':
                    strong_indicators += 1
                
                # Dialogue active if at least 1 strong indicator
                if strong_indicators >= 1:
                    return next_state
                return None
            return check_fn
        
        def trans_detect_naming_screen(next_state: str) -> Callable:
            """
            Transition when we detect the actual naming/gender screen.
            The naming screen has distinct visual characteristics - detect by what we SEE, not milestones.
            
            CRITICAL: Do NOT check PLAYER_NAME_SET milestone here!
            The milestone gets set BEFORE we see the keyboard (game auto-assigns default name).
            We need to detect the keyboard is visible and handle it, regardless of milestone state.
            """
            def check_fn(s, v):
                location = s.get('player', {}).get('location', '')
                dialogue = (v.get('on_screen_text', {}).get('dialogue', '') or '').upper()
                menu_title = (v.get('on_screen_text', {}).get('menu_title', '') or '').upper()
                screen_context = v.get('screen_context', '').lower()
                
                # CRITICAL: Check for naming keyboard visually, not by milestone
                # Indicators: "YOUR NAME?" in text + screen_context='menu' + keyboard visible
                has_your_name = "YOUR NAME" in dialogue or "YOUR NAME" in menu_title
                is_menu_context = screen_context == 'menu'
                
                # Also check for alphabet keyboard indicators in menu_title
                has_keyboard_letters = any(letter_set in menu_title for letter_set in ['A B C', 'ABCDEFG', 'NAME INPUT'])
                
                if has_your_name and (is_menu_context or has_keyboard_letters):
                    print(f"🔍 [TRANS_NAMING] Naming keyboard detected visually!")
                    print(f"   dialogue='{dialogue[:40]}...', menu_title='{menu_title[:40]}...', screen='{screen_context}'")
                    return next_state
                
                # Check for gender/naming screen indicators
                # Gender question should appear when we're NOT just in title sequence anymore
                if "BOY" in dialogue and "GIRL" in dialogue and 'TITLE_SEQUENCE' not in location:
                    print(f"🔍 [TRANS_NAMING] Gender question detected outside title sequence!")
                    return next_state
                    return next_state
                
                # Naming keyboard has distinctive menu title
                if "NAME" in menu_title and len(menu_title) > 3:
                    print(f"🔍 [TRANS_NAMING] Naming screen menu detected: {menu_title}")
                    return next_state
                
                return None
            return check_fn
        
        def trans_naming_complete_and_visual_cleared(next_state: str) -> Callable:
            """
            Transition after naming is complete AND naming screen is visually gone.
            
            CRITICAL SAFETY CHECK: This prevents transitioning to S3_TRUCK_RIDE while
            we're still on the naming keyboard. We check BOTH:
            1. Milestone PLAYER_NAME_SET is complete (name has been set in memory)
            2. Visual indicators show naming screen is GONE (no "YOUR NAME?" visible)
            
            This ensures we don't try to navigate while still on the naming screen.
            """
            def check_fn(s, v):
                name_milestone_done = s.get('milestones', {}).get('PLAYER_NAME_SET', {}).get('completed', False)
                dialogue = (v.get('on_screen_text', {}).get('dialogue', '') or '').upper()
                menu_title = (v.get('on_screen_text', {}).get('menu_title', '') or '').upper()
                screen_context = v.get('screen_context', '').lower()
                
                # Check if naming screen is visually gone
                has_your_name = "YOUR NAME" in dialogue or "YOUR NAME" in menu_title
                has_keyboard = any(letters in menu_title for letters in ['A B C', 'NAME INPUT'])
                is_still_naming = has_your_name or has_keyboard or screen_context == 'menu'
                
                print(f"🔍 [S2_TRANS] Milestone={name_milestone_done}, still_naming={is_still_naming}")
                print(f"   dialogue='{dialogue[:30]}...', menu='{menu_title[:30]}...', context='{screen_context}'")
                
                # CRITICAL: Only transition if milestone is done AND naming screen is visually cleared
                if name_milestone_done and not is_still_naming:
                    print(f"✅ [S2_TRANS] Naming complete AND visually cleared → {next_state}")
                    return next_state
                elif name_milestone_done and is_still_naming:
                    print(f"⏳ [S2_TRANS] Milestone done but naming screen still visible - waiting...")
                    return None
                else:
                    return None
            return check_fn
        
        def trans_name_set_plus_frames(s, v, frames_to_wait: int = 10):
            """
            Transition after PLAYER_NAME_SET milestone completes AND we've waited additional frames.
            This handles the delay between name being set and location actually changing.
            
            Strategy:
            1. Wait for PLAYER_NAME_SET milestone
            2. Count frames after milestone completes
            3. After N frames, check location and transition appropriately
            """
            name_milestone_done = s.get('milestones', {}).get('PLAYER_NAME_SET', {}).get('completed', False)
            location = s.get('player', {}).get('location', '')
            
            # Initialize frame counter on first call
            if not hasattr(trans_name_set_plus_frames, '_frames_since_name_set'):
                trans_name_set_plus_frames._frames_since_name_set = 0
                trans_name_set_plus_frames._name_was_set = False
            
            # Reset counter if milestone wasn't set before but is now
            if name_milestone_done and not trans_name_set_plus_frames._name_was_set:
                print(f"🔍 [S1_TRANS] PLAYER_NAME_SET milestone just completed! Starting frame counter.")
                trans_name_set_plus_frames._frames_since_name_set = 0
                trans_name_set_plus_frames._name_was_set = True
            
            # If name is set, increment counter
            if name_milestone_done:
                trans_name_set_plus_frames._frames_since_name_set += 1
                frames_waited = trans_name_set_plus_frames._frames_since_name_set
                print(f"🔍 [S1_TRANS] Name set, waited {frames_waited}/{frames_to_wait} frames. Location: '{location}'")
                
                # After waiting enough frames, check location and transition
                if frames_waited >= frames_to_wait:
                    print(f"🔍 [S1_TRANS] Waited {frames_to_wait} frames, checking location for transition...")
                    
                    # Check for truck/van
                    if 'MOVING_VAN' in location or 'VAN' in location or 'TRUCK' in location:
                        print(f"🔍 [S1_TRANS] ✅ Found truck/van in location! Transitioning to S3_TRUCK_RIDE")
                        # Reset for next time
                        trans_name_set_plus_frames._frames_since_name_set = 0
                        trans_name_set_plus_frames._name_was_set = False
                        return 'S3_TRUCK_RIDE'
                    
                    # Check for house
                    if 'HOUSE 1F' in location or 'BRENDAN' in location:
                        print(f"🔍 [S1_TRANS] ✅ Found house in location! Transitioning to S4_MOM_DIALOG_1F")
                        # Reset for next time
                        trans_name_set_plus_frames._frames_since_name_set = 0
                        trans_name_set_plus_frames._name_was_set = False
                        return 'S4_MOM_DIALOG_1F'
                    
                    # Still in TITLE_SEQUENCE after waiting - wait more, but only up to a hard cap.
                    # The memory reader has map-buffer corruption during the intro animation so
                    # location may never update away from TITLE_SEQUENCE.  After 25 frames total
                    # we know name confirmation is done and the truck/intro cutscene has started.
                    hard_cap = frames_to_wait + 15  # 25 frames total
                    if frames_waited >= hard_cap:
                        print(f"🔍 [S1_TRANS] ✅ Hard cap reached ({frames_waited} frames) — forcing S3_TRUCK_RIDE")
                        trans_name_set_plus_frames._frames_since_name_set = 0
                        trans_name_set_plus_frames._name_was_set = False
                        return 'S3_TRUCK_RIDE'
                    print(f"🔍 [S1_TRANS] ⚠️ Still in '{location}' after {frames_waited} frames, waiting (cap={hard_cap})...")
            
            return None
            
        def trans_in_battle(next_state: str) -> Callable:
            """Transition when in battle."""
            def check_fn(s, v):
                if s.get('game', {}).get('in_battle', False):
                    return next_state
                return None
            return check_fn

        def trans_area_and_dialogue(x_range: List[int], y_range: List[int], next_state: str) -> Callable:
            """
            Transition when in specific AREA and dialogue appears.
            CRITICAL FIX: Handles "adjacent-interact" bug by checking area, not exact position.
            ALSO: Ignores player monologue hallucinations - only real NPC dialogue triggers transition.
            """
            def check_fn(s, v):
                pos = s.get('player', {}).get('position', {})
                x, y = pos.get('x', -1), pos.get('y', -1)
                
                # Check if in area
                in_area = x in x_range and y in y_range
                if not in_area:
                    return None
                
                # Check for dialogue - prefer VLM, fall back to memory-based game_state
                text_box_visible = v.get('visual_elements', {}).get('text_box_visible', False)
                game_state = s.get('game', {}).get('game_state', '')
                memory_dialogue = (game_state == 'dialog')

                if not text_box_visible and not memory_dialogue:
                    return None

                # CRITICAL: Ignore player monologue hallucinations (VLM only; memory fallback skips this)
                if text_box_visible:
                    dialogue_text = v.get('on_screen_text', {}).get('dialogue') or ''  # Handle None from hallucination filter
                    is_player_monologue = dialogue_text.strip().upper().startswith('PLAYER:')
                    if is_player_monologue:
                        print(f"🔍 [TRANS_AREA_DIALOGUE] Player monologue detected - ignoring (likely VLM hallucination)")
                        return None

                source = 'VLM' if text_box_visible else 'memory(game_state=dialog)'
                # Real dialogue in the area - transition!
                print(f"🔍 [TRANS_AREA_DIALOGUE] In area, real dialogue detected ({source}) - transitioning to {next_state}")
                return next_state
            return check_fn
        
        def trans_left_area_or_no_dialogue(x_range: List[int], y_range: List[int], next_state: str) -> Callable:
            """
            Transition when player has LEFT specific area OR all dialogue cleared.
            This prevents premature transitions while dialogue is blocking movement.
            Used for post-interaction dialogue like May's pokeball.
            """
            def check_fn(s, v):
                pos = s.get('player', {}).get('position', {})
                x, y = pos.get('x', -1), pos.get('y', -1)
                
                # Check if we've left the interaction area
                left_area = x not in x_range or y not in y_range
                
                # Check all dialogue indicators
                screen_context = v.get('screen_context', '').lower()
                text_box_visible = v.get('visual_elements', {}).get('text_box_visible', False)
                continue_prompt_visible = v.get('visual_elements', {}).get('continue_prompt_visible', False)
                game_state = s.get('game', {}).get('game_state', '').lower()
                no_dialogue = not text_box_visible and screen_context != 'dialogue' and game_state != 'dialog' and not continue_prompt_visible
                
                # Transition if we've left the area OR dialogue is fully cleared
                if left_area or no_dialogue:
                    print(f"🔍 [TRANS_LEFT_AREA] Pos ({x},{y}), left_area={left_area}, no_dialogue={no_dialogue}")
                    return next_state
                return None
            return check_fn
        
        def trans_left_area_only(x_range: List[int], y_range: List[int], next_state: str) -> Callable:
            """
            Transition ONLY when player has physically left the area.
            This is the most robust check - if dialogue is blocking, player can't move.
            Once dialogue clears, player can move and will leave the area.
            DO NOT use perception checks - they are unreliable due to filtering.
            """
            def check_fn(s, v):
                pos = s.get('player', {}).get('position', {})
                x, y = pos.get('x', -1), pos.get('y', -1)
                
                # Only transition if we've physically left the interaction area
                left_area = x not in x_range or y not in y_range
                
                if left_area:
                    print(f"🔍 [TRANS_LEFT_AREA_ONLY] Pos ({x},{y}) - left area, transitioning to {next_state}")
                    return next_state
                return None
            return check_fn

        def trans_position_area(x_range: List[int], y_range: List[int], next_state: str) -> Callable:
            """
            Transition when player ENTERS the specified area.
            Use this when dialogue will auto-trigger at a position and you want to transition 
            immediately upon arrival, not wait for dialogue to appear/clear.
            """
            def check_fn(s, v):
                pos = s.get('player', {}).get('position', {})
                x, y = pos.get('x', -1), pos.get('y', -1)
                
                # Transition if we're in the area
                in_area = x in x_range and y in y_range
                
                if in_area:
                    print(f"🔍 [TRANS_POSITION_AREA] Pos ({x},{y}) - in area, transitioning to {next_state}")
                    return next_state
                return None
            return check_fn

        def trans_no_dialogue_and_not_in_battle(next_state: str) -> Callable:
            """Transition when dialogue is done AND not in battle."""
            def check_fn(s, v):
                if not v.get('visual_elements', {}).get('text_box_visible', False) and not s.get('game', {}).get('in_battle', False):
                    return next_state
                return None
            return check_fn

        def trans_no_dialogue_and_no_nickname_text(next_state: str) -> Callable:
            """Transition when dialogue is done AND nickname text is gone."""
            def check_fn(s, v):
                dialogue = v.get('on_screen_text', {}).get('dialogue', '') or ''
                if not v.get('visual_elements', {}).get('text_box_visible', False) and "NICKNAME" not in dialogue.upper():
                    return next_state
                return None
            return check_fn

        def trans_nickname_or_dialogue_done(nickname_state: str, no_nickname_state: str) -> Callable:
            """
            Transition for S23_BIRCH_DIALOG_2 with two possible paths:
            1. If "NICKNAME" appears in dialogue/menu_title → go to nickname_state (S24)
            2. If dialogue clears AND no nickname text → go to no_nickname_state (S25)
            
            This handles the case where nickname screen may or may not appear after Birch's lab dialogue.
            """
            def check_fn(s, v):
                dialogue = v.get('on_screen_text', {}).get('dialogue', '') or ''
                menu_title = v.get('on_screen_text', {}).get('menu_title', '') or ''
                text_box_visible = v.get('visual_elements', {}).get('text_box_visible', False)
                
                # Check if nickname screen appeared
                if "NICKNAME" in dialogue.upper() or "NICKNAME" in menu_title.upper():
                    print(f"🔍 [TRANS_NICKNAME_OR_DONE] NICKNAME detected → {nickname_state}")
                    return nickname_state
                
                # Check if dialogue completely cleared (no nickname)
                if not text_box_visible:
                    print(f"🔍 [TRANS_NICKNAME_OR_DONE] Dialogue cleared, no nickname → {no_nickname_state}")
                    return no_nickname_state
                
                # Still in dialogue, keep waiting
                return None
            return check_fn

        # --- State Machine Definition ---

        # Pre-instantiate S0 transition functions so their internal state (e.g. the
        # _title_seq_presses counter in trans_has_dialogue) persists across steps.
        # If these are created inside the lambda they'd be re-created every call and
        # any accumulated counter state would be lost.
        _s0_trans_naming   = trans_detect_naming_screen('S2_GENDER_NAME_SELECT')
        _s0_trans_dialogue = trans_has_dialogue('S1_PROF_DIALOG')

        return {
            # === Phase 1: Title & Naming ===
            'S0_TITLE_SCREEN': BotState(
                name='S0_TITLE_SCREEN',
                description='Title screen - press A, but detect naming screen for shortcut',
                action_fn=action_clear_dialogue,  # Use clear_dialogue to handle naming shortcut
                # Transition to S1 if dialogue appears, OR to S2 if naming screen appears
                next_state_fn=lambda s, v: (
                    _s0_trans_naming(s, v) or
                    _s0_trans_dialogue(s, v)
                )
            ),
            'S1_PROF_DIALOG': BotState(
                name='S1_PROF_DIALOG',
                description='Professor Birch intro cutscene - press A until naming keyboard opens (PLAYER_NAME_SET)',
                action_fn=action_clear_dialogue,
                # Transition to S2 the moment PLAYER_NAME_SET fires — that milestone fires
                # when the naming screen opens, so S2/action_special_naming takes over
                # immediately and handles the keyboard correctly.
                next_state_fn=lambda s, v: (
                    'S2_GENDER_NAME_SELECT'
                    if s.get('milestones', {}).get('PLAYER_NAME_SET', {}).get('completed', False)
                    else None
                ),
                max_attempts=200,
                timeout_seconds=600.0,
            ),
            'S2_GENDER_NAME_SELECT': BotState(
                name='S2_GENDER_NAME_SELECT',
                description='Gender/name selection + all post-naming professor dialogue',
                action_fn=action_special_naming,
                # Stay in S2 until the location actually changes away from TITLE_SEQUENCE.
                # That only happens after every post-naming dialogue box is cleared and
                # the truck cutscene loads.  This works regardless of whether VLM is on or off.
                next_state_fn=lambda s, v: (
                    'S3_TRUCK_RIDE'
                    if s.get('player', {}).get('location', 'TITLE_SEQUENCE') not in ('TITLE_SEQUENCE', '', None)
                    else None
                ),
                max_attempts=200,
                timeout_seconds=600.0,
            ),

            # === Phase 2: Truck & House (The "Stuck in House" Fix) ===
            'S3_TRUCK_RIDE': BotState(
                name='S3_TRUCK_RIDE',
                description='Inside the moving van',
                action_fn=action_nav(NavigationGoal(x=8, y=1, map_location='MOVING_VAN', description="Exit Van")),
                next_state_fn=trans_location_contains('HOUSE 1F', 'S4_MOM_DIALOG_1F')  # Matches both PLAYERS_HOUSE_1F and BRENDANS_HOUSE_1F
            ),
            'S4_MOM_DIALOG_1F': BotState(
                name='S4_MOM_DIALOG_1F',
                description='Mom dialogue after truck ride (1F) - multi-page dialogue, then TV cutscene starts',
                action_fn=action_clear_dialogue_persistent,  # Use persistent for multi-page dialogue
                # CRITICAL: After Mom's dialogue, TV cutscene starts immediately (Prof Birch intro)
                # We can't wait for "no dialogue" because the TV keeps playing
                # Instead, check if we're at position (8,7) or (8,8) - means Mom dialogue finished and we auto-walked in
                # The TV cutscene dialogue will be handled by S5's navigation (it clears dialogue before moving)
                next_state_fn=lambda s, v: (
                    'S5_NAV_TO_STAIRS_1F' if (
                        s.get('player', {}).get('position', {}).get('y') in [7, 8] and
                        s.get('player', {}).get('position', {}).get('x') == 8 and
                        'HOUSE 1F' in s.get('player', {}).get('location', '')
                    ) else None
                )
            ),
            'S5_NAV_TO_STAIRS_1F': BotState(
                name='S5_NAV_TO_STAIRS_1F',
                description='Navigate to stairs on 1F',
                action_fn=action_nav(NavigationGoal(x=8, y=2, map_location='PLAYERS_HOUSE_1F', description="Go to Stairs")),
                next_state_fn=trans_location_contains('2F', 'S6_NAV_TO_CLOCK')  # Check for 2F (works for both PLAYERS_HOUSE and BRENDANS_HOUSE)
            ),
            
            # === THE CRITICAL FIX: S6 -> S7 with area-based transition ===
            'S6_NAV_TO_CLOCK': BotState(
                name='S6_NAV_TO_CLOCK',
                description='Navigate to clock in 2F bedroom and interact',
                action_fn=action_nav(NavigationGoal(x=5, y=1, map_location='PLAYERS_HOUSE_2F', description="Interact with Clock", should_interact=True)),
                # CRITICAL: Transition when in clock AREA and dialogue appears (not exact position)
                # This handles the "adjacent-interact" bug where agent presses A from (6,1) or (5,2)
                next_state_fn=trans_area_and_dialogue(x_range=[4, 5, 6], y_range=[1, 2], next_state='S7_SET_CLOCK')
            ),
            'S7_SET_CLOCK': BotState(
                name='S7_SET_CLOCK',
                description='Setting the clock and subsequent Mom dialogue',
                action_fn=action_special_clock,
                # min_wait_steps=15: when VLM is down, visuals appear clear immediately but
                # game_state='dialog' is REAL (post-clock Mom dialogue). The default 2-step escape
                # would prematurely start S8 with dialogue still active. 15 steps gives
                # action_special_clock time to press A through all post-clock dialogue pages.
                next_state_fn=trans_no_dialogue('S8_NAV_TO_STAIRS_2F', min_wait_steps=15)
            ),
            # === Phase 3 (Post-Clock): Exit House and Visit Rival ===
            # SPLIT INTO TWO STATES for batched navigation
            'S8_NAV_TO_STAIRS_2F': BotState(
                name='S8_NAV_TO_STAIRS_2F',
                description='Navigate to stairs on 2F after setting clock',
                action_fn=action_nav(NavigationGoal(x=7, y=1, map_location='PLAYERS_HOUSE_2F', description="2F Stairs")),
                next_state_fn=trans_location_contains('1F', 'S8B_NAV_TO_DOOR_1F')
            ),
            'S8B_NAV_TO_DOOR_1F': BotState(
                name='S8B_NAV_TO_DOOR_1F',
                description='Navigate to exit door on 1F',
                action_fn=action_nav(NavigationGoal(x=8, y=9, map_location='PLAYERS_HOUSE_1F', description="Exit House")),
                next_state_fn=trans_location_exact('LITTLEROOT TOWN', 'S9_NAV_TO_MAYS_HOUSE')
            ),
            
            # === Phase 3: Rival's House ===
            'S9_NAV_TO_MAYS_HOUSE': BotState(
                name='S9_NAV_TO_MAYS_HOUSE',
                description="Navigate to May's house",
                action_fn=action_nav(NavigationGoal(x=14, y=8, map_location='LITTLEROOT_TOWN', description="Go to May's House")),
                next_state_fn=trans_location_contains('MAYS HOUSE 1F', 'S10_MAYS_MOTHER_DIALOG')
            ),
            'S10_MAYS_MOTHER_DIALOG': BotState(
                name='S10_MAYS_MOTHER_DIALOG',
                description="May's mother dialogue (1F) - multi-page with '...' continuation",
                action_fn=action_clear_dialogue_persistent,  # Use persistent to handle multi-page dialogue
                next_state_fn=trans_no_dialogue('S11_NAV_TO_STAIRS_MAYS_HOUSE')  # Wait 3 steps for multi-page dialogue
            ),
            'S11_NAV_TO_STAIRS_MAYS_HOUSE': BotState(
                name='S11_NAV_TO_STAIRS_MAYS_HOUSE',
                description="Navigate to stairs in May's house (1F)",
                action_fn=action_nav(NavigationGoal(x=2, y=2, map_location='LITTLEROOT TOWN MAYS HOUSE 1F', description='Go to Stairs')),
                next_state_fn=trans_location_contains('MAYS HOUSE 2F', 'S11B_NAV_TO_POKEBALL')
            ),
            'S11B_NAV_TO_POKEBALL': BotState(
                name='S11B_NAV_TO_POKEBALL',
                description='Navigate to Pokéball on 2F and interact to trigger May',
                action_fn=action_nav(NavigationGoal(x=5, y=4, map_location='MAYS_HOUSE_2F', description="Interact with Pokéball", should_interact=True)),
                next_state_fn=trans_left_area_or_no_dialogue(x_range=[4, 5, 6], y_range=[3, 4, 5], next_state='S12_MAY_DIALOG')
            ),
            'S12_MAY_DIALOG': BotState(
                name='S12_MAY_DIALOG',
                description='May dialogue (2F) - clear dialogue then move LEFT to exit area',
                action_fn=action_clear_dialogue_then_try_move('LEFT'),
                next_state_fn=trans_left_area_only(x_range=[4, 5, 6], y_range=[3, 4, 5], next_state='S13_NAV_TO_STAIRS_2F')
            ),
            'S13_NAV_TO_STAIRS_2F': BotState(
                name='S13_NAV_TO_STAIRS_2F',
                description="Navigate to stairs on May's house 2F",
                action_fn=action_nav(NavigationGoal(x=1, y=1, map_location='LITTLEROOT TOWN MAYS HOUSE 2F', description='Go to Stairs')),
                next_state_fn=trans_location_contains('MAYS HOUSE 1F', 'S14A_MAY_DOWNSTAIRS_DIALOG')
            ),
            'S14A_MAY_DOWNSTAIRS_DIALOG': BotState(
                name='S14A_MAY_DOWNSTAIRS_DIALOG',
                description="May follows downstairs - move around until dialogue triggers, then clear it",
                action_fn=action_wander_until_dialogue,  # Wander RIGHT until May appears, then clear her dialogue
                next_state_fn=trans_no_dialogue('S14_NAV_TO_EXIT_MAYS_HOUSE', min_wait_steps=3)
            ),
            'S14_NAV_TO_EXIT_MAYS_HOUSE': BotState(
                name='S14_NAV_TO_EXIT_MAYS_HOUSE',
                description="Navigate to exit on May's house 1F",
                action_fn=action_nav(NavigationGoal(x=2, y=9, map_location='LITTLEROOT TOWN MAYS HOUSE 1F', description="Exit May's House")),
                next_state_fn=trans_location_exact('LITTLEROOT TOWN', 'S15_NAV_TO_NPC_NORTH')
            ),
            # SPLIT S15 into THREE states to avoid walking back into May's house
            # Problem: Door at (14,9) warps back into house when pressing UP
            # Solution: Move AWAY from x=14 first, then navigate north
            'S15_NAV_TO_NPC_NORTH': BotState(
                name='S15_NAV_TO_NPC_NORTH',
                description='Step LEFT away from door to avoid re-entering',
                action_fn=action_nav(NavigationGoal(x=11, y=11, map_location='LITTLEROOT TOWN', description='Move west away from door', should_interact=False)),
                next_state_fn=lambda s, v: (
                    'S15B_NAV_NORTH_CONTINUED' if s.get('player', {}).get('position', {}).get('x', 0) <= 12
                    else None
                )
            ),
            'S15B_NAV_NORTH_CONTINUED': BotState(
                name='S15B_NAV_NORTH_CONTINUED',
                description='Navigate north to NPC area - dialogue auto-triggers',
                action_fn=action_nav(NavigationGoal(x=11, y=1, map_location='LITTLEROOT TOWN', description='Walk north (no interaction)', should_interact=False)),
                next_state_fn=trans_position_area(x_range=[10, 11, 12], y_range=[1, 2, 3], next_state='S16_NPC_DIALOG')  # Transition when reaching area, not when dialogue appears
            ),
            'S16_NPC_DIALOG': BotState(
                name='S16_NPC_DIALOG',
                description='NPC dialogue - clear dialogue then move UP',
                action_fn=action_clear_dialogue_then_try_move('UP'),
                next_state_fn=trans_left_area_only(x_range=[10, 11, 12], y_range=[1, 2, 3], next_state='S17_NAV_TO_ROUTE_101')
            ),
            'S17_NAV_TO_ROUTE_101': BotState(
                name='S17_NAV_TO_ROUTE_101',
                description='Move UP to Route 101 - Birch cutscene auto-triggers, dialogue system handles it',
                action_fn=action_simple(['UP']),  # Move UP to trigger map transition
                # CRITICAL: Can't use trans_has_dialogue because bot yields to dialogue BEFORE checking transitions
                # Instead, detect Route 101 by location name (may be ROUTE 101 or corrupted TITLE_SEQUENCE)
                next_state_fn=lambda s, v: (
                    'S19_NAV_TO_BAG' if 'ROUTE' in s.get('player', {}).get('location', '').upper() or
                                       'TITLE' in s.get('player', {}).get('location', '').upper()
                    else None
                )
            ),
            # S18_BIRCH_DIALOG removed - dialogue system automatically handles Birch's "H-help me!" cutscene
            # After dialogue clears, agent transitions directly to S19_NAV_TO_BAG
            'S19_NAV_TO_BAG': BotState(
                name='S19_NAV_TO_BAG',
                description="Navigate to Birch's bag on ground and interact with it",
                action_fn=action_nav(NavigationGoal(x=7, y=14, map_location='ROUTE 101', description="Interact with Birch's Bag", should_interact=True)),
                next_state_fn=trans_area_and_dialogue(x_range=[6, 7, 8], y_range=[13, 14, 15], next_state='S20_INTERACT_BAG')
            ),
            'S20_INTERACT_BAG': BotState(
                name='S20_INTERACT_BAG',
                description='Interact with bag to trigger starter menu',
                action_fn=action_clear_dialogue,
                next_state_fn=lambda s, v: (
                    'S24_NICKNAME' if "NICKNAME" in (v.get('on_screen_text', {}).get('dialogue', '') or '').upper()
                    else ('S23_BIRCH_DIALOG_2' if (
                        s.get('milestones', {}).get('STARTER_CHOSEN', {}).get('completed', False) and
                        'PROFESSOR BIRCHS LAB' in s.get('player', {}).get('location', '')
                    )
                    else ('S21_STARTER_SELECT' if "Choose a" in (v.get('on_screen_text', {}).get('dialogue', '') or '')
                    else None))
                )
            ),
            'S21_STARTER_SELECT': BotState(
                name='S21_STARTER_SELECT',
                description='Pokemon Selection from bag',
                action_fn=action_special_starter,
                next_state_fn=trans_in_battle('S22_FIRST_BATTLE')
            ),
            'S22_FIRST_BATTLE': BotState(
                name='S22_FIRST_BATTLE',
                description='First battle (Poochyena)',
                action_fn=action_pass_to_battle_bot,
                next_state_fn=trans_milestone_complete('STARTER_CHOSEN', 'S23_BIRCH_DIALOG_2')
            ),
            'S23_BIRCH_DIALOG_2': BotState(
                name='S23_BIRCH_DIALOG_2',
                description='Dialogue with Birch after battle (in lab) - may or may not show nickname screen',
                action_fn=action_clear_dialogue,
                next_state_fn=trans_nickname_or_dialogue_done(nickname_state='S24_NICKNAME', no_nickname_state='COMPLETED')
            ),
            'S24_NICKNAME': BotState(
                name='S24_NICKNAME',
                description='Nickname starter screen',
                action_fn=action_special_nickname,
                next_state_fn=trans_no_dialogue_and_no_nickname_text('COMPLETED')
            ),
            # S25_LEAVE_LAB removed — ObjectiveManager handles lab exit navigation
            
            # === Final State ===
            'COMPLETED': BotState(
                name='COMPLETED',
                description='Opening sequence complete - hand off to VLM/A*',
                action_fn=lambda s, v: None,
                next_state_fn=lambda s, v: None
            )
        }

    def get_state_summary(self) -> Dict[str, Any]:
        """Get current state summary for debugging/monitoring"""
        state = self.states.get(self.current_state_name)
        elapsed = time.time() - self.state_entry_time if state else 0
        
        return {
            'current_state': self.current_state_name,
            'state_description': state.description if state else 'Unknown',
            'attempt_count': self.state_attempt_count,
            'max_attempts': state.max_attempts if state else 0,
            'elapsed_seconds': elapsed,
            'timeout_seconds': state.timeout_seconds if state else 0,
            'last_action': self.last_action,
            'state_history_length': len(self.state_history)
        }
    
    def reset(self):
        """Reset the opener bot to initial state"""
        self.current_state_name = 'S0_TITLE_SCREEN'
        self.state_entry_time = time.time()
        self.state_attempt_count = 0
        self.last_action = None
        self.state_history.clear()
        logger.info("[OPENER BOT] Reset to S0_TITLE_SCREEN state")


# === Global Instance Management ===

_global_opener_bot: Optional[OpenerBot] = None


def get_opener_bot() -> OpenerBot:
    """Get or create the global opener bot instance"""
    global _global_opener_bot
    if _global_opener_bot is None:
        _global_opener_bot = OpenerBot()
    return _global_opener_bot
