#!/usr/bin/env python3
"""
MANUAL SCRIPTED TEST - Proves the system works

Sends button commands directly via server API (headless mode)
The server will process the action queue and execute the buttons

Sequence: A, A, UP, UP, LEFT, RIGHT, DOWN
Expected: After 2 A presses, at least ONE directional input should move the player
"""

import subprocess
import time
import requests
import sys

if __name__ == "__main__":
    print("=" * 70)
    print("✅ PASSING TEST: Manual Button Sequence")
    print("=" * 70)
    print("This test will PASS if player moves from (12,12) after button sequence")
    print("Using: python -m server.app (headless server)")
    print("Sequence: A, A, UP, UP, LEFT, RIGHT, DOWN")
    print("=" * 70)

    # Start server in headless mode (no client, so server processes actions directly)
    cmd = ["python", "-m", "server.app", "--port", "8000", "--load-state", "tests/save_states/dialog.state"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        print("\n⏳ Starting server (waiting 8 seconds for initialization)...")
        time.sleep(8)

        # Sequence with adequate wait times
        # IMPORTANT: Each action takes ~0.45s (12 hold frames + 24 release frames at 80 FPS)
        # So we need to wait AT LEAST 0.5s per action
        sequence = [
            ("A", 3.0),      # Dismiss dialogue page 1 - wait 3s
            ("A", 3.0),      # Dismiss dialogue page 2 - wait 3s
            ("UP", 3.0),     # Try up - wait 3s
            ("UP", 3.0),     # Try up again - wait 3s
            ("LEFT", 3.0),   # Try left - wait 3s
            ("RIGHT", 3.0),  # Try right - wait 3s
            ("DOWN", 3.0),   # Try down - wait 3s
        ]

        initial_pos = None
        final_pos = None

        print("\n🎮 EXECUTING BUTTON SEQUENCE:")
        print("(Each action takes ~0.45s, waiting 3s between to ensure processing)\n")

        for i, (button, wait) in enumerate(sequence, 1):
            # Send button - CRITICAL: Must use "buttons" (list), not "action" (string)
            try:
                resp = requests.post("http://localhost:8000/action", json={"buttons": [button]}, timeout=2)
                if resp.status_code == 200:
                    print(f"[{i}/{len(sequence)}] ✓ Sent: {button:<6}", end="", flush=True)
                else:
                    print(f"[{i}/{len(sequence)}] ✗ Failed to send {button} (status={resp.status_code})")
                    continue
            except Exception as e:
                print(f"[{i}/{len(sequence)}] ✗ Failed to send {button}: {e}")
                continue

            time.sleep(wait)

            # Check position
            try:
                resp = requests.get("http://localhost:8000/state", timeout=2)
                if resp.status_code == 200:
                    state = resp.json()
                    pos = state["player"]["position"]
                    if isinstance(pos, dict):
                        current_pos = (pos["x"], pos["y"])
                    else:
                        current_pos = tuple(pos[:2])

                    # Get dialogue state
                    in_dialog = state["game"].get("in_dialog", False)
                    movement_enabled = state["game"].get("movement_enabled", True)
                    game_state = state["game"].get("game_state", "unknown")

                    if initial_pos is None:
                        initial_pos = current_pos
                    final_pos = current_pos

                    if current_pos != (12, 12):
                        print(f" → {current_pos} ✅ MOVED! (dialog={in_dialog}, move={movement_enabled})")
                    else:
                        print(f" → {current_pos} (dialog={in_dialog}, move={movement_enabled}, state={game_state})")
            except Exception as e:
                print(f" (state check failed: {e})")

        print("\n" + "=" * 70)
        if final_pos and final_pos != (12, 12):
            print("✅ TEST PASSED!")
            print(f"   Player moved from (12,12) to {final_pos}")
            sys.exit(0)
        else:
            print("❌ TEST FAILED!")
            print(f"   Player never moved from (12,12)")
            print(f"   This means either:")
            print(f"     1. Dialogue didn't dismiss (still blocking movement)")
            print(f"     2. Movement inputs aren't being processed")
            print(f"     3. Player is surrounded by collision/walls")
            sys.exit(1)

    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except Exception:
            process.kill()
