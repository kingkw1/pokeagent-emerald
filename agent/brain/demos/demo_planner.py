# agent/brain/demos/demo_planner.py
"""
Phase 1 demo: GoalManager detects a blocker, RecoveryPlanner generates a
recovery plan via LLM (or mock), and the new task is injected back into the
goal stack.

Usage (from project root):
    python -m agent.brain.demos.demo_planner          # mock mode
    python -m agent.brain.demos.demo_planner --live   # live mode (Gemini)
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path so `agent.*` imports work
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agent.brain.goal_manager import GoalManager
from agent.brain.planner import RecoveryPlanner


def run_demo(live: bool = False):
    print("==================================================")
    print("🧠 POKEMON AGENT: REASONING ENGINE DEMO (PHASE 1)")
    print("==================================================")

    # 1. Initialise components
    vlm = None
    if live:
        from utils.vlm import VLM
        vlm = VLM(model_name="gemini-2.0-flash", backend="gemini")
        print("   Mode: LIVE (Gemini API)")
    else:
        print("   Mode: MOCK (no API call)")

    gm = GoalManager()
    planner = RecoveryPlanner(vlm=vlm)

    print(f"\n[Status] Initial: {gm.current_directive}")

    # 2. Simulate getting blocked by the Old Man
    mock_perception = {
        "visual_data": {
            "screen_context": "overworld",
            "on_screen_text": {
                "dialogue": "Wait! Don't go out into the tall grass! "
                            "I'll show you how to catch Pokemon!",
                "speaker": "Old Man",
            },
        }
    }

    print("\n--- AGENT RECEIVES VISUAL DATA ---")
    gm.update(mock_perception)

    # 3. If blocked, trigger the Planner
    active_task = gm.state["sub_tasks"][0]
    if active_task["status"] == "BLOCKED":
        print("\n--- TRIGGERING RECOVERY PLANNER ---")

        plan = planner.generate_recovery_plan(
            current_goal=active_task["task"],
            blocker_reason="NPC Dialogue Keyword Detected",
            blocker_context=active_task["blocker_context"],
        )

        print(f"🤖 LLM REASONING: {plan['reasoning']}")
        print(f"✅ NEW TASK GENERATED: {plan['recovery_task']}")

        # 4. Inject the recovery task into the GoalManager
        gm.add_recovery_task(plan["recovery_task"])

    print(f"\n[Status] Final: {gm.current_directive}")
    print("==================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1 Recovery Planner Demo")
    parser.add_argument("--live", action="store_true",
                        help="Call the real Gemini API instead of using mock responses")
    args = parser.parse_args()
    run_demo(live=args.live)
