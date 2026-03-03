# agent/brain/demos/demo_full_flow.py
"""
Phase 2 end-to-end RAG demo: Seeds memories, triggers a blocker, queries
ChromaDB via semantic search, and generates a recovery plan.

Usage (from project root):
    python -m agent.brain.demos.demo_full_flow          # mock mode
    python -m agent.brain.demos.demo_full_flow --live   # live mode (Gemini)
"""
import argparse
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from agent.objective_manager import ObjectiveManager
from agent.brain.planner import RecoveryPlanner
from agent.brain.memory import EpisodicMemory

import os
_DEMO_DB_PATH = os.path.join(_PROJECT_ROOT, "memory_db_full_flow")


def run_demo(live: bool = False):
    print("==================================================")
    print("🧠 POKEMON AGENT: END-TO-END RAG DEMO")
    print("==================================================")

    # 1. Initialize Components
    memory = EpisodicMemory(db_path=_DEMO_DB_PATH)
    memory.clear_memory()  # Start fresh

    vlm = None
    if live:
        from utils.vlm import VLM
        vlm = VLM(model_name="gemini-2.0-flash", backend="gemini")
        print("   Mode: LIVE (Gemini API)")
    else:
        print("   Mode: MOCK (no API call)")

    om = ObjectiveManager()
    planner = RecoveryPlanner(vlm=vlm, memory=memory, verbose=True)

    # 2. The "Learning" Phase
    # The agent "experiences" the game rules and logs them to ChromaDB
    print("\n--- PHASE 1: LEARNING (LOGGING MEMORIES) ---")
    facts = [
        "To pass the Old Man in Viridian/Oldale, you must talk to him to watch the tutorial.",
        "Small trees can be cut using HM01 Cut.",
        "Ledges are one-way jumps.",
        "The Sketch Artist blocks Route 103 until you beat the Rival.",
    ]
    for fact in facts:
        memory.log_event(fact, {"type": "mechanic"})
        print(f"📝 Logged: {fact}")

    # 3. The "Problem" Phase
    # The agent gets blocked
    print("\n--- PHASE 2: THE BLOCKER ---")
    mock_perception = {
        "visual_data": {
            "screen_context": "overworld",
            "on_screen_text": {
                "dialogue": "Wait! Don't go out there! It's dangerous!",
                "speaker": "Old Man",
            },
        }
    }
    om._scan_dialogue_for_blockers(mock_perception)

    # 4. The "Solution" Phase (RAG + Planning)
    if om.is_blocked:
        print("\n--- PHASE 3: RAG & PLANNING ---")

        active = om.get_active_objectives()
        current_goal = active[0].description if active else "Reach Petalburg City"

        plan = planner.generate_recovery_plan(
            current_goal=current_goal,
            blocker_reason="NPC Dialogue Keyword",
            blocker_context=om._blocker_state.get("context", ""),
        )

        print(f"🤖 LLM REASONING: {plan['reasoning']}")
        print(f"✅ NEW TASK: {plan['recovery_task']}")

        # Validate that RAG actually happened
        if "tutorial" in plan["reasoning"].lower() or "talk" in plan["recovery_task"].lower():
            print("\n🏆 SUCCESS: The agent used the Retrieved Memory to solve the problem!")
        else:
            print("\n⚠️  WARNING: The plan seems generic. RAG might not have retrieved the right key.")
    else:
        print("\n❌ FAIL: ObjectiveManager did not detect the blocker.")

    # Cleanup demo DB
    shutil.rmtree(_DEMO_DB_PATH, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end RAG demo for the Pokemon Agent")
    parser.add_argument("--live", action="store_true", help="Use real Gemini API")
    args = parser.parse_args()
    run_demo(live=args.live)
