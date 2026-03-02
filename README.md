# PokéAgent: A Hierarchical Neuro-Symbolic Agent

**An autonomous AI agent that plays Pokémon Emerald using a "Fast Brain / Slow Brain" hybrid architecture.**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-green.svg)
![Status](https://img.shields.io/badge/status-Active_Development-orange.svg)

## 🤖 Overview

This project implements a **Hierarchical Control System** designed to solve complex, long-horizon RPG tasks in real-time. Unlike traditional RL agents (which struggle with long-term planning) or pure LLM agents (which are too slow and expensive for 60 FPS gameplay), PokéAgent utilizes a **Neuro-Symbolic Architecture**.

The agent splits cognition into two distinct systems:
1.  **System 1 (Fast Brain):** Deterministic, high-frequency modules for navigation, combat, and menu macros.
2.  **System 2 (Slow Brain):** A Latent Reasoning Engine powered by **Gemini Pro** and **RAG (ChromaDB)** that handles strategy, exceptions, and blockers.

## 🧠 Architecture

The agent operates on a **"Router-Executor"** pattern. The central **Goal Manager** acts as the executive router, dispatching tasks to specialized sub-systems based on the current state.

### 1. The Executive Layer (Goal Manager)
* **Role:** The "Traffic Cop" of the agent.
* **Function:** Tracks high-level objectives (e.g., "Obtain first badge") and monitors perception for "Blockers" (e.g., dialogue boxes, unseen obstacles).
* **Logic:** Uses a Finite State Machine (FSM) to switch between Navigation, Combat, and Planning modes.

### 2. System 1: The "Fast Brain" (Real-Time Execution)
* **Navigation Planner:** A graph-based **A* pathfinder** that routes the agent through the game world with pixel-perfect precision.
* **Battle Engine:** A hybrid heuristic/RL model that handles turn-based combat, type matching, and item usage at 60 FPS.
* **Scripted Macros:** Deterministic execution for UI-heavy tasks (e.g., setting the clock, naming the character).

### 3. System 2: The "Slow Brain" (Reasoning & Recovery)
* **Trigger:** Activates *only* when the Goal Manager detects a blocker (e.g., "Wait! You can't go there!").
* **Perception:** Uses a **Vision Language Model (VLM)** to interpret the screen and extract semantic context.
* **Memory (RAG):** Queries a local **Vector Database (ChromaDB)** to retrieve relevant game knowledge or past experiences.
* **Planning:** Generates a structured JSON recovery plan (e.g., *"The context implies a trainer battle is required. Switch to Battle Mode."*) and pushes it to the Goal Manager.

## 📂 Key Features

* **Retrieval-Augmented Generation (RAG):** The agent "remembers" game rules and past events using semantic vector search.
* **Hybrid Perception:** Combines direct memory access (for precise coordinates) with visual analysis (OCR & VLM) for dialogue and context understanding.
* **Resilient Pathfinding:** A multi-stage navigation system that handles map transitions, ledges, and cutscene triggers.
* **Cost-Efficient Design:** LLM inference is used only on demand (handling exceptions), keeping operating costs low while maintaining high intelligence.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/kingkw1/pokeagent-emerald.git
cd pokeagent

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

```

### Configuration

1. **ROM File:** Place your legally obtained `Pokemon Emerald.gba` file in the root directory.
2. **API Keys:** Create a `.env` file and add your Google Gemini API key:

```env
GOOGLE_API_KEY=your_key_here
```


## 🚀 Usage

To run the agent in autonomous mode:

```bash
python run.py --agent-auto
```

**Debug Tools:**

* `python inspect_brain.py`: View the contents of the semantic memory (ChromaDB).
* `python demo_rag_memory.py`: Run a standalone test of the retrieval system.

## 🔮 Roadmap

* **Phase 1 (Complete):** Navigation, Basic Combat, VLM Perception.
* **Phase 2 (Current):** RAG Integration, Dynamic "Blocker" Handling, Goal Manager Refactor.
* **Phase 3 (Next):** **"Semantic Twitch Plays Pokémon"** — Integrating a task queue to allow external users (or chat) to inject natural language goals (e.g., *"Catch a Pikachu"*) which the agent interprets and executes autonomously.

## 📄 License

This project is open-source and available under the MIT License.
