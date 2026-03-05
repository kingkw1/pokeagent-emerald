# PokéAgent Emerald

**An autonomous AI agent that plays Pokémon Emerald using a hybrid hierarchical controller — combining deterministic programmatic logic with Vision Language Model (VLM) reasoning.**

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10--3.11-green.svg)
![Status](https://img.shields.io/badge/status-Active_Development-orange.svg)

## Overview

PokéAgent is a **Hierarchical Neuro-Symbolic Agent** designed to solve complex, long-horizon RPG tasks in real-time. It tackles the challenge of autonomous gameplay by splitting cognition into two distinct systems: a **"Fast Brain"** (deterministic controllers for navigation, combat, menus) and a **"Slow Brain"** (an on-demand LLM/VLM reasoning layer powered by RAG that handles exceptions, blockers, and strategic pivots).

### Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Fast by default** | Programmatic controllers handle navigation, combat, and menus deterministically |
| **Smart when needed** | VLM reasoning activates only on blockers or uncertain states — keeping cost and latency low |
| **Memory-augmented** | ChromaDB-backed episodic memory enables RAG-powered recovery planning |
| **Agentic Routing** | A central Executive FSM parses VLM data to issue strict directives, turning "God Class" monoliths into a clean routing switchboard |

## Architecture

PokéAgent uses a **Router-Executor** pattern where a central `ObjectiveManager` acts as the executive brain, issuing structured `Directive` objects that are executed by specialized deterministic controllers.

### System 1 / System 2 Design

1. **ObjectiveManager (Executive Router):** Reads milestone progression, generates a `Directive` (e.g., `goal_coords=(0, 8, 'ROUTE_102')`), and dispatches it to the appropriate controller.
2. **System 1 — Fast Brain (Execution):** `NavigationPlanner` (A\* pathfinding), `Battle Engine` (dual-mode: Heuristic / RL architecture), and `OpenerBot` (intro FSM) execute directives at high frequency with zero LLM calls.
3. **System 2 — Slow Brain (Recovery):** When the agent gets stuck (oscillation detection) or hits a dialogue blocker, the `RecoveryPlanner` queries `EpisodicMemory` (ChromaDB) via RAG and asks the LLM for a recovery plan. The resulting task is pushed onto a recovery stack that **pre-empts** normal milestone navigation — the agent executes the brain's plan before resuming the main quest.

```
┌─────────────────────────────────────────────────────────────┐
│                   ObjectiveManager                          │
│                  (Executive Router)                         │
│   Milestones → Directive → Dispatch (+ Recovery Priority)   │
├─────────────────────────────┬───────────────────────────────┤
│     System 1 (Fast Brain)   │    System 2 (Slow Brain)      │
│  ┌─────────┬──────────────┐ │  ┌───────────┬──────────────┐ │
│  │ Nav     │ Battle Engine│ │  │ Episodic  │ Recovery     │ │
│  │ Planner │ (Heuristic + │ │  │ Memory    │ Planner      │ │
│  │ (A*/BFS)│  RL arch.)   │ │  │ (ChromaDB)│ (RAG+Gemini) │ │
│  └─────────┴──────────────┘ │  └───────────┴──────────────┘ │
└─────────────────────────────┴───────────────────────────────┘
```

### Slow Brain Trigger Flow

The Slow Brain activates on three trigger types:

1. **Battle transitions** — When `in_battle` flips `True`, the `RecoveryPlanner` fires (RAG query → LLM). The recovery task (e.g., "Win the battle") is auto-completed when the battle ends.
2. **Dialogue blockers** — If NPC dialogue contains blocking keywords ("wait", "stop", "dangerous"), the `RecoveryPlanner` generates a recovery task.
3. **Navigation stuck** — If position oscillation is detected (≤2 unique positions over 6 steps while not in battle), `signal_blocker("Navigation Stuck")` fires, triggering RAG + LLM recovery on the next step.

Recovery tasks are consumed at the **top** of `get_next_action_directive()`, pre-empting milestone navigation. Once the recovery task completes, normal progression resumes.

### Specialized Controllers

1. **Opener Bot** — A 20+ state finite state machine that handles the game intro sequence (title screen → starter Pokémon selection). Hands off to the main agent after the `STARTER_CHOSEN` milestone. See [docs/OPENER_BOT.md](docs/OPENER_BOT.md).

2. **Dual-Mode Battle Engine** — A combat controller built on a Strategy-pattern architecture with two hot-swappable backends:
   - **Heuristic Agent (Active):** A robust rule-based engine with type-effectiveness matrix, trainer vs. wild classification, behavioral stuck-detection (switches strategy after repeated failed run attempts), and memory-based HP/PP tracking with VLM fallback. This is the production combat controller.
   - **RL Agent (Integration In Progress):** A Proximal Policy Optimization (PPO) model trained via `sb3-contrib MaskablePPO` on curriculum battle scenarios. A trained prototype (`emerald_curriculum_v1`) exists; observation alignment and live-agent data bridging are the remaining integration steps. See [agent/combat/RL_BATTLE_BOT_PLAN.md](agent/combat/RL_BATTLE_BOT_PLAN.md).

3. **Navigation System** — Two-tier pathfinding with NPC-seeking capability:
   - **Global:** Server-side A\* over the explored world graph with grass avoidance, ledge handling, and portal detection. Batched movement execution (up to 15 steps).
   - **Local:** BFS fallback over a 15×15 visible tile grid.
   - **NPC Targeting (Planned):** Dynamic NPC detection via `gObjectEvents` memory parsing + VLM semantic identification. Replaces hardcoded NPC coordinates in milestones with runtime-resolved positions.
   - See [docs/PATHFINDING_SUMMARY.md](docs/PATHFINDING_SUMMARY.md).

4. **Objective Manager** — Milestone-driven progression through 40+ predefined milestones derived from official speedrun splits. Provides goal coordinates and interaction flags via a tactical directive system. See [docs/DIRECTIVE_SYSTEM.md](docs/DIRECTIVE_SYSTEM.md).

5. **Perception Module** — Layered extraction pipeline:
   - **Primary:** VLM structured JSON extraction (Qwen2-VL-2B-Instruct, ~2.3 s local inference)
   - **Secondary:** OCR with Pokémon-specific colour matching (pytesseract)
   - **Tertiary:** Programmatic heuristics (red triangle detection, dialogue border matching)

6. **Brain (Memory & Recovery Planning)** — Episodic memory backed by **ChromaDB** with `all-MiniLM-L6-v2` embeddings. On every step the brain logs dialogue, detects blockers (keyword matching + position oscillation), and — when triggered — runs a RAG query → LLM recovery-planning pipeline. Recovery tasks are pushed onto a priority stack inside `ObjectiveManager` and executed before milestone navigation resumes. See [agent/brain/README.md](agent/brain/README.md).

### VLM Integration

The default VLM backend is **Google Gemini Flash** (`gemini-2.0-flash`). The system also supports OpenAI, OpenRouter, Ollama, and local HuggingFace models — selectable via CLI flags.

## Key Features

- **Retrieval-Augmented Generation (RAG):** The agent remembers past dialogue and events in a persistent vector database, querying them by semantic similarity to inform recovery plans.
- **Hybrid Perception:** Combines emulator memory reads (precise coordinates), VLM analysis (scene understanding), and OCR (text extraction).
- **Resilient Pathfinding:** Multi-tier navigation handles map transitions, warps, ledges, and cutscene triggers.
- **Client-Server Architecture:** FastAPI emulator server + Pygame client — supports headless operation and web-based stream visualization.
- **Cost-Efficient Design:** LLM inference fires only on demand (exceptions & blockers), keeping API costs low.

## Installation

### Prerequisites

- **Python 3.10 or 3.11** (3.12+ is not supported)
- A legally obtained Pokémon Emerald GBA ROM
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system
- (Optional) A Google Gemini API key for the default VLM backend

### Setup

```bash
# Clone the repository
git clone https://github.com/kingkw1/pokeagent-emerald.git
cd pokeagent-emerald

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
# .venv\Scripts\activate   # Windows

# Install dependencies (uv is recommended)
uv sync          # preferred
# or: pip install -r requirements.txt
```

### Configuration

1. **ROM File:** Place your ROM as `Emerald-GBAdvance/rom.gba` (this is the default path).
2. **API Keys:** Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_key_here
```

## Usage

### Autonomous Mode

```bash
python run.py --agent-auto
```

### Common Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--rom PATH` | `Emerald-GBAdvance/rom.gba` | Path to the GBA ROM |
| `--load-state PATH` | — | Load a save state on startup |
| `--backend NAME` | `gemini` | VLM backend (`gemini`, `openai`, `openrouter`, `local`, `ollama`) |
| `--model-name NAME` | `gemini-2.0-flash` | Model to use |
| `--manual` | off | Start in manual (keyboard) mode |
| `--headless` | off | Run without the Pygame display |
| `--simple` | off | Simple mode — direct frame→action without the full 4-module pipeline |
| `--record` | off | Record gameplay video |
| `--no-ocr` | off | Disable OCR dialogue detection |
| `--port PORT` | `8000` | FastAPI server port |

### Debug & Inspection Tools

```bash
# Inspect the contents of the ChromaDB episodic memory
python -m agent.brain.demos.inspect_brain

# Run a standalone RAG retrieval demo
python -m agent.brain.demos.demo_rag_memory
```

## Project Structure

```
pokeagent-emerald/
├── run.py                  # Main entry point — starts server + client
├── agent/                  # Core agent code
│   ├── __init__.py         # Agent class, module wiring
│   ├── action.py           # Master controller (priority delegation)
│   ├── battle_bot.py       # Rule-based battle engine
│   ├── navigation_planner.py
│   ├── objective_manager.py # Milestone-driven progression
│   ├── opener_bot.py       # Intro-sequence state machine
│   ├── perception.py       # Layered perception pipeline
│   ├── planning.py         # Programmatic planning (zero VLM calls)
│   ├── memory.py           # Legacy memory module
│   ├── location_graph.py   # World graph + BFS routing
│   ├── brain/              # Memory & RAG subsystem (ChromaDB)
│   └── combat/             # Dual-mode battle architecture (Heuristic active, RL integration in progress)
├── server/                 # FastAPI emulator server + Pygame client
│   ├── app.py              # Headless GBA emulator server
│   ├── client.py           # Pygame display + agent loop
│   ├── frame_server.py     # Lightweight frame server for web streaming
│   └── stream.html         # Web-based stream visualization UI
├── utils/                  # VLM backends, OCR, helpers
├── models/                 # Trained model artifacts (PPO, perception — experimental)
├── pokemon_env/            # GBA emulator bindings
├── Emerald-GBAdvance/      # ROM, save states, milestone configs
├── docs/                   # Architecture docs, competition guidelines
├── examples/               # Integration examples (e.g. OpenerBot quickstart)
├── data/                   # Perception seed data, curated screenshots
├── memory_db/              # ChromaDB persistent storage
└── tests/                  # Test suite
```

## Roadmap

- [x] Opener Bot (title → starter selection)
- [x] Rule-based Battle Bot with type effectiveness
- [x] Global A\* + local BFS navigation
- [x] Milestone-driven Objective Manager (40+ milestones)
- [x] VLM perception pipeline (Qwen2-VL + OCR + heuristics)
- [x] Episodic memory with ChromaDB RAG
- [x] **Phase 1: Brain Consolidation:** Merge the legacy `ObjectiveManager` (milestones) and `GoalManager` (RAG) into a single Executive Router.
- [x] **Phase 2: Controller Hollowing:** Refactor the 5,000-line `action.py` into a clean switchboard that delegates to specialized handlers. *(Complete — 4,972→584 lines, −88%. Extracted 6 modules: `pathfinding.py`, `stuck_handler.py`, `vlm_action.py`, `directive_nav.py`, `intro_handler.py`, `vlm_prompt.py`)*
- [x] **Phase 2.5: Slow Brain Wiring:** Connected `RecoveryPlanner` output to execution. Recovery tasks now pre-empt milestone navigation in `get_next_action_directive()`. Wired oscillation detector → `signal_blocker()` for navigation failures. Stripped wasteful VLM call from `planning.py` (now fully programmatic). Passed `ObjectiveManager` directly to `action_step()` instead of via function attribute.
- [ ] **Phase 3: The "Strangler Fig" Deprecation:** Phase out the hardcoded `OpenerBot` by transitioning its movement and battle logic to the dynamic A\* and RL systems.
- [ ] **Phase 3.5: RL Combat Integration:** Complete observation alignment, bridge the live-agent RAM data to the training observation format, and transition the default combat backend from the Heuristic Agent to the trained RL neural network. Includes an automated **data-collection pipeline** using the Pygame client to harvest diverse battle states from manual human play, feeding them into the `stable-retro` PPO training curriculum. See [agent/combat/RL_BATTLE_BOT_PLAN.md](agent/combat/RL_BATTLE_BOT_PLAN.md).
- [ ] **Phase 4: Proactive Strategic Planning (Walkthrough RAG):**
  - [x] **4.1:** Knowledge base preparation — Bulbapedia walkthrough chunked & embedded into ChromaDB (`strategy_guide` collection).
  - [x] **4.2:** Strategic planner — RAG-driven `get_next_directive()` queries walkthrough text & resolves location names to `LOCATION_GRAPH` keys.
  - [x] **4.3a:** Shadow mode — RAG planner runs alongside milestones, logging comparison to `shadow_comparison.jsonl`.
  - [x] **4.3b:** RAG-primary with milestone fallback — RAG drives navigation, milestones catch failures.
  - [ ] **4.3c:** RAG-only — Remove milestone list once behavioural evaluation confirms end-to-end corridor completion. Milestones retained as silent last-resort fallback.
  - [ ] **4.4: Local NPC Navigation (Three-Tier):** Enable the agent to navigate to specific NPCs without hardcoded coordinates. *(a)* Fix `gObjectEvents` struct parsing for reliable NPC positions from memory. *(b)* VLM semantic identification — match walkthrough NPC names against `visible_entities`. *(c)* VLM bounding-box fallback for script-spawned NPCs. See [agent/brain/PLAN.MD](agent/brain/PLAN.MD).
- [ ] **Phase 5: Semantic Twitch Plays Pokémon:** Implement a task queue API to allow stream viewers to inject natural language goals (e.g., *"Catch a Pikachu"*) directly into the Goal Manager.

## Documentation

See the [docs/](docs/) directory for detailed design documents:

- [ARCHITECTURAL_BLUEPRINT.md](docs/ARCHITECTURAL_BLUEPRINT.md) — Full system architecture and implementation status
- [DIRECTIVE_SYSTEM.md](docs/DIRECTIVE_SYSTEM.md) — Objective Manager and tactical directives
- [PATHFINDING_SUMMARY.md](docs/PATHFINDING_SUMMARY.md) — Navigation system internals
- [OPENER_BOT.md](docs/OPENER_BOT.md) — Intro-sequence state machine
- [DIALOGUE_SYSTEM.md](docs/DIALOGUE_SYSTEM.md) — Dialogue detection and handling
## License

This project is available under the [MIT License](LICENSE).
