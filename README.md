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

3. **Navigation System** — Two-tier pathfinding with NPC-aware obstacle avoidance:
   - **Global:** Server-side A\* over the explored world graph with grass avoidance, ledge handling, portal detection, and **NPC obstacle injection**. Batched movement execution (up to 15 steps).
   - **Local:** BFS fallback over a 15×15 visible tile grid.
   - **NPC Targeting:** Dynamic NPC detection via `gObjectEvents` memory parsing. `_resolve_npc_coords()` locates NPCs by role using the `NpcRegistry` (adaptive discovery) with `graphics_id` fallback. NPC positions are injected as obstacles into A\* pathfinding to prevent the agent from walking through stationary NPCs.
   - **Graph-Derived Coordinates:** Building entrances, interior exits, and POI positions are resolved at runtime from `LOCATION_GRAPH` portal/warp metadata via `get_entrance_coords()`, `get_interior_exit_coords()`, and `get_poi_coords()` — eliminating hardcoded coordinate constants.
   - See [docs/PATHFINDING_SUMMARY.md](docs/PATHFINDING_SUMMARY.md).

4. **Objective Manager** — Milestone-driven progression through 40+ predefined milestones derived from official speedrun splits. Milestone `target_coords` are resolved dynamically from `LOCATION_GRAPH` via `target_coords_fn` lambdas. Provides goal coordinates and interaction flags via a tactical directive system. RAG-primary mode (Phase 4.3b) allows the walkthrough planner to override milestone targets. See [docs/DIRECTIVE_SYSTEM.md](docs/DIRECTIVE_SYSTEM.md).

5. **Perception Module** — Layered extraction pipeline:
   - **Primary:** VLM structured JSON extraction (Qwen2-VL-2B-Instruct, ~2.3 s local inference)
   - **Secondary:** OCR with Pokémon-specific colour matching (pytesseract)
   - **Tertiary:** Programmatic heuristics (red triangle detection, dialogue border matching)

6. **Brain (Memory & Recovery Planning)** — Episodic memory backed by **ChromaDB** with `all-MiniLM-L6-v2` embeddings. On every step the brain logs dialogue, detects blockers (keyword matching + position oscillation), and — when triggered — runs a RAG query → LLM recovery-planning pipeline. Recovery tasks are pushed onto a priority stack inside `ObjectiveManager` and executed before milestone navigation resumes. See [agent/brain/README.md](agent/brain/README.md).

### VLM Integration

The default VLM backend is **Google Gemini Flash** (`gemini-2.5-flash`). The system also supports OpenAI, OpenRouter, Ollama, and local HuggingFace models — selectable via CLI flags.

## Key Features

- **Retrieval-Augmented Generation (RAG):** The agent remembers past dialogue and events in a persistent vector database, querying them by semantic similarity to inform recovery plans.
- **Hybrid Perception:** Combines emulator memory reads (precise coordinates), VLM analysis (scene understanding), and OCR (text extraction).
- **Resilient Pathfinding:** Multi-tier navigation handles map transitions, warps, ledges, and cutscene triggers.
- **Client-Server Architecture:** FastAPI emulator server + Pygame client — supports headless operation and web-based stream visualization.
- **Cost-Efficient Design:** LLM inference fires only on demand (exceptions & blockers), keeping API costs low.

## Installation

### Prerequisites

- **Python 3.10** (pinned in `.python-version`; 3.11 also works, 3.12+ is **not** supported)
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager (recommended)
- A legally obtained **Pokémon Emerald GBA ROM**
- **[Tesseract OCR](https://github.com/tesseract-ocr/tesseract)** installed on your system
- **[libmgba](https://mgba.io/)** native shared library (see System Dependencies below)
- **(Optional)** A Google Gemini API key for the default VLM backend
- **(Optional)** An NVIDIA GPU with CUDA for local VLM inference and RL model training

### System Dependencies

Install system-level packages before the Python setup:

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install -y tesseract-ocr python3.10 python3.10-venv git

# Install the native mGBA shared library (required by the mgba Python package)
wget https://github.com/mgba-emu/mgba/releases/download/0.10.5/mGBA-0.10.5-ubuntu64-focal.tar.xz
tar -xf mGBA-0.10.5-ubuntu64-focal.tar.xz
sudo dpkg -i mGBA-0.10.5-ubuntu64-focal/libmgba.deb
sudo apt --fix-broken install -y   # resolves any missing deps (ffmpeg libs, etc.)

# On Ubuntu 22.04+, you may also need a libzip compatibility symlink:
sudo ln -sf /lib/x86_64-linux-gnu/libzip.so.4 /lib/x86_64-linux-gnu/libzip.so.5 && sudo ldconfig

# macOS (Homebrew)
brew install tesseract python@3.10 git mgba
```

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/kingkw1/pokeagent-emerald.git
cd pokeagent-emerald

# 2. Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create the virtual environment and install all dependencies
#    uv reads .python-version (3.10) and pyproject.toml automatically
uv sync

# 4. Activate the virtual environment
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

# (Alternative: skip uv and use pip)
# python3.10 -m venv .venv && source .venv/bin/activate
# pip install -r requirements.txt
```

### Configuration

1. **ROM File:** Place your Pokémon Emerald ROM at `Emerald-GBAdvance/rom.gba`.
2. **API Keys:** Create a `.env` file in the project root:

```env
# Required for the default Gemini VLM backend (either name works)
GEMINI_API_KEY=your_gemini_api_key_here
# GOOGLE_API_KEY=your_google_api_key_here  # also accepted

# Optional — only if using the OpenAI or OpenRouter backends
OPENAI_API_KEY=your_openai_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
```

3. **Save States (optional):** The `Emerald-GBAdvance/` directory ships with several `.state` files at different game checkpoints. You can use `--load-state` to start from any of them.

### Verify Installation

```bash
# Confirm Python version
python --version   # should be 3.10.x

# Run the test suite
pytest

# Quick smoke test — start in manual (keyboard) mode
python run.py --manual --load-state Emerald-GBAdvance/truck_start.state
```

### Transferring to Another Machine

The git repo contains the source code and tracked assets, but several large or sensitive items are **gitignored** and must be copied manually.

#### What Git Already Tracks

Everything in the repo — source code, docs, save states (`.state`), milestone configs, `uv.lock`, etc. Just `git clone` (or `git pull`) on the new machine to get these.

#### What You Must Copy Manually

| Item | Path | Size | Why |
|------|------|------|-----|
| **GBA ROM** | `Emerald-GBAdvance/rom.gba` | ~16 MB | Gitignored (`.gba`) — legally required to supply your own |
| **Trained models** | `models/` | ~20 GB | Gitignored — perception checkpoints (`perception_v0.1`, `perception_v0.2_qwen_final`) and RL models (`PPO/`, `PPO_Masked/`) |
| **Episodic memory DB** | `memory_db/` | ~7 MB | Gitignored — ChromaDB vector store with NPC registry and past episodes |
| **Agent cache** | `.pokeagent_cache/` | ~1 MB | Gitignored — checkpoint state, milestone progress, map stitcher data |
| **Environment file** | `.env` | tiny | Gitignored — API keys (Gemini, OpenAI, etc.) |

#### Transfer Steps

```bash
# === On the OLD machine ===

# 1. Make sure all code changes are committed and pushed
cd /path/to/pokeagent-emerald
git add -A && git commit -m "sync before transfer" && git push

# 2. Archive the gitignored files you need on the new machine
tar czf pokeagent-extras.tar.gz \
    Emerald-GBAdvance/rom.gba \
    models/ \
    memory_db/ \
    .pokeagent_cache/ \
    .env

# Transfer pokeagent-extras.tar.gz to the new machine (USB, scp, cloud, etc.)

# === On the NEW machine ===

# 3. Clone the repo
git clone https://github.com/kingkw1/pokeagent-emerald.git
cd pokeagent-emerald

# 4. Extract the gitignored extras into the repo root
tar xzf /path/to/pokeagent-extras.tar.gz

# 5. Install system deps + Python environment (see System Dependencies section above)
sudo apt install -y tesseract-ocr
# Install native mGBA library (see System Dependencies for full instructions)
wget https://github.com/mgba-emu/mgba/releases/download/0.10.5/mGBA-0.10.5-ubuntu64-focal.tar.xz
tar -xf mGBA-0.10.5-ubuntu64-focal.tar.xz
sudo dpkg -i mGBA-0.10.5-ubuntu64-focal/libmgba.deb
sudo apt --fix-broken install -y
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 6. Verify
source .venv/bin/activate
pytest
python run.py --manual --load-state Emerald-GBAdvance/truck_start.state
```

> **Tip — travelling light:** If you skip `models/` (~20 GB), the agent still works — it just won't have local perception/RL models. The default Gemini VLM backend runs remotely and doesn't need them. You can always copy the models over later.

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
| `--load-checkpoint` | off | Resume from the last `.pokeagent_cache/` checkpoint |
| `--backend NAME` | `gemini` | VLM backend (`gemini`, `openai`, `openrouter`, `local`, `ollama`) |
| `--model-name NAME` | `gemini-2.5-flash` | Model to use |
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
  - [x] **4.4a:** Fix `gObjectEvents` struct parsing — reliable NPC position reading from emulator memory.
  - [x] **4.4b:** Dynamic NPC targeting — `_resolve_npc_coords()` pipeline replaces hardcoded NPC coordinates in milestones.
  - [ ] **4.4c:** VLM bounding-box fallback for script-spawned NPCs not in `gObjectEvents`.
  - [x] **4.4d:** NPC Identity Registry — adaptive discovery replaces hardcoded `graphics_id` constants.
  - [x] **4.4e:** NPC obstacle injection — A\* pathfinding treats detected NPCs as impassable tiles.
  - [x] **4.4f:** Graph-derived coordinates — building entrances, interior exits, milestone `target_coords`, and NPC fallback positions resolved from `LOCATION_GRAPH` portal/POI metadata.
- [ ] **Phase 5: Post-Opening Strangler Fig** — Remove location-specific hardcoded blocks so RAG + generic systems drive all post-opening navigation:
  - [x] **5.1:** Milestone-index gate — opener-only special cases (Birch Lab, Petalburg City/Gym, Rival Battle) skipped when past opening sequence.
  - [x] **5.2:** Remove location-specific waypoint hacks — Route 104 South NPC avoidance and Petalburg Woods obstacle zone waypoints deleted; agent navigates via A\* + RAG.
  - [x] **5.3:** Generic Pokemon Center healing — `_check_healing_needed()` + `find_nearest_pokemon_center()` work in any city, replacing the Rustboro-specific healing block.
  - [ ] **5.4:** Gym navigation via RAG + A\* — replace Roxanne waypoint sequence with generic gym pathfinding.
  - [ ] **5.5:** End-to-end validation from `06_road` split to Roxanne completion with zero location-specific special cases.
- [ ] **Phase X: Semantic Twitch Plays Pokémon:** Implement a task queue API to allow stream viewers to inject natural language goals (e.g., *"Catch a Pikachu"*) directly into the Goal Manager.

## Documentation

See the [docs/](docs/) directory for detailed design documents:

- [ARCHITECTURAL_BLUEPRINT.md](docs/ARCHITECTURAL_BLUEPRINT.md) — Full system architecture and implementation status
- [DIRECTIVE_SYSTEM.md](docs/DIRECTIVE_SYSTEM.md) — Objective Manager and tactical directives
- [PATHFINDING_SUMMARY.md](docs/PATHFINDING_SUMMARY.md) — Navigation system internals
- [OPENER_BOT.md](docs/OPENER_BOT.md) — Intro-sequence state machine
- [DIALOGUE_SYSTEM.md](docs/DIALOGUE_SYSTEM.md) — Dialogue detection and handling
## License

This project is available under the [MIT License](LICENSE).
