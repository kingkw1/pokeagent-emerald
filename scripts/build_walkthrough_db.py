#!/usr/bin/env python3
"""
Build the Walkthrough Knowledge Base — Phase 4.1

One-time (or repeatable) script that:
1. Fetches Pokémon Emerald walkthrough parts from Bulbapedia's MediaWiki API.
2. Chunks each part at ``==Heading==`` boundaries.
3. Cleans MediaWiki markup → plain prose.
4. Embeds and stores every chunk in the ``strategy_guide`` ChromaDB collection.

Usage::

    python scripts/build_walkthrough_db.py              # fetch + embed all parts
    python scripts/build_walkthrough_db.py --parts 1 3  # only parts 1 and 3
    python scripts/build_walkthrough_db.py --dry-run     # preview chunks, don't embed
    python scripts/build_walkthrough_db.py --rebuild     # wipe collection first

The walkthrough parts on Bulbapedia (21 total):
    Part  1 — Littleroot Town → Petalburg City
    Part  2 — Route 104 → Rustboro City
    Part  3 — Rustboro City → Dewford Town
    Part  4 — Dewford Town → Slateport City
    Part  5 — Slateport City
    Part  6 — Route 110 → Mauville City
    Part  7 — Mauville City → Route 111
    Part  8 — Route 112 → Lavaridge Town
    Part  9 — Petalburg City → Route 118
    Part 10 — Route 119 → Fortree City
    Part 11 — Route 120 → Lilycove City
    Part 12 — Mt. Pyre → Lilycove City
    Part 13 — Mossdeep City
    Part 14 — Seafloor Cavern → Sootopolis City
    Part 15 — Route 126 → Pacifidlog Town
    Part 16 — Ever Grande City (Victory Road)
    Part 17 — Pokémon League (Elite Four & Champion)
    Part 18 — Post-game: Southern Island → New Mauville
    Part 19 — Post-game: Sky Pillar → Shoal Cave
    Part 20 — Post-game: Safari Zone → Trainer Hill
    Part 21 — Post-game: S.S. Tidal → Battle Frontier
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

# Ensure the project root is on sys.path so agent.brain imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.brain.walkthrough_db import WalkthroughDB, chunk_wikitext, SUPPLEMENTAL_CHUNKS
from agent.location_graph import LOCATION_GRAPH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Bulbapedia URL template
# ============================================================================

_WIKI_RAW_URL = (
    "https://bulbapedia.bulbagarden.net/w/index.php"
    "?title=Walkthrough:Pok%C3%A9mon_Emerald/Part_{part}"
    "&action=raw"
)

# Total walkthrough parts on Bulbapedia (Parts 1-21)
MAX_PARTS = 21

# Polite request headers
_HEADERS = {
    "User-Agent": "PokeAgentEmerald/1.0 (walkthrough-scraper; +https://github.com/kingkw1/pokeagent-emerald)",
}


def fetch_wikitext(part: int, retries: int = 3) -> Optional[str]:
    """Fetch raw wikitext for a walkthrough part from Bulbapedia.

    Returns the raw text or ``None`` on failure.
    """
    # Primary URL: Walkthrough namespace (confirmed working)
    # Fallback: Appendix namespace (older, may redirect)
    url_patterns = [
        (
            "https://bulbapedia.bulbagarden.net/w/index.php"
            f"?title=Walkthrough:Pok%C3%A9mon_Emerald/Part_{part}"
            "&action=raw"
        ),
        (
            "https://bulbapedia.bulbagarden.net/w/index.php"
            f"?title=Appendix:Emerald_walkthrough/Section_{part}"
            "&action=raw"
        ),
    ]

    for url in url_patterns:
        for attempt in range(1, retries + 1):
            try:
                req = Request(url, headers=_HEADERS)
                with urlopen(req, timeout=30) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                    # Detect redirect stubs (e.g. "#REDIRECT [[...]]")
                    if raw.strip().upper().startswith("#REDIRECT"):
                        logger.warning(
                            f"Part {part}: redirect stub ({len(raw)} chars) — {url}"
                        )
                        break  # try next URL pattern
                    # Bulbapedia returns a short error page if the title doesn't exist
                    if len(raw) < 200 and "no article" in raw.lower():
                        logger.warning(f"Part {part}: URL returned 'no article' — {url}")
                        break  # try next URL pattern
                    logger.info(f"Part {part}: fetched {len(raw):,} chars from {url}")
                    return raw
            except URLError as exc:
                logger.warning(
                    f"Part {part}: attempt {attempt}/{retries} failed — {exc}"
                )
                if attempt < retries:
                    time.sleep(2 * attempt)  # polite exponential backoff
            except Exception as exc:
                logger.error(f"Part {part}: unexpected error — {exc}")
                break

    logger.error(f"Part {part}: all fetch attempts failed.")
    return None


# ============================================================================
# Offline fallback — embedded walkthrough data for Parts 1-3
# ============================================================================
# If Bulbapedia is unreachable, we use curated walkthrough summaries that
# cover Littleroot → Rustboro (matching MILESTONE_PROGRESSION scope).
# These are written in the same ==Heading== format as the wiki.
# ============================================================================

_OFFLINE_WALKTHROUGH: dict[int, str] = {
    1: """
==Littleroot Town==
You begin the game riding in the back of a moving truck. When the truck stops, you'll be in Littleroot Town. Enter your new house — your Mom will greet you and tell you to set the clock upstairs. After going to your room and setting the clock, head back downstairs. Your Mom will mention that Professor Birch lives next door. Go outside and visit the house next door (your rival's house). Go upstairs to meet your rival, but they're not home. Head back outside and go north toward Route 101.

==Route 101==
When you try to walk into the tall grass on Route 101, you'll hear someone cry for help. Professor Birch is being chased by a wild Poochyena! He'll ask you to grab a Pokéball from his bag on the ground. Choose one of three starter Pokémon:
- Treecko (Grass type)
- Torchic (Fire type)
- Mudkip (Water type)
After choosing your starter and defeating the wild Poochyena, Professor Birch will thank you and invite you back to his lab. Head north through Route 101 to reach Oldale Town. The route is straightforward — just walk north through the tall grass.

==Oldale Town==
Welcome to Oldale Town! There's a Pokémon Center here where you can heal your Pokémon for free — use it whenever your Pokémon are hurt. There's also a Poké Mart, but it has limited stock initially. A man near the north exit will show you around. Head to Route 103 to the north to find your rival for your first battle.
""",
    2: """
==Route 103==
Walk north from Oldale Town to reach Route 103. You'll find your rival here. Talk to them to trigger your first rival battle. Your rival will have the starter Pokémon that has a type advantage over yours (if you chose Treecko, they have Torchic, etc.). After winning (or losing — the story continues either way), your rival will suggest heading back to Birch's Lab.

==Birch's Lab==
Return south through Route 103 and Oldale Town, then south through Route 101 to Littleroot Town. Enter Professor Birch's Lab. The Professor will give you a Pokédex, and your rival will give you 5 Poké Balls. Now your Pokémon journey truly begins! Head back north through Route 101 to Oldale Town, then west through Route 102 toward Petalburg City.

==Route 102==
Route 102 connects Oldale Town (east) to Petalburg City (west). There are several trainers here who will challenge you:
- Youngster Calvin (Poochyena)
- Bug Catcher Rick (Wurmple)
- Youngster Allen (Zigzagoon)
You can catch wild Pokémon in the tall grass here: Zigzagoon, Wurmple, Lotad, Seedot, Ralts (rare), Poochyena, and Surskit (rare). Walk west through the route, battling trainers along the way, to reach Petalburg City.

==Petalburg City==
Petalburg City is home to the Petalburg Gym, led by your father Norman. When you enter town, head to the gym. Inside, you'll meet your Dad and a boy named Wally. Your Dad will ask you to help Wally catch his first Pokémon. You'll escort Wally to Route 102 where he catches a Ralts. After returning to the gym, your Dad will tell you to challenge the Rustboro Gym first. Head west from Petalburg to reach Route 104.
""",
    3: """
==Route 104 (South)==
Head west from Petalburg City to Route 104's southern section. There's a beach area and a few trainers. Rich Boy Winston has a Zigzagoon. Continue north and you'll reach the entrance to Petalburg Woods.

==Petalburg Woods==
Petalburg Woods is a maze-like forest area. Follow the main path north. Inside the forest, you'll encounter:
- Bug Catcher Lyle (Wurmple)
- Bug Catcher James (Nincada)
You'll also find a Devon Researcher being harassed by a Team Aqua Grunt. Battle the Grunt (he has a Poochyena) to save the researcher. The grateful researcher gives you a Great Ball. Continue north through the woods to exit onto Route 104 North.

Wild Pokémon in the woods include: Zigzagoon, Wurmple, Silcoon, Cascoon, Taillow, Shroomish, and Slakoth (rare).

==Route 104 (North)==
After exiting Petalburg Woods, you'll be on the northern section of Route 104. There are a few trainers here:
- Lass Haley (Lotad, Shroomish)
- Twins Gina & Mia (Seedot, Lotad)
The Pretty Petal Flower Shop is here — talk to one of the sisters to receive the Wailmer Pail. Continue north to reach Rustboro City.

==Rustboro City==
Rustboro City is a large city with many important locations:
- Rustboro Gym (Rock-type, led by Roxanne)
- Pokémon Center (heal your team here before the gym!)
- Devon Corporation (important later)
- Pokémon School

Head to the Pokémon Center first to heal up. When you're ready, enter the Rustboro Gym. The gym uses Rock-type Pokémon. If you chose Treecko or Mudkip, you'll have a type advantage. If you chose Torchic, consider catching a Lotad or Shroomish in the forest for this battle.

Gym Leader Roxanne has:
- Geodude (Lv. 12)
- Nosepass (Lv. 15)

After defeating Roxanne, you'll receive the Stone Badge and TM39 (Rock Tomb). Exit the gym and head south to continue your adventure.
""",
}


def generate_location_graph_chunks() -> List[dict]:
    """Convert LOCATION_GRAPH portal data into RAG-ready text chunks.

    One chunk per location.  Format::

        <display_name> (<key>): <description>
        Portals:
          <direction> → <neighbor_key>  (entry=<coords>, exit=<coords>, type=<type>)
          ...

    These topology chunks give the HTN Supervisor precise navigation data
    (entry/exit tiles, connectivity) from structured game-world knowledge,
    replacing the 5 hand-written SUPPLEMENTAL_CHUNKS.
    """
    chunks: List[dict] = []
    for key, node in LOCATION_GRAPH.items():
        lines = [
            f"{node['display_name']} ({key}): {node.get('description', '')}",
        ]
        portals = node.get("portals") or {}
        if portals:
            lines.append("Portals:")
            for neighbor_key, portal in portals.items():
                req = portal.get("requirements")
                req_str = f"  [requires: {req}]" if req else ""
                lines.append(
                    f"  {portal.get('direction', '?')} → {neighbor_key}"
                    f"  (entry={portal.get('entry_coords')}, "
                    f"exit={portal.get('exit_coords')}, "
                    f"type={portal.get('type', '?')}){req_str}"
                )
        chunks.append({
            "text": "\n".join(lines),
            "metadata": {
                "location_key": key,
                "display_name": node["display_name"],
                "map_id": node.get("map_id"),
                "source": "LOCATION_GRAPH",
                "supplemental": True,
                "is_topology": True,
                "has_battle": False,
            },
        })
    return chunks


def get_walkthrough_text(part: int, offline: bool = False) -> Optional[str]:
    """Fetch walkthrough text online, falling back to offline if available."""
    if offline:
        text = _OFFLINE_WALKTHROUGH.get(part)
        if text:
            logger.info(f"Part {part}: using offline walkthrough ({len(text):,} chars)")
            return text
        logger.warning(f"Part {part}: no offline text available.")
        return None

    text = fetch_wikitext(part)
    if text:
        return text

    # Online fetch failed — try offline fallback
    fallback = _OFFLINE_WALKTHROUGH.get(part)
    if fallback:
        logger.info(f"Part {part}: online fetch failed, using offline fallback.")
        return fallback
    return None


# ============================================================================
# Main
# ============================================================================

def build_database(
    parts: Optional[List[int]] = None,
    dry_run: bool = False,
    rebuild: bool = False,
    offline: bool = False,
    db_path: str = "./memory_db",
) -> int:
    """Fetch, chunk, and embed walkthrough parts. Returns total chunks added."""
    db = WalkthroughDB(db_path=db_path)

    if rebuild:
        logger.info("Rebuilding: clearing existing strategy_guide collection...")
        db.clear()

    if parts is None:
        parts = list(range(1, MAX_PARTS + 1))

    total_chunks = 0

    for part_num in parts:
        raw = get_walkthrough_text(part_num, offline=offline)
        if not raw:
            continue

        chunks = chunk_wikitext(raw, part_number=part_num)
        logger.info(f"Part {part_num}: {len(chunks)} chunks extracted.")

        if dry_run:
            for i, ch in enumerate(chunks):
                meta = ch["metadata"]
                preview = ch["text"][:120].replace("\n", " ")
                print(
                    f"  [{i}] location={meta.get('location', '?')!r}  "
                    f"battle={meta.get('has_battle')}  "
                    f"len={len(ch['text'])}  "
                    f"preview={preview!r}..."
                )
            total_chunks += len(chunks)
            continue

        added = db.add_chunks(chunks)
        total_chunks += added
        logger.info(f"Part {part_num}: {added} chunks embedded.")

        # Polite delay between fetches
        time.sleep(1)

    if not dry_run:
        added = db.add_chunks(SUPPLEMENTAL_CHUNKS)
        total_chunks += added
        logger.info(f"Supplemental chunks: {added} embedded.")

        # Phase 5.5 — LOCATION_GRAPH topology chunks.
        # These provide precise portal coordinates and connectivity for the 21
        # nodes covering Littleroot Town → Rustboro Gym.  Once validated they
        # supersede all SUPPLEMENTAL_CHUNKS above (comment out that block then).
        topology = generate_location_graph_chunks()
        added = db.add_chunks(topology)
        total_chunks += added
        logger.info(f"LOCATION_GRAPH topology chunks: {added} embedded ({len(topology)} nodes).")

    action = "previewed" if dry_run else "embedded"
    logger.info(
        f"Done. {total_chunks} chunks {action}. "
        f"Collection size: {db.count()}"
    )
    return total_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Build the Walkthrough RAG knowledge base."
    )
    parser.add_argument(
        "--parts",
        type=int,
        nargs="+",
        default=None,
        help="Walkthrough part numbers to process (default: all 1–21).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview chunks without embedding.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe the strategy_guide collection before building.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use offline walkthrough text only (no network requests).",
    )
    parser.add_argument(
        "--db-path",
        default="./memory_db",
        help="Path to ChromaDB persistent storage (default: ./memory_db).",
    )
    args = parser.parse_args()

    total = build_database(
        parts=args.parts,
        dry_run=args.dry_run,
        rebuild=args.rebuild,
        offline=args.offline,
        db_path=args.db_path,
    )

    print(f"\n{'='*60}")
    print(f"Walkthrough KB build complete: {total} chunks processed.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
