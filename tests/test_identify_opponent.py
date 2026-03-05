"""
Tests for _identify_opponent() — the single source of truth for opponent species.

Verifies the priority chain:
  1. VLM visible_entities  (visually confirmed)
  2. Dialogue history      ("sent out X", "Wild X appeared")
  3. RAM opponent_pokemon   (last resort — proven unreliable)
"""

import unittest
from agent.battle_bot import BattleBot


class TestIdentifyOpponent(unittest.TestCase):

    def setUp(self):
        self.bot = BattleBot()

    # ---------------------------------------------------------------
    # Helper builders
    # ---------------------------------------------------------------

    @staticmethod
    def _visual(entities=None):
        """Build minimal visual_data with visible_entities."""
        return {
            "visible_entities": entities or [],
            "screen_context": "battle",
            "on_screen_text": {
                "dialogue": "",
                "raw_dialogue": "",
                "speaker": None,
                "menu_title": None,
            },
        }

    @staticmethod
    def _state(party=None):
        """Build minimal state_data with player party."""
        if party is None:
            party = ["TREECKO"]
        # Convert string species names to dicts matching real party format
        party_dicts = [{"species_name": s} for s in party]
        return {
            "game": {
                "battle_info": {
                    "player_pokemon": {
                        "species": "TREECKO",
                        "level": 7,
                    },
                }
            },
            "player": {"party": party_dicts},
        }

    @staticmethod
    def _battle_info(species="Unknown", types=None):
        """Build battle_info dict with opponent_pokemon from RAM."""
        return {
            "player_pokemon": {
                "species": "TREECKO",
                "level": 7,
                "current_hp": 20,
                "max_hp": 20,
            },
            "opponent_pokemon": {
                "species": species,
                "types": types or [],
            },
        }

    # ---------------------------------------------------------------
    # Priority 1 — VLM visible_entities
    # ---------------------------------------------------------------

    def test_vlm_takes_priority_over_ram(self):
        """VLM says POOCHYENA, RAM says TAILLOW → must return POOCHYENA."""
        visual = self._visual(entities=["POOCHYENA", "TREECKO"])
        state = self._state(party=["TREECKO"])
        battle = self._battle_info(species="TAILLOW", types=["NORMAL", "FLYING"])

        species, types = self.bot._identify_opponent(visual, state, battle)
        self.assertEqual(species, "POOCHYENA")
        # Types from RAM are still collected
        self.assertEqual(types, ["NORMAL", "FLYING"])

    def test_vlm_alone_no_ram(self):
        """VLM finds opponent, RAM is empty."""
        visual = self._visual(entities=["ZIGZAGOON", "TREECKO"])
        state = self._state(party=["TREECKO"])
        battle = self._battle_info(species="Unknown", types=[])

        species, types = self.bot._identify_opponent(visual, state, battle)
        self.assertEqual(species, "ZIGZAGOON")
        self.assertEqual(types, [])

    def test_vlm_filters_player_party(self):
        """VLM sees both player and opponent — only returns opponent."""
        visual = self._visual(entities=["TREECKO", "WURMPLE"])
        state = self._state(party=["TREECKO"])
        battle = self._battle_info(species="Unknown")

        species, _ = self.bot._identify_opponent(visual, state, battle)
        self.assertEqual(species, "WURMPLE")

    # ---------------------------------------------------------------
    # Priority 2 — Dialogue history
    # ---------------------------------------------------------------

    def test_dialogue_used_when_vlm_empty(self):
        """VLM has no entities, but dialogue has 'sent out POOCHYENA'."""
        visual = self._visual(entities=[])
        state = self._state()
        battle = self._battle_info(species="Unknown")

        self.bot._dialogue_history = [
            "YOUNGSTER CALVIN sent out POOCHYENA!",
        ]

        species, _ = self.bot._identify_opponent(visual, state, battle)
        self.assertEqual(species, "POOCHYENA")

    def test_dialogue_used_when_vlm_only_sees_player(self):
        """VLM only sees TREECKO (our pokemon), dialogue has opponent."""
        visual = self._visual(entities=["TREECKO"])
        state = self._state(party=["TREECKO"])
        battle = self._battle_info(species="Unknown")

        self.bot._dialogue_history = [
            "Wild ZIGZAGOON appeared!",
        ]

        species, _ = self.bot._identify_opponent(visual, state, battle)
        self.assertEqual(species, "ZIGZAGOON")

    # ---------------------------------------------------------------
    # Priority 3 — RAM (last resort)
    # ---------------------------------------------------------------

    def test_ram_used_when_vlm_and_dialogue_both_empty(self):
        """No VLM entities, no dialogue — RAM is only option."""
        visual = self._visual(entities=[])
        state = self._state()
        battle = self._battle_info(species="TAILLOW", types=["NORMAL", "FLYING"])

        self.bot._dialogue_history = []
        self.bot._opponent_species_from_dialogue = ""

        species, types = self.bot._identify_opponent(visual, state, battle)
        self.assertEqual(species, "TAILLOW")
        self.assertEqual(types, ["NORMAL", "FLYING"])

    def test_ram_species_prefix_filtered(self):
        """RAM species starting with 'Species_' is treated as Unknown."""
        visual = self._visual(entities=[])
        state = self._state()
        battle = self._battle_info(species="Species_278", types=[])

        self.bot._dialogue_history = []
        self.bot._opponent_species_from_dialogue = ""

        species, _ = self.bot._identify_opponent(visual, state, battle)
        self.assertEqual(species, "Unknown")

    # ---------------------------------------------------------------
    # Types from RAM always collected
    # ---------------------------------------------------------------

    def test_ram_types_collected_regardless_of_species_source(self):
        """Even when VLM identifies species, types from RAM are returned."""
        visual = self._visual(entities=["POOCHYENA", "TREECKO"])
        state = self._state(party=["TREECKO"])
        battle = self._battle_info(species="TAILLOW", types=["DARK"])

        species, types = self.bot._identify_opponent(visual, state, battle)
        self.assertEqual(species, "POOCHYENA")
        self.assertEqual(types, ["DARK"])

    # ---------------------------------------------------------------
    # Cache update
    # ---------------------------------------------------------------

    def test_dialogue_cache_updated_when_vlm_finds_species(self):
        """VLM identifies species → dialogue cache is updated to match."""
        visual = self._visual(entities=["SHROOMISH", "TREECKO"])
        state = self._state(party=["TREECKO"])
        battle = self._battle_info(species="Unknown")

        self.bot._opponent_species_from_dialogue = "ZIGZAGOON"  # stale cache

        self.bot._identify_opponent(visual, state, battle)
        self.assertEqual(self.bot._opponent_species_from_dialogue, "SHROOMISH")

    # ---------------------------------------------------------------
    # All-unknown
    # ---------------------------------------------------------------

    def test_all_sources_unknown(self):
        """No VLM, no dialogue, no RAM → returns Unknown."""
        visual = self._visual(entities=[])
        state = self._state()
        battle = self._battle_info(species="Unknown")

        self.bot._dialogue_history = []
        self.bot._opponent_species_from_dialogue = ""

        species, types = self.bot._identify_opponent(visual, state, battle)
        self.assertEqual(species, "Unknown")
        self.assertEqual(types, [])


if __name__ == "__main__":
    unittest.main()
