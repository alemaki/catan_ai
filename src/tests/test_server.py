import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import unittest
from unittest.mock import patch, MagicMock

from catanatron.models.player import Color, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer

from utils.model_player import ModelPlayer
from utils.player_constants import model_players
from ui.server import app, extended_player_factory


class TestGetModelsEndpoint(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_returns_200(self):
        self.assertEqual(self.client.get("/api/models").status_code, 200)

    def test_contains_all_registered_models(self):
        data = json.loads(self.client.get("/api/models").data)
        self.assertEqual(set(data), set(model_players.keys()))

    def test_no_duplicates(self):
        data = json.loads(self.client.get("/api/models").data)
        self.assertEqual(len(data), len(set(data)))


class TestExtendedPlayerFactory(unittest.TestCase):

    def _key(self, type_str, color=Color.BLUE):
        return (type_str, color)

    def test_human_is_value_function_player_not_bot(self):
        player = extended_player_factory(self._key("HUMAN"))
        self.assertIsInstance(player, ValueFunctionPlayer)
        self.assertFalse(player.is_bot)

    def test_random_is_random_player(self):
        self.assertIsInstance(extended_player_factory(self._key("RANDOM")), RandomPlayer)

    def test_catanatron_is_alpha_beta_player(self):
        self.assertIsInstance(extended_player_factory(self._key("CATANATRON")), AlphaBetaPlayer)

    def test_unknown_type_raises_value_error(self):
        with self.assertRaises(ValueError):
            extended_player_factory(self._key("UNKNOWN_BOT"))


class TestPostGameEndpoint(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def _create_game(self, players):
        return self.client.post(
            "/api/games",
            data=json.dumps({"players": players}),
            content_type="application/json",
        )

    def test_create_game_returns_200(self):
        self.assertEqual(self._create_game(["HUMAN", "RANDOM"]).status_code, 200)

    def test_create_game_returns_game_id(self):
        data = json.loads(self._create_game(["HUMAN", "RANDOM"]).data)
        self.assertIn("game_id", data)
        self.assertIsInstance(data["game_id"], str)

    def test_create_two_random_game(self):
        self.assertEqual(self._create_game(["RANDOM", "RANDOM"]).status_code, 200)

    def test_missing_players_returns_400(self):
        resp = self.client.post("/api/games", data=json.dumps({}), content_type="application/json")
        self.assertEqual(resp.status_code, 400)

    def test_empty_body_returns_400(self):
        resp = self.client.post("/api/games")
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
