import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import torch
from catanatron import Game
from catanatron.models.player import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.features import get_feature_ordering

from utils.model_player import ModelPlayer, ActionSelectableModel

EXPECTED_OBS_LEN = len(get_feature_ordering(num_players=2, map_type="BASE"))


class FirstValid(ActionSelectableModel):
    def select_action(self, observation, valid_actions, device="cpu"):
        return valid_actions[0]


class InvalidAction(ActionSelectableModel):
    def select_action(self, observation, valid_actions, device="cpu"):
        return 999999


def make_game(model):
    player = ModelPlayer(model, Color.BLUE)
    return Game(players=[player, WeightedRandomPlayer(Color.RED)]), player


# Kind of useless but what can you do
class TestModelPlayerProperties(unittest.TestCase):

    def test_is_bot_true_by_default(self):
        player = ModelPlayer(FirstValid(), Color.BLUE)
        self.assertTrue(player.is_bot)

    def test_is_bot_can_be_set_false(self):
        player = ModelPlayer(FirstValid(), Color.BLUE, is_bot=False)
        self.assertFalse(player.is_bot)

    def test_color_is_set(self):
        player = ModelPlayer(FirstValid(), Color.BLUE)
        self.assertEqual(player.color, Color.BLUE)


class TestModelPlayerObservation(unittest.TestCase):

    def setUp(self):
        self.game, self.player = make_game(FirstValid())

    # this might fail randomly, but not sure
    def test_length_matches_feature_ordering(self):
        self.assertEqual(self.player.get_observation(self.game).shape[0], EXPECTED_OBS_LEN)

    def test_values_are_finite(self):
        obs = self.player.get_observation(self.game)
        self.assertTrue(torch.all(torch.isfinite(obs)))


class TestModelPlayerDecide(unittest.TestCase):

    # peak testing
    def test_game_completes_without_error(self):
        game, _ = make_game(FirstValid())
        game.play()

    def test_decide_calls_select_action(self):
        chosen_actions = []

        class TrackingModel(ActionSelectableModel):
            def select_action(self, observation, valid_actions, device="cpu"):
                chosen = valid_actions[0]
                chosen_actions.append(chosen)
                return chosen

        game, player = make_game(TrackingModel())
        game.play()
        self.assertGreater(len(chosen_actions), 0)

    def test_invalid_model_raises_value_error(self):
        game, _ = make_game(InvalidAction())
        with self.assertRaises(ValueError):
            game.play()


if __name__ == "__main__":
    unittest.main()
