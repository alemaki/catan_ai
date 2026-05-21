import sys
sys.path.append('../') # literally the most retarded language ever
import unittest
import gymnasium
import catanatron.gym
from models.dqn import reward_function
from utils.utils import create_random_players_env
from catanatron import Color, Game

class TestRewardFunction(unittest.TestCase):

    def setUp(self):
        self.env: gymnasium.Env = create_random_players_env(reward_function)
        self.game: Game = self.env.unwrapped.game
        self.p0_color: Color = self.game.state.colors[0]
        reward_function.last_points = 0

    def test_first_call_gives_zero(self):
        self.assertEqual(reward_function(self.game, self.p0_color), 0)

    def test_first_point_gives_no_reward(self):
        reward_function(self.game, self.p0_color)  # initialise last_points
        self.game.state.player_state["P0_ACTUAL_VICTORY_POINTS"] = 1
        self.assertEqual(reward_function(self.game, self.p0_color), 0)

    def test_single_point_gain_rewards_10(self):
        self.game.state.player_state["P0_ACTUAL_VICTORY_POINTS"] = 1
        reward_function(self.game, self.p0_color)
        self.game.state.player_state["P0_ACTUAL_VICTORY_POINTS"] = 2
        self.assertEqual(reward_function(self.game, self.p0_color), 10)

    def test_multi_point_gain_scales_reward(self):
        self.game.state.player_state["P0_ACTUAL_VICTORY_POINTS"] = 1
        reward_function(self.game, self.p0_color)
        self.game.state.player_state["P0_ACTUAL_VICTORY_POINTS"] = 4
        self.assertEqual(reward_function(self.game, self.p0_color), 30)

    def test_no_negative_reward_for_vp_loss(self):
        """Reward should be floored at 0, never punish VP loss (e.g. longest road stolen)."""
        self.game.state.player_state["P0_ACTUAL_VICTORY_POINTS"] = 5
        reward_function(self.game, self.p0_color)
        self.game.state.player_state["P0_ACTUAL_VICTORY_POINTS"] = 3
        self.assertGreaterEqual(reward_function(self.game, self.p0_color), 0)

    def test_reward_for_won_game(self):
        self.game.state.player_state["P0_ACTUAL_VICTORY_POINTS"] = 10
        self.assertGreaterEqual(reward_function(self.game, self.p0_color), 100)

    def test_reward_for_lost_game(self):
        self.game.state.player_state["P1_ACTUAL_VICTORY_POINTS"] = 10
        self.assertLessEqual(reward_function(self.game, self.p0_color), -100)

if __name__ == "__main__":
    unittest.main()