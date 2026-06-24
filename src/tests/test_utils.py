import unittest
import torch
import random
import gymnasium
import catanatron.gym
from catanatron.state_functions import player_key
from models.dqn import reward_function, reset_reward_function, valid_actions_to_mask, ReplayMemory, NStepBuffer, DQNTransition, DQN, optimize_model, BATCH_SIZE
from utils.utils import create_random_players_env, create_game_stats
from utils.constants import MAX_ACTION_COUNT, VP_REWARD, CITY_REWARD, ROAD_REWARD, WIN_REWARD
from catanatron import Color, Game

class TestRewardFunction(unittest.TestCase):

    def setUp(self):
        self.env: gymnasium.Env = create_random_players_env(reward_function)
        self.game: Game = self.env.unwrapped.game
        self.agent_color: Color = Color.BLUE
        self.key = player_key(self.game.state, self.agent_color)

    def test_first_call_gives_zero(self):
        self.assertEqual(reward_function(self.game, self.agent_color), 0)

    def test_first_point_gives_no_reward(self):
        reward_function(self.game, self.agent_color)  # initialise last_points
        self.game.state.player_state[f"{self.key}_ACTUAL_VICTORY_POINTS"] = 1
        self.assertEqual(reward_function(self.game, self.agent_color), 0)

    def test_single_point_gain_rewards_VP_REWARD(self):
        self.game.state.player_state[f"{self.key}_ACTUAL_VICTORY_POINTS"] = 1
        reward_function(self.game, self.agent_color)
        self.game.state.player_state[f"{self.key}_ACTUAL_VICTORY_POINTS"] = 2
        self.assertEqual(reward_function(self.game, self.agent_color), VP_REWARD)

    def test_multi_point_gain_scales_reward(self):
        self.game.state.player_state[f"{self.key}_ACTUAL_VICTORY_POINTS"] = 1
        reward_function(self.game, self.agent_color)
        self.game.state.player_state[f"{self.key}_ACTUAL_VICTORY_POINTS"] = 4
        self.assertEqual(reward_function(self.game, self.agent_color), VP_REWARD*3)

    def test_no_negative_reward_for_vp_loss(self):
        """Reward should be floored at 0, will not punish VP loss (e.g. longest road stolen)."""
        self.game.state.player_state[f"{self.key}_ACTUAL_VICTORY_POINTS"] = 5
        reward_function(self.game, self.agent_color)
        self.game.state.player_state[f"{self.key}_ACTUAL_VICTORY_POINTS"] = 3
        self.assertGreaterEqual(reward_function(self.game, self.agent_color), 0)

    def test_reward_for_won_game(self):
        self.game.state.player_state[f"{self.key}_ACTUAL_VICTORY_POINTS"] = 10
        self.assertGreaterEqual(reward_function(self.game, self.agent_color), WIN_REWARD)

    def test_reward_for_lost_game(self):
        self.game.state.player_state[f"{"P0" if self.key == "P1" else "P1"}_ACTUAL_VICTORY_POINTS"] = 10
        self.assertLessEqual(reward_function(self.game, self.agent_color), -WIN_REWARD)

    def test_reset_restarts_vp_tracking_from_one(self):
        self.game.state.player_state[f"{self.key}_ACTUAL_VICTORY_POINTS"] = 5
        reward_function(self.game, self.agent_color)
        reset_reward_function()
        # last_points is back to 1; 5 VPs now yields 4 * VP_REWARD
        reward = reward_function(self.game, self.agent_color)
        self.assertEqual(reward, VP_REWARD * 4)

class TestValidActionsToMask(unittest.TestCase):

    def test_valid_actions_get_zero(self):
        mask = valid_actions_to_mask([0, 5, 100])
        self.assertEqual(mask[0].item(), 0.0)
        self.assertEqual(mask[5].item(), 0.0)
        self.assertEqual(mask[100].item(), 0.0)

    def test_invalid_actions_get_large_negative(self):
        mask = valid_actions_to_mask([0])
        self.assertAlmostEqual(mask[1].item(), -1e9)

    def test_default_length_matches_max_action_count(self):
        mask = valid_actions_to_mask([0, 1])
        self.assertEqual(len(mask), MAX_ACTION_COUNT)

    def test_custom_action_dim(self):
        mask = valid_actions_to_mask([0, 2], action_dim=5)
        self.assertEqual(len(mask), 5)
        self.assertEqual(mask[0].item(), 0.0)
        self.assertAlmostEqual(mask[1].item(), -1e9)

    def test_all_valid_gives_all_zeros(self):
        mask = valid_actions_to_mask(list(range(MAX_ACTION_COUNT)))
        self.assertTrue(torch.all(mask == 0))

    def test_no_valid_gives_all_negative(self):
        mask = valid_actions_to_mask([])
        self.assertTrue(torch.all(mask == -1e9))

class TestReplayMemory(unittest.TestCase):

    OBS_SIZE = 10

    def setUp(self):
        self.memory = ReplayMemory(100)

    def _make_transition(self, reward=1.0, done=False):
        obs = [0.0] * self.OBS_SIZE
        return ReplayMemory.create_dqn_transition(obs, [0, 1, 2], 0, obs, [0, 1, 2], reward, done)

    def test_empty_memory_has_zero_length(self):
        self.assertEqual(len(self.memory), 0)

    def test_push_increases_length(self):
        self.memory.push(self._make_transition())
        self.assertEqual(len(self.memory), 1)

    def test_push_multiple_increases_length(self):
        for _ in range(5):
            self.memory.push(self._make_transition())
        self.assertEqual(len(self.memory), 5)

    def test_capacity_is_respected(self):
        mem = ReplayMemory(10)
        for _ in range(15):
            mem.push(self._make_transition())
        self.assertEqual(len(mem), 10)

    def test_sample_returns_correct_batch_size(self):
        for _ in range(20):
            self.memory.push(self._make_transition())
        batch = self.memory.sample(10)
        self.assertEqual(len(batch), 10)

    def test_sample_raises_when_not_enough(self):
        self.memory.push(self._make_transition())
        with self.assertRaises(ValueError):
            self.memory.sample(10)

    def test_create_transition_returns_tensors(self):
        t = self._make_transition()
        self.assertIsInstance(t.observation, torch.Tensor)
        self.assertIsInstance(t.action, torch.Tensor)
        self.assertIsInstance(t.reward, torch.Tensor)
        self.assertIsInstance(t.done, torch.Tensor)

    def test_create_transition_observation_shape(self):
        t = self._make_transition()
        self.assertEqual(t.observation.shape[0], self.OBS_SIZE)

    def test_create_transition_done_false_is_zero(self):
        t = self._make_transition(done=False)
        self.assertEqual(t.done.item(), 0.0)

    def test_create_transition_done_true_is_one(self):
        t = self._make_transition(done=True)
        self.assertEqual(t.done.item(), 1.0)

    def test_create_transition_reward_stored_correctly(self):
        t = self._make_transition(reward=42.0)
        self.assertAlmostEqual(t.reward.item(), 42.0)

    def test_mask_shape_in_transition(self):
        t = self._make_transition()
        self.assertEqual(t.valid_actions_mask.shape[0], MAX_ACTION_COUNT)
        self.assertEqual(t.next_valid_actions_mask.shape[0], MAX_ACTION_COUNT)

    def test_mask_values_in_transition(self):
        obs = [0.0] * self.OBS_SIZE
        valid = [0, 2, 5]
        t = ReplayMemory.create_dqn_transition(obs, valid, 0, obs, valid, 1.0, False)
        self.assertEqual(t.valid_actions_mask[0].item(), 0.0)
        self.assertEqual(t.valid_actions_mask[2].item(), 0.0)
        self.assertAlmostEqual(t.valid_actions_mask[1].item(), -1e9)
        self.assertAlmostEqual(t.valid_actions_mask[3].item(), -1e9)

class TestRewardFunctionRoads(unittest.TestCase):

    def setUp(self):
        self.env: gymnasium.Env = create_random_players_env(reward_function)
        self.game: Game = self.env.unwrapped.game
        self.agent_color: Color = Color.BLUE
        self.key = player_key(self.game.state, self.agent_color)
        reset_reward_function()

    def test_first_road_gives_no_reward(self):
        self.game.state.player_state[f"{self.key}_ROADS_AVAILABLE"] -= 1
        reward = reward_function(self.game, self.agent_color)
        self.assertEqual(reward, 0)

    def test_road_built_gives_reward(self):
        self.game.state.player_state[f"{self.key}_ROADS_AVAILABLE"] -= 2
        reward = reward_function(self.game, self.agent_color)
        self.assertEqual(reward, ROAD_REWARD)

    def test_road_reward_not_given_twice(self):
        self.game.state.player_state[f"{self.key}_ROADS_AVAILABLE"] -= 2
        reward_function(self.game, self.agent_color)
        second_reward = reward_function(self.game, self.agent_color)
        self.assertEqual(second_reward, 0)

    def test_two_roads_give_double_reward(self):
        self.game.state.player_state[f"{self.key}_ROADS_AVAILABLE"] -= 3
        reward = reward_function(self.game, self.agent_color)
        self.assertEqual(reward, ROAD_REWARD * 2)

    def test_vp_and_road_rewards_combine(self):
        self.game.state.player_state[f"{self.key}_ROADS_AVAILABLE"] -= 2
        self.game.state.player_state[f"{self.key}_ACTUAL_VICTORY_POINTS"] += 2
        reward = reward_function(self.game, self.agent_color)
        self.assertEqual(reward, ROAD_REWARD + VP_REWARD)

    def test_no_road_reward_when_roads_unchanged(self):
        reward = reward_function(self.game, self.agent_color)
        self.assertEqual(reward, 0)

    def test_reset_clears_road_tracking(self):
        self.game.state.player_state[f"{self.key}_ROADS_AVAILABLE"] -= 2
        reward_function(self.game, self.agent_color)
        # Without reset, same state gives 0
        self.assertEqual(reward_function(self.game, self.agent_color), 0)
        # After reset, same state gives non-zero (tracking restarted from 1)
        reset_reward_function()
        self.assertGreater(reward_function(self.game, self.agent_color), 0)

class TestCreateGameStats(unittest.TestCase):

    def setUp(self):
        env = create_random_players_env()
        env.reset()
        self.game = env.unwrapped.game

    def test_returns_all_expected_keys(self):
        stats = create_game_stats(self.game)
        expected = {
            "game_turn", "finished", "mp_won",
            "mp_public_vps", "mp_actual_vps",
            "mp_cities", "mp_settlements", "mp_roads",
            "mp_has_road", "mp_has_army",
            "mp_total_resources", "mp_dev_cards_in_hand", "mp_dev_cards_played",
        }
        self.assertEqual(set(stats.keys()), expected)

    def test_game_not_finished_at_start(self):
        self.assertFalse(create_game_stats(self.game)["finished"])

    def test_mp_not_won_at_start(self):
        self.assertFalse(create_game_stats(self.game)["mp_won"])

    def test_game_turn_is_non_negative_int(self):
        stats = create_game_stats(self.game)
        self.assertIsInstance(stats["game_turn"], int)
        self.assertGreaterEqual(stats["game_turn"], 0)

    def test_vp_counts_are_non_negative(self):
        stats = create_game_stats(self.game)
        self.assertGreaterEqual(stats["mp_actual_vps"], 0)
        self.assertGreaterEqual(stats["mp_public_vps"], 0)

    def test_road_and_building_counts_non_negative(self):
        stats = create_game_stats(self.game)
        self.assertGreaterEqual(stats["mp_roads"], 0)
        self.assertGreaterEqual(stats["mp_settlements"], 0)
        self.assertGreaterEqual(stats["mp_cities"], 0)

    def test_resource_and_card_counts_non_negative(self):
        stats = create_game_stats(self.game)
        self.assertGreaterEqual(stats["mp_total_resources"], 0)
        self.assertGreaterEqual(stats["mp_dev_cards_in_hand"], 0)
        self.assertGreaterEqual(stats["mp_dev_cards_played"], 0)

    def test_road_and_army_flags_are_bool(self):
        stats = create_game_stats(self.game)
        self.assertIsInstance(stats["mp_has_road"], bool)
        self.assertIsInstance(stats["mp_has_army"], bool)

    def test_mp_roads_counts_built_roads(self):
        key = player_key(self.game.state, Color.BLUE)
        before = create_game_stats(self.game)["mp_roads"]
        self.game.state.player_state[f"{key}_ROADS_AVAILABLE"] -= 3
        self.assertEqual(create_game_stats(self.game)["mp_roads"], before + 3)

    def test_mp_actual_vps_reflects_player_state(self):
        key = player_key(self.game.state, Color.BLUE)
        self.game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] = 7
        self.assertEqual(create_game_stats(self.game)["mp_actual_vps"], 7)

    def test_finished_and_mp_won_when_main_player_reaches_10_vps(self):
        key = player_key(self.game.state, Color.BLUE)
        self.game.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] = 10
        stats = create_game_stats(self.game)
        self.assertTrue(stats["finished"])
        self.assertTrue(stats["mp_won"])

    def test_finished_true_mp_won_false_when_enemy_wins(self):
        enemy_key = player_key(self.game.state, Color.RED)
        self.game.state.player_state[f"{enemy_key}_ACTUAL_VICTORY_POINTS"] = 10
        stats = create_game_stats(self.game)
        self.assertTrue(stats["finished"])
        self.assertFalse(stats["mp_won"])

class TestCreateGameStatsPlayerIdentity(unittest.TestCase):
    """Verify create_game_stats always reads from the requested player, never a wrong one."""

    def setUp(self):
        env = create_random_players_env()
        env.reset()
        self.game = env.unwrapped.game
        self.blue_key = player_key(self.game.state, Color.BLUE)
        self.enemy_color = next(c for c in self.game.state.colors if c != Color.BLUE)
        self.enemy_key = player_key(self.game.state, self.enemy_color)

    def test_default_reads_blue_vps_not_enemy(self):
        self.game.state.player_state[f"{self.enemy_key}_ACTUAL_VICTORY_POINTS"] = 9
        stats = create_game_stats(self.game)
        self.assertNotEqual(stats["mp_actual_vps"], 9)

    def test_blue_vp_change_reflected_by_default(self):
        self.game.state.player_state[f"{self.blue_key}_ACTUAL_VICTORY_POINTS"] = 7
        self.assertEqual(create_game_stats(self.game)["mp_actual_vps"], 7)

    def test_enemy_road_change_not_reflected_in_blue_stats(self):
        before = create_game_stats(self.game)["mp_roads"]
        self.game.state.player_state[f"{self.enemy_key}_ROADS_AVAILABLE"] -= 5
        self.assertEqual(create_game_stats(self.game)["mp_roads"], before)

    def test_blue_road_change_reflected_by_default(self):
        before = create_game_stats(self.game)["mp_roads"]
        self.game.state.player_state[f"{self.blue_key}_ROADS_AVAILABLE"] -= 3
        self.assertEqual(create_game_stats(self.game)["mp_roads"], before + 3)

    def test_enemy_settlement_change_not_reflected_in_blue_stats(self):
        before = create_game_stats(self.game)["mp_settlements"]
        self.game.state.player_state[f"{self.enemy_key}_SETTLEMENTS_AVAILABLE"] -= 2
        self.assertEqual(create_game_stats(self.game)["mp_settlements"], before)

    def test_blue_settlement_change_reflected_by_default(self):
        before = create_game_stats(self.game)["mp_settlements"]
        self.game.state.player_state[f"{self.blue_key}_SETTLEMENTS_AVAILABLE"] -= 1
        self.assertEqual(create_game_stats(self.game)["mp_settlements"], before + 1)

    def test_enemy_city_change_not_reflected_in_blue_stats(self):
        before = create_game_stats(self.game)["mp_cities"]
        self.game.state.player_state[f"{self.enemy_key}_CITIES_AVAILABLE"] -= 2
        self.assertEqual(create_game_stats(self.game)["mp_cities"], before)

    def test_blue_city_change_reflected_by_default(self):
        before = create_game_stats(self.game)["mp_cities"]
        self.game.state.player_state[f"{self.blue_key}_CITIES_AVAILABLE"] -= 1
        self.assertEqual(create_game_stats(self.game)["mp_cities"], before + 1)

    def test_enemy_resource_change_not_reflected_in_blue_stats(self):
        from catanatron import RESOURCES
        before = create_game_stats(self.game)["mp_total_resources"]
        for resource in RESOURCES:
            self.game.state.player_state[f"{self.enemy_key}_{resource}_IN_HAND"] = 3
        self.assertEqual(create_game_stats(self.game)["mp_total_resources"], before)

    def test_blue_resource_change_reflected_by_default(self):
        from catanatron import RESOURCES
        for resource in RESOURCES:
            self.game.state.player_state[f"{self.blue_key}_{resource}_IN_HAND"] = 0
        self.game.state.player_state[f"{self.blue_key}_{list(RESOURCES)[0]}_IN_HAND"] = 4
        self.assertEqual(create_game_stats(self.game)["mp_total_resources"], 4)

    def test_mp_won_false_when_only_enemy_reaches_10_vps(self):
        self.game.state.player_state[f"{self.enemy_key}_ACTUAL_VICTORY_POINTS"] = 10
        stats = create_game_stats(self.game)
        self.assertTrue(stats["finished"])
        self.assertFalse(stats["mp_won"])

    def test_mp_won_true_when_blue_reaches_10_vps(self):
        self.game.state.player_state[f"{self.blue_key}_ACTUAL_VICTORY_POINTS"] = 10
        stats = create_game_stats(self.game)
        self.assertTrue(stats["finished"])
        self.assertTrue(stats["mp_won"])

    def test_explicit_blue_matches_default(self):
        stats_default = create_game_stats(self.game)
        stats_explicit = create_game_stats(self.game, Color.BLUE)
        self.assertEqual(stats_default, stats_explicit)

    def test_explicit_enemy_color_reads_enemy_vps(self):
        self.game.state.player_state[f"{self.enemy_key}_ACTUAL_VICTORY_POINTS"] = 5
        stats = create_game_stats(self.game, agent_color=self.enemy_color)
        self.assertEqual(stats["mp_actual_vps"], 5)

    def test_explicit_enemy_color_does_not_return_blue_vps(self):
        self.game.state.player_state[f"{self.blue_key}_ACTUAL_VICTORY_POINTS"] = 8
        self.game.state.player_state[f"{self.enemy_key}_ACTUAL_VICTORY_POINTS"] = 3
        stats = create_game_stats(self.game, agent_color=self.enemy_color)
        self.assertNotEqual(stats["mp_actual_vps"], 8)
        self.assertEqual(stats["mp_actual_vps"], 3)

    def test_blue_and_enemy_stats_differ_when_state_differs(self):
        self.game.state.player_state[f"{self.blue_key}_ACTUAL_VICTORY_POINTS"] = 8
        self.game.state.player_state[f"{self.enemy_key}_ACTUAL_VICTORY_POINTS"] = 3
        blue_stats = create_game_stats(self.game, Color.BLUE)
        enemy_stats = create_game_stats(self.game, self.enemy_color)
        self.assertNotEqual(blue_stats["mp_actual_vps"], enemy_stats["mp_actual_vps"])

    def test_explicit_enemy_road_change_reflected(self):
        before = create_game_stats(self.game, agent_color=self.enemy_color)["mp_roads"]
        self.game.state.player_state[f"{self.enemy_key}_ROADS_AVAILABLE"] -= 2
        after = create_game_stats(self.game, agent_color=self.enemy_color)["mp_roads"]
        self.assertEqual(after, before + 2)

    def test_explicit_enemy_road_not_reflected_in_blue(self):
        before = create_game_stats(self.game)["mp_roads"]
        self.game.state.player_state[f"{self.enemy_key}_ROADS_AVAILABLE"] -= 2
        self.assertEqual(create_game_stats(self.game)["mp_roads"], before)
