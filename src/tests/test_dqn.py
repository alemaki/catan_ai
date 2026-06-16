import unittest
import torch
import random
import gymnasium
import catanatron.gym
from catanatron.state_functions import player_key
from models.dqn import reward_function, reset_reward_function, valid_actions_to_mask, ReplayMemory, DQN, optimize_model, BATCH_SIZE
from utils.utils import create_random_players_env, create_game_stats
from utils.constants import MAX_ACTION_COUNT, VP_REWARD, CITY_REWARD, ROAD_REWARD, WIN_REWARD
from catanatron import Color, Game


class TestRewardFunction(unittest.TestCase):

    def setUp(self):
        self.env: gymnasium.Env = create_random_players_env(reward_function)
        self.game: Game = self.env.unwrapped.game
        self.agent_color: Color = self.game.state.colors[0]
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
        self.game.state.player_state["P1_ACTUAL_VICTORY_POINTS"] = 10
        self.assertLessEqual(reward_function(self.game, self.agent_color), -WIN_REWARD)

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
        return ReplayMemory.create_transition(obs, [0, 1, 2], 0, obs, [0, 1, 2], reward, done)

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

class TestDQN(unittest.TestCase):

    OBSERVATION_SIZE = 20
    ACTION_SIZE = 10

    def setUp(self):
        self.net = DQN(self.OBSERVATION_SIZE, self.ACTION_SIZE)

    def test_forward_single_output_shape(self):
        obs = torch.zeros(1, self.OBSERVATION_SIZE)
        out = self.net(obs)
        self.assertEqual(out.shape, (1, self.ACTION_SIZE))

    def test_forward_batch_output_shape(self):
        obs = torch.zeros(32, self.OBSERVATION_SIZE)
        out = self.net(obs)
        self.assertEqual(out.shape, (32, self.ACTION_SIZE))

    def test_select_action_greedy_returns_valid(self):
        valid = [2, 5, 7]
        obs = [0.0] * self.OBSERVATION_SIZE
        action = self.net.select_action(obs, valid, epsilon=0.0)
        self.assertIn(action, valid)

    def test_select_action_random_returns_valid(self):
        valid = [1, 3, 9]
        obs = [0.0] * self.OBSERVATION_SIZE
        for _ in range(30):
            action = self.net.select_action(obs, valid, epsilon=1.0)
            self.assertIn(action, valid)

    def test_masking_blocks_invalid_action(self):
        # Zero advantage weights, bias strongly prefers action 1 (invalid)
        with torch.no_grad():
            self.net.advantage_stream.weight.zero_()
            self.net.advantage_stream.bias.zero_()
            self.net.advantage_stream.bias[1] = 1000.0
        obs = [0.0] * self.OBSERVATION_SIZE
        action = self.net.select_action(obs, valid_actions=[0], epsilon=0.0)
        self.assertEqual(action, 0)

    def test_select_action_single_valid_always_returned(self):
        obs = [0.0] * self.OBSERVATION_SIZE
        for epsilon in [0.0, 0.5, 1.0]:
            action = self.net.select_action(obs, [4], epsilon=epsilon)
            self.assertEqual(action, 4)

    def test_select_action_greedy_is_deterministic(self):
        obs = [0.5] * self.OBSERVATION_SIZE
        valid = [0, 1, 2, 3]
        first = self.net.select_action(obs, valid, epsilon=0.0)
        for _ in range(10):
            self.assertEqual(self.net.select_action(obs, valid, epsilon=0.0), first)

    def test_dueling_uniform_advantage_gives_equal_q_values(self):
        # With zeroed advantage stream all Q(s,a) = V(s) + 0 - mean(0) = V(s)
        with torch.no_grad():
            self.net.advantage_stream.weight.zero_()
            self.net.advantage_stream.bias.zero_()
        obs = torch.zeros(1, self.OBSERVATION_SIZE)
        q = self.net(obs)
        self.assertTrue(torch.allclose(q, q[0, 0].expand_as(q), atol=1e-5))

class TestOptimizeModel(unittest.TestCase):
    # Use MAX_ACTION_COUNT so the mask shape matches the network output
    OBS_SIZE = 20

    def setUp(self):
        self.policy_net = DQN(self.OBS_SIZE, MAX_ACTION_COUNT)
        self.target_net = DQN(self.OBS_SIZE, MAX_ACTION_COUNT)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=1e-3)
        self.memory = ReplayMemory(1000)

    def _fill_memory(self, n):
        obs = [0.0] * self.OBS_SIZE
        valid = list(range(MAX_ACTION_COUNT))
        for _ in range(n):
            action = random.randint(0, MAX_ACTION_COUNT - 1)
            t = ReplayMemory.create_transition(
                obs, valid, action, obs, valid,
                float(random.random()), random.random() > 0.9
            )
            self.memory.push(t)

    def test_returns_zero_when_memory_too_small(self):
        loss = optimize_model(self.optimizer, self.policy_net, self.target_net, self.memory)
        self.assertEqual(loss, 0)

    def test_returns_float_loss_with_enough_samples(self):
        self._fill_memory(BATCH_SIZE + 10)
        loss = optimize_model(self.optimizer, self.policy_net, self.target_net, self.memory)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)

    def test_policy_parameters_change_after_step(self):
        self._fill_memory(BATCH_SIZE + 10)
        params_before = [p.clone() for p in self.policy_net.parameters()]
        optimize_model(self.optimizer, self.policy_net, self.target_net, self.memory)
        changed = any(
            not torch.equal(before, after)
            for before, after in zip(params_before, self.policy_net.parameters())
        )
        self.assertTrue(changed)

    def test_target_parameters_unchanged_after_step(self):
        self._fill_memory(BATCH_SIZE + 10)
        target_before = [p.clone() for p in self.target_net.parameters()]
        optimize_model(self.optimizer, self.policy_net, self.target_net, self.memory)
        for before, after in zip(target_before, self.target_net.parameters()):
            self.assertTrue(torch.equal(before, after))

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

class TestTrainingSmoke(unittest.TestCase):

    def test_three_episodes_complete_without_error(self):
        env = create_random_players_env(reward_function)
        observation, _ = env.reset()
        obs_size = observation.shape[0]

        policy_net = DQN(obs_size, MAX_ACTION_COUNT)
        target_net = DQN(obs_size, MAX_ACTION_COUNT)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        memory = ReplayMemory(500)
        optimizer = torch.optim.AdamW(policy_net.parameters(), lr=1e-3)

        for _ in range(3):
            obs, info = env.reset()
            obs = obs / 50
            reward_function.last_points = 0
            reward_function.last_roads = 0
            prev_obs, prev_info = obs, info
            done = False
            while not done:
                action = policy_net.select_action(prev_obs, info["valid_actions"], epsilon=1.0)
                obs, reward, terminated, truncated, info = env.step(action)
                obs = obs / 50
                done = terminated or truncated
                t = ReplayMemory.create_transition(
                    prev_obs, prev_info["valid_actions"],
                    action,
                    obs, info["valid_actions"],
                    reward, done
                )
                memory.push(t)
                prev_obs, prev_info = obs, info
                optimize_model(optimizer, policy_net, target_net, memory)

        env.close()


if __name__ == "__main__":
    unittest.main()
