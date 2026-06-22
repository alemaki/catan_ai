import unittest
import torch
import random
import gymnasium
import catanatron.gym
from catanatron.state_functions import player_key
from models.dqn import ReplayMemory, NStepBuffer, DQNTransition, DQN, optimize_model, BATCH_SIZE
from utils.utils import *
from utils.constants import MAX_ACTION_COUNT, VP_REWARD, CITY_REWARD, ROAD_REWARD, WIN_REWARD
from catanatron import Color, Game

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
            t = ReplayMemory.create_dqn_transition(
                obs, valid, action, obs, valid,
                float(random.random()), random.random() > 0.9
            )
            self.memory.push(t)

    def test_returns_zero_when_memory_too_small(self):
        loss, mean_max_q = optimize_model(self.optimizer, self.policy_net, self.target_net, self.memory)
        self.assertEqual(loss, 0)
        self.assertEqual(mean_max_q, 0.0)

    def test_returns_float_loss_with_enough_samples(self):
        self._fill_memory(BATCH_SIZE + 10)
        loss, _ = optimize_model(self.optimizer, self.policy_net, self.target_net, self.memory)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)

    def test_returns_float_mean_max_q_with_enough_samples(self):
        self._fill_memory(BATCH_SIZE + 10)
        _, mean_max_q = optimize_model(self.optimizer, self.policy_net, self.target_net, self.memory)
        self.assertIsInstance(mean_max_q, float)

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

    def test_loss_is_non_negative(self):
        self._fill_memory(BATCH_SIZE + 10)
        loss, _ = optimize_model(self.optimizer, self.policy_net, self.target_net, self.memory)
        self.assertGreaterEqual(loss, 0.0)

    def test_exactly_batch_size_samples_triggers_optimization(self):
        self._fill_memory(BATCH_SIZE)
        loss, _ = optimize_model(self.optimizer, self.policy_net, self.target_net, self.memory)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)

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
            obs = obs
            reset_reward_function()
            prev_obs, prev_info = obs, info
            done = False
            while not done:
                action = policy_net.select_action(prev_obs, info["valid_actions"], epsilon=1.0)
                obs, reward, terminated, truncated, info = env.step(action)
                obs = obs
                done = terminated or truncated
                t = ReplayMemory.create_dqn_transition(
                    prev_obs,
                    prev_info["valid_actions"],
                    action,
                    obs,
                    info["valid_actions"],
                    reward,
                    done
                )
                memory.push(t)
                prev_obs, prev_info = obs, info
                optimize_model(optimizer, policy_net, target_net, memory)

        env.close()


class TestNStepBuffer(unittest.TestCase):
    N = 3
    GAMMA = 0.9
    OBS_SIZE = 4

    def setUp(self):
        self.buf = NStepBuffer(n=self.N, gamma=self.GAMMA)

    def _step(self, reward=0.0, done=False, obs_id=0):
        obs = [float(obs_id)] * self.OBS_SIZE
        next_obs = [float(obs_id + 1)] * self.OBS_SIZE
        return (obs, [0, 1], 0, reward, next_obs, [0, 1], done)

    def _fill(self, n=None):
        count = n if n is not None else self.N
        for i in range(count):
            self.buf.append(*self._step(reward=1.0, obs_id=i))

    def test_not_ready_when_empty(self):
        self.assertFalse(self.buf.ready())

    def test_not_ready_below_n(self):
        self._fill(self.N - 1)
        self.assertFalse(self.buf.ready())

    def test_ready_at_n(self):
        self._fill()
        self.assertTrue(self.buf.ready())

    def test_ready_above_n(self):
        self._fill()
        self.buf.append(*self._step())
        self.assertTrue(self.buf.ready())

    def test_pop_returns_transition(self):
        self._fill()
        self.assertIsInstance(self.buf.pop(), DQNTransition)

    def test_pop_decrements_buffer(self):
        self._fill()
        self.buf.pop()
        self.assertEqual(len(self.buf.buffer), self.N - 1)

    def test_pop_accumulated_return(self):
        for r in [1.0, 2.0, 3.0]:
            self.buf.append(*self._step(reward=r))
        expected_R = 1.0 + 0.9 * 2.0 + 0.9**2 * 3.0
        self.assertAlmostEqual(self.buf.pop().reward.item(), expected_R, places=5)

    def test_pop_uses_first_obs(self):
        self._fill()
        self.assertAlmostEqual(self.buf.pop().observation[0].item(), 0.0)

    def test_pop_uses_last_next_obs(self):
        self._fill()
        self.assertAlmostEqual(self.buf.pop().next_observation[0].item(), float(self.N))

    def test_pop_done_false_for_non_terminal(self):
        self._fill()
        self.assertEqual(self.buf.pop().done.item(), 0.0)

    def test_done_cuts_return_at_terminal(self):
        self.buf.append(*self._step(reward=1.0, done=False))
        self.buf.append(*self._step(reward=10.0, done=True))
        self.buf.append(*self._step(reward=99.0, done=False))
        expected_R = 1.0 + 0.9 * 10.0
        self.assertAlmostEqual(self.buf.pop().reward.item(), expected_R, places=5)

    def test_done_sets_done_true_in_transition(self):
        self.buf.append(*self._step(done=False))
        self.buf.append(*self._step(done=True))
        self.buf.append(*self._step(done=False))
        self.assertEqual(self.buf.pop().done.item(), 1.0)

    def test_done_uses_terminal_as_next_obs(self):
        self.buf.append(*self._step(obs_id=0, done=False))
        self.buf.append(*self._step(obs_id=1, done=True))
        self.buf.append(*self._step(obs_id=2, done=False))
        self.assertAlmostEqual(self.buf.pop().next_observation[0].item(), 2.0)

    def test_flush_empty_returns_empty(self):
        self.assertEqual(self.buf.flush(), [])

    def test_flush_returns_all_remaining(self):
        self._fill(2)
        self.assertEqual(len(self.buf.flush()), 2)

    def test_flush_clears_buffer(self):
        self._fill(2)
        self.buf.flush()
        self.assertFalse(self.buf.ready())
        self.assertEqual(len(self.buf.buffer), 0)

    def test_flush_each_return_accumulated_from_position(self):
        self.buf.append(*self._step(reward=1.0, done=False))
        self.buf.append(*self._step(reward=2.0, done=False))
        self.buf.append(*self._step(reward=3.0, done=True))
        ts = self.buf.flush()
        self.assertEqual(len(ts), 3)
        self.assertAlmostEqual(ts[0].reward.item(), 1.0 + 0.9*2.0 + 0.9**2*3.0, places=5)
        self.assertAlmostEqual(ts[1].reward.item(), 2.0 + 0.9*3.0, places=5)
        self.assertAlmostEqual(ts[2].reward.item(), 3.0, places=5)

    def test_flush_win_reward_included_in_all_transitions(self):
        # win reward at terminal must propagate into preceding transitions
        self.buf.append(*self._step(reward=0.0, done=False))
        self.buf.append(*self._step(reward=200.0, done=True))
        ts = self.buf.flush()
        self.assertAlmostEqual(ts[0].reward.item(), 0.9 * 200.0, places=4)
        self.assertAlmostEqual(ts[1].reward.item(), 200.0, places=4)

    def test_flush_all_done_true_when_terminal_in_window(self):
        self.buf.append(*self._step(done=False))
        self.buf.append(*self._step(done=True))
        ts = self.buf.flush()
        for t in ts:
            self.assertEqual(t.done.item(), 1.0)

    def test_clear_empties_buffer(self):
        self._fill()
        self.buf.clear()
        self.assertFalse(self.buf.ready())
        self.assertEqual(len(self.buf.buffer), 0)

    def test_n_equals_one_acts_as_single_step(self):
        buf = NStepBuffer(n=1, gamma=self.GAMMA)
        buf.append(*self._step(reward=5.0, done=False))
        self.assertTrue(buf.ready())
        self.assertAlmostEqual(buf.pop().reward.item(), 5.0, places=5)

    def test_single_step_flush(self):
        self.buf.append(*self._step(reward=7.0, done=True))
        ts = self.buf.flush()
        self.assertEqual(len(ts), 1)
        self.assertAlmostEqual(ts[0].reward.item(), 7.0, places=5)


if __name__ == "__main__":
    unittest.main()
