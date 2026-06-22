import unittest
import torch
from models.reinforce import REINFORCEAgent, compute_returns, reinforce_update, GAMMA
from utils.utils import ReplayMemory, REINFORCEState, create_random_players_env, reset_reward_function
from utils.constants import MAX_ACTION_COUNT

OBS_SIZE = 20
ACTION_SIZE = MAX_ACTION_COUNT


def _make_memory(rewards, dones=None, obs_size=OBS_SIZE):
    if dones is None:
        dones = [False] * (len(rewards) - 1) + [True]
    mem = ReplayMemory(len(rewards) + 1)
    valid = list(range(ACTION_SIZE))
    for r, d in zip(rewards, dones):
        mem.push(ReplayMemory.create_reinforce_state(
            [0.0] * obs_size, valid, 0, float(r), bool(d)
        ))
    return mem


class TestReplayMemoryREINFORCEState(unittest.TestCase):

    OBS_SIZE = 10

    def _make(self, reward=1.0, done=False, action=2):
        obs = [0.0] * self.OBS_SIZE
        return ReplayMemory.create_reinforce_state(obs, [0, 1, 2], action, reward, done)

    def test_returns_reinforce_state_namedtuple(self):
        self.assertIsInstance(self._make(), REINFORCEState)

    def test_observation_is_float32_tensor(self):
        s = self._make()
        self.assertIsInstance(s.observation, torch.Tensor)
        self.assertEqual(s.observation.dtype, torch.float32)

    def test_action_is_long_tensor(self):
        self.assertEqual(self._make().action.dtype, torch.long)

    def test_reward_stored_correctly(self):
        self.assertAlmostEqual(self._make(reward=5.5).reward.item(), 5.5)

    def test_done_false_stored_as_zero(self):
        self.assertEqual(self._make(done=False).done.item(), 0.0)

    def test_done_true_stored_as_one(self):
        self.assertEqual(self._make(done=True).done.item(), 1.0)

    def test_mask_valid_actions_are_zero(self):
        obs = [0.0] * self.OBS_SIZE
        s = ReplayMemory.create_reinforce_state(obs, [0, 3, 5], 0, 0.0, False)
        for idx in [0, 3, 5]:
            self.assertEqual(s.valid_actions_mask[idx].item(), 0.0)

    def test_mask_invalid_actions_are_large_negative(self):
        obs = [0.0] * self.OBS_SIZE
        s = ReplayMemory.create_reinforce_state(obs, [0], 0, 0.0, False)
        self.assertAlmostEqual(s.valid_actions_mask[1].item(), -1e9)

    def test_mask_shape_matches_max_action_count(self):
        self.assertEqual(self._make().valid_actions_mask.shape[0], MAX_ACTION_COUNT)

    def test_observation_shape(self):
        self.assertEqual(self._make().observation.shape[0], self.OBS_SIZE)

    def test_no_value_or_log_prob_fields(self):
        s = self._make()
        self.assertFalse(hasattr(s, "value"))
        self.assertFalse(hasattr(s, "log_prob"))


class TestComputeReturns(unittest.TestCase):

    def test_single_step_return_equals_reward(self):
        """G_0 = r_0 (only one step, gamma^0 * r_0)."""
        returns = compute_returns([7.0])
        self.assertAlmostEqual(returns[0].item(), 7.0, places=5)

    def test_two_step_returns(self):
        """G_0 = r_0 + gamma*r_1;  G_1 = r_1."""
        r0, r1 = 1.0, 2.0
        returns = compute_returns([r0, r1])
        self.assertAlmostEqual(returns[1].item(), r1, places=5)
        self.assertAlmostEqual(returns[0].item(), r0 + GAMMA * r1, places=5)

    def test_three_step_gamma_n_formula(self):
        """G_t = sum_{k=0}^{T-t-1} gamma^k * r_{t+k}"""
        r = [1.0, 2.0, 3.0]
        returns = compute_returns(r)
        expected_G0 = r[0] + GAMMA * r[1] + GAMMA**2 * r[2]
        expected_G1 = r[1] + GAMMA * r[2]
        expected_G2 = r[2]
        self.assertAlmostEqual(returns[0].item(), expected_G0, places=4)
        self.assertAlmostEqual(returns[1].item(), expected_G1, places=4)
        self.assertAlmostEqual(returns[2].item(), expected_G2, places=4)

    def test_later_steps_have_smaller_returns(self):
        """With positive rewards, G_0 > G_1 > ... > G_T-1."""
        returns = compute_returns([1.0, 1.0, 1.0, 1.0])
        for i in range(len(returns) - 1):
            self.assertGreater(returns[i].item(), returns[i + 1].item())

    def test_output_length_matches_input(self):
        self.assertEqual(len(compute_returns([1.0, 2.0, 3.0])), 3)

    def test_returns_are_finite(self):
        returns = compute_returns([1.0, -0.5, 2.0, 0.0])
        self.assertTrue(torch.all(torch.isfinite(returns)))

    def test_zero_reward_trajectory_gives_zero_returns(self):
        returns = compute_returns([0.0, 0.0, 0.0])
        self.assertTrue(torch.all(returns == 0.0))

    def test_single_large_terminal_reward_propagates_back(self):
        """Win reward at the last step should discount back to earlier steps."""
        rewards = [0.0, 0.0, 100.0]
        returns = compute_returns(rewards)
        self.assertAlmostEqual(returns[2].item(), 100.0, places=4)
        self.assertAlmostEqual(returns[1].item(), GAMMA * 100.0, places=4)
        self.assertAlmostEqual(returns[0].item(), GAMMA**2 * 100.0, places=4)


class TestREINFORCEAgent(unittest.TestCase):

    def setUp(self):
        self.agent = REINFORCEAgent(OBS_SIZE, ACTION_SIZE)

    def test_forward_output_shape_single(self):
        self.assertEqual(self.agent(torch.zeros(1, OBS_SIZE)).shape, (1, ACTION_SIZE))

    def test_forward_output_shape_batch(self):
        self.assertEqual(self.agent(torch.zeros(8, OBS_SIZE)).shape, (8, ACTION_SIZE))

    def test_forward_is_finite(self):
        out = self.agent(torch.randn(4, OBS_SIZE))
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_select_action_returns_valid_action(self):
        valid = [0, 5, 42, 100]
        action = self.agent.select_action([0.0] * OBS_SIZE, valid)
        self.assertIn(action, valid)

    def test_select_action_always_valid_over_many_calls(self):
        valid = [1, 7, 99]
        for _ in range(30):
            self.assertIn(self.agent.select_action([0.0] * OBS_SIZE, valid), valid)

    def test_select_action_mask_blocks_invalid_actions(self):
        with torch.no_grad():
            self.agent.policy_head.weight.zero_()
            self.agent.policy_head.bias.zero_()
            self.agent.policy_head.bias[5] = 1000.0
        for _ in range(20):
            self.assertEqual(self.agent.select_action([0.0] * OBS_SIZE, [0]), 0)

    def test_select_action_single_valid_always_returned(self):
        for _ in range(20):
            self.assertEqual(self.agent.select_action([0.0] * OBS_SIZE, [42]), 42)

    def test_select_action_returns_int(self):
        action = self.agent.select_action([0.0] * OBS_SIZE, [0, 1, 2])
        self.assertIsInstance(action, int)


class TestREINFORCEUpdate(unittest.TestCase):

    def setUp(self):
        self.agent = REINFORCEAgent(OBS_SIZE, ACTION_SIZE)
        self.optimizer = torch.optim.AdamW(self.agent.parameters(), lr=1e-3)
        self.memory = _make_memory(rewards=[1.0] * 10)

    def test_returns_float(self):
        self.assertIsInstance(reinforce_update(self.agent, self.optimizer, self.memory), float)

    def test_parameters_change_after_update(self):
        params_before = [p.clone() for p in self.agent.parameters()]
        reinforce_update(self.agent, self.optimizer, self.memory)
        changed = any(not torch.equal(b, a) for b, a in zip(params_before, self.agent.parameters()))
        self.assertTrue(changed)

    def test_loss_is_finite(self):
        loss = reinforce_update(self.agent, self.optimizer, self.memory)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))

    def test_multiple_updates_do_not_error(self):
        for _ in range(3):
            reinforce_update(self.agent, self.optimizer, self.memory)

    def test_sparse_reward_episode(self):
        """All-zero rewards except terminal win — still produces a finite update."""
        mem = _make_memory(rewards=[0.0] * 9 + [20.0])
        loss = reinforce_update(self.agent, self.optimizer, mem)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))


class TestREINFORCESmoke(unittest.TestCase):

    def test_one_episode_completes_without_error(self):
        from catanatron import Color
        env = create_random_players_env(num_enemies=1)
        obs, info = env.reset()
        reset_reward_function()

        agent = REINFORCEAgent(obs.shape[0], MAX_ACTION_COUNT)
        optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-3)
        memory = ReplayMemory(4096)

        done = False
        while not done:
            action = agent.select_action(obs, info["valid_actions"])
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            memory.push(ReplayMemory.create_reinforce_state(
                obs, info["valid_actions"], action, reward, done
            ))
            obs, info = next_obs, next_info

        loss = reinforce_update(agent, optimizer, memory)
        self.assertIsInstance(loss, float)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        env.close()


if __name__ == "__main__":
    unittest.main()
