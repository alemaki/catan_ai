import unittest
import torch
from models.ppo import PPOActor, PPOCritic, compute_gae, ppo_update, GAMMA, LAMBDA
from utils.utils import ReplayMemory, PPOState, create_random_players_env, reset_reward_function
from utils.constants import MAX_ACTION_COUNT

OBS_SIZE = 20
ACTION_SIZE = MAX_ACTION_COUNT


def _make_ppo_memory(rewards, values, dones, obs_size=OBS_SIZE):
    T = len(rewards)
    mem = ReplayMemory(T + 1)
    valid = list(range(ACTION_SIZE))
    for r, v, d in zip(rewards, values, dones):
        s = ReplayMemory.create_ppo_state(
            [0.0] * obs_size, valid, 0, float(r),
            torch.tensor(float(v)), torch.tensor(-1.0), bool(d)
        )
        mem.push(s)
    return mem


class TestReplayMemoryPPOState(unittest.TestCase):

    OBS_SIZE = 10

    def _make(self, reward=1.0, done=False, action=0, log_prob=-1.5, value=0.5):
        obs = [0.0] * self.OBS_SIZE
        return ReplayMemory.create_ppo_state(
            obs, [0, 1, 2], action, reward,
            torch.tensor(value), torch.tensor(log_prob), done
        )

    def test_returns_ppo_state_namedtuple(self):
        self.assertIsInstance(self._make(), PPOState)

    def test_observation_is_float32_tensor(self):
        s = self._make()
        self.assertIsInstance(s.observation, torch.Tensor)
        self.assertEqual(s.observation.dtype, torch.float32)

    def test_action_is_long_tensor(self):
        self.assertEqual(self._make().action.dtype, torch.long)

    def test_reward_stored_correctly(self):
        self.assertAlmostEqual(self._make(reward=7.5).reward.item(), 7.5)

    def test_done_false_stored_as_zero(self):
        self.assertEqual(self._make(done=False).done.item(), 0.0)

    def test_done_true_stored_as_one(self):
        self.assertEqual(self._make(done=True).done.item(), 1.0)

    def test_log_prob_stored_correctly(self):
        self.assertAlmostEqual(self._make(log_prob=-2.3).log_prob.item(), -2.3, places=5)

    def test_value_stored_correctly(self):
        self.assertAlmostEqual(self._make(value=3.7).value.item(), 3.7, places=5)

    def test_mask_valid_actions_are_zero(self):
        obs = [0.0] * self.OBS_SIZE
        s = ReplayMemory.create_ppo_state(obs, [0, 2, 4], 0, 0.0, torch.tensor(0.0), torch.tensor(0.0), False)
        for idx in [0, 2, 4]:
            self.assertEqual(s.valid_actions_mask[idx].item(), 0.0)

    def test_mask_invalid_actions_are_large_negative(self):
        obs = [0.0] * self.OBS_SIZE
        s = ReplayMemory.create_ppo_state(obs, [0], 0, 0.0, torch.tensor(0.0), torch.tensor(0.0), False)
        self.assertAlmostEqual(s.valid_actions_mask[1].item(), -1e9)

    def test_observation_shape(self):
        self.assertEqual(self._make().observation.shape[0], self.OBS_SIZE)

    def test_mask_shape_matches_max_action_count(self):
        self.assertEqual(self._make().valid_actions_mask.shape[0], MAX_ACTION_COUNT)


class TestPPOActor(unittest.TestCase):

    def setUp(self):
        self.actor = PPOActor(OBS_SIZE, ACTION_SIZE)

    def test_forward_output_shape_single(self):
        out = self.actor(torch.zeros(1, OBS_SIZE))
        self.assertEqual(out.shape, (1, ACTION_SIZE))

    def test_forward_output_shape_batch(self):
        out = self.actor(torch.zeros(16, OBS_SIZE))
        self.assertEqual(out.shape, (16, ACTION_SIZE))

    def test_forward_outputs_valid_probabilities(self):
        probs = self.actor(torch.randn(4, OBS_SIZE))
        self.assertTrue(torch.all(probs >= 0))
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5))

    def test_select_action_returns_valid_action(self):
        action, _ = self.actor.select_training_action([0.0] * OBS_SIZE, [0, 5, 100, 200])
        self.assertIn(action, [0, 5, 100, 200])

    def test_select_action_always_valid_over_many_calls(self):
        valid = [1, 3, 7]
        for _ in range(30):
            action, _ = self.actor.select_training_action([0.0] * OBS_SIZE, valid)
            self.assertIn(action, valid)

    def test_select_action_mask_blocks_invalid_actions(self):
        with torch.no_grad():
            self.actor.advantage_stream.weight.zero_()
            self.actor.advantage_stream.bias.zero_()
            self.actor.advantage_stream.bias[5] = 1000.0
        for _ in range(20):
            action, _ = self.actor.select_training_action([0.0] * OBS_SIZE, valid_actions=[0])
            self.assertEqual(action, 0)

    def test_select_action_log_prob_is_tensor(self):
        _, log_prob = self.actor.select_training_action([0.0] * OBS_SIZE, [0, 1])
        self.assertIsInstance(log_prob, torch.Tensor)

    def test_select_action_log_prob_is_non_positive(self):
        _, log_prob = self.actor.select_training_action([0.0] * OBS_SIZE, [0, 1, 2])
        self.assertLessEqual(log_prob.item(), 0.0)

    def test_select_action_single_valid_always_returned(self):
        for _ in range(20):
            action, _ = self.actor.select_training_action([0.0] * OBS_SIZE, [42])
            self.assertEqual(action, 42)

    def test_select_action_single_valid_log_prob_near_zero(self):
        """Only one valid action means probability 1 → log_prob ≈ 0."""
        _, log_prob = self.actor.select_training_action([0.0] * OBS_SIZE, [0])
        self.assertAlmostEqual(log_prob.item(), 0.0, places=4)

    def test_forward_output_is_finite(self):
        out = self.actor(torch.randn(4, OBS_SIZE))
        self.assertTrue(torch.all(torch.isfinite(out)))


class TestPPOCritic(unittest.TestCase):

    def setUp(self):
        self.critic = PPOCritic(OBS_SIZE)

    def test_forward_output_shape_single(self):
        self.assertEqual(self.critic(torch.zeros(1, OBS_SIZE)).shape, (1, 1))

    def test_forward_output_shape_batch(self):
        self.assertEqual(self.critic(torch.zeros(8, OBS_SIZE)).shape, (8, 1))

    def test_forward_scalar_after_squeeze(self):
        val = self.critic(torch.zeros(1, OBS_SIZE)).squeeze()
        self.assertEqual(val.shape, torch.Size([]))

    def test_forward_is_finite(self):
        out = self.critic(torch.randn(4, OBS_SIZE))
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_forward_different_inputs_give_different_outputs(self):
        v1 = self.critic(torch.zeros(1, OBS_SIZE))
        v2 = self.critic(torch.ones(1, OBS_SIZE))
        self.assertFalse(torch.equal(v1, v2))


class TestComputeGAE(unittest.TestCase):

    def _mem(self, rewards, values, dones):
        return _make_ppo_memory(rewards, values, dones)

    def test_returns_tuple_of_two_tensors(self):
        adv, vt = compute_gae(self._mem([1.0], [0.5], [True]))
        self.assertIsInstance(adv, torch.Tensor)
        self.assertIsInstance(vt, torch.Tensor)

    def test_output_length_matches_memory_size(self):
        adv, vt = compute_gae(self._mem([1.0, 2.0, 3.0], [0.5] * 3, [False, False, True]))
        self.assertEqual(adv.shape[0], 3)
        self.assertEqual(vt.shape[0], 3)

    def test_value_targets_equal_advantages_plus_values(self):
        """value_target = advantage + V(s) is a hard mathematical identity."""
        values = [0.3, 0.8, 1.2]
        adv, vt = compute_gae(self._mem([1.0, 0.5, 2.0], values, [False, False, True]))
        v = torch.tensor(values, dtype=torch.float32)
        self.assertTrue(torch.allclose(vt, adv + v, atol=1e-5))

    def test_single_step_advantage_equals_reward_minus_value(self):
        adv, _ = compute_gae(self._mem([5.0], [2.0], [True]))
        self.assertAlmostEqual(adv[0].item(), 3.0, places=5)

    def test_single_step_value_target_equals_reward(self):
        """T=1: value_target = (r-V) + V = r, regardless of done."""
        _, vt = compute_gae(self._mem([5.0], [2.0], [True]))
        self.assertAlmostEqual(vt[0].item(), 5.0, places=5)

    def test_terminal_step_no_bootstrap(self):
        """done=True → adv = r - V with no next-value bootstrapping."""
        adv, _ = compute_gae(self._mem([3.0], [1.0], [True]))
        self.assertAlmostEqual(adv[0].item(), 2.0, places=5)

    def test_non_terminal_two_step_advantage(self):
        """done=False bootstraps V(s_{t+1}) into adv[t]."""
        rewards = [1.0, 2.0]
        values  = [3.0, 4.0]
        adv, _ = compute_gae(self._mem(rewards, values, [False, False]))
        # t=1 (T-1): adv = r1 - v1 = -2.0
        self.assertAlmostEqual(adv[1].item(), -2.0, places=5)
        # t=0: delta = r0 + GAMMA*v1 - v0; gae = delta + GAMMA*LAMBDA*adv[1]
        expected = (1.0 + GAMMA * 4.0 - 3.0) + GAMMA * LAMBDA * (-2.0)
        self.assertAlmostEqual(adv[0].item(), expected, places=4)

    def test_terminal_cuts_bootstrap_and_gae_propagation(self):
        """done=True at t: adv[t] = r[t] - v[t], and t+1's gae does not flow into t-1."""
        # 3 steps: done=[False, True, False]
        # t=1 done=True → mask=0 → adv[1] = r[1] - v[1] = 5.0 exactly
        # t=0: bootstraps v[1] and uses adv[1]=5.0 (not the raw adv[2])
        rewards = [1.0, 10.0, 3.0]
        values  = [2.0,  5.0, 6.0]
        adv, _ = compute_gae(self._mem(rewards, values, [False, True, False]))
        self.assertAlmostEqual(adv[1].item(), 5.0, places=5)
        expected_adv0 = (1.0 + GAMMA * 5.0 - 2.0) + GAMMA * LAMBDA * 5.0
        self.assertAlmostEqual(adv[0].item(), expected_adv0, places=4)

    def test_all_terminal_advantages_are_reward_minus_value(self):
        """Every step done=True → each adv[t] = r[t] - v[t] independently."""
        rewards = [1.0, 2.0, 3.0]
        values  = [0.5, 1.0, 1.5]
        adv, _ = compute_gae(self._mem(rewards, values, [True, True, True]))
        for i, (r, v) in enumerate(zip(rewards, values)):
            self.assertAlmostEqual(adv[i].item(), r - v, places=5)

    def test_advantages_and_value_targets_are_finite(self):
        adv, vt = compute_gae(self._mem([1.0, -0.5, 2.0], [0.5, 1.0, 0.8], [False, False, True]))
        self.assertTrue(torch.all(torch.isfinite(adv)))
        self.assertTrue(torch.all(torch.isfinite(vt)))


class TestPPOUpdate(unittest.TestCase):

    def setUp(self):
        self.actor = PPOActor(OBS_SIZE, ACTION_SIZE)
        self.critic = PPOCritic(OBS_SIZE)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=1e-3)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=1e-3)
        self.memory = _make_ppo_memory(
            rewards=[1.0] * 10,
            values=[0.5] * 10,
            dones=[False] * 9 + [True],
        )

    def test_returns_two_floats(self):
        result = ppo_update(self.actor, self.critic, self.actor_opt, self.critic_opt, self.memory)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], float)
        self.assertIsInstance(result[1], float)

    def test_actor_parameters_change_after_update(self):
        params_before = [p.clone() for p in self.actor.parameters()]
        ppo_update(self.actor, self.critic, self.actor_opt, self.critic_opt, self.memory)
        changed = any(not torch.equal(b, a) for b, a in zip(params_before, self.actor.parameters()))
        self.assertTrue(changed)

    def test_critic_parameters_change_after_update(self):
        params_before = [p.clone() for p in self.critic.parameters()]
        ppo_update(self.actor, self.critic, self.actor_opt, self.critic_opt, self.memory)
        changed = any(not torch.equal(b, a) for b, a in zip(params_before, self.critic.parameters()))
        self.assertTrue(changed)

    def test_losses_are_finite(self):
        actor_loss, critic_loss = ppo_update(self.actor, self.critic, self.actor_opt, self.critic_opt, self.memory)
        self.assertTrue(torch.isfinite(torch.tensor(actor_loss)))
        self.assertTrue(torch.isfinite(torch.tensor(critic_loss)))

    def test_critic_loss_is_non_negative(self):
        """Critic uses MSE — always >= 0."""
        _, critic_loss = ppo_update(self.actor, self.critic, self.actor_opt, self.critic_opt, self.memory)
        self.assertGreaterEqual(critic_loss, 0.0)

    def test_multiple_updates_do_not_error(self):
        for _ in range(3):
            ppo_update(self.actor, self.critic, self.actor_opt, self.critic_opt, self.memory)


class TestPPOTrainingSmoke(unittest.TestCase):

    def test_one_episode_completes_without_error(self):
        from catanatron import Color
        env = create_random_players_env(num_enemies=1)
        obs, info = env.reset()
        reset_reward_function()

        actor = PPOActor(obs.shape[0], MAX_ACTION_COUNT)
        critic = PPOCritic(obs.shape[0])
        actor_opt = torch.optim.AdamW(actor.parameters(), lr=1e-3)
        critic_opt = torch.optim.AdamW(critic.parameters(), lr=1e-3)
        memory = ReplayMemory(4096)

        done = False
        while not done:
            action, log_prob = actor.select_training_action(obs, info["valid_actions"])
            with torch.no_grad():
                value = critic(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze()
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            memory.push(ReplayMemory.create_ppo_state(obs, info["valid_actions"], action, reward, value, log_prob, done))
            obs, info = next_obs, next_info

        actor_loss, critic_loss = ppo_update(actor, critic, actor_opt, critic_opt, memory)
        self.assertIsInstance(actor_loss, float)
        self.assertIsInstance(critic_loss, float)
        env.close()


if __name__ == "__main__":
    unittest.main()
