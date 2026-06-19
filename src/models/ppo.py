import random
import torch.nn
from utils.constants import *
from utils.utils import *

BATCH_SIZE = 64
GAMMA = 0.99
LAMBDA = 0.95

class PPOActor(torch.nn.Module):
    def __init__(self, observation_shape, actions_shape):
        super().__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(observation_shape, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
        )
        self.advantage_stream = torch.nn.Linear(256, actions_shape)
        self.softmax = torch.nn.Softmax(dim=-1)

    """
    Called with either one element to determine next action, or a batch
    during optimization. Returns tensor([[left0exp,right0exp]...]).
    """
    def forward(self, observation):
        x = self.linear(observation)
        advantage = self.advantage_stream(x)
        return self.softmax(advantage)

    def select_action(self, observation: list, valid_actions: list, device="cpu") -> tuple[int, torch.Tensor]:
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = self.linear(observation)
            logits = self.advantage_stream(logits).squeeze(0)

        mask = valid_actions_to_mask(valid_actions, logits.shape[0], device=device)
        logits = logits + mask

        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob


class PPOCritic(torch.nn.Module):


    def __init__(self, observation_shape):
        super().__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(observation_shape, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
        )
        self.value = torch.nn.Linear(256, 1)

    """
    Called with either one element to determine next action, or a batch
    during optimization. Returns tensor([[left0exp,right0exp]...]).
    """
    def forward(self, observation):
        x = self.linear(observation)
        return self.value(x)
    
PPO_EPOCHS = 10
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01

def compute_gae(memory: ReplayMemory, device="cpu"):
    all_transitions = memory.get_all()
    batch = PPOState(*zip(*all_transitions))

    rewards = torch.stack(batch.reward).to(device)
    values = torch.stack(batch.value).to(device)

    T = len(all_transitions)
    advantages = torch.zeros(T, device=device)

    gae = rewards[-1] - values[-1]
    advantages[-1] = gae
    for t in reversed(range(T - 1)):
        delta = rewards[t] + GAMMA * values[t + 1] - values[t]
        gae = delta + GAMMA * LAMBDA * gae
        advantages[t] = gae

    value_targets = advantages + values
    return advantages, value_targets


def ppo_update(actor: PPOActor, critic: PPOCritic, actor_optimizer, critic_optimizer, memory: ReplayMemory, device="cpu"):
    advantages, value_targets = compute_gae(memory, device)

    all_transitions = memory.get_all()
    batch = PPOState(*zip(*all_transitions))

    observations   = torch.stack(batch.observation).to(device)
    actions        = torch.stack(batch.action).to(device)
    old_log_probs  = torch.stack(batch.log_prob).to(device)
    valid_masks    = torch.stack(batch.valid_actions_mask).to(device)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    T = len(all_transitions)
    total_actor_loss = 0.0
    total_critic_loss = 0.0

    for _ in range(PPO_EPOCHS):
        order = torch.randperm(T)
        for start in range(0, T, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]

            mb_obs          = observations[idx]
            mb_actions      = actions[idx]
            mb_old_log_prob = old_log_probs[idx]
            mb_masks        = valid_masks[idx]
            mb_advantages   = advantages[idx]
            mb_targets      = value_targets[idx]

            # recompute log probs and entropy under current actor
            logits = actor.advantage_stream(actor.linear(mb_obs)) + mb_masks
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            # clipped actor loss
            ratio = torch.exp(new_log_probs - mb_old_log_prob)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # critic loss
            new_values = critic(mb_obs).squeeze(-1)
            critic_loss = torch.nn.functional.mse_loss(new_values, mb_targets)

            loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            actor_optimizer.step()
            critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

    return total_actor_loss, total_critic_loss
