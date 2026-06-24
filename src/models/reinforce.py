import torch
import torch.nn as nn
from utils.constants import *
from utils.utils import valid_actions_to_mask, ReplayMemory, REINFORCEState
from utils.model_player import ActionSelectableModel

GAMMA = 0.999


class REINFORCEAgent(nn.Module, ActionSelectableModel):
    def __init__(self, obs_size, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(512, num_actions)

    def forward(self, obs):
        x = self.network(obs)
        return self.policy_head(x)

    def select_action(self, obs, valid_actions, device="cpu"):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = self.forward(obs_tensor).squeeze(0)
        mask = valid_actions_to_mask(valid_actions, logits.shape[0], device=device)
        dist = torch.distributions.Categorical(logits=logits + mask)
        return dist.sample().item()


def compute_returns(rewards, gamma=GAMMA, device="cpu"):
    """
    Monte Carlo returns
    """
    T = len(rewards)
    returns = torch.zeros(T)
    G = 0.0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns.to(device)


def reinforce_update(agent, optimizer, memory: ReplayMemory, device="cpu"):
    all_transitions = memory.get_all()
    batch = REINFORCEState(*zip(*all_transitions))

    observations = torch.stack(batch.observation).to(device)
    actions = torch.stack(batch.action).to(device)
    masks = torch.stack(batch.valid_actions_mask).to(device)
    rewards = [t.item() for t in batch.reward]

    logits = agent(observations) + masks
    dist = torch.distributions.Categorical(logits=logits)
    log_probs = dist.log_prob(actions)

    returns = compute_returns(rewards, device = device)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    loss = -(log_probs * returns).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
    optimizer.step()

    return loss.item()
