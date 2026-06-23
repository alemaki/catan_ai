import random
import torch.nn
from collections import deque
from utils.constants import *
from utils.utils import *
from utils.model_player import ActionSelectableModel

DOUBLE_DQN = True

class NStepBuffer:
    def __init__(self, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self.buffer = deque()

    def append(self, obs, valid_actions, action, reward, next_obs, next_valid_actions, done):
        self.buffer.append((obs, valid_actions, action, reward, next_obs, next_valid_actions, done))

    def ready(self) -> bool:
        return len(self.buffer) >= self.n

    def _build_transition(self, device):
        R = 0.0
        end_idx = len(self.buffer) - 1
        for i, (_, _, _, r, _, _, done) in enumerate(self.buffer):
            R += (self.gamma ** i) * r
            if done:
                end_idx = i
                break

        obs, valid, action, _, _, _, _ = self.buffer[0]
        _, _, _, _, next_obs, next_valid, done = self.buffer[end_idx]

        return ReplayMemory.create_dqn_transition(
            obs, valid, action, next_obs, next_valid, R, done, device=device
        )

    def pop(self, device="cpu") -> DQNTransition:
        t = self._build_transition(device)
        self.buffer.popleft()
        return t

    def flush(self, device="cpu") -> list:
        transitions = []
        while self.buffer:
            transitions.append(self._build_transition(device))
            self.buffer.popleft()
        return transitions

    def clear(self):
        self.buffer.clear()


class DQN(torch.nn.Module, ActionSelectableModel):


    def __init__(self, observation_shape, actions_shape):
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(observation_shape, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
        )
        # Dueling streams
        self.value_stream = torch.nn.Linear(256, 1)
        self.advantage_stream = torch.nn.Linear(256, actions_shape)

    """
    Called with either one element to determine next action, or a batch
    during optimization. Returns tensor([[left0exp,right0exp]...]).
    """
    def forward(self, observation):
        x = self.shared(observation)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        # Dueling: Q = V + (A - mean(A))
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

    def select_action(self, observation: list, valid_actions: list, device = "cpu",  epsilon: float = 0.0) -> int:
        # Random choice for espilon start
        if random.random() < epsilon:
            return random.choice(valid_actions)


        with torch.no_grad():
            # Transalte observation
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)

            # Get result of the forward
            q_values = self(observation).squeeze(0)
            mask = valid_actions_to_mask(valid_actions, q_values.shape[0], device = device)
            q_values = q_values + mask

            return torch.argmax(q_values).item()

BATCH_SIZE = 64
GAMMA = 0.99

def optimize_model(optimizer, policy_net: DQN, target_net: DQN, memory: ReplayMemory, n: int = 1) -> tuple:
    if len(memory) < BATCH_SIZE:
        return 0, 0.0

    transitions = memory.sample(BATCH_SIZE)
    batch = DQNTransition(*zip(*transitions))

    observation_batch = torch.stack(batch.observation)
    action_batch = torch.stack(batch.action).unsqueeze(1)
    reward_batch = torch.stack(batch.reward)
    next_observation_batch = torch.stack(batch.next_observation)
    done_batch = torch.stack(batch.done)

    # Q(s,a)
    observation_action_values = policy_net(observation_batch).gather(1, action_batch).squeeze()

    # Q target
    with torch.no_grad():
        next_mask_batch = torch.stack(batch.next_valid_actions_mask)
        if DOUBLE_DQN:
            # Take the next actions the policy predicts and evaluate them with the target netowrk to get the next_q_vals
            next_best_actions = (policy_net(next_observation_batch) + next_mask_batch).argmax(dim=1, keepdim=True)
            max_next_q = target_net(next_observation_batch).gather(1, next_best_actions).squeeze()
        else:
            # Evaluate the next action with the target network and take the best values from it
            next_q = target_net(next_observation_batch)
            masked_next_q = next_q + next_mask_batch
            max_next_q = masked_next_q.max(dim=1).values

        expected_q = reward_batch + (GAMMA ** n) * max_next_q * (1 - done_batch)
        mean_max_q = max_next_q.mean().item()

    loss = torch.nn.functional.smooth_l1_loss(observation_action_values, expected_q)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()

    return loss.item(), mean_max_q
