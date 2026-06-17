import random
import torch.nn
from utils.constants import MAX_ACTION_COUNT
from collections import namedtuple, deque
from catanatron import Game, Color, RESOURCES
from utils.constants import WIN_REWARD, VP_REWARD, CITY_REWARD, ROAD_REWARD
from catanatron.state import PLAYER_INITIAL_STATE
from catanatron.state_functions import player_key

Transition = namedtuple("Transition",
                            ("observation",
                             "valid_actions_mask",
                             "action",
                             "next_observation",
                             "next_valid_actions_mask",
                             "reward",
                             "done"
                            )
                        )

"""

"""
def reward_function(game: Game, agent_color: Color):
    reward = 0
    ps = game.state.player_state
    key = player_key(game.state, agent_color)

    # VP gain (primary)
    vp_gain = ps[f"{key}_ACTUAL_VICTORY_POINTS"] - reward_function.last_points
    reward += max(0, vp_gain) * VP_REWARD
    reward_function.last_points = max(1, ps[f"{key}_ACTUAL_VICTORY_POINTS"])

    # Resource diversity (encourages varied production) (secondary)
    # resources = [ps[f"{key}_{r}_IN_HAND"] for r in RESOURCES]
    # diversity = sum(1 for r in resources if r > 0)
    # reward += (diversity - reward_function.last_diversity) * 0.1
    # reward_function.last_diversity = diversity

    # Building progress (secondary)
    roads_built = PLAYER_INITIAL_STATE["ROADS_AVAILABLE"] - ps[f"{key}_ROADS_AVAILABLE"]
    reward += max(0, (roads_built - reward_function.last_roads) * ROAD_REWARD)
    reward_function.last_roads = max(1, roads_built)

    # Win/loss (primary)
    if game.winning_color() is not None:
        reward += WIN_REWARD if game.winning_color() == agent_color else -WIN_REWARD
        reset_reward_function()

    return reward

def reset_reward_function():
    reward_function.last_points = 1
    reward_function.last_roads = 1

reset_reward_function()

def valid_actions_to_mask(valid_actions, action_dim = MAX_ACTION_COUNT, device = "cpu"):
    mask = torch.full((action_dim,), -1e9, device=device, dtype=torch.float32)
    mask[valid_actions] = 0
    return mask

"""

"""
class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque([], maxlen = capacity)

    @staticmethod
    def create_transition(observation, valid_actions, action, next_observation, next_valid_actions, reward, done, device = "cpu"):
        valid_actions_mask = valid_actions_to_mask(valid_actions, device = device) # makes tesnor
        next_valid_actions_mask = valid_actions_to_mask(next_valid_actions, device = device)
        return Transition(
            torch.tensor(observation, dtype=torch.float32, device=device),
            valid_actions_mask,
            torch.tensor(action, dtype=torch.long, device=device),
            torch.tensor(next_observation, dtype=torch.float32, device=device),
            next_valid_actions_mask,
            torch.tensor(reward, dtype=torch.float32, device=device),
            torch.tensor(done, dtype=torch.float32, device=device),
        )

    def push(self, transition: Transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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

        return ReplayMemory.create_transition(
            obs, valid, action, next_obs, next_valid, R, done, device=device
        )

    def pop(self, device="cpu") -> Transition:
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


class DQN(torch.nn.Module):


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

    def select_action(self, observation: list, valid_actions: list, epsilon: float, device = "cpu") -> int:
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
    batch = Transition(*zip(*transitions))

    observation_batch = torch.stack(batch.observation)
    action_batch = torch.stack(batch.action).unsqueeze(1)
    reward_batch = torch.stack(batch.reward)
    next_observation_batch = torch.stack(batch.next_observation)
    done_batch = torch.stack(batch.done)

    # Q(s,a)
    observation_action_values = policy_net(observation_batch).gather(1, action_batch).squeeze()

    # Q target
    with torch.no_grad():
        next_q = target_net(next_observation_batch)

        next_mask_batch = torch.stack(batch.next_valid_actions_mask)
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
