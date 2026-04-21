import random
import torch.nn
from utils.constants import MAX_ACITON_COUNT
from collections import namedtuple, deque
from catanatron import Game, Color
from utils.constants import device

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
def reward_function(game: Game, p0_color: Color):
    reward = 0
    if  reward_function.last_points == 0: # move back to 1? so we don't reward first move.
         reward_function.last_points = 1

    # VP gain. Can happen with settlements, castles, longest roads, biggest army, or just vp gain from casino
    reward += (game.state.player_state["P0_ACTUAL_VICTORY_POINTS"] - reward_function.last_points) * 10 # increase reward
    reward_function.last_points = game.state.player_state["P0_ACTUAL_VICTORY_POINTS"]

    # Win/loss
    if not (game.winning_color() is None):
        if game.winning_color() == p0_color:
            reward += 100
        else:
            reward -= 100 # slap it in the face
        reward_function.last_points = 0

    return reward

reward_function.last_points = 0 # static

def valid_actions_to_mask(valid_actions, action_dim = MAX_ACITON_COUNT):
    mask = torch.full((action_dim,), -1e9, device=device)
    mask[valid_actions] = 0
    return mask

"""

"""
class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque([], maxlen = capacity)

    @staticmethod
    def create_transition(observation, valid_actions, action, next_observation, next_valid_actions, reward, done):
        valid_actions_mask = valid_actions_to_mask(valid_actions)
        next_valid_actions_mask = valid_actions_to_mask(next_valid_actions)
        return Transition(
            torch.tensor(observation, dtype=torch.float32, device=device),
            torch.tensor(valid_actions_mask, dtype=torch.float32, device=device),
            torch.tensor(action, dtype=torch.long, device=device),
            torch.tensor(next_observation, dtype=torch.float32, device=device),
            torch.tensor(next_valid_actions_mask, dtype=torch.float32, device=device),
            torch.tensor(reward, dtype=torch.float32, device=device),
            torch.tensor(done, dtype=torch.float32, device=device),
        )

    def push(self, transition: Transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(torch.nn.Module):

    def __init__(self, observation_shape, actions_shape):
        super(DQN, self).__init__()
        self.layer1 = torch.nn.Linear(observation_shape, 32)
        self.layer2 = torch.nn.Linear(32, 32)
        self.layer3 = torch.nn.Linear(32, actions_shape)

    """
    Called with either one element to determine next action, or a batch
    during optimization. Returns tensor([[left0exp,right0exp]...]).
    """
    def forward(self, observation):
        observation = torch.nn.functional.relu(self.layer1(observation))
        observation = torch.nn.functional.relu(self.layer2(observation))
        return self.layer3(observation)

    def select_action(self, observation: list, valid_actions: list, epsilon: float) -> int:
        # Random choice for espilon start
        if random.random() < epsilon:
            return random.choice(valid_actions)


        with torch.no_grad():
            # Transalte observation
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)

            # Get result of the forward
            q_values = self(observation).squeeze(0)
            mask = valid_actions_to_mask(valid_actions)
            q_values = q_values + mask

            return torch.argmax(q_values).item()

"""

"""
def mask_q_values(q_values: torch.Tensor, valid_actions: list) -> torch.Tensor:
    # Mask for unplayable actions.
    mask = torch.full_like(q_values, -1e9)
    mask[valid_actions] = 0
    return q_values + mask

BATCH_SIZE = 64
GAMMA = 0.99

def optimize_model(optimizer, policy_net: DQN, target_net: DQN, memory: ReplayMemory) -> float:
    if len(memory) < BATCH_SIZE:
        return 0

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

        expected_q = reward_batch + GAMMA * max_next_q * (1 - done_batch)

    loss = torch.nn.functional.smooth_l1_loss(observation_action_values, expected_q)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()

    return loss.item()
