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


    def __init__(self, observation_shape, actions_shape, neurons: int = 512):
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(observation_shape, neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(neurons, neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(neurons, neurons//2),
            torch.nn.ReLU(),
        )
        # Dueling streams
        self.value_stream = torch.nn.Linear(neurons//2, 1)
        self.advantage_stream = torch.nn.Linear(neurons//2, actions_shape)

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

# Example config
DQN_TRAINING_CONFIG = {
    "reward_function"   : None,
    "learning_rate"     : 3e-4,
    "gamma"             : 0.999,
    "memory"            : 100_000,
    "neurons"           : 512, # I think this shouldn't be here, but whatever.
    "n_steps"           : 1,
    "should_save"       : False,
    "save_model_path"   : MODELS_SAVE_PATH,
    "save_model_folder" : "", # leave empty for no save
    "save_stats_path"   : STATS_SAVE_PATH,
    "save_stats_name"   : "", # leave empty for no save
    "starting_episode"  : 0,
    "ending_episode"    : 15_000,
    "save_factor"       : 5, # will save that many times factoring ending_episode. (e.g. ending episode = 1K, factor 5 will save on 2K, 4K, 6K, 8K 10K)
    "load_saved_model"  : "",# leave empty for no load
    "epsilon_start"     : 1.0,
    "eps_min"           : 0.001,
    "eps_steps_decay"   : 1_000_000, # Million steps to reach eps_min
    "target_update_episodes" : 5,
    "optimize_model"    : True,
    "enemies"           : [WeightedRandomPlayer(Color.RED)],
    "on_episode_end"    : None, # function for episode end, args: episode, policy_net, target_net, memory, epsilon
} 


def dqn_run(dqn_config: dict):
    env = create_players_env(reward_function=dqn_config.get("reward_function", None), enemies=dqn_config.get("enemies", []))
    observation, _ = env.reset()

    policy_net = DQN(observation.shape[0], MAX_ACTION_COUNT, neurons=dqn_config["neurons"]).to(device)
    target_net = DQN(observation.shape[0], MAX_ACTION_COUNT, neurons=dqn_config["neurons"]).to(device)
    if dqn_config.get("load_saved_model", "") != "":
        path = os.path.join(MODELS_SAVE_PATH, dqn_config.get("load_saved_model"))
        policy_net.load_state_dict(torch.load(path, map_location=device))

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    memory = ReplayMemory(dqn_config["memory"])
    n_step_buffer = NStepBuffer(n=dqn_config["n_steps"], gamma=dqn_config["gamma"])

    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=dqn_config["learning_rate"], amsgrad=True)

    epsilon: float  = dqn_config["epsilon_start"]
    EPS_MIN: float = dqn_config["eps_min"]
    EPS_DECAY_STEP: float = (EPS_MIN / epsilon) ** (1 / dqn_config["eps_steps_decay"])

    policy_net.train()

    for episode in range(dqn_config["starting_episode"], dqn_config["ending_episode"] + 1):
        observation, info = env.reset()
        reset_reward_function()
        n_step_buffer.clear()
        done: bool = False
        prev_observation: list = observation
        prev_info: dict = info
        total_loss: float = 0
        total_reward: float = 0
        total_mean_max_q: float = 0
        n_opt_steps: int = 0
        while not done:
            with torch.no_grad():
                action = policy_net.select_action(prev_observation, info["valid_actions"], device = device, epsilon = epsilon)

            observation, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            total_reward += reward

            n_step_buffer.append(
                prev_observation,
                prev_info["valid_actions"],
                action,
                reward,
                observation,
                info["valid_actions"],
                done,
            )

            if n_step_buffer.ready():
                memory.push(n_step_buffer.pop(device))

            prev_observation = observation
            prev_info = info

            if dqn_config.get("optimize_model", True):
                loss, mean_max_q = optimize_model(optimizer, policy_net, target_net, memory, n=dqn_config["n_steps"])
                total_loss += loss
                if loss > 0:
                    total_mean_max_q += mean_max_q
                    n_opt_steps += 1

            # Epsilon update
            epsilon = max(EPS_MIN, epsilon * EPS_DECAY_STEP)

        # flush remaining steps from the n-step buffer into memory
        for t in n_step_buffer.flush(device):
            memory.push(t)

        # save stats for game
        ep_mean_max_q = total_mean_max_q / max(n_opt_steps, 1)
        game_stats: dict = create_game_stats(env.unwrapped.game, Color.BLUE)

        if dqn_config.get("save_stats_name", "") != "":
            save_stats(
                    game_stats,
                    episode,
                    total_loss,
                    dqn_config.get("save_stats_name"),
                    epsilon=epsilon,
                    total_reward=total_reward,
                    mean_max_q=ep_mean_max_q)

        # update target network
        if episode % dqn_config["target_update_episodes"] == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if dqn_config.get("save_model_folder", "") != "" and\
            episode % (dqn_config["ending_episode"] // dqn_config["save_factor"]) == 0 and\
            episode != 0:
            save_model(target_net, f"{dqn_config['save_model_folder']}/dqn_episode_{episode}.pt")

        if dqn_config.get("on_episode_end") is not None:
            dqn_config.get("on_episode_end")(episode, policy_net, target_net, memory, epsilon)

    env.close()
