import copy
import torch
from catanatron import Color
from utils.constants import *
from utils.utils import *
from utils.model_player import ModelPlayer
from models.dqn import *

LEARNING_RATE = 3e-4
MEMORY = 100_000
EPISODES = 15_000
N_STEPS = 1
OPPONENT_SYNC_EVERY = 100

CHECKPOINT = "dqn_smaller_stats_dueling_1v1_better_reward/dqn_episode_15000.pt"
STATS_FILE = "saved_self_play/dqn_selfplay.json"
REAL_STATS_FILE = "dqn_selfplay.json"
SAVE_PREFIX = "dqn_selfplay"

# Hack the observation size a little. TODO: fix.
env = create_random_players_env(reward_function=reward_function, num_enemies=0)
observation, _ = env.reset()

obs_size = observation.shape[0]

policy_net = DQN(obs_size, MAX_ACTION_COUNT).to(device)
policy_net.load_state_dict(torch.load(os.path.join(MODELS_SAVE_PATH, CHECKPOINT), map_location=device))

target_net = DQN(obs_size, MAX_ACTION_COUNT).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

opponent_net = copy.deepcopy(policy_net)
opponent_net.eval()

env.close()

opponent_player = ModelPlayer(opponent_net, Color.RED, device=device)
env = gymnasium.make(
    "catanatron/Catanatron-v0",
    config={"enemies": [opponent_player], "reward_function": reward_function},
)

memory = ReplayMemory(MEMORY)
n_step_buffer = NStepBuffer(n=N_STEPS, gamma=GAMMA)
optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)

# TODO: should that even be here
epsilon: float = 0.1
EPS_MIN: float = 0.001
EPS_DECAY_STEP: float = (EPS_MIN / epsilon) ** (1 / 500_000)

policy_net.train()


def log_games_with_random(model: DQN, number_of_games = OPPONENT_SYNC_EVERY):
    random_env = create_random_players_env(num_enemies=1, reward_function = reward_function)
    #TODO: A lot of code repetition with the dqn. FIX!
    for episode in range(number_of_games + 1):
        observation, info = random_env.reset()
        reset_reward_function()
        n_step_buffer.clear()
        done: bool = False
        prev_observation: list = observation
        prev_info: dict = info
        total_loss: float = 0
        total_reward: float = 0

        while not done:
            with torch.no_grad():
                action = model.select_action(prev_observation, info["valid_actions"], device = device, epsilon = 0)

            observation, reward, terminated, truncated, info = random_env.step(action)
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
    
        # flush remaining steps from the n-step buffer into memory
        for t in n_step_buffer.flush(device):
            memory.push(t)

        # save stats for game
        game_stats: dict = create_game_stats(random_env.unwrapped.game, Color.BLUE)
        save_stats(game_stats, episode, total_loss, REAL_STATS_FILE,
                epsilon=epsilon, total_reward=total_reward, mean_max_q=ep_mean_max_q)


for episode in range(EPISODES + 1):
    observation, info = env.reset()
    reset_reward_function()
    n_step_buffer.clear()
    done: bool = False
    prev_observation = observation
    prev_info = info
    total_loss: float = 0
    total_reward: float = 0
    total_mean_max_q: float = 0
    n_opt_steps: int = 0

    while not done:
        with torch.no_grad():
            action = policy_net.select_action(prev_observation, info["valid_actions"], device=device, epsilon=epsilon)

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

        loss, mean_max_q = optimize_model(optimizer, policy_net, target_net, memory, n=N_STEPS)
        total_loss += loss
        if loss > 0:
            total_mean_max_q += mean_max_q
            n_opt_steps += 1

        epsilon = max(EPS_MIN, epsilon * EPS_DECAY_STEP)

    for t in n_step_buffer.flush(device):
        memory.push(t)

    ep_mean_max_q = total_mean_max_q / max(n_opt_steps, 1)
    game_stats = create_game_stats(env.unwrapped.game, Color.BLUE)
    save_stats(game_stats, episode, total_loss, STATS_FILE,
               epsilon=epsilon, total_reward=total_reward, mean_max_q=ep_mean_max_q)

    if episode % 5 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if episode % OPPONENT_SYNC_EVERY == 0 and episode != 0:
        opponent_net.load_state_dict(policy_net.state_dict())
        opponent_net.eval()
        print(f"[episode {episode}] opponent synced to current policy")
        # Time to evaluate how better we are:
        log_games_with_random(policy_net)

    if episode % (EPISODES // 5) == 0 and episode != 0:
        save_model(target_net, f"{SAVE_PREFIX}/dqn_episode_{episode}.pt")

env.close()
