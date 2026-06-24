import torch
from utils.constants import *
from utils.utils import *
from models.dqn import *

LEARNING_RATE = 3e-4
MEMORY = 100_000
EPISODES = 15_000
N_STEPS = 1
STARTING_EPISODE = 6001
env = create_random_players_env(reward_function=reward_function, num_enemies=1)
observation, _ = env.reset()

policy_net = DQN(observation.shape[0], MAX_ACTION_COUNT).to(device)
target_net = DQN(observation.shape[0], MAX_ACTION_COUNT).to(device)
policy_net.load_state_dict(torch.load(os.path.join(MODELS_SAVE_PATH, "d3qn_smaller_stats_1v1_better_reward/dqn_episode_6000.pt"), map_location=device))

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

memory = ReplayMemory(MEMORY)
n_step_buffer = NStepBuffer(n=N_STEPS, gamma=GAMMA)

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)

epsilon: float  = 1.0
EPS_MIN: float = 0.001
EPS_DECAY_STEP: float = (EPS_MIN / 1.0) ** (1 / 1_000_000) # Million steps to reach eps_min
NORMALIZATION: bool = False
epsilon = max(EPS_MIN, epsilon * (EPS_DECAY_STEP**(STARTING_EPISODE*200)))

policy_net.train()

for episode in range(STARTING_EPISODE, EPISODES + 1):
    observation, info = env.reset()
    reset_reward_function()
    n_step_buffer.clear()
    done: bool = False
    if NORMALIZATION:
        observation /= 50
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
        if NORMALIZATION:
            observation /= 50
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

        # Epsilon update
        epsilon = max(EPS_MIN, epsilon * EPS_DECAY_STEP)

    # flush remaining steps from the n-step buffer into memory
    for t in n_step_buffer.flush(device):
        memory.push(t)

    # save stats for game
    ep_mean_max_q = total_mean_max_q / max(n_opt_steps, 1)
    game_stats: dict = create_game_stats(env.unwrapped.game, Color.BLUE)
    save_stats(game_stats, episode, total_loss, "d3qn_smaller_stats_1v1_better_reward.json",
               epsilon=epsilon, total_reward=total_reward, mean_max_q=ep_mean_max_q)

    # update target network occasionally
    if episode % 5 == 0:
        target_net.load_state_dict(policy_net.state_dict())
    if episode % (EPISODES // 5) == 0 and episode != 0:
        save_model(target_net, f"d3qn_smaller_stats_1v1_better_reward/dqn_episode_{episode}.pt")

env.close()
