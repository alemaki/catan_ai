import random
import torch
import gymnasium
import catanatron.gym
import os
import json
from utils.constants import device, MAX_ACTION_COUNT
from utils.utils import create_game_stats, save_model, save_stats, create_random_players_env
from models.dqn import reward_function, reset_reward_function, Transition, ReplayMemory, NStepBuffer, DQN, optimize_model, GAMMA

LEARNING_RATE = 3e-4
MEMORY = 100_000
EPISODES = 10_000
N_STEPS = 10

env = create_random_players_env(reward_function= reward_function)
observation, _ = env.reset()

policy_net = DQN(observation.shape[0], MAX_ACTION_COUNT).to(device)
target_net = DQN(observation.shape[0], MAX_ACTION_COUNT).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

memory = ReplayMemory(MEMORY)
n_step_buffer = NStepBuffer(n=N_STEPS, gamma=GAMMA)

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)

epsilon: float  = 1.0
EPS_MIN: float = 0.001
EPS_DECAY_STEP: float = (EPS_MIN / 1.0) ** (1 / 1_000_000) # Million steps to reach eps_min
NORMALIZATION: bool = False

policy_net.train()

for episode in range(EPISODES + 1):
    observation, info = env.reset()
    reset_reward_function()
    n_step_buffer.clear()
    done: bool = False
    if NORMALIZATION:
        observation /= 50
    prev_observation: list = observation
    prev_info: dict = info
    total_loss: float = 0
    while not done:
        with torch.no_grad():
            action = policy_net.select_action(prev_observation, info["valid_actions"], epsilon, device = device)

        observation, reward, terminated, truncated, info = env.step(action)
        if NORMALIZATION:
            observation /= 50
        done = terminated or truncated

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

        total_loss = total_loss + optimize_model(optimizer, policy_net, target_net, memory, n=N_STEPS)

        # Epsilon update
        epsilon = max(EPS_MIN, epsilon * EPS_DECAY_STEP)

    # flush remaining steps from the n-step buffer into memory
    for t in n_step_buffer.flush(device):
        memory.push(t)

    # save stats for game
    game_stats: dict = create_game_stats(env.unwrapped.game)
    save_stats(game_stats, episode, total_loss, "dqn_stats_dueling_model_10_step.json")

    # update target network occasionally
    if episode % 5 == 0:
        target_net.load_state_dict(policy_net.state_dict())
    if episode % (EPISODES // 5) == 0 and episode != 0:
        save_model(target_net, f"dqn_stats_dueling_model_10_step/dqn_episode_{episode}.pt")

env.close()
