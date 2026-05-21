import random
import torch
import gymnasium
import catanatron.gym
import os
import json
from utils.constants import device, MAX_ACTION_COUNT
from utils.utils import create_game_stats, save_model, save_stats, create_random_players_env
from models.dqn import reward_function, Transition, ReplayMemory, DQN, optimize_model

LEARNING_RATE = 3e-4
MEMORY = 100_000
EPISODES = 10_000

env = create_random_players_env()
observation, _ = env.reset()

policy_net = DQN(observation.shape[0], MAX_ACTION_COUNT).to(device)
target_net = DQN(observation.shape[0], MAX_ACTION_COUNT).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

memory = ReplayMemory(MEMORY)

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)

epsilon: float  = 1.0
EPS_DECAY: float = 0.996
EPS_MIN: float = 0.01
NORMALIZATION: bool = True

policy_net.train()

for episode in range(EPISODES + 1):
    observation, info = env.reset()
    done: bool = False

    prev_observation: list = observation
    prev_info: dict = info
    total_loss: float = 0
    while not done:
        with torch.no_grad():
            action = policy_net.select_action(prev_observation, info["valid_actions"], epsilon)

        observation, reward, terminated, truncated, info = env.step(action)
        if NORMALIZATION:
            observation /= 50
        done = terminated or truncated

        transition: Transition = ReplayMemory.create_transition(prev_observation, prev_info["valid_actions"], action, observation, info["valid_actions"], reward, done)
        memory.push(transition)

        prev_observation = observation
        prev_info = info

        total_loss = total_loss + optimize_model(optimizer, policy_net, target_net, memory)

    # Epsilon update
    epsilon = max(EPS_MIN, epsilon * EPS_DECAY)

    # save stats for game
    game_stats: dict = create_game_stats(env.unwrapped.game)
    save_stats(game_stats, episode, total_loss, "dqn_stats_256_neurons_model_default_reward.json")

    # update target network occasionally
    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())
    if episode % (EPISODES // 5) == 0 and episode != 0:
        save_model(target_net, f"256_neurons_model_default_reward/dqn_episode_{episode}.pt")

env.close()