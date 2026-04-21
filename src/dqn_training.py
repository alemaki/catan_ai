import random
import torch
import gymnasium
import catanatron.gym
import os
import json
from catanatron import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
from utils.constants import device, MAX_ACITON_COUNT
from utils.utils import create_game_stats, save_model, save_stats
from models.dqn import reward_function, Transition, ReplayMemory, DQN, optimize_model


LEARNING_RATE = 3e-4

env = gymnasium.make(
    "catanatron/Catanatron-v0",
    config={
        "enemies": [
            WeightedRandomPlayer(Color.RED),
            WeightedRandomPlayer(Color.WHITE),
            WeightedRandomPlayer(Color.ORANGE),
        ],
        "reward_function": reward_function,
    },
)

observation, _ = env.reset()

policy_net = DQN(observation.shape[0], MAX_ACITON_COUNT).to(device)
target_net = DQN(observation.shape[0], MAX_ACITON_COUNT).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

memory = ReplayMemory(10000)

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)

epsilon = 1.0
EPS_DECAY = 0.995
EPS_MIN = 0.01

for episode in range(10001):
    observation, info = env.reset()
    done: bool = False

    prev_observation: list = observation
    prev_info: dict = info
    total_loss: float = 0
    while not done:
        action = policy_net.select_action(prev_observation, info["valid_actions"], epsilon)

        observation, reward, terminated, truncated, info = env.step(action)
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
    save_stats(game_stats, episode, total_loss, "dqn_stats_32_neurons_model_default_reward.json")

    # update target network occasionally
    if episode % 100 == 0:
        target_net.load_state_dict(policy_net.state_dict())
    if episode % 2000 == 0 and episode != 0:
        save_model(target_net, f"32_neurons_model_default_reward/dqn_episode_{episode}.pt")

env.close()