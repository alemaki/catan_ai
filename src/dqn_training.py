import random
import torch
import gymnasium
import catanatron.gym
from catanatron import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
from utils.constants import device, MAX_ACTIONS
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

policy_net = DQN(observation.shape[0], MAX_ACTIONS).to(device)
target_net = DQN(observation.shape[0], MAX_ACTIONS).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

memory = ReplayMemory(10000)

optimizer = torch.optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)

epsilon = 1.0
EPS_DECAY = 0.995
EPS_MIN = 0.01

for episode in range(5000):
    observation, info = env.reset()
    done = False

    while not done:
        action = policy_net.select_action(observation, info["valid_actions"], epsilon)

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated

        memory.push(Transition(observation, info["valid_actions"], action, next_obs, next_info["valid_actions"], reward, done))

        observation = next_obs
        info = next_info

        optimize_model(optimizer, policy_net, target_net, memory)

    epsilon = max(EPS_MIN, epsilon * EPS_DECAY)

    # update target network occasionally
    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())

env.close()