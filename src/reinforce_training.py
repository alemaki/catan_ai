import torch
from catanatron import Color
from utils.constants import *
from utils.utils import *
from models.reinforce import REINFORCEAgent, reinforce_update

LEARNING_RATE = 9e-4
ROLLOUT_CAPACITY = 4096
EPISODES = 20_000

env = create_random_players_env(num_enemies=1, reward_function=reward_function)
observation, _ = env.reset()

agent = REINFORCEAgent(observation.shape[0], MAX_ACTION_COUNT).to(device)
optimizer = torch.optim.AdamW(agent.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(ROLLOUT_CAPACITY)

agent.train()

for episode in range(EPISODES + 1):
    observation, info = env.reset()
    reset_reward_function()
    done = False
    total_reward = 0.0

    while not done:
        action = agent.select_action(observation, info["valid_actions"], device=device)
        next_observation, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        memory.push(ReplayMemory.create_reinforce_state(
            observation,
            info["valid_actions"],
            action,
            reward,
            done,
            device=device,
        ))

        observation = next_observation
        info = next_info

    loss = reinforce_update(agent, optimizer, memory, device=device)
    memory.clear()

    game_stats = create_game_stats(env.unwrapped.game, Color.BLUE)
    save_stats(game_stats, episode, abs(loss),
               "reinforce_1v1_better_reward.json",
               total_reward=total_reward, loss_is_mean=True)

    if episode % (EPISODES // 5) == 0 and episode != 0:
        save_model(agent, f"reinforce_1v1_better_reward/agent_episode_{episode}.pt")

env.close()
