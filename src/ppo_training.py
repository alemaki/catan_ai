import torch
from utils.constants import *
from utils.utils import *
from models.ppo import PPOActor, PPOCritic, ppo_update

LEARNING_RATE = 9e-4
ROLLOUT_CAPACITY = 4096
EPISODES = 15_000

env = create_random_players_env(num_enemies=1, reward_function = reward_function)
observation, _ = env.reset()

actor  = PPOActor(observation.shape[0], MAX_ACTION_COUNT).to(device)
critic = PPOCritic(observation.shape[0]).to(device)

actor_optimizer  = torch.optim.AdamW(actor.parameters(),  lr=LEARNING_RATE)
critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=LEARNING_RATE)

memory = ReplayMemory(ROLLOUT_CAPACITY)

actor.train()
critic.train()

for episode in range(EPISODES + 1):
    observation, info = env.reset()
    reset_reward_function()
    done = False
    total_reward = 0.0

    while not done:
        action, log_prob = actor.select_training_action(observation, info["valid_actions"], device=device)

        with torch.no_grad():
            obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
            value = critic(obs_tensor).squeeze()

        next_observation, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        memory.push(ReplayMemory.create_ppo_state(
            observation,
            info["valid_actions"],
            action,
            reward,
            value,
            log_prob,
            done,
            device=device,
        ))

        observation = next_observation
        info = next_info

    actor_loss, critic_loss = ppo_update(
        actor, critic, actor_optimizer, critic_optimizer, memory, device=device
    )
    memory.clear()

    game_stats = create_game_stats(env.unwrapped.game, Color.BLUE)

    save_stats(game_stats, episode, abs(actor_loss) + abs(critic_loss), "ppo_smaller_stats_1v1_better_reward.json",
               total_reward=total_reward)

    if episode % (EPISODES // 5) == 0 and episode != 0:
        save_model(actor,  f"ppo_smaller_1v1_better_reward/actor_episode_{episode}.pt")
        save_model(critic, f"ppo_smaller_1v1_better_reward/critic_episode_{episode}.pt")

env.close()
