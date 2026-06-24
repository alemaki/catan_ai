import torch
from catanatron import Color
from utils.constants import *
from utils.utils import *
from utils.model_player import ModelPlayer
from models.dqn import DQN, dqn_run, DQN_TRAINING_CONFIG

NEURONS = 512
MEMORY_SIZE = 100_000
EPISODES = 15_000
N_STEPS = 1
OPPONENT_SYNC_EVERY = 100

CHECKPOINT = "dqn_bigger_stats_dueling_1v1_better_reward/dqn_episode_4000.pt"
STATS_FILE = "saved_self_play/dqn_bigger_selfplay.json"
REAL_STATS_FILE = "dqn_bigger_selfplay.json"
SAVE_PREFIX = "dqn_bigger_selfplay"

tmp_env = create_random_players_env(num_enemies=1)
obs, _ = tmp_env.reset()
obs_size = obs.shape[0]
tmp_env.close()

opponent_net = DQN(obs_size, MAX_ACTION_COUNT, neurons=NEURONS).to(device)
opponent_net.load_state_dict(torch.load(os.path.join(MODELS_SAVE_PATH, CHECKPOINT), map_location=device))
opponent_net.eval()

opponent_player = ModelPlayer(opponent_net, Color.RED, device=device)


def log_games_with_random(model: DQN, number_of_games: int = OPPONENT_SYNC_EVERY, starting_episode: int = 0):
    """Evaluate model against random opponents and log stats."""
    eval_env = create_random_players_env(num_enemies=1, reward_function=reward_function)

    for episode in range(number_of_games + 1):
        observation, info = eval_env.reset()
        reset_reward_function()
        done = False
        prev_observation = observation
        total_reward = 0.0

        while not done:
            with torch.no_grad():
                action = model.select_action(prev_observation, info["valid_actions"], device=device, epsilon=0)

            observation, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            total_reward += reward
            prev_observation = observation

        game_stats = create_game_stats(eval_env.unwrapped.game, Color.BLUE)
        save_stats(game_stats, episode + starting_episode, 0.0, REAL_STATS_FILE,
                   epsilon=0.0, total_reward=total_reward, mean_max_q=0.0)

    eval_env.close()


def on_episode_end(episode, policy_net, target_net, memory, epsilon):
    if episode % OPPONENT_SYNC_EVERY == 0 and episode != 0:
        opponent_net.load_state_dict(policy_net.state_dict())
        opponent_net.eval()
        print(f"[episode {episode}] opponent synced to current policy")
        log_games_with_random(policy_net, number_of_games=OPPONENT_SYNC_EVERY,
                              starting_episode=episode - OPPONENT_SYNC_EVERY)


config = {
    **DQN_TRAINING_CONFIG,
    "reward_function"       : reward_function,
    "load_saved_model"      : CHECKPOINT,
    "starting_episode"      : 0,
    "ending_episode"        : EPISODES,
    "epsilon_start"         : 0.1,
    "eps_min"               : 0.001,
    "eps_steps_decay"       : 500_000,
    "neurons"               : NEURONS,
    "memory"                : MEMORY_SIZE,
    "n_steps"               : N_STEPS,
    "learning_rate"         : 3e-4,
    "save_stats_name"       : STATS_FILE,
    "save_model_folder"     : SAVE_PREFIX,
    "save_factor"           : 5,
    "enemies"               : [opponent_player],
    "on_episode_end"        : on_episode_end
}

dqn_run(config)
