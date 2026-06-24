from utils.constants import *
from utils.utils import *
from models.dqn import dqn_run, DQN_TRAINING_CONFIG

STARTING_EPISODE = 6001
_EPS_MIN = 0.001
_EPS_STEPS = 1_000_000
_raw_decay = (_EPS_MIN / 1.0) ** (1 / _EPS_STEPS)
starting_epsilon = max(_EPS_MIN, 1.0 * (_raw_decay ** (STARTING_EPISODE * 200)))

config = {
    **DQN_TRAINING_CONFIG,
    "reward_function"   : reward_function,
    "load_saved_model"  : "d3qn_smaller_stats_1v1_better_reward/dqn_episode_6000.pt",
    "starting_episode"  : STARTING_EPISODE,
    "ending_episode"    : 15_000,
    "epsilon_start"     : starting_epsilon,
    "eps_min"           : _EPS_MIN,
    "eps_steps_decay"   : _EPS_STEPS,
    "save_stats_name"   : "d3qn_smaller_stats_1v1_better_reward.json",
    "save_model_folder" : "d3qn_smaller_stats_1v1_better_reward",
    "enemies"           : [WeightedRandomPlayer(Color.RED)],
}

dqn_run(config)
