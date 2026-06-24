from utils.constants import *
from utils.utils import *
from models.ppo import ppo_run, PPO_TRAINING_CONFIG

config = {
    **PPO_TRAINING_CONFIG,
    "reward_function"   : reward_function,
    "ending_episode"    : 18_000,
    "save_stats_name"   : "ppo_bigger_stats_1v1_better_reward.json",
    "save_model_folder" : "ppo_bigger_1v1_better_reward",
    "enemies"           : [WeightedRandomPlayer(Color.RED)],
}

ppo_run(config)
