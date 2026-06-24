from utils.constants import *
from utils.utils import *
from models.ppo import ppo_run, PPO_TRAINING_CONFIG

config = {
    **PPO_TRAINING_CONFIG,
    "reward_function"   : None,
    "save_stats_name"   : "ppo_bigger_stats_1v1_default_reward.json",
    "save_model_folder" : "ppo_bigger_1v1_default_reward",
    "enemies"           : [WeightedRandomPlayer(Color.RED)],
}

ppo_run(config)
