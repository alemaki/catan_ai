import torch
from collections import namedtuple

# (26.03.2026) https://catanatron.readthedocs.io/en/latest/catanatron.gym.envs.html#catanatron.gym.envs.catanatron.gym.CatanatronEnv.action_space
# Integers from the [0, 327]
MAX_ACTION = 327
MAX_ACTION_COUNT = 328

WIN_REWARD = 20
VP_REWARD = 3
CITY_REWARD = 1
ROAD_REWARD = 0.6

MODELS_SAVE_PATH = "./models/saved/"
STATS_SAVE_PATH = "./models/stats/"

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
