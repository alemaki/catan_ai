import torch
from collections import namedtuple

# (26.03.2026) https://catanatron.readthedocs.io/en/latest/catanatron.gym.envs.html#catanatron.gym.envs.catanatron.gym.CatanatronEnv.action_space
# Integers from the [0, 327]
MAX_ACTIONS = 327

MODELS_SAVE_PATH = "./models/saved/"
STATS_SAVE_PATH = "./models/stats/"

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
