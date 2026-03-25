import random
import gymnasium
import catanatron.gym
from catanatron import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
from utils.constants import device

from catanatron.features import player_features

env = gymnasium.make(
    "catanatron/Catanatron-v0",
    config={
        "enemies": [
            WeightedRandomPlayer(Color.RED),
            WeightedRandomPlayer(Color.ORANGE),
            WeightedRandomPlayer(Color.WHITE),
        ],
    },
)

def get_feature_index_map(env):
    # Access internal game (hacky but works)
    game = env.unwrapped.game

    # Rebuild feature dict manually (using same functions)
    features = {}
    features.update(player_features(game, Color.BLUE))
    # TODO: also include other feature extractors if needed

    keys = sorted(features.keys())

    return {k: i for i, k in enumerate(keys)}

observation, info = env.reset()

for _ in range(5000):
    action = random.choice(info["valid_actions"])

    observation, reward, terminated, truncated, info = env.step(action) # CHECKME: reward is it good?
    done = terminated or truncated
    if done:
        break
        observation, info = env.reset()

feature_map = get_feature_index_map(env)

vp_index = feature_map["P0_ACTUAL_VPS"]
pvp_index = feature_map["P0_PUBLIC_VPS"]
print(vp_index)
print(pvp_index)
print(feature_map["P0_SETTLEMENTS_LEFT"])
print(feature_map["P0_ROADS_LEFT"])
print(observation[vp_index])
print(observation[pvp_index])
print(env.observation_space)
print(observation.shape)
print(observation)
print(info)

env.close()