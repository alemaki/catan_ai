import os
import json
import torch
import gymnasium
import random
import catanatron.gym
from collections import namedtuple, deque
from catanatron import Game, Color, RESOURCES
from catanatron.features import feature_extractors
from catanatron.players.weighted_random import WeightedRandomPlayer, Player
from catanatron.state import PLAYER_INITIAL_STATE
from catanatron.state_functions import player_key
from catanatron.models.enums import DEVELOPMENT_CARDS
from utils.constants import *

feature_index_map_ref: dict | None = None
game_ref: Game | None = None

def create_players_env(reward_function=None, enemies: list[Player] = []) -> gymnasium.Env:
    if len(enemies) == 0:
        raise "should have some enemies"
    config = {"enemies": enemies}
    if reward_function is not None:
        config["reward_function"] = reward_function
    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config=config,
    )
    return env

def create_random_players_env(reward_function=None, num_enemies: int = 3) -> gymnasium.Env:
    colors = [Color.RED, Color.WHITE, Color.ORANGE]
    enemies = [WeightedRandomPlayer(c) for c in colors[:num_enemies]]
    return create_players_env(reward_function=reward_function, enemies=enemies)

def get_feature_index_map(game: Game, agent_color: Color) -> dict:
    assert isinstance(game, Game)

    # Remember game for future ref
    if (game_ref is None) or (game_ref != game):
        game_ref = game
        feature_index_map_ref = None

    # Return cache when applicable
    if not (feature_index_map_ref is None):
        return feature_index_map_ref

    # Build full feature dict exactly like env
    record: dict = {}
    for extractor in feature_extractors:
        record.update(extractor(game, agent_color))

    # Sort keys exactly like env
    keys: list = sorted(record.keys())
    feature_index_map_ref = {k: i for i, k in enumerate(keys)}

    return feature_index_map_ref

def create_game_stats(game: Game, agent_color = Color.BLUE) -> dict:
    key: str = player_key(game.state, agent_color)
    ps: dict = game.state.player_state
    result: dict = {}
    # Current turn
    result["game_turn"] = game.state.num_turns
    # If game ended
    result["finished"] = not (game.winning_color() is None)
    # If main player won
    result["mp_won"] = game.winning_color() == agent_color
    # Main player public victory point count
    result["mp_public_vps"] = ps[f"{key}_VICTORY_POINTS"]
    # Main player actual victory point count
    result["mp_actual_vps"] = ps[f"{key}_ACTUAL_VICTORY_POINTS"]
    # Main player city count
    result["mp_cities"] = PLAYER_INITIAL_STATE["CITIES_AVAILABLE"] - ps[f"{key}_CITIES_AVAILABLE"]
    # Main player settlement count
    result["mp_settlements"] = PLAYER_INITIAL_STATE["SETTLEMENTS_AVAILABLE"] - ps[f"{key}_SETTLEMENTS_AVAILABLE"]
    # Main player road count
    result["mp_roads"] = PLAYER_INITIAL_STATE["ROADS_AVAILABLE"] - ps[f"{key}_ROADS_AVAILABLE"]
    # If main player has longest road
    result["mp_has_road"] = ps[f"{key}_HAS_ROAD"]
    # If main player has largest army
    result["mp_has_army"] = ps[f"{key}_HAS_ARMY"]
    # Main player's resources left
    result["mp_total_resources"] = 0
    for resource in RESOURCES:
        result["mp_total_resources"] += ps[f"{key}_{resource}_IN_HAND"]

    # Main player's dev cards left
    result["mp_dev_cards_in_hand"] = 0
    for dev_card in DEVELOPMENT_CARDS:
        result["mp_dev_cards_in_hand"] += ps[f"{key}_{dev_card}_IN_HAND"]

    # Main player's dev cards used total
    result["mp_dev_cards_played"] = 0
    for dev_card in DEVELOPMENT_CARDS:
        result["mp_dev_cards_played"] += ps[f"{key}_PLAYED_{dev_card}"]

    return result

def save_stats(game_stats: dict, episode: int, total_loss: float, filepath: str,
               epsilon: float = 0.0, total_reward: float = 0.0, mean_max_q: float = 0.0, loss_is_mean = False):
    stats_path = os.path.join(STATS_SAVE_PATH, filepath)
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)

    entry = {
        "episode": episode,
        "epsilon": epsilon,
        "total_loss_for_game": total_loss,
        "total_reward": total_reward,
        "mean_max_q": mean_max_q,
        "stats": game_stats,
    }

    print("Loss for entire game (mean):", total_loss / (1 if loss_is_mean else game_stats["game_turn"]))

    with open(stats_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

def save_model(model, filepath: str):
    model_path = os.path.join(MODELS_SAVE_PATH, filepath)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)


def get_knight_position_value(game: Game, agent_color: Color):
    board = game.state.board
    robber_coord = board.robber_coordinate
    tile = board.map.land_tiles[robber_coord]
    node_ids = tile.nodes.values()

    value = 0
    for node_id in node_ids:
        building = board.buildings.get(node_id)
        if building is not None:
            color, building_type = building
            factor = ROBBER_HURT_CITY_FACTOR if building_type == "CITY" else 1
            if color != agent_color:
                value += ROBBER_HURT_ENEMY_VALUE * factor
            elif color == agent_color:
                value += ROBBER_HURT_SELF_VALUE * factor
    return value

def reward_function(game: Game, agent_color: Color):
    reward = 0
    ps = game.state.player_state
    key = player_key(game.state, agent_color)

    # VP gain (primary signal)
    vp_gain = ps[f"{key}_ACTUAL_VICTORY_POINTS"] - reward_function.last_points
    reward += max(0, vp_gain) * VP_REWARD
    reward_function.last_points = max(1, ps[f"{key}_ACTUAL_VICTORY_POINTS"])

    # Roads
    roads_built = PLAYER_INITIAL_STATE["ROADS_AVAILABLE"] - ps[f"{key}_ROADS_AVAILABLE"]
    reward += max(0, (roads_built - reward_function.last_roads) * ROAD_REWARD)
    reward_function.last_roads = roads_built

    # Cities
    cities_built = PLAYER_INITIAL_STATE["CITIES_AVAILABLE"] - ps[f"{key}_CITIES_AVAILABLE"]
    reward += max(0, (cities_built - reward_function.last_cities) * CITY_REWARD)
    reward_function.last_cities = cities_built

    # Settlements built beyond initial 2
    settlements_built = PLAYER_INITIAL_STATE["SETTLEMENTS_AVAILABLE"] - ps[f"{key}_SETTLEMENTS_AVAILABLE"]
    extra_settlements = max(0, settlements_built - 2)
    reward += max(0, (extra_settlements - reward_function.last_settlements) * SETTLEMENT_REWARD)
    reward_function.last_settlements = extra_settlements

    # Removed since the agent might be punished in weird situations
    # # Penalize road spam when cities/settlements are available
    # if roads_built >= (cities_built + settlements_built)*2.5 + 3:
    #     reward -= ROAD_SPAM_PENALTY

    if reward_function.was_moving_knight:
        reward += get_knight_position_value(game, agent_color) * ROBBER_REWARD
        reward_function.was_moving_knight = False
    elif game.state.is_moving_knight:
        reward_function.was_moving_knight = True


    # Win/loss
    if game.winning_color() is not None:
        reward += WIN_REWARD if game.winning_color() == agent_color else -WIN_REWARD

    return reward

def reset_reward_function():
    reward_function.last_points = 1
    reward_function.last_roads = 1
    reward_function.last_cities = 0
    reward_function.last_settlements = 0
    reward_function.was_moving_knight = False

reset_reward_function()


def valid_actions_to_mask(valid_actions, action_dim = MAX_ACTION_COUNT, device = "cpu"):
    mask = torch.full((action_dim,), -1e9, device=device, dtype=torch.float32)
    mask[valid_actions] = 0
    return mask


REINFORCEState = namedtuple("REINFORCEState",
                            ("observation",
                             "valid_actions_mask",
                             "action",
                             "reward",
                             "done"
                            )
                        )

PPOState = namedtuple("PPOState",
                            ("observation",
                             "valid_actions_mask",
                             "action",
                             "reward",
                             "value",
                             "log_prob",
                             "done"
                            )
                        )

DQNTransition = namedtuple("DQNTransition",
                            ("observation",
                             "valid_actions_mask",
                             "action",
                             "next_observation",
                             "next_valid_actions_mask",
                             "reward",
                             "done"
                            )
                        )

"""

"""
class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque([], maxlen = capacity)

    @staticmethod
    def create_dqn_transition(observation, valid_actions, action, next_observation, next_valid_actions, reward, done, device = "cpu"):
        valid_actions_mask = valid_actions_to_mask(valid_actions, device = device) # makes tesnor
        next_valid_actions_mask = valid_actions_to_mask(next_valid_actions, device = device)
        return DQNTransition(
            torch.tensor(observation, dtype=torch.float32, device=device),
            valid_actions_mask,
            torch.tensor(action, dtype=torch.long, device=device),
            torch.tensor(next_observation, dtype=torch.float32, device=device),
            next_valid_actions_mask,
            torch.tensor(reward, dtype=torch.float32, device=device),
            torch.tensor(done, dtype=torch.float32, device=device),
        )

    @staticmethod
    def create_reinforce_state(observation, valid_actions, action, reward, done, device="cpu"):
        valid_actions_mask = valid_actions_to_mask(valid_actions, device=device)
        to_tensor = lambda x, dtype: x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=dtype, device=device)
        return REINFORCEState(
            to_tensor(observation, dtype=torch.float32),
            valid_actions_mask,
            to_tensor(action, dtype=torch.long),
            to_tensor(reward, dtype=torch.float32),
            to_tensor(done, dtype=torch.float32),
        )

    @staticmethod
    def create_ppo_state(observation, valid_actions, action, reward, value, log_prob, done, device="cpu"):
        valid_actions_mask = valid_actions_to_mask(valid_actions, device=device)
        to_tensor = lambda x, dtype: x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=dtype, device=device)
        return PPOState(
            to_tensor(observation, dtype=torch.float32),
            valid_actions_mask,
            to_tensor(action, dtype=torch.long),
            to_tensor(reward, dtype=torch.float32),
            to_tensor(value, dtype=torch.float32),
            to_tensor(log_prob, dtype=torch.float32),
            to_tensor(done, dtype=torch.float32),
        )

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get_all(self):
        return list(self.memory)

    def clear(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)
