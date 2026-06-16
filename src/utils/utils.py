import os
import json
import torch
import gymnasium
from catanatron.features import feature_extractors
from catanatron import Game, Color, RESOURCES
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.state import PLAYER_INITIAL_STATE
from catanatron.state_functions import player_key
from catanatron.models.enums import DEVELOPMENT_CARDS
from utils.constants import STATS_SAVE_PATH, MODELS_SAVE_PATH

feature_index_map_ref: dict | None = None
game_ref: Game | None = None

def create_random_players_env(reward_function = None) -> gymnasium.Env:
    config = {
            "enemies": [
                WeightedRandomPlayer(Color.RED),
                WeightedRandomPlayer(Color.WHITE),
                WeightedRandomPlayer(Color.ORANGE),
            ],
        }
    if reward_function is not None:
        config["reward_function"] = reward_function

    env = gymnasium.make(
        "catanatron/Catanatron-v0",
        config = config,
    )
    return env

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

def create_game_stats(game: Game) -> dict:
    agent_color: Color = game.state.colors[0]
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

def save_stats(game_stats: dict, episode: int, total_loss: float, filepath: str):
    stats_path = os.path.join(STATS_SAVE_PATH, filepath)
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)

    entry = {
        "episode": episode,
        "total_loss_for_game": total_loss,
        "stats": game_stats,
    }

    print("Loss for entire game (mean):", total_loss / game_stats["game_turn"])

    with open(stats_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

def save_model(model, filepath: str):
    model_path = os.path.join(MODELS_SAVE_PATH, filepath)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)