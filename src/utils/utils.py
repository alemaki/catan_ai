
import os 
import json
import torch
from catanatron.features import feature_extractors
from catanatron import Game, Color, RESOURCES
from catanatron.state import PLAYER_INITIAL_STATE
from catanatron.models.enums import DEVELOPMENT_CARDS
from utils.constants import STATS_SAVE_PATH, MODELS_SAVE_PATH

feature_index_map_ref: dict | None = None
game_ref: Game | None = None

def get_feature_index_map(game: Game) -> dict:
    assert (game is Game)

    # Remember game for future ref
    if (game_ref is None) or (game_ref != game):
        game_ref = game
        feature_index_map_ref = None

    # Return cache when applicable
    if not (feature_index_map_ref is None):
        return feature_index_map_ref

    p0_color: Color = game.state.colors[0]
    # Build full feature dict exactly like env
    record: dict = {}
    for extractor in feature_extractors:
        record.update(extractor(game, p0_color))

    # Sort keys exactly like env
    keys: list = sorted(record.keys())
    feature_index_map_ref = {k: i for i, k in enumerate(keys)}

    return feature_index_map_ref

def create_game_stats(game: Game) -> dict:
    result: dict = {}
    # Current turn
    result["game_turn"] = game.state.num_turns
    # If game ended
    result["finished"] = not (game.winning_color() is None)
    # If main player won
    result["mp_won"] = game.winning_color() == game.state.colors[0]
    # Main player public victory point count
    result["mp_public_vps"] = game.state.player_state["P0_VICTORY_POINTS"]
    # Main player actual victory point count
    result["mp_actual_vps"] = game.state.player_state["P0_ACTUAL_VICTORY_POINTS"]
    # Main player city count
    result["mp_cities"] = PLAYER_INITIAL_STATE["CITIES_AVAILABLE"] - game.state.player_state["P0_CITIES_AVAILABLE"] # TODO: find a better way to get these
    # Main player settlement count
    result["mp_settlements"] = PLAYER_INITIAL_STATE["SETTLEMENTS_AVAILABLE"] - game.state.player_state["P0_SETTLEMENTS_AVAILABLE"] # TODO: find a better way to get these
    # Main player road count
    result["mp_roads"] = PLAYER_INITIAL_STATE["ROADS_AVAILABLE"] - game.state.player_state["P0_ROADS_AVAILABLE"] # TODO: find a better way to get these
    # If main player has longest road
    result["mp_has_road"] = game.state.player_state["P0_HAS_ROAD"]
    # If main player has largest army
    result["mp_has_army"] = game.state.player_state["P0_HAS_ARMY"]
    # Main player's resources left
    result["mp_total_resources"] = 0
    for resource in RESOURCES:
        result["mp_total_resources"] += game.state.player_state[f"P0_{resource}_IN_HAND"]

    # Main player's dev cards left
    result["mp_dev_cards_in_hand"] = 0
    for dev_card in DEVELOPMENT_CARDS:
        result["mp_dev_cards_in_hand"] += game.state.player_state[f"P0_{dev_card}_IN_HAND"]

    # Main player's dev cards used total
    result["mp_dev_cards_played"] = 0
    for dev_card in DEVELOPMENT_CARDS:
        result["mp_dev_cards_played"] += game.state.player_state[f"P0_PLAYED_{dev_card}"]
    
    return result

def save_stats(game_stats, episode, filepath: str):
    stats_path = os.path.join(STATS_SAVE_PATH, filepath)
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)

    entry = {
        "episode": episode,
        "stats": game_stats,
    }

    with open(stats_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

def save_model(model, filepath: str):
    model_path = os.path.join(MODELS_SAVE_PATH, filepath)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)