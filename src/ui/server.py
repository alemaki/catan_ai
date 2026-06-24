import os
import sys

# Ensure src/ is on the path regardless of where this file lives, retarded language
SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
os.chdir(SRC)

import torch
from flask import jsonify
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.features import get_feature_ordering
import catanatron.web.api as catanapi
from catanatron.web import create_app

from utils.model_player import ModelPlayer
from utils.constants import MODELS_SAVE_PATH, MAX_ACTION_COUNT, device
from utils.player_constants import model_players

OBSERVATION_SHAPE = len(get_feature_ordering(num_players=2, map_type="BASE"))


def load_model_player(model_name: str, color: Color) -> ModelPlayer:
    config = model_players[model_name]
    model = config["model_type"](
        OBSERVATION_SHAPE, MAX_ACTION_COUNT, neurons=config["neurons"]
    ).to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(MODELS_SAVE_PATH, config["save_path"]),
            map_location=device,
        )
    )
    model.eval()
    return ModelPlayer(model, color, is_bot=True, device=device)


def extended_player_factory(player_key):
    type_str, color = player_key
    if type_str == "CATANATRON":
        return AlphaBetaPlayer(color, 2, True)
    elif type_str == "RANDOM":
        return RandomPlayer(color)
    elif type_str == "HUMAN":
        return ValueFunctionPlayer(color, is_bot=False)
    elif type_str.startswith("MODEL:"):
        model_name = type_str[len("MODEL:"):]
        return load_model_player(model_name, color)
    else:
        raise ValueError(f"Unknown player type: {type_str!r}")


catanapi.player_factory = extended_player_factory

app = create_app()


@app.route("/api/models")
def get_models_endpoint():
    return jsonify(list(model_players.keys()))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
