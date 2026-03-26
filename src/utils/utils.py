
from catanatron.features import feature_extractors
from catanatron import Game, Color

feature_index_map_ref: dict or None = None
game_ref: Game or None = None

def get_feature_index_map(game: Game) -> int:
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