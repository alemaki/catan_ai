from torch import Tensor
from catanatron import Game
from catanatron.features import create_sample, get_feature_ordering
from catanatron.models.player import Player

class ActionSelectableModel():
    def select_action(self, observation: list, valid_actions: list, device: str = "cpu") -> int:
        raise NotImplementedError

class ModelPlayer(Player):
    """
    Player that can use a model to predict a playable action.
    """
    def __init__(self, model: ActionSelectableModel, color, is_bot=True, device="cpu"):
        super().__init__(color, is_bot)
        self.model = model
        self.device = device
        self.features = get_feature_ordering(num_players=2, map_type="BASE") # Always expect the 1v1, TODO: fix later

    def set_model(self, model):
        self.model = model

    def decide(self, game: Game, playable_actions):
        observation = self._get_observation(game)
        return self.model.select_action(observation, playable_actions, device=self.device)
    
    def _get_observation(self, game: Game) -> Tensor:
        sample = create_sample(game, self.color)
        return Tensor([float(sample[i]) for i in self.features])