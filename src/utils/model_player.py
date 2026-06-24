from torch import Tensor
from catanatron import Game, Action
from catanatron.features import create_sample, get_feature_ordering
from catanatron.models.player import Player
from catanatron.gym.envs.catanatron_env import to_action_space

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

    def get_valid_actions(self, playable_actions)-> list[int]:
        return list(map(to_action_space, playable_actions))

    def decide(self, game: Game, playable_actions: list[Action]) -> Action:
        observation = self.get_observation(game)
        action_map = {
            to_action_space(action): action
            for action in playable_actions
        }
        chosen_action_id = self.model.select_action(
            observation,
            list(action_map.keys()),
            device=self.device,
        )
        # Paranoia
        if chosen_action_id not in action_map:
            raise ValueError(
                f"Model selected invalid action {chosen_action_id}. "
                f"Valid actions are {list(action_map.keys())}"
            )

        return action_map[chosen_action_id]
    
    def get_observation(self, game: Game) -> Tensor:
        sample = create_sample(game, self.color)
        return Tensor([float(sample[i]) for i in self.features])