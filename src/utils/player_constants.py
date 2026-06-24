from models.dqn import DQN
from models.ppo import PPOActor
from models.reinforce import REINFORCEAgent

from catanatron.models.player import RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.value import ValueFunctionPlayer



cpu_players = {
    "RandomPlayer": RandomPlayer,
    "Weighted Random Player" : WeightedRandomPlayer,
    "Victory Point Player" : VictoryPointPlayer,
    "Alpha Beta Player" : AlphaBetaPlayer,
    #"Same Turn Alpha Beta Player" : SameTurnAlphaBetaPlayer, # idk what those two do so I don't include them in results player does so I 
    #"Greedy Playouts Player" : GreedyPlayoutsPlayer,
    "MCTS Player" : MCTSPlayer,
    "Value Function Player" : ValueFunctionPlayer,
}

model_players = {
    "D3QN (Dueling, Double)" : {
        "model_type": DQN,
        "save_path": "d3qn_smaller_stats_1v1_better_reward/dqn_episode_12000.pt",
        "neurons": 512,
        "comment": "Dueling + Double DQN, small network. Trained 12K episodes vs WeightedRandom with shaped reward.",
    },
    "D2QN Big Selfplay (Dueling)" : {
        "model_type": DQN,
        "save_path": "dqn_bigger_selfplay/dqn_episode_15000.pt",
        "neurons": 1024,
        "comment": "Dueling DQN, big network. Trained 15K episodes via self-play (opponent synced every 100 episodes).",
    },
    "D2QN Small Selfplay (Dueling)" : {
        "model_type": DQN,
        "save_path": "dqn_selfplay/dqn_episode_9000.pt",
        "neurons": 512,
        "comment": "Dueling DQN, small network. Trained 9K episodes via self-play (opponent synced every 100 episodes).",
    },
    "D2QN Small (Dueling)" : {
        "model_type": DQN,
        "save_path": "dqn_smaller_stats_dueling_1v1_better_reward/dqn_episode_15000.pt",
        "neurons": 512,
        "comment": "Dueling DQN, small network. Trained 15K episodes vs WeightedRandom with shaped reward.",
    },
    "D2QN Big (Dueling)" : {
        "model_type": DQN,
        "save_path": "dqn_bigger_stats_dueling_1v1_better_reward/dqn_episode_4000.pt",
        "neurons": 1024,
        "comment": "Dueling DQN, big network. Only 4K episodes — undertrained vs WeightedRandom, needs more training.",
    },
    "PPO DefaultR Big" : {
        "model_type": PPOActor,
        "save_path": "ppo_bigger_1v1_default_reward/actor_episode_15000.pt",
        "neurons": 1024,
        "comment": "PPO Actor, big network. Trained 10K episodes with default catanatron reward vs WeightedRandom. Needs more training.",
    },
    "PPO BetterR Big" : {
        "model_type": PPOActor,
        "save_path": "ppo_bigger_1v1_better_reward/actor_episode_18000.pt",
        "neurons": 1024,
        "comment": "PPO Actor, big network. Trained 18K episodes with shaped reward vs WeightedRandom.",
    },
    "PPO BetterR Small" : {
        "model_type": PPOActor,
        "save_path": "ppo_smaller_1v1_better_reward/actor_episode_15000.pt",
        "neurons": 512,
        "comment": "PPO Actor, small network. Trained 15K episodes with shaped reward vs WeightedRandom.",
    },
    # "REINFORCE" : { # REINFORCE IS A weak model so don't include him
    #     "model_type": REINFORCEAgent,
    #     "save_path": "reinforce_1v1_better_reward/agent_episode_16000.pt",
    #     "neurons": 1024,
    #     "comment": "Vanilla REINFORCE (Monte Carlo policy gradient), small network. Trained 16K episodes with shaped reward vs WeightedRandom.",
    # },
}