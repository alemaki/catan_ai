from utils.utils import *
from utils.constants import *
from utils.model_player import *
from catanatron.game import Game, TURNS_LIMIT
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.minimax import AlphaBetaPlayer, SameTurnAlphaBetaPlayer
from catanatron.players.playouts import GreedyPlayoutsPlayer
from catanatron.players.mcts import MCTSPlayer
from catanatron.players.value import ValueFunctionPlayer
from itertools import combinations
from collections import defaultdict
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter

from models.dqn import DQN
from models.ppo import PPOActor
from models.reinforce import REINFORCEAgent

PLAYOUT_GAMES = 1000

colors = [Color.BLUE, Color.RED]

cpu_players = {
    "Weighted Random Player" : WeightedRandomPlayer,
    "Victory Point Player" : VictoryPointPlayer,
    "Alpha Beta Player" : AlphaBetaPlayer,
    "Same Turn Alpha Beta Player" : SameTurnAlphaBetaPlayer,
    "Greedy Playouts Player" : GreedyPlayoutsPlayer,
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
        "save_path": "ppo_bigger_1v1_default_reward/actor_episode_10000.pt",
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
    "REINFORCE" : {
        "model_type": REINFORCEAgent,
        "save_path": "reinforce_1v1_better_reward/agent_episode_16000.pt",
        "neurons": 512,
        "comment": "Vanilla REINFORCE (Monte Carlo policy gradient), small network. Trained 16K episodes with shaped reward vs WeightedRandom.",
    },
}

env = create_random_players_env(reward_function=None, num_enemies=1)
observation, _ = env.reset()
env.close()

OBSERVATION_SHAPE = observation.shape[0]

def create_model_player(model_stats: dict, color: Color) -> ModelPlayer:
    model = model_stats["model_type"](OBSERVATION_SHAPE, MAX_ACTION_COUNT, neurons = model_stats["neurons"]).to(device)
    model.load_state_dict(torch.load(os.path.join(MODELS_SAVE_PATH, model_stats["save_path"]), map_location=device))
    model.eval()
    player = ModelPlayer(model, color, is_bot = True, device = device)
    return player

# Let the games begin
results = [

]

def play_games(player1: Player, player2: Player) -> tuple[int, int]:
    player1_wins = 0
    player2_wins = 0
    for i in range(PLAYOUT_GAMES):
        game: Game = Game(players = [player1, player2])
        game.play()
        while game.state.num_turns == TURNS_LIMIT:
            game: Game = Game(players = [player1, player2])
            game.play()
        if game.winning_color() == player1.color:
            player1_wins += 1
        else:
            player2_wins += 1

    return (player1_wins, player2_wins)

for (name1, name2) in combinations(model_players, 2):
    player1: ModelPlayer = create_model_player(model_players[name1], Color.BLUE)
    player2: ModelPlayer = create_model_player(model_players[name2], Color.RED)
    
    player1_wins, player2_wins = play_games(player1, player2)
    print(f"{name1} vs {name2} - {player1_wins} : {player2_wins}")

    # TODO!!: this is a bit ugly.
    # TODO: Also, should I log more information?
    results.append([name1, name2, player1_wins, player2_wins])


# TODO: some function to combine, not two fors?
for model_player_name in model_players:
    for cpu_player_name in cpu_players:
        player1: ModelPlayer = create_model_player(model_players[model_player_name], Color.BLUE)
        player2: ModelPlayer = cpu_players[cpu_player_name](Color.RED)

        player1_wins, player2_wins = play_games(player1, player2)
        print(f"{model_player_name} vs {cpu_player_name} - {player1_wins} : {player2_wins}")

        results.append([model_player_name, cpu_player_name, player1_wins, player2_wins])


for (name1, name2) in combinations(cpu_players, 2):
        player1: ModelPlayer = cpu_players[name1](Color.BLUE)
        player2: ModelPlayer = cpu_players[name2](Color.RED)

        player1_wins, player2_wins = play_games(player1, player2)
        print(f"{name1} vs {name2} - {player1_wins} : {player2_wins}")

        results.append([name1, name2, player1_wins, player2_wins])

# The code below is AI generated----------------------------------------------

def save_results_to_excel(results, filename="model_evaluation.xlsx"):
    all_model_names = list(model_players.keys())
    all_cpu_names   = list(cpu_players.keys())
    all_names       = all_model_names + all_cpu_names

    matchup = {}
    for name1, name2, w1, w2 in results:
        matchup[(name1, name2)] = (w1, w2)
        matchup[(name2, name1)] = (w2, w1)

    total_wins  = defaultdict(int)
    total_games = defaultdict(int)
    for name1, name2, w1, w2 in results:
        total_wins[name1]  += w1;  total_wins[name2]  += w2
        total_games[name1] += w1 + w2;  total_games[name2] += w1 + w2

    MODEL_FILL  = PatternFill("solid", fgColor="FFE699")  # gold
    HEADER_FILL = PatternFill("solid", fgColor="2F5496")  # dark blue
    header_font = Font(bold=True, color="FFFFFF")
    bold_font   = Font(bold=True)
    center      = Alignment(horizontal="center", vertical="center")

    wb = Workbook()

    # ── Sheet 1: Summary ──────────────────────────────────────────────────────
    ws = wb.active
    ws.title = "Summary"

    for col, h in enumerate(["Player", "Type", "Wins", "Losses", "Total Games", "Win %", "Comment"], 1):
        c = ws.cell(row=1, column=col, value=h)
        c.fill = HEADER_FILL;  c.font = header_font;  c.alignment = center

    for row, name in enumerate(all_names, 2):
        is_model = name in model_players
        fill     = MODEL_FILL if is_model else None
        w = total_wins[name];  g = total_games[name]
        comment  = model_players[name]["comment"] if is_model else ""
        row_data = [name, "Model" if is_model else "CPU", w, g - w, g,
                    round(w / g * 100, 1) if g else 0.0, comment]
        for col, val in enumerate(row_data, 1):
            c = ws.cell(row=row, column=col, value=val)
            if fill: c.fill = fill

    ws.column_dimensions["A"].width = 35
    ws.column_dimensions["B"].width = 10
    ws.column_dimensions["F"].width = 10
    ws.column_dimensions["G"].width = 45

    # ── Sheet 2: Matchup Matrix ───────────────────────────────────────────────
    ws2 = wb.create_sheet("Matchup Matrix")

    ws2.cell(row=1, column=1, value="vs →").font = bold_font
    for col, name in enumerate(all_names, 2):
        c = ws2.cell(row=1, column=col, value=name)
        c.fill = MODEL_FILL if name in model_players else HEADER_FILL
        c.font = Font(bold=True) if name in model_players else header_font
        c.alignment = Alignment(horizontal="center", textRotation=45)

    for row, row_name in enumerate(all_names, 2):
        is_model_row = row_name in model_players
        c = ws2.cell(row=row, column=1, value=row_name)
        c.font = bold_font
        if is_model_row: c.fill = MODEL_FILL

        for col, col_name in enumerate(all_names, 2):
            if row_name == col_name:
                ws2.cell(row=row, column=col, value="—").alignment = center
                continue
            key = (row_name, col_name)
            if key in matchup:
                w, l = matchup[key]
                total = w + l
                pct = round(w / total * 100, 1) if total else 0.0
                c = ws2.cell(row=row, column=col, value=f"{pct}%\n({w}-{l})")
                c.alignment = Alignment(horizontal="center", vertical="center", wrapText=True)
                if is_model_row: c.fill = MODEL_FILL
            else:
                ws2.cell(row=row, column=col, value="N/A").alignment = center

    ws2.column_dimensions["A"].width = 35
    ws2.row_dimensions[1].height = 90
    for col in range(2, len(all_names) + 2):
        ws2.column_dimensions[get_column_letter(col)].width = 14

    path = os.path.join(STATS_SAVE_PATH, filename)
    wb.save(path)
    print(f"Saved evaluation to {path}")


save_results_to_excel(results)