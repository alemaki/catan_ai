import random
import torch.nn
from utils.constants import *
from utils.utils import *
from utils.model_player import ActionSelectableModel

BATCH_SIZE = 64
GAMMA = 0.999
LAMBDA = 0.99

class PPOActor(torch.nn.Module, ActionSelectableModel):
    def __init__(self, observation_shape, actions_shape, neurons=512):
        super().__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(observation_shape, neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(neurons, neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(neurons, neurons//2),
            torch.nn.ReLU(),
        )
        self.advantage_stream = torch.nn.Linear(neurons//2, actions_shape)
        self.softmax = torch.nn.Softmax(dim=-1)

    """
    This forward is a bit worthless, since the advantage expects the valid_actions_mask
    """
    def forward(self, observation):
        x = self.linear(observation)
        advantage = self.advantage_stream(x)
        return self.softmax(advantage)

    def select_training_action(self, observation: list, valid_actions: list, device="cpu") -> tuple[int, torch.Tensor]:
        observation = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = self.linear(observation)
            logits = self.advantage_stream(logits).squeeze(0)

        mask = valid_actions_to_mask(valid_actions, logits.shape[0], device=device)
        logits = logits + mask

        probs = self.softmax(logits)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob
        
    """ This is used for the ModelPlayer. """
    def select_action(self, observation: list, valid_actions: list, device="cpu") -> tuple[int, torch.Tensor]:
        x, _ = self.select_training_action(observation, valid_actions, device=device)
        return x

class PPOCritic(torch.nn.Module):


    def __init__(self, observation_shape, neurons=512):
        super().__init__()
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(observation_shape, neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(neurons, neurons),
            torch.nn.ReLU(),
            torch.nn.Linear(neurons, neurons//2),
            torch.nn.ReLU(),
        )
        self.value = torch.nn.Linear(neurons//2, 1)

    """
    Called with either one element to determine next action, or a batch
    during optimization. Returns tensor([[left0exp,right0exp]...]).
    """
    def forward(self, observation):
        x = self.linear(observation)
        return self.value(x)
    
PPO_EPOCHS = 1
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.05

def compute_gae(memory: ReplayMemory, device="cpu"):
    all_transitions = memory.get_all()
    batch = PPOState(*zip(*all_transitions))

    rewards = torch.stack(batch.reward).to(device)
    values  = torch.stack(batch.value).to(device)
    dones   = torch.stack(batch.done).to(device)

    T = len(all_transitions)
    advantages = torch.zeros(T, device=device)

    gae = 0.0
    for t in reversed(range(T)):
        next_value = 0.0 if t == T - 1 else values[t + 1].item()
        mask = 1.0 - dones[t].item()
        delta = rewards[t] + GAMMA * next_value * mask - values[t]
        gae   = delta + GAMMA * LAMBDA * mask * gae
        advantages[t] = gae

    value_targets = advantages + values
    return advantages, value_targets

def ppo_update(actor: PPOActor, critic: PPOCritic, actor_optimizer, critic_optimizer, memory: ReplayMemory, device="cpu"):
    advantages, value_targets = compute_gae(memory, device)

    all_transitions = memory.get_all()
    batch = PPOState(*zip(*all_transitions))

    observations = torch.stack(batch.observation).to(device)
    actions = torch.stack(batch.action).to(device)
    old_log_probs = torch.stack(batch.log_prob).to(device)
    valid_masks = torch.stack(batch.valid_actions_mask).to(device)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    T = len(all_transitions)
    total_actor_loss = 0.0
    total_critic_loss = 0.0

    for _ in range(PPO_EPOCHS):
        order = torch.randperm(T)
        for start in range(0, T, BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]

            mb_obs          = observations[idx]
            mb_actions      = actions[idx]
            mb_old_log_prob = old_log_probs[idx]
            mb_masks        = valid_masks[idx]
            mb_advantages   = advantages[idx]
            mb_targets      = value_targets[idx]

            # recompute log probs and entropy under current actor
            logits = actor.advantage_stream(actor.linear(mb_obs)) + mb_masks
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(mb_actions)
            entropy = dist.entropy().mean()

            # clipped actor loss
            ratio = torch.exp(new_log_probs - mb_old_log_prob)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # critic loss
            new_values = critic(mb_obs).squeeze(-1)
            critic_loss = torch.nn.functional.mse_loss(new_values, mb_targets)

            loss = actor_loss + VALUE_COEF * critic_loss - ENTROPY_COEF * entropy

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            actor_optimizer.step()
            critic_optimizer.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

    return total_actor_loss, total_critic_loss


PPO_TRAINING_CONFIG = {
    "reward_function"   : None,
    "learning_rate"     : 9e-4,
    "actor_neurons"     : 1024,
    "critic_neurons"    : 1024,
    "rollout_capacity"  : 4096,
    "save_model_folder" : "",  # leave empty for no save
    "save_stats_name"   : "",  # leave empty for no save
    "starting_episode"  : 0,
    "ending_episode"    : 15_000,
    "save_factor"       : 5,
    "load_saved_actor"  : "",  # leave empty for no load
    "load_saved_critic" : "",  # leave empty for no load
    "enemies"           : [WeightedRandomPlayer(Color.RED)],
    "on_episode_end"    : None, # function to run on episode end, args: episode, actor, critic, memory
}


def ppo_run(ppo_config: dict):
    env = create_players_env(reward_function=ppo_config.get("reward_function", None), enemies=ppo_config.get("enemies", []))
    observation, _ = env.reset()

    actor  = PPOActor(observation.shape[0], MAX_ACTION_COUNT, ppo_config["actor_neurons"]).to(device)
    critic = PPOCritic(observation.shape[0], ppo_config["critic_neurons"]).to(device)

    if ppo_config.get("load_saved_actor", "") != "":
        path = os.path.join(MODELS_SAVE_PATH, ppo_config["load_saved_actor"])
        actor.load_state_dict(torch.load(path, map_location=device))
    if ppo_config.get("load_saved_critic", "") != "":
        path = os.path.join(MODELS_SAVE_PATH, ppo_config["load_saved_critic"])
        critic.load_state_dict(torch.load(path, map_location=device))

    actor_optimizer  = torch.optim.AdamW(actor.parameters(),  lr=ppo_config["learning_rate"])
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=ppo_config["learning_rate"])

    memory = ReplayMemory(ppo_config["rollout_capacity"])

    actor.train()
    critic.train()

    for episode in range(ppo_config["starting_episode"], ppo_config["ending_episode"] + 1):
        observation, info = env.reset()
        reset_reward_function()
        done = False
        total_reward = 0.0

        while not done:
            action, log_prob = actor.select_training_action(observation, info["valid_actions"], device=device)

            with torch.no_grad():
                obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(device)
                value = critic(obs_tensor).squeeze()

            next_observation, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            memory.push(ReplayMemory.create_ppo_state(
                observation,
                info["valid_actions"],
                action,
                reward,
                value,
                log_prob,
                done,
                device=device,
            ))

            observation = next_observation
            info = next_info

        actor_loss, critic_loss = ppo_update(
            actor, critic, actor_optimizer, critic_optimizer, memory, device=device
        )
        memory.clear()

        game_stats = create_game_stats(env.unwrapped.game, Color.BLUE)

        if ppo_config.get("save_stats_name", "") != "":
            save_stats(game_stats, episode, abs(actor_loss) + abs(critic_loss),
                       ppo_config["save_stats_name"], total_reward=total_reward)

        if ppo_config.get("save_model_folder", "") != "" and \
                episode % (ppo_config["ending_episode"] // ppo_config["save_factor"]) == 0 and \
                episode != 0:
            save_model(actor,  f"{ppo_config['save_model_folder']}/actor_episode_{episode}.pt")
            save_model(critic, f"{ppo_config['save_model_folder']}/critic_episode_{episode}.pt")

        if ppo_config.get("on_episode_end") is not None:
            ppo_config.get("on_episode_end")(episode, actor, critic, memory)

    env.close()
