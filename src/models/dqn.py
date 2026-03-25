from collections import namedtuple, deque

Transition = namedtuple("Transition",
                        ("state", "action", "next_state", "reward", "done"))

"""
    Create transition for the replay memory.
    args:
    prev_observation,
    action,
    observation,
    done
    N - number of players
"""

def get_vp(observation, N) -> int:
    return observation[207 + 181*N] # MUST check if this is true

def get_vp_gain(prev_observation, observation, N) -> int:
    return get_vp(observation, N) - get_vp(prev_observation, N)

def create_transition(prev_observation, action, observation, done, N) -> Transition:
    reward = 0
    # VP gain. Can happen with settlements, castles, longest roads, biggest army, or just vp gain from casino
    reward += get_vp_gain(prev_observation, observation, N) * 1.0
    # Win/loss
    if done:
        if get_vp(observation, N) == 10:
            reward += 5
        else:
            reward -= 5

    return return Transition(prev_observation, action, observation, reward, done)

class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque([], maxlen = capacity)

    def push(self, transition: Transition):
        pass
