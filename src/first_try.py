import random
import gymnasium
import catanatron.gym

env = gymnasium.make("catanatron/Catanatron-v0")
observation, info = env.reset()
print(info)
print(observation)
for _ in range(1000):
    action = random.choice(info["valid_actions"])

    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    if done:
        observation, info = env.reset()
print(info)
print(observation)
env.close()