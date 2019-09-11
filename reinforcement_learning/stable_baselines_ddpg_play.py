"""
Envs. & models to be loaded in pairs for parameter compatibility

------------------------------------------------------------------------------
env = AirSimGym("../data/lm/lm_7_5.txt", continious=True, scale_reward=True)
model = DDPG.load("./models/ddpg_final_8hrs.pkl", env=env)
------------------------------------------------------------------------------
"""


import numpy as np

from airsim_env import AirSimGym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DDPG

if __name__ == "__main__":
    env = AirSimGym(continuous=True, scale_reward=True)
    env = DummyVecEnv([lambda: env])
    model = DDPG.load("./models/best_model.pkl", env=env)

    for i in range(10): # play for 10 episodes
        done = False
        total_reward = 0
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            total_reward += rewards[0]

        print(f"Reward for episode {i} = {total_reward}.")