"""
Envs. & models to be loaded in pairs for parameter compatibility

------------------------------------------------------------------------------------------------------------------------------
steering_angles = np.array([-0.7, -0.25, 0.0, 0.25, 0.7])
env = AirSimGym("../data/lm/lm_7_5.txt", continious=False, off_road_dist=3.9, max_speed=8.0, scale_reward=True, steering_angles=steering_angles)
model = DQN.load("./models/best_model_dqn.pkl", env=env)
------------------------------------------------------------------------------------------------------------------------------
"""

import numpy as np

from airsim_env import AirSimGym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN

if __name__ == "__main__":
    steering_angles = np.array([-0.7, -0.25, 0.0, 0.25, 0.7])
    env = AirSimGym(continuous=False, off_road_dist=3.9, max_speed=8.0, scale_reward=True, steering_angles=steering_angles)
    env = DummyVecEnv([lambda: env])

    model = DQN.load("./models/best_model_dqn.pkl", env=env)
    
    for i in range(10): # play for 10 episodes
        done = False
        total_reward = 0
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            total_reward += rewards[0]

        print(f"Reward for episode {i} = {total_reward}.")
        