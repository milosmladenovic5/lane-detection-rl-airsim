from airsim_env import AirSimEnv

import gym
import numpy as np

from baselines.ddpg.policies import MlpPolicy
from baselines.common.vec_env import DummyVecEnv
from baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from baselines import DDPG



def main():
    steering_angles = np.array([-0.7, -0.5, -0.25, 0.0, 0.25, 0.5, 0.7])
    env = AirSimGym(continuous=False, off_road_dist=2.9, max_speed=4.5, scale_reward=True, steering_angles=steering_angles)
    env = DummyVecEnv([lambda: env])

    # the noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    model = DDPG(MlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise)
    model.learn(total_timesteps=40000)

    del model # remove to demonstrate saving and loading

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        #env.render()


if __name__ == "__main__":
    main()