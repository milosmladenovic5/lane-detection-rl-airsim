import gym
from baselines import deepq
from airsim_env import AirSimGym
import numpy as np

def callback(lcl, _glb):
    # stop training if reward exceeds 199
    #is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return False

def main():
    steering_angles = np.array([-0.65, -0.5, -0.25, -0.1, 0.0, 1.0, 0.25, 0.5, 0.65])
    env = AirSimGym(continuous=False, off_road_dist=2.9, max_speed=4.5, scale_reward=True, steering_angles=steering_angles)
    #model = deepq.load_act("airsim_dqn_test_model.pkl")
    model = deepq.learn(
        env,
        network="cnn",
        lr=0.0011,
        total_timesteps=0,
        buffer_size=65000,
        exploration_fraction=0.1,
        exploration_final_eps=0.015,
        print_freq=1,
        callback=callback
        ,load_path='airsim_dqn_test_model_2019189.pkl'
    )

    print("Loading model from airsim_dqn_test_model.pkl")
    
    for i in range (0, 18):
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(model(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)

if __name__ == "__main__":
    main()
