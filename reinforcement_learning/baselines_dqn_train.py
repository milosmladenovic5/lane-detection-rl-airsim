import gym
from baselines import deepq
from airsim_env import AirSimGym
import numpy as np

def callback(lcl, _glb):
    # stop training if reward exceeds 199
    #is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return False

def main():
    steering_angles = np.array([-0.7, -0.5, -0.25, 0.0, 0.25, 0.5, 0.7])
    env = AirSimGym(continuous=False, off_road_dist=2.9, max_speed=3.6, scale_reward=True, steering_angles=steering_angles)
    act = deepq.learn(
        env,
        network="cnn",
        lr=1e-3,
        total_timesteps=300000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=1,
        callback=callback
        #,load_path='airsim_dqn_test_model.pkl'
    )
    print("Saving model to airsim_dqn_test_model.pkl")
    act.save("airsim_dqn_test_model.pkl")

if __name__ == "__main__":
    main()
