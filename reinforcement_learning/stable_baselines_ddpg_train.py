import gym
import numpy as np

from airsim_env import AirSimGym
from stable_baselines.ddpg.policies import LnCnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DDPG
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from datetime import datetime

VER_NO = 1
best_mean_reward, n_steps = -np.inf, 0
log_dir = "./log_dir/"


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN and others) or after n steps (see ACER or PPO2)
    :params _locals: (dict)
            _globals: (dict)
    """
    global n_steps, best_mean_reward
    # Print stats every 1000 calls
    if (n_steps + 1) % 1000 == 0:
        # Evaluate policy performance
        x, y = ts2xy(load_results(log_dir), "timesteps")
        if len(x) > 0:
            mean_reward = np.mean(y[-100:])
            print(x[-1], "timesteps")
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals["self"].save("./models/best_model.pkl")
            
            print("-" * 90)
    n_steps += 1
    return True

N_ROLLOUT_STEPS = 200

if __name__ == "__main__":

    env = AirSimGym(continuous=True, scale_reward=True, pause_after=N_ROLLOUT_STEPS)
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    model = DDPG(LnCnnPolicy,\
                 env,\
                 verbose=1,
                 buffer_size=5000,\
                 random_exploration=0.015,\
                 tensorboard_log="./tensorboard",\
                 actor_lr=0.0009,\
                 critic_lr=0.001,
                 nb_rollout_steps=N_ROLLOUT_STEPS)
    
    start_date = datetime.now()
    model.learn(total_timesteps=25000, callback=callback)
    end_date = datetime.now()
    hours = int((end_date - start_date).total_seconds()) // 3600

    model.save(f"./models/ddpg_final_ver{VER_NO}_{hours}hrs.pkl")
