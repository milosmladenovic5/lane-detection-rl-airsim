import numpy as np

from airsim_env import AirSimGym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import CnnPolicy, LnCnnPolicy
from stable_baselines import DQN
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from datetime import datetime

VER_NO = 2
best_mean_reward, n_steps = -np.inf, 0
log_dir = "./log_dir/"


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
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
                _locals["self"].save("./models/best_model_dqn.pkl")
            
            print("-" * 90)
    n_steps += 1
    return True

if __name__ == "__main__":
    steering_angles = np.array([-0.7, -0.5, -0.25, 0.0, 0.25, 0.5, 0.7])
    env = AirSimGym(continuous=False, off_road_dist=2.9, max_speed=3.6, scale_reward=True, steering_angles=steering_angles)
    env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    model = DQN(LnCnnPolicy, \
                env,\
                verbose=1,\
                buffer_size=80000,\
                learning_rate=0.001,\
                train_freq=5,\
                batch_size=64,\
                tensorboard_log="./tensorboard",\
                checkpoint_path=log_dir,\
                exploration_fraction=0.63,\
                exploration_final_eps=0.1,\
                prioritized_replay=True)

    start_date = datetime.now()
    #model = DQN.load(log_dir + "best_model.pkl", env=env)
    model.learn(total_timesteps=500000, log_interval=200, callback=callback)
    end_date = datetime.now()
    hours = int((end_date - start_date).total_seconds()) // 3600
    model.save(f"./models/dqn_final_ver{VER_NO}_{hours}hrs.pkl")
