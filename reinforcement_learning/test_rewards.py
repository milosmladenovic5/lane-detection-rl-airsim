from airsim_env import AirSimGym
import airsim
import time

if __name__ == "__main__":
    env = AirSimGym("../data/lm/lm_7_5.txt", api_control=False, scale_reward=True)
    while True:
        ob, reward, done, _ = env.step(None) # comment out first two lines in step
        print(reward, done)
        time.sleep(0.1)

