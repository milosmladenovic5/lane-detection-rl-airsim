import sys
sys.path.insert(0, "../")

import airsim
import numpy as np
import math
import time
import gym
from gym import spaces
from math import sin, cos, sqrt, atan2, radians
from general_tools.road_lines import load_road_lines
from general_tools.utils import *
from scipy.spatial.distance import euclidean
import cv2

import warnings
warnings.simplefilter("ignore")

region_of_interest_vertices = [
    (0, 35),
    (0, 65),
    (255, 65),
    (35, 255),
    (205, 0)
]

TARGET_LATITUDE =  47.645778419398034
TARGET_LONGITUDE = -122.13657136870823


HEIGHT = 65
WIDTH  = 255
CENTER = np.array([WIDTH // 2, HEIGHT // 2])

MAX_DISTANCE_ALLOWED = 60

GRAY_DIFFERENCE_THRESHOLD = 25
LEFT_RIGHT_DIFF_TRESHOLD = 18

LEFT_PIXEL_DISTANCE = 50
RIGHT_PIXEL_DISTANCE = 50

def region_of_interest(img, vertices):
        # Define a blank matrix that matches the image height/width.
        mask = np.zeros_like(img)    # Retrieve the number of color channels of the image.
        #channel_count = img.shape[2]    # Create a match color with the same color channel counts.
        match_mask_color = 255 # (255,) * channel_count - set this when image is not grayscale

        # Fill inside the polygon
        cv2.fillPoly(mask, vertices, match_mask_color)

        # Returning the image only where mask pixels match
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

def brightness(img, val):
    assert val > 0
#   uint8 overflow handling ...
#   e: 222 + 50 = 16
    limit = 255 - val
    img[img > limit] = 255
    img[img < limit] += val

def val_diff_greater_than_threshold(threshold, first_val, second_val, third_val):
    diff_first_second = 0
    diff_second_third = 0
    diff_first_third = 0

    diff_first_second = first_val - second_val if first_val > second_val else second_val - first_val
    diff_second_third = second_val - third_val if second_val > third_val else third_val - second_val
    diff_first_third = first_val - third_val if first_val > third_val else third_val - first_val 
    
    return diff_first_second > threshold or diff_second_third > threshold or diff_first_third > threshold 

def diff_between_two_rgb_threshold(threshold, r1, g1, b1, r2, g2, b2):
    diff_r1_r2 = 0
    diff_g1_g2 = 0
    diff_b1_b2 = 0

    diff_r1_r2 = r1 - r2 if r1 > r2 else r2 - r1
    diff_g1_g2 = g1 - g2 if g1 > g2 else g2 - g1 
    diff_b1_b2 = b1 - b2 if b1 > b2 else b2 - b1

    return diff_r1_r2 > threshold or diff_g1_g2 > threshold or diff_b1_b2 > threshold

def get_distance_from_target(gp):
    R = 6373.0

    lat1 = radians(gp.latitude)
    lon1 = radians(gp.longitude)
    lat2 = radians(TARGET_LATITUDE)
    lon2 = radians(TARGET_LONGITUDE)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

def get_distance(gp1, gp2):
    R = 6373.0

    lat1 = radians(gp1.latitude)
    lon1 = radians(gp1.longitude)
    lat2 = radians(gp2.latitude)
    lon2 = radians(gp2.longitude)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


class AirSimGym(gym.Env):
    def __init__(self, crop_h1=70, crop_h2=135, crop_w1=0, crop_w2=255, n_channels_env=3,\
                 continuous=True, low=(-1.0, -1.0), high=(1.0, 1.0), action_map=(True, False), steering_angles=3, obs_is_image_only=True, sleep_after_action=0.02,\
                 hold_mid_reward_decay_rate=1.0, speed_reward_decay_rate=1.2, off_road_reward=-8.0, off_road_dist=4.2, crashed_reward=-20.0, stuck_reward=-25.0,\
                 goal_point_exists=True, goal=None, goal_reached_reward=25.0, goal_reached_dist=5.0, hold_mid_reward_mul=10.0, speed_reward_mul=7.0,\
                 max_speed=10.0, min_allowed_speed=1.7, steps_at_min_speed_allowed=55, scale_reward=False, api_control=True, pause_after=None, ip="", port=41451):
        """        
        :param crop_h1: (int) - lower bound of the height final image to be used in observation
        :param crop_h2: (int) - upper bound of the height final image to be used in observation
        :param crop_w1: (int) - lower bound of the width final image to be used in observation
        :param crop_w2: (int) - upper bound of the width final image to be used in observation
        :param n_channels_env: (int) - number of channels that env. img contains
        :param continious: (bool) - determines if the action space in continious or not
        :param low: (tuple) - lowest values continious actions can take between [-1., 1.] 
        :param high: (tuple) - highest values continious actions can take between [-1, 1]
        :param action_map: (tuple) - determines if actions should be mapped between [0, 1]
        :param steering_angles: (int or np.array) - steering angles to be used if the action space is descrete
        :param obs_is_image_only: (bool) - use only env. scene as the observation
        :param sleep_after_action: (float) - sleep time after action signal is sent to the AirSim simulator
        :param hold_mid_reward_decay_rate: (float) - decay rate of the reward for holding middle road
        :param speed_reward_decay_rate: (float) - decay rat eof the reward for going fast
        :param off_road_reward: (float) - reward for going of the road (ending the episode)
        :param off_road_dist: (float) - minimum distance from the nearest road line to be considered off road
        :param crashed_reward: (float) - reward for crashing
        :param stuck_reward: (float) - reward for getting stuck (or sitting in one place for too long)
        :param goal_point_exists: (bool) - determines if goal point is used for calculating the reward
        :param goal: (np.array) - coordinates of the goal point
        :param goal_reached_reward: (float) - reward for reaching the end goal (if it exists)
        :param goal_reached_dist: (float) - minimum distance from the goal to consider it reached
        :param hold_mid_reward_mul: (float) - hold mid reward multiplier
        :param speed_reward_mul: (float) - speed reward multiplier
        :param max_speed: (float) - maximum allowed speed
        :param min_allowed_speed: (float) - minimum allowed speed
        :param steps_at_min_speed_allowed: (int) - number of timesteps agent can spend at minimum allowed speed
        :param scale_reward: (bool) - determines whether the reward should be scaled
        :param pause_after: (int) - number of steps after which AirSim simulator should be paused
        :param ip: (str) - IP address of the AirSim simulator
        :param port: (int) - port of the AirSim simulator

        """
        super(type(self), self).__init__()
        
        assert min_allowed_speed > 0
        assert max_speed > min_allowed_speed
        assert len(low) >= 2
        assert len(high) >= 2
        assert len(action_map) >= 2
    
        self.continious = continuous
        self.low = low
        self.high = high
        self.crop_h1 = crop_h1
        self.crop_h2 = crop_h2
        self.crop_w1 = crop_w1
        self.crop_w2 = crop_w2
        self.n_channels_env = n_channels_env # this parameter referes for the number of channels of the ENV
        self.height = crop_h2 - crop_h1
        self.width = crop_w2 - crop_w1
        self.max_speed = max_speed
        self.min_allowed_speed = min_allowed_speed
        self.steps_at_min_speed_allowed = steps_at_min_speed_allowed
        self.steps_at_min_speed_count = 0
        self.sleep_after_action = sleep_after_action
        self.hold_mid_reward_decay_rate = hold_mid_reward_decay_rate
        self.speed_reward_decay_rate = speed_reward_decay_rate
        self.off_road_reward = off_road_reward
        self.off_road_dist = off_road_dist
        self.crashed_reward = crashed_reward
        self.stuck_reward = stuck_reward
        self.goal_reached_reward = goal_reached_reward
        self.goal_reached_dist = goal_reached_dist
        self.hold_mid_reward_mul = hold_mid_reward_mul
        self.speed_reward_mul = speed_reward_mul
        self.goal_point_exists = goal_point_exists
        self.scale_reward = scale_reward
        self.pause_after = pause_after
        self.timesteps = 0
        self.sim_paused = False

        self.client = airsim.CarClient(ip=ip, port=port)
        self.client.confirmConnection()
        self.client.enableApiControl(api_control)

        self.off_road_cumulative_reward = 0
        
        self.home_point = self.client.getHomeGeoPoint()
        print (f"HomePoint is {self.home_point}")
        self.target_distance_from_home = get_distance_from_target(self.home_point)
        self.last_known_distance = self.target_distance_from_home
        
        if self.continious:
            self.action_space = spaces.Box(low=np.array(self.low), high=np.array(self.high), dtype=np.float32)
            self._act = self._act_continuous
            self._throttle_map = map_to_range if action_map[0] else float
            self._steering_map = map_to_range if action_map[1] else float
            self._speed_contrib = self._speed_contrib_continious
        else:
            if isinstance(steering_angles, int):
                self.steering_angles = create_symmetric_range(steering_angles)
            else:
                self.steering_angles = steering_angles
            self.action_space = spaces.Discrete(self.steering_angles.shape[0])
            self._act = self._act_discrete
            self._speed_contrib = self._speed_contrib_discrete

        if obs_is_image_only:
            self.observation_space = spaces.Box(low=0, high=255,\
                                        shape=(self.height, self.width, 3),\
                                        dtype=np.uint8)
            
            self._get_observation = self._get_image
        else:
            # maybe sometime in the future
            raise NotImplementedError("TBD")

        if self.goal_point_exists:
            self.goal = np.array([393.7127990722656, -125.40431213378906, 8.464661598205566]) # if goal is not passed as parameter, then set it to the last point of the road

        self.reward_scale_factor = 1.0
        # if self.scale_reward:
        #     regular_max_reward = self.hold_mid_reward_mul + self.speed_reward_mul
        #     self.reward_scale_factor = np.abs(max([regular_max_reward, self.off_road_reward, self.stuck_reward, self.crashed_reward, self.goal_reached_reward],\
        #                                              key=lambda x: np.abs(x)))
            
        #     assert self.reward_scale_factor > 0.0
        # else:
        #     self.reward_scale_factor = 1.0
    
    def __del__(self):
        try:
            self.client.enableApiControl(False)
        except:
            print("Couldn't restore AirSim control.")
        
    def render(self, mode="human", close=False):
        return

    def detect_lines(self, img):
        frame = cv2.GaussianBlur(img, (5, 5), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low_yellow = np.array([18, 25, 140])
        up_yellow = np.array([48, 200, 180])
        mask = cv2.inRange(hsv, low_yellow, up_yellow)
        
        edges = cv2.Canny(mask, 100, 150)

        #cropped_image = region_of_interest(edges, np.array([region_of_interest_vertices], np.int32))
        lines = cv2.HoughLinesP(edges, 2, np.pi/180, 25, maxLineGap=3)

        return lines

    def detect_lines_hough_regular(self, img):
        frame = cv2.GaussianBlur(img, (5, 5), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low_yellow = np.array([18, 25, 140])
        up_yellow = np.array([48, 255, 255])
        mask = cv2.inRange(hsv, low_yellow, up_yellow)
        
        edges = cv2.Canny(mask, 75, 150)

        cropped_image = region_of_interest(edges, np.array([region_of_interest_vertices], np.int32))
        lines = cv2.HoughLines(cropped_image, 1, np.pi/180, 12)

        return lines

    def calculate_distance(self):
        img = np.array(self._get_image())
        #img.setflags(write=1)
        #brightness(img, 50)
        lines = self.detect_lines(img)

        if lines is  None:
            brightness(img, 18)
            lines = self.detect_lines(img)
        
        if lines is not None:
            nearest_line_distance = MAX_DISTANCE_ALLOWED
            for line in lines:
                line_start_x = line[0][0]
                line_start_y = line[0][1]

                red_left = img[line_start_y][line_start_x - LEFT_PIXEL_DISTANCE if line_start_x - LEFT_PIXEL_DISTANCE > 0 else line_start_x][0]
                green_left = img[line_start_y][line_start_x - LEFT_PIXEL_DISTANCE if line_start_x - LEFT_PIXEL_DISTANCE > 0 else line_start_x][1]
                blue_left = img[line_start_y][line_start_x - LEFT_PIXEL_DISTANCE if line_start_x - LEFT_PIXEL_DISTANCE > 0 else line_start_x][2]

                red_right = img[line_start_y][line_start_x + RIGHT_PIXEL_DISTANCE if line_start_x + RIGHT_PIXEL_DISTANCE < WIDTH else line_start_x][0]
                green_right = img[line_start_y][line_start_x + RIGHT_PIXEL_DISTANCE if line_start_x + RIGHT_PIXEL_DISTANCE < WIDTH else line_start_x][1]
                blue_right = img[line_start_y][line_start_x + RIGHT_PIXEL_DISTANCE if line_start_x + RIGHT_PIXEL_DISTANCE < WIDTH else line_start_x][2]

                if not (val_diff_greater_than_threshold(GRAY_DIFFERENCE_THRESHOLD, red_left, green_left, blue_left) \
                    or val_diff_greater_than_threshold(GRAY_DIFFERENCE_THRESHOLD, red_right, green_right, blue_right) \
                    or diff_between_two_rgb_threshold(LEFT_RIGHT_DIFF_TRESHOLD, red_left, green_left, blue_left, red_right, green_right, blue_right)):      

                    nearest_line_distance = min(nearest_line_distance,\
                                                distance_from_the_line(np.array([line[0][0], line[0][1]]), np.array([line[0][2], line[0][3]]), CENTER))

            return nearest_line_distance/MAX_DISTANCE_ALLOWED
        else:
            return -1

    def calculate_distance_regular_hough_transform(self):
        img = self._get_image()
        img.setflags(write=1)
        #brightness(img, 50)
        lines = self.detect_lines_hough_regular(img)

        if lines is  None:
            brightness(img, 25)
            lines = self.detect_lines_hough_regular(img)
        
        if lines is not None:
            nearest_line_distance = np.inf
            for rho, theta in lines[0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                nearest_line_distance = min(nearest_line_distance,\
                                        distance_from_the_line(np.array([x1, y1]), np.array([x2, y2]), CENTER))
                #print(nearest_line_distance/MAX_DISTANCE_ALLOWED)
            return nearest_line_distance/MAX_DISTANCE_ALLOWED
        else:
            return -1

    def step(self, action):
        if (self.sim_paused):
            self.client.simPause(False)

        self._act(action)
        time.sleep(self.sleep_after_action)

        self.current_position = self._pos_as_np_array()
        self.car_state = self.client.getCarState()
        
        if (self.car_state.speed < self.min_allowed_speed):
            self.steps_at_min_speed_count += 1
        else:
            self.steps_at_min_speed_count = 0 # reset the counter

        reward, done = self._compute_reward()
        self.timesteps += 1
        
        if self.pause_after != None and (self.timesteps % self.pause_after == 0):
            self.client.simPause(True)
            self.sim_paused = True

        return self._get_observation(),\
               reward,\
               done,\
               {}

    def reset(self):
        self.client.reset()
        self.client.armDisarm(True)
        self.client.enableApiControl(True)
        self.steps_at_min_speed_count = 0

        return self._get_observation()

    def _goal_reached(self):
        """
        Checks whether the end goal is reached based on the env. parameters
        """
        return euclidean(self.current_position, self.goal) <= self.goal_reached_dist
    
    def _pos_as_np_array(self):
        """
            Returns the current position of the agent in engine space
            as numpy array for more convenient calculations
        """
        cp = self.client.simGetGroundTruthKinematics().position
        return np.array([cp.x_val, cp.y_val, cp.z_val])

    def _compute_reward(self):
        """
        Computes the reward value for the agent and decides if the current episode should finish
        Reward is based on the state of the agent (stuck, crashed, off road..)
        """
        geo_point_current = self.client.simGetGroundTruthEnvironment().geo_point 
        distance_from_target = get_distance_from_target(geo_point_current)

        if (self._goal_reached() if self.goal_point_exists else False):
            reward, done = self.goal_reached_reward, True     
        elif self.client.simGetCollisionInfo().has_collided:
            reward, done = self.crashed_reward, True
        elif self.steps_at_min_speed_count == self.steps_at_min_speed_allowed:
            reward, done =  self.stuck_reward, True   
        else:
            nearest_line_distance = self.calculate_distance()
                
            if nearest_line_distance == -1:
                self.off_road_cumulative_reward += -1 
                if self.off_road_cumulative_reward == -5:
                    self.off_road_cumulative_reward = 0
                    reward, done = self.off_road_reward, True
                else:
                    reward, done = -1, False
            else:
                #self.off_road_cumulative_reward = 0
                closer_to_goal_contrib = 1 if distance_from_target < self.last_known_distance else 0 
                reward, done = (self.hold_mid_reward_mul * np.exp(-nearest_line_distance * self.hold_mid_reward_decay_rate) + self._speed_contrib() + closer_to_goal_contrib), False

        return reward / self.reward_scale_factor, done

    def _speed_contrib_continious(self):
        """
        Calculates the speed contribution to the end reward
        This function is invoked only if the env. action space in continious
        """
        speed = self.car_state.speed
        if speed <= self.min_allowed_speed:
            return self.stuck_reward / 2.0
        else:
            return self.speed_reward_mul * np.exp(-(1.0 / speed) * self.speed_reward_decay_rate)
    
    def _speed_contrib_discrete(self):
        """
        Calculates the speed contribution to the end reward
        This function is invoked only if the env. action space in descrete
        Currently this function only returns 0.0 since in discrete env. speed is always constant
        """
        return 0.0

    def _get_image(self):
        """
        Returns the cropped front camera image
        """
        image_response = self.client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])[0]
        image1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
        image_rgba = image1d.reshape(image_response.height, image_response.width, self.n_channels_env) # some envs. use diffrent number of channels (like l. mountains and nh)

        return image_rgba[self.crop_h1:self.crop_h2, self.crop_w1:self.crop_w2, 0:3]


    def _act_discrete(self, action):
        """
        Performs the discrete action

        :param action: (int) action to perform
        """
        self.car_state = self.client.getCarState()
        current_speed = self.car_state.speed
        
        throttle = 0.8 if current_speed <= self.max_speed else 0.0
        steering = self.steering_angles[action]

        self.client.setCarControls(airsim.CarControls(throttle=throttle, steering=steering))


    def _act_continuous(self, action):
        """
        Performs the continious actions
        TODO: add brake and parametrize map_to_range
        
        :param action: (np.array) - actions to perform
        """
        self.car_state = self.client.getCarState()
        current_speed = self.car_state.speed

        throttle = self._throttle_map(action[0])
        throttle = 0.0 if current_speed >= self.max_speed else throttle
        steering = self._steering_map(action[1])

        self.client.setCarControls(airsim.CarControls(throttle=throttle, steering=steering))
