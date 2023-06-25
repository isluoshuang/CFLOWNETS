import numpy as np
from numpy import random
from gym import spaces
from gym import Env
import math

class PointEnv_MultiStep_Two_goal(Env):
    def __init__(self):
        self.threshold = 10.0
        high = np.array(
            [self.threshold, self.threshold], dtype=np.float32,
        )
        low = np.array(
            [0.0, 0.0], dtype=np.float32,
        )
        self.goal1 = [5.0, 10.0]
        self.goal2 = [10.0, 5.0]
        self.observation_space = spaces.Box(low=low, high=high, shape=(2,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        self.state = [0.0, 0.0]
        self.time = 1.0
        self.cnt_step = 0
        self.step_max = 12

    def step(self, action):
        action = (action + 1) * 45.0
        if action < 0.0: action = 0.0
        if action > 90.0: action = 90.0
        x = self.state[0]
        y = self.state[1]
        costheta = math.cos(action*math.pi/180.0)
        sintheta = math.sin(action*math.pi/180.0)
        x = x + costheta * self.time
        y = y + sintheta * self.time

        done = False
        self.cnt_step += 1
        self.state = np.array([x, y])
        reward = 0
        if self.cnt_step == self.step_max:
            dist1 = ((x - self.goal1[0]) ** 2 + (y - self.goal1[1]) ** 2) ** 0.3
            dist2 = ((x - self.goal2[0]) ** 2 + (y - self.goal2[1]) ** 2) ** 0.3
            reward1 = 1.0 / (dist1 + 0.5)
            reward2 = 1.0 / (dist2 + 0.5)
            reward = reward1 + reward2
            done = True
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.zeros((1,2), dtype=np.float32)[0]
        self.cnt_step = 0
        return self.state


