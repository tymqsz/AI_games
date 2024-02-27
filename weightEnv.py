import gymnasium as gym
from gymnasium import spaces
import numpy as np

class WeightEnv(gym.Env):
    def __init__(self, const_weight_loss=False):
        super().__init__()
        
        self.observation_space = spaces.Discrete(120)
        self.action_space = spaces.Discrete(4)

        self.weight_gain = {0: 0, 1:1, 2:2, 3:3}
        
        self.weight = 30
        self.min_weight = 70
        self.max_weight = 80
        self.const_weight_loss = const_weight_loss
        self.step_target = 500
        self.step_cnt = 0

        self.eaten_cnt = 0
    
    def _get_obs(self):
        return self.weight 
    
    def reset(self, seed=None):
        super().reset(seed=seed)

        self.weight = np.random.choice([x for x in range(50, 90)])
        self.step_cnt = 0
        self.eaten_cnt = 0

        return self._get_obs()
    
    def step(self, action):
        action = int(action)
        self.weight += self.weight_gain[action]
        
        self.step_cnt += 1
    
        if self.weight >= self.min_weight and self.weight <= self.max_weight:
            reward = 1
        else:
            reward = 0

        weight_loss = np.random.choice([0, 1, 2, 3])

        self.weight -= weight_loss

        observation = self._get_obs()
        terminated = (self.step_cnt == self.step_target or self.weight > 120 or self.weight <= 0)
        
        return observation, reward, terminated