import numpy as np


class OUNoise:
    '''Define Ornstein-Uhlenbeck Process for Exploration Noise'''
    def __init__(self, action_dim:int, mu:float=0, theta:float=0.15, sigma:float=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
