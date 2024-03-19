import numpy as np

class Qlearn():
    def __init__(self, env):
        self.Q = np.zeros(shape=(env.observation_space.n, env.action_space.n))
        
        self.env = env

    def train(self, n_iter=1000, gamma=0.99, learning_rate=0.05, eps_init=1, eps_decay=0.0001, eps_min=0.05):
        i = 0
        eps = eps_init

        observation = self.env.reset()
        while i < n_iter:
            x = np.random.random()
            if x < eps:
                action = self.env.action_space.sample()
            else:
                action = np.argmax(self.Q[observation])
            
            new_observation, reward, end_of_ep = self.env.step(action)

            if end_of_ep:
                observation = self.env.reset()
                i += 1
                continue
            else:
                observation = new_observation

            max_q = np.argmax(self.Q[new_observation])

            observation = observation[0]
            self.Q[observation][action] = (1-learning_rate)*self.Q[observation][action] + learning_rate*(reward + gamma*max_q)

            
            i += 1

            eps = max(eps_min, eps*np.exp(-eps_decay))

    def predict(self, observation):
        return np.argmax(self.Q[observation])


