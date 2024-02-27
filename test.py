from Q_learn import Qlearn
from weightEnv import WeightEnv
import matplotlib.pyplot as plt

env = WeightEnv()

model = Qlearn(env=env)

model.train(n_iter=100000)


test_env = WeightEnv()
episodes = 100
reward = 0
for epis in range(episodes):
    done = False
    obs = test_env.reset()
    while not done:
        action = model.predict(obs)
        #action = test_env.action_space.sample()
        obs, rew, done = test_env.step(action)

        reward += rew
print(f"avg reward: {reward/episodes}")