from environment import PongEnv
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = PongEnv()

obs = env.observation_space.shape[0]
act = env.action_space.n

# Define the model
model = Sequential()
model.add(Flatten(input_shape=(1,)))
model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Flatten())
model.add(Dense(act, activation="linear"))

# Create the DQNAgent
agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=3),  # Match the window_length with the input shape
    policy=BoltzmannQPolicy(),
    nb_actions=3,
    target_model_update=0.01
)

agent.compile(Adam(), metrics=["mae"])

# Training loop
history = agent.fit(env, nb_steps=1000, verbose=1)

# Evaluate the agent if needed
# scores = agent.test(env, nb_episodes=10, visualize=False)
