import torch
from torch import nn
import torch.optim as optim
import numpy as np

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, lr, device, optim, loss_fn):
        super().__init__()

        # Sequential NN definition
        self.network = nn.Sequential(
            nn.Linear(n_observations, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        ).to(device)

        # optimizer, loss_fn set to provided else defaulted to Adam and Mean Squared Error
        self.optimizer = optim.Adam(self.parameters(), lr=lr) if optim is None else optim
        self.loss_fn = nn.MSELoss() if loss_fn is None else loss_fn

    def forward(self, x):
        logits = self.network(x)

        return logits


class Agent():
    def __init__(self, env, n_observations, n_actions, lr=1e-4, batch_size=128, memory_cap=10000, gamma=0.99):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # hyperparameters
        self.epsilon = 1
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.02
        self.gamma = gamma
        self.batch_size = batch_size

        self.env = env

        # init DQN
        self.DQN = DQN(n_observations, n_actions, lr, self.device)

        # replay memory as a set of 5 numpy arrays
        self.memory_index = 0
        self.memory_size = 0
        self.memory_capacity = memory_cap
        self.state_memory = np.zeros((self.memory_capacity, n_observations), dtype=np.float32)
        self.next_state_memory = np.zeros((self.memory_capacity, n_observations), dtype=np.float32)
        self.action_memory = np.zeros((self.memory_capacity,), dtype=np.int32)
        self.reward_memory = np.zeros((self.memory_capacity,), dtype=np.float32)
        self.done_memory = np.zeros((self.memory_capacity,), dtype=np.bool_)

    def store_memory(self, state, next_state, action, reward, done):
        # set index to fit into size of memory (self.memory_capacity)
        idx = self.memory_index % self.memory_capacity if self.memory_index != 0 else 0

        self.state_memory[idx] = state
        self.next_state_memory[idx] = next_state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.done_memory[idx] = done

        self.memory_index += 1
        
        # update fullness of memory
        self.memory_size = min(self.memory_size + 1, self.memory_capacity)

    def act(self, state):
        # epsilon-greedy based action selection
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                Q_values = self.DQN.forward(torch.tensor(np.array([state]),device=self.device))

                return torch.argmax(Q_values).item()
    
    def learn(self):
        if self.memory_size >= self.batch_size:
            # select batch of memories
            chosen_indices = np.random.randint(self.memory_size-1, size=self.batch_size)

            state_batch = torch.tensor(self.state_memory[chosen_indices],device=self.device)
            next_state_batch = torch.tensor(self.next_state_memory[chosen_indices],device=self.device)
            action_batch = torch.tensor(self.action_memory[chosen_indices], device=self.device)
            reward_batch = torch.tensor(self.reward_memory[chosen_indices],device=self.device)
            done_batch = torch.tensor(self.done_memory[chosen_indices],device=self.device)


            # calculate Q-values of next states
            with torch.no_grad():
                next_Q_pred = self.DQN.forward(next_state_batch)
                next_Q_pred[done_batch] = 0

            # calculate Q-values of current states
            current_Q_values = self.DQN.forward(state_batch)  

            # calculate target_Q_values based on rewards and maximum next state Q-values
            target_Q_values = current_Q_values.clone()
            updated_Q_values = reward_batch + self.gamma*torch.max(next_Q_pred, dim=1)[0]
            target_Q_values[torch.arange(len(target_Q_values)), action_batch] = updated_Q_values
            
            # calculate loss and optimize
            loss = self.DQN.loss_fn(current_Q_values, target_Q_values)
            loss.backward()
            self.DQN.optimizer.step()
            self.DQN.optimizer.zero_grad()

            # update epsilon value
            self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)