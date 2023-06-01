from score import ScoreEnv
from collections import deque
import random
import torch
from torch.nn import Module, Linear
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import csv



class QNetwork(Module):
    def __init__(self):
        super().__init__()
        self.fc = Linear(66, 128)
        self.fcQ1 = Linear(128, 256)
        self.fcQ2 = Linear(256, 256)
        self.fcQ3 = Linear(256, 20)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        x = self.fcQ1(x)
        x = F.relu(x)
        x = self.fcQ2(x)
        x = F.relu(x)
        x = self.fcQ3(x)

        return x

# network and optimizer
Q = QNetwork()
optimizer = torch.optim.Adam(Q.parameters(), lr=0.0005)

# target network
Q_target = QNetwork()

replay_memory = deque(maxlen=1000000)  # replay buffer
discount = 0.99  # discount factor gamma
mini_batch_size = 32

def update_Q():

    loss = 0
    for state, action, state_next, reward, done in random.sample(replay_memory, min(mini_batch_size, len(replay_memory))):
        with torch.no_grad():
            if done:
                target = reward
            else:
                target = reward + discount * torch.max(Q_target(state_next))

        loss = loss + (target - Q(state)[action]) ** 2

    loss = loss / mini_batch_size

    optimizer.zero_grad()

    loss.backward()

def epsilonGreedy(epsilon):

    assert epsilon >= 0 and epsilon <= 1, "invalid epsilon value"

    if random.random() < epsilon :
        return random.randint(0, 19)
    else :
        return torch.argmax(Q(state)).item()

def stateDecoding(state):

    # one hot decoding
    fret_state = env.one_hot_decoding(state[0:21])
    finger_state = env.one_hot_decoding(state[21:26])
    string_state = env.one_hot_decoding(state[26:30])

    # fret number decoding
    assert fret_state in range(0, 21), "invalid fret number"
    fret_number = fret_state

    # finger number decoding
    assert finger_state in (0, 5), "invalid finger number"
    finger_number = finger_state

    # string number decoding
    assert string_state in (0, 4), "invalid string number"
    string_number = string_state + 1

    return fret_number, finger_number, string_number

# score environment
env = ScoreEnv()

# for computing average reward over 100 episodes
reward_history_100 = deque(maxlen=100)
reward_history = []

# for updating target network
target_interval = 100
target_counter = 0

episodes = 2000
max_iter = env.episode_length
total_sample = 0

# train : DQN
for episode in range(episodes):
    # sum of accumulated rewards
    G = 0

    # get initial observation
    observation, _ = env.reset()
    state = torch.tensor(observation, dtype=torch.float32)

    while env.t <= max_iter:
        # choose action with epsilon greedy policy
        action = epsilonGreedy(epsilon=np.clip(1 - total_sample / (max_iter*1500) + 0.05, 0, 1))

        # get next observation and current reward for the chosen action
        observation_next, reward, done, _ = env.step(action)
        state_next = torch.tensor(observation_next, dtype=torch.float32)
        total_sample += 1

        # Compute G_0
        G = G + (discount ** (env.t - 1)) * reward

        # Store transition into the replay memory
        replay_memory.append([state, action, state_next, reward, done])

        update_Q()

        # periodically update target network
        target_counter = target_counter + 1
        if target_counter % target_interval == 0:
            Q_target.load_state_dict(Q.state_dict())

        if done:
            break

        # pass observation to the next step
        observation = observation_next
        state = state_next

        # compute average reward
    reward_history_100.append(G)
    reward_history.append(G)
    avg = sum(reward_history_100) / len(reward_history_100)
    if episode % 10 == 0:
        print('episode: {}, reward: {:.1f}, avg: {:.1f}'.format(episode, G, avg))

# plot objective
t = range(episodes)
plt.plot(t, np.array(reward_history), 'b', linewidth = 2, label = 'DQN')
plt.legend(prop={'size':12})
plt.xlabel('Iteration')
plt.ylabel('Total rewards')
plt.savefig('return_DQN.png')

results = []

# inference
while not done:
    state = torch.tensor(state, dtype=torch.float32)
    action = torch.argmax(Q(state)).item()
    state, reward, done, _ = env.step(action)

    state_decoded = stateDecoding(state)
    results.append([state[0], state[1]])

# save results as csv file
with open("result.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(results)






