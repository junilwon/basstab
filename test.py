from score import ScoreEnv, generate_score
from collections import deque
import random
import torch
import torch.nn as nn
from torch.nn import Module, Linear
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import csv
import os
import glob
from main import validActionSample, stateDecoding

env = ScoreEnv()
inf = 99999999
episodes = 10000
max_iter = env.episode_length
total_sample = 0
eps = 1
last_episode_id = 0
train = True

class QNetwork(Module):
    def __init__(self):
        super().__init__()
        self.fc = Linear(66, 128)
        self.fcQ1 = Linear(128, 256)
        self.fcQ2 = Linear(256, 20)

        # self.prelu = nn.PReLU()
        # self.prelu1 = nn.PReLU()
        # self.prelu2 = nn.PReLU()

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        x = self.fcQ1(x)
        x = F.relu(x)
        x = self.fcQ2(x)

        return x

def validActionSample(epsilon, state, Q):

    next_melody = env.score[env.t]
    valid_action = []
    invalid_action = np.zeros(env.nA)
    for a in range(env.nA):
        finger = a // 4
        string = a % 4
        fret = next_melody - 5 * string
        if fret < 0 or fret > 20 or (finger == 0 and fret > 0):
            invalid_action[a] = 1
        else:
            valid_action.append(a)

    invalid_action = torch.Tensor(invalid_action)

    curr_Q = Q(state)
    curr_Q = curr_Q - inf * invalid_action

    if random.random() < epsilon:
        return random.choice(valid_action)
    else:
        return torch.argmax(curr_Q).item()


Q = QNetwork()
# network and optimizer
optimizer = torch.optim.Adam(Q.parameters(), lr=0.001)
ckpt_folder = os.path.join('./ckpt_DQN')
replay_memory = deque(maxlen=1000000)  # replay buffer
discount = 0.99  # discount factor gamma
mini_batch_size = 32

results = []
# call latest checkpoint
# if len(glob.glob(os.path.join(ckpt_folder, '*.pt'))) > 0:
# load the last ckpt
# checkpoint = torch.load(glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1], map_location=torch.device('cpu'))
checkpoint = torch.load('./ckpt_DQN/ckpt_00000901.pt', map_location=torch.device('cpu'))
Q.load_state_dict(checkpoint['Q_model'])

# inference
observation, _ = env.reset()
state = torch.tensor(observation, dtype=torch.float32)
done = False
G = 0
while not done:
    state = torch.tensor(state, dtype=torch.float32)

    action = validActionSample(epsilon=0, state=state, Q=Q)

    state, reward, done, _ = env.step(action)
    # G = G + (discount ** (env.t - 1)) * reward
    G = G + reward
    state_decoded = stateDecoding(state)
    results.append([state_decoded[0], state_decoded[2], state_decoded[1], reward])

# save results as csv file
with open('result.txt', 'w', encoding='UTF-8') as f:
    score = generate_score()
    del score[-1]
    assert len(score) == len(results), f"length of score and results are different : {len(score)} and {len(results)}"

    print(G)
    for i in range(len(results)):
        # f.write(f'Melody : {score[i]}\n{results[i][0]} fret, {results[i][1]} string\n')
        f.write(f'Melody : {score[i]}\n{results[i][0]} fret, {results[i][1]} string\n{results[i][2]} finger\ngot {results[i][3]} reward\n\n')
