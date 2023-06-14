from score import ScoreEnv, generate_score
from collections import deque
import random
import torch
from torch.nn import Module, Linear
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
import csv
import os
import glob


class QNetwork(Module):
    def __init__(self):
        super().__init__()
        self.fc = Linear(66, 128)
        self.fcQ1 = Linear(128, 256)
        # self.fcQ2 = Linear(256, 256)
        self.fcQ2 = Linear(256, 20)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        x = self.fcQ1(x)
        x = F.relu(x)
        # x = self.fcQ2(x)
        # x = F.relu(x)
        x = self.fcQ2(x)


        return x

# network and optimizer
Q = QNetwork()
optimizer = torch.optim.Adam(Q.parameters(), lr=0.0005)

# target network
Q_target = QNetwork()

replay_memory = deque(maxlen=1000000)  # replay buffer
discount = 0.99  # discount factor gamma
mini_batch_size = 32

def makeckptdir():
    # CHECKPOINT
    ckpt_folder = os.path.join('./ckpt_DQN')
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    if len(glob.glob(os.path.join(ckpt_folder, '*.pt'))) > 0:
        # load the last ckpt
        checkpoint = torch.load(glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1], map_location=torch.device('cpu'))

        Q.load_state_dict(checkpoint['Q_model'])
        Q_target.load_state_dict(checkpoint['Q_target_model'])


        last_episode_id = checkpoint['episode']
        reward_history = checkpoint['reward_history']
        reward_history_100 = reward_history[-100:]
        eps = checkpoint['epsilon']

    return ckpt_folder

def update_Q():
    states = []
    actions = []
    next_states = []
    dones = []
    rewards = []
    for state, action, state_next, reward, done in random.sample(replay_memory, min(mini_batch_size, len(replay_memory))):
        states.append(state)
        actions.append(action)
        next_states.append(state_next)
        dones.append(done)
        rewards.append(reward)

    states = torch.stack(states)
    actions = torch.tensor(actions).type(torch.int64)
    next_states = torch.stack(next_states)
    dones = torch.tensor(dones, dtype=torch.int64)
    rewards = torch.tensor(rewards)

    with torch.no_grad():
        target = rewards + (1-dones.view(-1, 1)) * discount * Q_target(next_states).max(1)[0] ##
    curr_Q = Q(states).gather(1, actions.view(-1, 1))
    loss = (target - curr_Q.squeeze()) ** 2
    loss = torch.mean(loss)

    optimizer.zero_grad()

    loss.backward()


def epsilonGreedy(epsilon):
    assert epsilon >= 0 and epsilon <= 1, "invalid epsilon value"
    if random.random() < epsilon :
        return random.randint(0, 19) ##
    else :
        return torch.argmax(Q(state)).item() ##

def stateDecoding(state):

    # one hot decoding
    fret_state = env.one_hot_decoding(state[0:21])
    finger_state = env.one_hot_decoding(state[21:26])
    string_state = env.one_hot_decoding(state[26:30])

    # fret number decoding
    assert fret_state in range(0, 21), f"invalid fret number {fret_state}"
    fret_number = fret_state

    # finger number decoding
    assert finger_state in range(0, 5), f"invalid finger number {finger_state}"
    finger_number = finger_state

    # string number decoding
    assert string_state in range(0, 4), f"invalid string number {string_state}"
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

episodes = 5000
max_iter = env.episode_length
total_sample = 0
eps = 1
last_episode_id = 0
ckpt_folder = makeckptdir()
train = True

# train : DQN
if train:
    for episode in range(last_episode_id, episodes):
        # sum of accumulated rewards
        G = 0

        # get initial observation
        observation, _ = env.reset()
        state = torch.tensor(observation, dtype=torch.float32)

        while env.t <= max_iter:
            # choose action with epsilon greedy policy
            eps = np.clip(1.1 - total_sample / (max_iter * 3000), 0.05, 1)
            action = epsilonGreedy(epsilon=eps)

            # get next observation and current reward for the chosen action
            observation_next, reward, done, _ = env.step(action)
            state_next = torch.tensor(observation_next, dtype=torch.float32)
            total_sample += 1

            # Compute G_0
            G = G + (discount ** (env.t - 1)) * reward

            # Store transition into the replay memory
            replay_memory.append([state, action, state_next, reward, done])
            if len(replay_memory) >= mini_batch_size:
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
            print('episode: {}, reward: {:.1f}, avg: {:.1f}, eps: {}'.format(episode, G, avg, eps))

        # checkpoint storing
        if episode % 100 == 1:
            # return plotting
            plt.figure()
            plt.plot(reward_history)
            plt.legend(['episode reward'], loc=1)
            plt.xlabel('m episode')
            plt.ylabel('reward')
            plt.savefig(os.path.join(ckpt_folder, 'rewards_' + str(episode).zfill(8) + '.jpg'))
            plt.close()

            # checkpoint storing
            torch.save({'episode': episode,
                        'reward_history': reward_history,
                        'Q_model': Q.state_dict(),
                        'Q_target_model': Q_target.state_dict(),
                        'epsilon': eps
                        },
                       os.path.join(ckpt_folder, 'ckpt_' + str(episode).zfill(8) + '.pt'))

else:
    results = []
    # call latest checkpoint
    if len(glob.glob(os.path.join(ckpt_folder, '*.pt'))) > 0:
        # load the last ckpt
        checkpoint = torch.load(glob.glob(os.path.join(ckpt_folder, '*.pt'))[-1], map_location=torch.device('cpu'))
        Q.load_state_dict(checkpoint['Q_model'])

    # inference
    observation, _ = env.reset()
    state = torch.tensor(observation, dtype=torch.float32)
    done = False
    G = 0
    while not done:
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.argmax(Q(state)).item()
        state, reward, done, _ = env.step(action)
        G = G + (discount ** (env.t - 1)) * reward
        state_decoded = stateDecoding(state)
        results.append([state_decoded[0], state_decoded[2]])

    # save results as csv file
    with open('result.txt', 'w', encoding='UTF-8') as f:
        score = generate_score()
        del score[-1]
        assert len(score) == len(results), f"length of score and results are different : {len(score)} and {len(results)}"

        for i in range(len(results)):
            f.write(f'Melody : {score[i]}\n{results[i][0]} fret, {results[i][1]} string\n\n')






