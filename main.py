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

# network and optimizer
Q = QNetwork()
optimizer = torch.optim.Adam(Q.parameters(), lr=0.001)

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

    loss = 0
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

    #################################################
    # action masking : targetQ -> -inf if
    invalid_actions = []
    for batch in range(mini_batch_size):
        invalid_action = np.zeros(20)
        next_melody = env.one_hot_decoding(next_states[batch, 30:66])
        for a in range(env.nA):
            finger = a // 4
            string = a % 4
            fret = next_melody - 5 * string
            if fret < 0 or fret > 20 or (finger == 0 and fret != 0):
                invalid_action[a] = 1
        invalid_actions.append(list(invalid_action))
    invalid_actions = torch.Tensor(invalid_actions)
    #################################################
    targetQ = Q_target(next_states)
    targetQ = targetQ - inf * invalid_actions

    curr_Q = Q(states)

    with torch.no_grad():
        target = rewards + (1-dones.view(-1, 1)) * discount * targetQ.max(1)[0]
    curr_Q = curr_Q.gather(1, actions.view(-1, 1))
    loss = (target - curr_Q.squeeze()) ** 2
    loss = torch.mean(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

# def epsilonGreedy(epsilon):
#     assert epsilon >= 0 and epsilon <= 1, "invalid epsilon value"
#     if random.random() < epsilon :
#         return random.randint(0, 19) ##
#     else :
#         return torch.argmax(Q(state)).item() ##

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
    string_number = string_state

    return fret_number, finger_number, string_number

# score environment
env = ScoreEnv()

# for computing average reward over 100 episodes
reward_history_100 = deque(maxlen=100)
reward_history = []

# for updating target network
target_interval = 100
target_counter = 0

inf = 99999999
episodes = 2000
max_iter = env.episode_length
total_sample = 0
eps = 1
last_episode_id = 0
ckpt_folder = makeckptdir()
train = True


if __name__ == "__main__":
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
                eps = np.clip(1.1 - total_sample / (max_iter * 500), 0.05, 1)

                action = validActionSample(eps)
                # action = epsilonGreedy(eps)

                # get next observation and current reward for the chosen action
                observation_next, reward, done, _ = env.step(action)
                state_next = torch.tensor(observation_next, dtype=torch.float32)
                total_sample += 1

                # Compute G_0
                # G = G + (discount ** (env.t - 1)) * reward
                G = G + reward

                # Store transition into the replay memory
                replay_memory.append([state, action, state_next, reward, done])
                if len(replay_memory) >= mini_batch_size:
                    loss = update_Q()

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
                print('episode: {}, reward: {:.1f}, avg: {:.1f}, eps: {:.3f}, train_loss : {:.3f}'.format(episode, G, avg, eps, loss))

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

            action = validActionSample(epsilon=100, state=state)

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





