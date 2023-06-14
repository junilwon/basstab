import numpy as np
from gym import Env, spaces

# loading bass score "Time of Our Life"
# validity checked
def generate_score():
    f = open("./score.txt", 'r')
    line = f.readline()
    line = line.replace(",", "").replace("\n", "").split(" ")
    score = list(map(int, line))
    score.append(0)
    return score

class ScoreEnv(Env):

    def __init__(self, score = None):
        if score is None:
            score = generate_score()
        self.score = score
        self.episode_length = len(score)

        # action & space dimension
        nA = 20
        nS = 66

        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

        self.last_hand_state = None

        self.t = 0  # timestep

    def compute_reward(self, curr_hand_state):

        # first step
        if self.last_hand_state is None :
            return 0

        # state download
        last_fret_state, last_finger_state, last_string_state = self.last_hand_state
        curr_fret_state, curr_finger_state, curr_string_state = curr_hand_state


        # impossible step
        if last_fret_state < 0 or last_fret_state > 20 or (last_finger_state == 0 and last_fret_state > 0) :
            return -100
        # normal steps
        elif last_finger_state == 0 :
            return 0
        elif last_finger_state == 1 :
            if curr_fret_state == last_fret_state and curr_string_state == last_string_state and curr_finger_state == 1:
                return 0
            elif curr_fret_state == last_fret_state and curr_string_state != last_string_state and (curr_finger_state in (2, 3, 4)):
                return 0
            elif curr_fret_state == last_fret_state + 1 and curr_finger_state in (2, 3, 4):
                return 0
            elif curr_fret_state == last_fret_state + 2 and curr_finger_state in (3, 4):
                return 0
            elif curr_fret_state == last_fret_state + 3 and curr_finger_state == 4:
                return 0
            else:
                return -1
        elif last_finger_state == 2 :
            if curr_fret_state == last_fret_state - 1 and curr_finger_state == 1:
                return 0
            elif curr_fret_state == last_fret_state and curr_string_state == last_string_state and curr_finger_state == 2:
                return 0
            elif curr_fret_state == last_fret_state and curr_string_state != last_string_state and curr_finger_state in (1, 3, 4):
                return 0
            elif curr_fret_state == last_fret_state + 1 and curr_finger_state in (3, 4):
                return 0
            elif curr_fret_state == last_fret_state + 2 and curr_finger_state == 4:
                return 0
            else:
                return -1
        elif last_finger_state == 3 :
            if curr_fret_state == last_fret_state - 2 and curr_finger_state == 1:
                return 0
            elif curr_fret_state == last_fret_state - 1 and curr_finger_state in (1, 2):
                return 0
            elif curr_fret_state == last_fret_state and curr_string_state == last_string_state and curr_finger_state == 3:
                return 0
            elif curr_fret_state == last_fret_state and curr_string_state != last_string_state and curr_finger_state in (1, 2, 4):
                return 0
            elif curr_fret_state == last_fret_state + 1 and curr_finger_state == 4:
                return 0
            else:
                return -1
        else:
            if curr_fret_state == last_fret_state - 3 and curr_finger_state == 1:
                return 0
            elif curr_fret_state == last_fret_state - 2 and curr_finger_state in (1, 2):
                return 0
            elif curr_fret_state == last_fret_state - 1 and curr_finger_state in (1, 2, 3):
                return 0
            elif curr_fret_state == last_fret_state and curr_string_state == last_string_state and curr_finger_state == 4:
                return 0
            elif curr_fret_state == last_fret_state and curr_string_state != last_string_state and curr_finger_state in (1, 2, 3):
                return 0
            else:
                return -1
    
    def one_hot_encoding(self, value, upper_bound):

        one_hot_vector = []
        
        for i in range(upper_bound):
            one_hot_vector.append(int(i == value))
        
        # if invalid input such as value < 0 is entered, will return zero vector

        return one_hot_vector

    def one_hot_decoding(self, one_hot_vector):

        for i in range(len(one_hot_vector)):
            if one_hot_vector[i] == 1:
                return i

        return 0

    def step(self, a):
        # a: 0~19

        # melody state download
        curr_melody_state = self.score[self.t]

        # calculate residual states from action and melody state
        finger_state = a // 4
        string_state = a % 4
        fret_state = curr_melody_state - 5 * string_state

        # compute reward
        curr_hand_state = [fret_state, finger_state, string_state]
        r = self.compute_reward(curr_hand_state)

        # time step
        self.t += 1
        self.last_hand_state = curr_hand_state

        # next melody update
        next_melody_state = self.score[self.t]

        # one-hot state transition
        fret_state = np.array(self.one_hot_encoding(fret_state, 21))
        finger_state = np.array(self.one_hot_encoding(finger_state, 5))
        string_state = np.array(self.one_hot_encoding(string_state, 4))
        next_melody_state = np.array(self.one_hot_encoding(next_melody_state, 36))
        
        self.s = np.hstack((fret_state, finger_state, string_state, next_melody_state))

        return self.s, r, self.t == self.episode_length - 1, {}

    def reset(self):
        self.t = 0
        self.s = np.zeros(21 + 5 + 4 + 36)
        self.s[-36:] = self.one_hot_encoding(self.score[0], 36)
        self.last_hand_state = None

        return self.s, {}

    def render(self):
        return

    def close(self):
        return