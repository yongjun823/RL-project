import gym
import numpy as np
import random
from logger import Logger


def rargmax(vector):
    """ Argmax that chooses randomly among eligible maximum indices. """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return random.choice(indices)


logger = Logger('./logs')
env = gym.make('Blackjack-v0')

# print(env.observation_space)
# Tuple(Discrete(32), Discrete(11), Discrete(2))
#       my sum      , dealer card , usable ace
# print(env.action_space)
# Discrete(2)

Q = np.zeros([32, 11, 2, 2])
r_list = []
num_episode = 10000000

for i in range(num_episode):
    s = env.reset()
    r_all = 0
    done = False

    while not done:
        action = rargmax(Q[s[0], s[1], int(s[2]), :])
        ns, reward, done, _ = env.step(action)

        Q[s[0], s[1], int(s[2]), action] = reward + np.max(Q[ns[0], ns[1], int(ns[2]), :])

        r_all += reward
        s = ns

    r_list.append(r_all)

    logger.scalar_summary('reward', r_all, i)

print('final success rate: {}'.format(sum(r_list) / num_episode))
