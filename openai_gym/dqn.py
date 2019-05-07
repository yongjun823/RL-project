import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from baselines.deepq.replay_buffer import ReplayBuffer

# https://github.com/higgsfield/RL-Adventure/blob/master/common/wrappers.py
from deep_rl.wrapper import make_atari, wrap_deepmind, wrap_pytorch


def epsilon_by_frame(x):
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 30000

    return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * x / epsilon_decay)


def plot(idx, rewards, losses):
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (idx, np.mean(rewards[-10:])))
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.show()


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class Model(nn.Module):

    def __init__(self, input_shape, num_actions):
        super(Model, self).__init__()

        self.batch_size = 64
        self.gamma = 0.95
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.replay_buffer = ReplayBuffer(size=60000)

        self.feature = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
        )
        # torch.Size([1, 64, 7, 7])

        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def action(self, s, epsilon):
        if random.random() > epsilon:
            s = torch.from_numpy(s).unsqueeze(0).to(device)
            q_value = self.forward(s)

            # maximum value index
            a = q_value.max(1)[1].data[0].cpu().detach().numpy()
        else:
            a = random.randrange(env.action_space.n)

        return a

    def compute_loss(self):
        s, a, r, n_s, d = self.replay_buffer.sample(self.batch_size)

        s = torch.from_numpy(s).to(device)
        n_s = torch.from_numpy(n_s).to(device)
        r = torch.from_numpy(np.float32(r)).to(device)
        a = torch.from_numpy(a).type(torch.LongTensor).to(device)
        d = torch.from_numpy(np.float32(d)).to(device)

        q_values = self.forward(s)
        next_q_values = self.forward(n_s)

        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]

        expected_q_value = r + self.gamma * next_q_value * (1 - d)

        loss = (q_value - expected_q_value.data).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss


if __name__ == '__main__':
    env_id = "MsPacmanNoFrameskip-v4"
    env = make_atari(env_id)
    env = wrap_deepmind(env, frame_stack=True, scale=True)
    env = wrap_pytorch(env)

    state_size, action_size = env.observation_space.shape, env.action_space.n

    total_episodes = 50  # Total episodes for training
    max_steps = 50000  # Max possible steps in an episode

    replay_initial = 10000

    model = Model(state_size, action_size).to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=1e-05)

    num_frames = 1400000
    losses = []
    all_rewards = []
    episode_reward = 0

    state = env.reset()

    for frame_idx in range(1, num_frames):
        action = model.action(state, epsilon_by_frame(frame_idx))

        # if epsilon_by_frame(frame_idx) < 0.1:
        #     env.render()

        next_state, reward, done, _ = env.step(action)
        model.replay_buffer.add(state, action, reward, next_state, float(done))

        state = next_state
        episode_reward += reward

        if done:
            state = env.reset()
            all_rewards.append(episode_reward)
            print('episode: {}, reward: {}, exploration: {}'.format(len(all_rewards), episode_reward,
                                                                    epsilon_by_frame(frame_idx)))
            episode_reward = 0

        if len(model.replay_buffer) > replay_initial:
            loss = model.compute_loss().item()
            print(loss)
            losses.append(loss)

        if frame_idx % 1000 == 0:
            print('frame idx {}'.format(frame_idx))

        if frame_idx % 10000 == 0:
            plot(frame_idx, all_rewards, losses)
