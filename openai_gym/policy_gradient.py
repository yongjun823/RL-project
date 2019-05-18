import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

env = gym.make('CartPole-v1')
env = env.unwrapped
# Policy gradient has high variance, seed for reproducability
env.seed(1)

## ENVIRONMENT Hyperparameters
state_size = 4
action_size = env.action_space.n

## TRAINING Hyperparameters
max_episodes = 300
learning_rate = 0.01
gamma = 0.95  # Discount rate


def discount_and_normalize_rewards(episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0

    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

    return discounted_episode_rewards


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x):
        x = self.model(x)

        return x

    def get_action(self, x):
        action_pb = self.forward(x)
        action_pb = f.softmax(action_pb, dim=1)
        action = torch.multinomial(action_pb, 1)

        action = action.cpu().numpy()[0][0]

        return action


class EpisodeData:

    def __init__(self) -> None:
        super().__init__()
        self.states, self.actions, self.rewards = [], [], []

    def clear(self):
        self.states, self.actions, self.rewards = [], [], []

    def add(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    pg_model = Model().to(device)
    optimizer = optim.Adam(params=pg_model.parameters(), lr=learning_rate)

    allRewards = []
    data = EpisodeData()
    crl = nn.CrossEntropyLoss(reduction='none')

    for episode in range(max_episodes):
        episode_rewards_sum = 0
        state = env.reset()

        # env.render()

        while True:
            torch_state = torch.from_numpy(state.reshape(1, 4).astype(np.float32)).to(device)
            action = pg_model.get_action(torch_state)

            new_state, reward, done, info = env.step(action)
            data.add(state, action, reward)

            state = new_state

            if done:
                episode_rewards_sum = np.sum(data.rewards)
                allRewards.append(episode_rewards_sum)

                optimizer.zero_grad()
                states = torch.FloatTensor(data.states).to(device)
                p_actions = pg_model(states)
                labels = torch.LongTensor(data.actions).to(device)

                pg_loss = crl(p_actions, labels)

                dis_rewards = discount_and_normalize_rewards(data.rewards)
                dis_rewards = torch.FloatTensor(dis_rewards).to(device)

                loss = torch.mean(pg_loss * dis_rewards)

                loss.backward()
                optimizer.step()

                if episode % 200 == 0:
                    print('episode: {} reward: {} all reward: {}'.format(episode, episode_rewards_sum,
                                                                         np.mean(allRewards[-10:])))
                data.clear()
                break
