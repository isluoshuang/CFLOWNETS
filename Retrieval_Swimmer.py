import gym
gym.make('Swimmer-v3')
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from gym.wrappers import TimeLimit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RewardShapeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.x = 0
        self.y = 0

    def reset(self, **kwargs):
        self.x = 0
        self.y = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        position_x = info['x_velocity']
        position_y = info['y_velocity']
        self.x += position_x
        self.y += position_y
        if done:
            return observation, (self.x**2 + self.y**2)**0.5, done, info
        else:
            return observation, 0.0, done, info


class Retrieval(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Retrieval, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, state_dim)

    def forward(self, state, action):
        sa = torch.cat([state, action], -1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)

        return q1


class TrainRetrieval(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):

        self.retrieval = Retrieval(state_dim, action_dim).to(device)
        self.retrieval_target = copy.deepcopy(self.retrieval)
        self.retrieval_optimizer = torch.optim.Adam(self.retrieval.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        pre_state = self.retrieval(next_state, action)

        # Compute retrieval loss
        retrieval_loss = F.mse_loss(pre_state, state)
        print(retrieval_loss)

        # Optimize the retrieval
        self.retrieval_optimizer.zero_grad()
        retrieval_loss.backward()
        self.retrieval_optimizer.step()

    def save(self, filename):
        torch.save(self.retrieval.state_dict(), filename + "_retrieval")
        torch.save(self.retrieval_optimizer.state_dict(), filename + "_retrieval_optimizer")


max_episode_steps = 50
env = RewardShapeWrapper(TimeLimit(gym.make('Swimmer-v3'), max_episode_steps=50))
test_env = RewardShapeWrapper(TimeLimit(gym.make('Swimmer-v3'), max_episode_steps=50))


action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
max_action = float(env.action_space.high[0])
hidden_dim = 256

policy = TrainRetrieval(state_dim, action_dim, max_action)


replay_buffer_size = 100000
replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

max_frames = 100000
start_timesteps = 1000

frame_idx = 0
rewards = []
test_rewards = []
batch_size = 256
test_epoch = 0
expl_noise = 0.1
episode_reward = 0.0
episode_timesteps = 0
episode_num = 0

state, done = env.reset(), False

while frame_idx < max_frames:

    episode_timesteps += 1
    action = env.action_space.sample()

    next_state, reward, done, _ = env.step(action)
    episode_reward += reward
    done_bool = float(done) if episode_timesteps < max_episode_steps else 0
    if done:
        replay_buffer.add(state, action, next_state, episode_reward, done_bool)
    else:
        replay_buffer.add(state, action, next_state, reward, done_bool)
    state = next_state

    if frame_idx >= start_timesteps:
        policy.train(replay_buffer, batch_size)

    if frame_idx >= start_timesteps and frame_idx % 10000 == 0:
        torch.save(policy.retrieval.state_dict(), 'retrieval_swimmer_sparse.pkl')

    if done:
        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    frame_idx += 1


