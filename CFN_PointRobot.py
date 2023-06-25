import math
import random
import os
import gym
import numpy as np
import csv
import torch
import torch.nn as nn

import torch.nn.functional as F
import pickle
from IPython.display import clear_output
from gym.wrappers import TimeLimit
from torch.distributions import Categorical
from PointRobotEnv import PointEnv_MultiStep_Two_goal
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

now_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


def save_variable(v,filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename

# 读取变量
def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


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


class Network(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Network, self).__init__()

        # Edge flow network architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], -1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.softplus(self.l3(q1))
        return q1


class CFN(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim,
            max_action,
            uniform_action_size,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):

        self.network = Network(state_dim, action_dim, hidden_dim).to(device)
        self.network_optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-4)
        self.retrieval = Retrieval(state_dim, action_dim).to(device)
        self.retrieval_optimizer = torch.optim.Adam(self.retrieval.parameters(), lr=3e-5)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.uniform_action_size = uniform_action_size
        self.uniform_action = np.random.uniform(low=-max_action, high=max_action, size=(uniform_action_size, action_dim))
        self.uniform_action = torch.Tensor(self.uniform_action).to(device)

    def select_action(self, state, is_max):
        sample_action = np.random.uniform(low=-self.max_action, high=self.max_action, size=(1000, action_dim))
        with torch.no_grad():
            sample_action = torch.Tensor(sample_action).to(device)
            state = torch.FloatTensor(state.reshape(1, -1)).repeat(1000, 1).to(device)
            edge_flow = self.network(state, sample_action).reshape(-1)
            if is_max == 0:
                idx = Categorical(edge_flow.float()).sample(torch.Size([1]))
                action = sample_action[idx[0]]
            elif is_max == 1:
                action = sample_action[edge_flow.argmax()]
        return action.cpu().data.numpy().flatten()

    def set_uniform_action(self):
        self.uniform_action = np.random.uniform(low=-max_action, high=max_action, size=(self.uniform_action_size, action_dim))
        self.uniform_action = torch.Tensor(self.uniform_action).to(device)
        return self.uniform_action

    def train(self, replay_buffer, frame_idx, batch_size=256, max_episode_steps=50, sample_flow_num=100):
        # Sample replay buffer
        state, action, reward, next_state, not_done = replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        not_done = torch.FloatTensor(np.float32(not_done)).to(device)

        with torch.no_grad():
            uniform_action = np.random.uniform(low=-max_action, high=max_action,
                                               size=(batch_size, max_episode_steps, sample_flow_num, action_dim))
            uniform_action = torch.Tensor(uniform_action).to(device)
            current_state = next_state.repeat(1, 1, sample_flow_num).reshape(batch_size, max_episode_steps,
                                                                             sample_flow_num, -1)
            inflow_state = self.retrieval(current_state, uniform_action)
            inflow_state = torch.cat([inflow_state, state.reshape(batch_size, max_episode_steps, -1, state_dim)], -2)
            uniform_action = torch.cat([uniform_action, action.reshape(batch_size, max_episode_steps, -1, action_dim)], -2)
        edge_inflow = self.network(inflow_state, uniform_action).reshape(batch_size, max_episode_steps, -1)
        epi = torch.Tensor([1.0]).repeat(batch_size*max_episode_steps).reshape(batch_size,-1).to(device)
        inflow = torch.log(torch.sum(torch.exp(torch.log(edge_inflow)), -1) + epi)

        with torch.no_grad():
            uniform_action = np.random.uniform(low=-max_action, high=max_action,
                                               size=(batch_size, max_episode_steps, sample_flow_num, action_dim))
            uniform_action = torch.Tensor(uniform_action).to(device)
            outflow_state = next_state.repeat(1, 1, (sample_flow_num+1)).reshape(batch_size, max_episode_steps, (sample_flow_num+1), -1)
            last_action = torch.Tensor([0.0]).reshape([1,1,1]).repeat(batch_size,1,1).to(device)
            last_action = torch.cat([action[:,1:,:], last_action], -2)
            uniform_action = torch.cat([uniform_action, last_action.reshape(batch_size, max_episode_steps, -1, action_dim)], -2)

        edge_outflow = self.network(outflow_state, uniform_action).reshape(batch_size, max_episode_steps, -1)

        outflow = torch.log(torch.sum(torch.exp(torch.log(edge_outflow)), -1) + epi)

        network_loss = F.mse_loss(inflow * not_done, outflow * not_done, reduction='none') + F.mse_loss(inflow * done_true, (torch.cat([reward[:,:-1],torch.log((reward*(sample_flow_num+1))[:,-1]).reshape(batch_size,-1)], -1)) * done_true, reduction='none')
        network_loss = torch.mean(torch.sum(network_loss, dim = 1))
        print(network_loss)
        self.network_optimizer.zero_grad()
        network_loss.backward()
        self.network_optimizer.step()

        if frame_idx % 5 == 0:
            pre_state = self.retrieval(next_state, action)
            retrieval_loss = F.mse_loss(pre_state, state)
            print(retrieval_loss)

            # Optimize the network
            self.retrieval_optimizer.zero_grad()
            retrieval_loss.backward()
            self.retrieval_optimizer.step()

writer = SummaryWriter(log_dir="runs/CFN_PointRobot_"+now_time)

max_episode_steps = 12
env = PointEnv_MultiStep_Two_goal()
test_env = PointEnv_MultiStep_Two_goal()

action_dim = env.action_space.shape[0]
state_dim = env.observation_space.shape[0]
max_action = float(env.action_space.high[0])
hidden_dim = 256
uniform_action_size = 1000

policy = CFN(state_dim, action_dim, hidden_dim, max_action, uniform_action_size)
policy.retrieval.load_state_dict(torch.load('Retrieval_PointRobot.pkl'))

replay_buffer_size = 8000
replay_buffer = ReplayBuffer(replay_buffer_size)

max_frames  = 1666
start_timesteps = 130
frame_idx = 0
rewards = []
test_rewards = []
x_idx = []
batch_size  = 128
test_epoch = 0
expl_noise = 0.4
sample_flow_num = 99
repeat_episode_num = 5
sample_episode_num = 1000

done_true = torch.zeros(batch_size, max_episode_steps).to(device)
for i in done_true:
    i[max_episode_steps-1] = 1

while frame_idx < max_frames:
    # print(frame_idx)
    state = env.reset()
    episode_reward = 0

    state_buf = []
    action_buf = []
    reward_buf = []
    next_state_buf = []
    done_buf = []

    for step in range(max_episode_steps):
        with torch.no_grad():
            action = policy.select_action(state, 0)

        next_state, reward, done, _ = env.step(action)
        done_bool = float(1. - done)

        state_buf.append(state)
        action_buf.append(action)
        reward_buf.append(reward)
        next_state_buf.append(next_state)
        done_buf.append(done_bool)

        state = next_state
        episode_reward += reward

        if done:
            frame_idx += 1
            print(frame_idx)
            replay_buffer.push(state_buf, action_buf, reward_buf, next_state_buf, done_buf)
            break

        if frame_idx >= start_timesteps and step % 2 == 0:
            policy.train(replay_buffer, frame_idx, batch_size, max_episode_steps, sample_flow_num)

    if frame_idx > start_timesteps and frame_idx % 25 == 0:
        print(frame_idx)
        test_epoch += 1
        avg_test_episode_reward = 0
        for i in range(repeat_episode_num):
            test_state = test_env.reset()
            test_episode_reward = 0
            for s in range(max_episode_steps):
                test_action = policy.select_action(np.array(test_state), 1)
                test_next_state, test_reward, test_done, _ = test_env.step(test_action)
                test_state = test_next_state
                test_episode_reward += test_reward
                if test_done:
                    break
            avg_test_episode_reward += test_episode_reward

        torch.save(policy.network.state_dict(), "runs/CFN_PointRobot_"+now_time+'.pkl')
        writer.add_scalar("CFN_PointRobot_reward", avg_test_episode_reward / repeat_episode_num, global_step=frame_idx * max_episode_steps)

    rewards.append(episode_reward)




