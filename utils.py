import numpy as np
import torch
import random


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.capacity = capacity
#         self.buffer = []
#         self.position = 0
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     def add(self, state, action, next_state, reward, done):
#         if len(self.buffer) < self.capacity:
#             self.buffer.append(None)
#         self.buffer[self.position] = (state, action, next_state, reward, done)
#         self.position = (self.position + 1) % self.capacity
#
#     def sample(self, batch_size):
#         batch = random.sample(self.buffer, batch_size)
#         state, action, next_state, reward, done = map(np.stack, zip(*batch))
#         return (
#             torch.FloatTensor(state).to(self.device),
#             torch.FloatTensor(action).to(self.device),
#             torch.FloatTensor(next_state).to(self.device),
#             torch.FloatTensor(reward).to(self.device),
#             torch.FloatTensor(done).to(self.device)
#         )
#
#     def __len__(self):
#         return len(self.buffer)