import numpy as np
from task import Task

# Updated to have a bias so it works with my task, which starts at zero. Otherwise action was always zero.
class PolicySearch_Agent():
	def __init__(self, task):
		# Task (environment) information
		self.task = task

		self.w = np.random.normal(size=(12, 4), scale=10)
		self.b = np.random.normal(size=(4,), scale=1)

		# Score tracker and learning parameters
		self.best_w = None
		self.best_b = None
		self.best_score = -np.inf
		self.noise_scale = 0.1

		# Episode variables
		self.reset_episode()

	def reset_episode(self):
		self.total_reward = 0.0
		self.count = 0
		state = self.task.reset()
		return state

	# Save experience and reward
	def step(self, reward, done):
		self.total_reward += reward
		self.count += 1

		# Learn, if at end of episode
		if done:
			self.learn()

	# Choose action based on given state and policy
	def act(self, state):		
		action = np.dot(state, self.w) + self.b # simple linear policy
		return action

	# Learn by random policy search, using a reward-based score
	def learn(self):
		self.score = self.total_reward / float(self.count) if self.count else 0.0
		if self.score > self.best_score:
			self.best_score = self.score
			self.best_w = self.w
			self.best_b = self.b
			self.noise_scale = max(0.5*self.noise_scale, 0.01)
		else:
			self.w = self.best_w
			self.b = self.best_b
			self.noise_scale = min(2*self.noise_scale, 3.2)
		self.w += self.noise_scale*np.random.normal(size=self.w.shape) # equal noise in all directions
		self.b += self.noise_scale*np.random.normal(size=self.b.shape)
		
