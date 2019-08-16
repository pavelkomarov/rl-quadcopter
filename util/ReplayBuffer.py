import random
from collections import namedtuple, deque

## Fixed-size buffer to store experience tuples.
class ReplayBuffer:

	## Initialize a ReplayBuffer object.
	# @param buffer_size: maximum size of buffer
	def __init__(self, buffer_size):
		self.memory = deque(maxlen=buffer_size)
		self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

	## Add a new experience to memory.
	# @param state, s
	# @param action, a
	# @param reward, r'
	# @param next_state s'
	# @param done Boolean, whether this step ends an episode
	def add(self, state, action, reward, next_state, done):
		self.memory.append(self.experience(state, action, reward, next_state, done))

	## Sample a batch of experiences from memory, uniformly at random.
	# @param batch_size The number of examples returned
	# @return A list of random experiences
	def sample(self, batch_size=64):
		return random.sample(self.memory, k=batch_size)

	## so python's builtin len() works with this object	
	# @return the current size of internal memory
	def __len__(self):
		return len(self.memory)
