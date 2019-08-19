import numpy as np
from util.ornstein_uhlenbeck import Noise
from util.Actor import Actor
from util.Critic import Critic
from util.ReplayBuffer import ReplayBuffer
from keras.models import load_model

## Reinforcement Learning agent that learns using DDPG.
class Agent():
	def __init__(self, task):
		self.task = task
		self.state_size = task.state_size
		self.action_size = task.action_size

		# "Note that we will need two copies of each model--one local and one target. This is an extension of the "Fixed
		# Q Targets" technique from Deep Q-Learning, and is used to decouple the parameters being updated from the ones
		# that are producing target values."
		# Actor (Policy) Model and Critic (Value) Model
		self.actor = Actor(self.state_size, self.action_size, task.action_low, task.action_high)
		self.actor_ = Actor(self.state_size, self.action_size, task.action_low, task.action_high) # target
		self.critic = Critic(self.state_size, self.action_size)
		self.critic_ = Critic(self.state_size, self.action_size)
		# Initialize target model parameters with local model parameters so they agree
		self.critic_.model.set_weights(self.critic.model.get_weights())
		self.actor_.model.set_weights(self.actor.model.get_weights())

		# Noise process
		self.noise = Noise(self.action_size, mu=0, theta=0.15, sigma=5)

		# Replay memory
		self.batch_size = 64
		self.memory = ReplayBuffer(100000)

		# Algorithm parameters
		self.gamma = 0.99  # discount factor
		self.tau = 0.01  # for soft update of target parameters

	def reset_episode(self):
		self.noise.reset()
		return self.task.reset()

	# V(s) <- V(s) + a*(r' + y*V(s') - V(s)) eqn. 6.2 in Sutton and Barto
	# A(s,a) = r' + y*V(s') - V(s) advantage function
	def step(self, state, action, reward, next_state, done):
		 # Save experience / reward
		self.memory.add(state, action, reward, next_state, done)

		# Learn, if enough samples are available in memory
		if len(self.memory) > self.batch_size:
			self.learn(self.memory.sample(self.batch_size))

	## Returns actions for given state(s) as per current policy.
	def act(self, state, train=True):
		state = np.reshape(state, (-1, self.state_size)) # 2D num states given x state size
		action = self.actor.model.predict(state)[0]
		noise = self.noise.sample() if train else 0 # add some noise for exploration
		return list(action + noise)

	## Update policy and value parameters using given batch of experience tuples.
	def learn(self, experiences):
		# Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
		states = np.vstack([e.state for e in experiences])
		actions = np.array([e.action for e in experiences]).astype(np.float32).reshape(-1, self.action_size)
		rewards = np.array([e.reward for e in experiences]).astype(np.float32).reshape(-1, 1)
		dones = np.array([e.done for e in experiences]).astype(np.uint8).reshape(-1, 1)
		next_states = np.vstack([e.next_state for e in experiences])

		# Get predicted next-state actions and Q values from target models
		#	 Q_targets_next = critic_(next_state, actor_(next_state))
		actions_next = self.actor_.model.predict_on_batch(next_states)
		Q_targets_next = self.critic_.model.predict_on_batch([next_states, actions_next])

		# Compute Q targets for current states and train critic model (local)
		Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
		self.critic.model.train_on_batch(x=[states, actions], y=Q_targets)

		# Train actor model (local)
		action_gradients = np.reshape(self.critic.get_action_gradients([states, actions, 0]), (-1, self.action_size))
		self.actor.train_fn([states, action_gradients, 1])  # custom training function

		# Soft-update target models
		self.soft_update_targets(self.critic, self.critic_)
		self.soft_update_targets(self.actor, self.actor_)   

	# "Notice that after training over a batch of experiences, we could just copy our newly learned weights (from the
	# local model) to the target model. However, individual batches can introduce a lot of variance into the process, so
	# it's better to perform a soft update, controlled by the parameter tau."
	def soft_update_targets(self, local, target):
		local_weights = np.array(local.model.get_weights())
		target_weights = np.array(target.model.get_weights())

		# w_t = tau*w_l + (1-tau)*w_t -> w_t += tau*(w_l - w_t), stepping magnitude tau from targets to local
		target.model.set_weights(self.tau*local_weights + (1 - self.tau)*target_weights)

	## Store the policy to a file
	def save_policy(self, fileName):
		self.actor.model.save(fileName)

	## load back the policy from the file
	def load_policy(self, fileName):
		self.actor.model = load_model(fileName)

