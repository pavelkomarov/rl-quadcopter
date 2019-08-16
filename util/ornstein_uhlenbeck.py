import numpy as np
import copy

# We'll use a specific noise process that has some desired properties, called the Ornstein–Uhlenbeck process. It
# essentially generates random samples from a Gaussian (Normal) distribution, but each sample affects the next one such
# that two consecutive samples are more likely to be closer together than further apart. In this sense, the process is
# Markovian in nature.
class Noise:

	## Initialize parameters and noise process.
	# @param size, how many dimensions to generate noise in
	# @param mu, center of the distribution
	# @param theta, speed of mean reversion. Higher pushes noise more toward the center faster over updates.
	# @param sigma, standard deviation of the distribution
	def __init__(self, size, mu, theta, sigma):
		self.mu = mu*np.ones(size)
		self.theta = theta
		self.sigma = sigma
		self.state = copy.copy(self.mu)

	## Reset the internal state (= noise) to mean (mu).
	def reset(self):
		self.state[:] = self.mu

	## Update internal state and return it as a noise sample.
	def sample(self):
		x = self.state
		dx = self.theta*(self.mu - x) + self.sigma*np.random.randn(len(x))
		self.state = x + dx
		return self.state

