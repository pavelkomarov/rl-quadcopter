import numpy as np
from util.physics_sim import PhysicsSim

class Task():
	"""Task (environment) that defines the goal and provides feedback to the agent."""
	# State = [x, y, z, phi, theta, psi, v_x, v_y, v_z, w_phi, w_theta, w_psi]
	# default task is start at rest at the origin and end at rest at (10,0,0)
	def __init__(self, init_state=[0.]*12, runtime=5., target_state=[0., 0., 10.]+[0.]*9):
		"""Initialize a Task object.
		Params
		======
			init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
			init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
			init_angle_velocities: initial radians/second for each of the three Euler angles
			runtime: time limit for each episode
			target_pos: target/goal (x,y,z) position for the agent
		"""
		self.init_state = init_state
		self.target_state = target_state
		
		self.sim = PhysicsSim(init_state[:6], init_state[6:9], init_state[9:], runtime)
		
		self.action_low = 0
		self.action_high = 900
		self.state_size = 12 # state now has 12 entries, not 6*3 repeats
		self.action_size = 4

	# reward based on distance to goal and instantaneous force applied (minimize accelerations, not velocities)
	def get_reward(self, rotor_speeds, done_):
		if done_ and self.sim.time < self.sim.runtime: # if the quadcopter crashes
			return -1, True # punish, and we're done

		if self.sim.pose[2] < 10: # if haven't reached the target height yet, give proportional reward
			# always +, more + closer to target
			#print(self.sim.v[2])
			return np.clip(self.sim.v[2]/10 + (10 - np.abs(self.target_state[2] - self.sim.pose[2]))/10, -1, 1), done_
		else: # if reach the target height, give bonus reward, and we're done
			return 1 + (self.sim.runtime - self.sim.time), True

	## Uses action to obtain next state, reward, done.
	# Action-repeats just does the same action three times and concatenates raw positions as a state. I've
	# chosen to instead get rid of this and return velocities directly, in keeping with my 12-entry state.
	# Might make this all noisier, but shouldnt I just be able to triple the interval between time steps?
	def step(self, rotor_speeds):
		#state = np.concatenate((self.sim.pose, self.sim.v, self.sim.angular_v))
		done_ = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
		reward, done = self.get_reward(rotor_speeds, done_)
		next_state = np.concatenate((self.sim.pose, self.sim.v, self.sim.angular_v))
		return next_state, reward, done, self.sim.time

	def reset(self):
		"""Reset the sim to start a new episode."""
		self.sim.reset()
		return self.init_state
