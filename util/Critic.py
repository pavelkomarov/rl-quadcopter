from keras import layers, models, optimizers, backend

## Q function, maps state, action -> value. Note there is a way to just use the delta of the Value function alone,
# which maps state -> value, in the agent, but this provided implementation uses Q.
class Critic:

	## Initialize parameters and build model.
	# @param state_size (int): Dimension of each state
	# @param action_size (int): Dimension of each action
	def __init__(self, state_size, action_size):
		# Define input layers
		states = layers.Input(shape=(state_size,), name='states')
		actions = layers.Input(shape=(action_size,), name='actions')

		# Add hidden layer(s) for state pathway
		net_states = layers.Dense(units=128, kernel_regularizer=layers.regularizers.l2(1e-6))(states)
		net_states = layers.BatchNormalization()(net_states)
		net_states = layers.Activation("relu")(net_states)
		net_states = layers.Dense(units=64, kernel_regularizer=layers.regularizers.l2(1e-6),
			activation='relu')(net_states)

		# Add hidden layer(s) for action pathway
		net_actions = layers.Dense(units=128, kernel_regularizer=layers.regularizers.l2(1e-6),
			activation='relu')(actions)
		net_actions = layers.Dense(units=64, kernel_regularizer=layers.regularizers.l2(1e-6),
			activation='relu')(net_actions)

		# Combine state and action pathways
		net = layers.Add()([net_states, net_actions])
		net = layers.Activation('relu')(net)

		# Add more layers to the combined network if needed

		# Add final output layer to prduce action values (Q values)
		Q_values = layers.Dense(units=1, name='q_values',
			kernel_initializer=layers.initializers.RandomUniform(minval=-0.005, maxval=0.005))(net)

		# Create Keras model
		self.model = models.Model(inputs=[states, actions], outputs=Q_values)

		# Define optimizer and compile model for training with built-in loss function
		optimizer = optimizers.Adam(lr=0.001)
		self.model.compile(optimizer=optimizer, loss='mse')

		# Compute action gradients (derivative of Q values w.r.t. to actions)
		action_gradients = backend.gradients(Q_values, actions)

		# Define an additional function to fetch action gradients (to be used by actor model)
		self.get_action_gradients = backend.function(
			inputs=[*self.model.input, backend.learning_phase()],
			outputs=action_gradients)
