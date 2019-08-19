from keras import layers, models, optimizers, backend

## Policy function, maps states -> actions. Deep deterministic policy gradients method.
class Actor:

	## Initialize parameters and build model.
	# @param state_size (int): Dimension of each state
	# @param action_size (int): Dimension of each action
	# @param action_low (array): Min value of each action dimension
	# @param action_high (array): Max value of each action dimension
	def __init__(self, state_size, action_size, action_low, action_high):
		# Define input layer (states)
		states = layers.Input(shape=(state_size,), name='states')
		# Simple network: "Also, what we know about good CNN design from supervised learning land doesn’t seem to apply
		# to reinforcement learning land, because you’re mostly bottlenecked by credit assignment / supervision bitrate,
		# not by a lack of a powerful representation. Your ResNets, batchnorms, or very deep networks have no power
		# here." -Andrej Karpathy
		net = layers.Dense(units=128, kernel_regularizer=layers.regularizers.l2(1e-6))(states)
		net = layers.BatchNormalization()(net)
		net = layers.Activation("relu")(net)
		net = layers.Dense(units=64, kernel_regularizer=layers.regularizers.l2(1e-6))(net)
		net = layers.BatchNormalization()(net) # batch norm to try to keep the actions from just hitting the rails
		net = layers.Activation("relu")(net)
		# Add final output layer with sigmoid activation
		raw_actions = layers.Dense(units=action_size, activation='sigmoid', name='raw_actions',
			kernel_initializer=layers.initializers.RandomUniform(minval=-0.005, maxval=0.005))(net)
		# Scale [0, 1] output for each action dimension to proper range
		action_range = action_high - action_low
		actions = layers.Lambda(lambda x: (x*action_range) + action_low, name='actions')(raw_actions)
		# Create Keras model
		self.model = models.Model(inputs=states, outputs=actions)

		# Define loss function using action value (Q value) gradients
		action_gradients = layers.Input(shape=(action_size,)) # Why the FUCK is this the loss function?
		loss = backend.mean(-action_gradients*actions)

		# Define optimizer and training function
		optimizer = optimizers.Adam(lr=0.001)
		updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
		self.train_fn = backend.function( # this is the legacy way to do this
			inputs=[self.model.input, action_gradients, backend.learning_phase()],
			outputs=[],
			updates=updates_op)
