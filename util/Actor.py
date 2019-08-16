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
		# Add hidden layers
		# TODO: Try different layer sizes, activations, add batch normalization, regularizers, etc.		
		net = layers.Dense(units=32, activation='relu')(states)
		net = layers.Dense(units=64, activation='relu')(net)
		net = layers.Dense(units=32, activation='relu')(net)
		# Add final output layer with sigmoid activation
		raw_actions = layers.Dense(units=action_size, activation='sigmoid', name='raw_actions')(net)
		# Scale [0, 1] output for each action dimension to proper range
		action_range = action_high - action_low
		actions = layers.Lambda(lambda x: (x*action_range) + action_low, name='actions')(raw_actions)
		# Create Keras model
		self.model = models.Model(inputs=states, outputs=actions)

		# Define loss function using action value (Q value) gradients
		# TODO: Incorporate any additional losses here (e.g. from regularizers)
		action_gradients = layers.Input(shape=(action_size,)) # Why the FUCK is this the loss function?
		loss = backend.mean(-action_gradients*actions)

		# Define optimizer and training function
		optimizer = optimizers.Adam() # this shit is all the legacy way to do this. What's the right way?
		updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
		self.train_fn = backend.function(
			inputs=[self.model.input, action_gradients, backend.learning_phase()],
			outputs=[],
			updates=updates_op)
