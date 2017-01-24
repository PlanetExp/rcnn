"""RCNN model
"""
import tensorflow as tf
from define_scope import define_scope  # custom decorators


class Model:
	def __init__(self, X, y, output_size=None,
		learning_rate=1e-5, learning_rate_decay=0.95,
		reg=1e-5, dropout=0.5, verbose=False):
		"""
		Initalize the model.

		Inputs:
		- output_size: number of classes C
		- learning_rate: Scalar giving learning rate for optimization.
		- learning_rate_decay: Scalar giving factor used to decay the learning rate
		  after each epoch.
		"""
		self.X = X
		self.y = y
		self.learning_rate = learning_rate
		self.learning_rate_decay = learning_rate_decay
		self.dropout = dropout

		# Store layers weight & bias
		self.params = {
			# input is [1, 9, 9, 1]
			# 3x3 conv, 1 input, 8 outputs
			'Wc1': tf.Variable(tf.random_normal([1, 1, 1, 32]), name='Wc1'),
			# 3x3 conv, 8 inputs, 16 outputs
			'Wc2': tf.Variable(tf.random_normal([3, 3, 32, 32]), name='Wc2'),  # shared
			# fully connected, 9*9*16 inputs, 512 outputs
			'Wd1': tf.Variable(tf.random_normal([9 * 9 * 32, 32])),
			# 512 inputs, 2 outputs (class prediction)
			'Wout': tf.Variable(tf.random_normal([32, output_size])),  # n_classes

			# biases
			'bc1': tf.Variable(tf.random_normal([32])),
			'bc2': tf.Variable(tf.random_normal([32])),
			'bd1': tf.Variable(tf.random_normal([32])),
			'bout': tf.Variable(tf.random_normal([output_size]))  # n_classes
		}

		# Instantiate functions once
		# self.loss
		# self.inference
		# self.train
		# self.predict
	
	@define_scope
	def inference(self):
		"""
		Setting up inference of model

		Returns:
			logits
		"""
		# Create some wrappers for simplicity
		def conv2d(X, W, b, strides=1):
			# Conv2D wrapper, with bias and relu activation
			X = tf.nn.conv2d(X, W, strides=[1, strides, strides, 1], padding='SAME')
			X = tf.nn.bias_add(X, b)
			return tf.nn.relu(X)

		def weight_variable(shape):
			"""Create a weight variable with appropriate initialization."""
			initial = tf.truncated_normal(shape, stddev=0.1)
			return tf.Variable(initial)

		def bias_variable(shape):
			"""Create a bias variable with appropriate initialization."""
			initial = tf.constant(0.1, shape=shape)
			return tf.Variable(initial)

		def variable_summaries(var, name):
			"""Attach a lot of summaries to a Tensor."""
			with tf.name_scope('summaries'):
				mean = tf.reduce_mean(var)
			tf.scalar_summary('mean/' + name, mean)
			with tf.name_scope('stddev'):
				stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			tf.scalar_summary('stddev/' + name, stddev)
			tf.scalar_summary('max/' + name, tf.reduce_max(var))
			tf.scalar_summary('min/' + name, tf.reduce_min(var))
			tf.histogram_summary(name, var)

		def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
			"""Rusable layer code for tensorboard naming

			See: https://github.com/tensorflow/tensorflow/blob/r0.11/tensorflow/examples/tutorials/mnist/mnist_with_summaries.py

			"""
			with tf.name_scope(layer_name):
				# This Variable will hold the state of the weights for the layer
				with tf.name_scope('weights'):
					weights = weight_variable([input_dim, output_dim])
					variable_summaries(weights, layer_name + '/weights')
				with tf.name_scope('biases'):
					biases = bias_variable([output_dim])
					variable_summaries(biases, layer_name + '/biases')
				with tf.name_scope('Wx_plus_b'):
					preactivate = tf.matmul(input_tensor, weights) + biases
					tf.histogram_summary(layer_name + '/pre_activations', preactivate)
			activations = act(preactivate, name='activation')
			tf.histogram_summary(layer_name + '/activations', activations)
			return activations

		def conv_relu(input_tensor, kernel_shape, bias_shape):
			# Create variable named "weights".
			weights = tf.get_variable("weights", kernel_shape,
				initializer=tf.random_normal_initializer())
			# Create variable named "biases".
			biases = tf.get_variable("biases", bias_shape,
				initializer=tf.constant_initializer(0.0))
			conv = tf.nn.conv2d(input_tensor, weights,
				strides=[1, 1, 1, 1], padding='SAME')
			return tf.nn.relu(conv + biases)

		def board_filter(input_board):
			with tf.variable_scope('conv1'):
				relu1 = conv_relu(input_board, [3, 3, 32, 32], [32])
			with tf.variable_scope('conv2'):
				return conv_relu(relu1, [3, 3, 32, 32], [32])


		# Unpack parameters
		X = self.X
		params = self.params

		# Convolution Layer
		with tf.variable_scope('conv1'):
			conv1 = conv_relu(X, [1, 1, 1, 32], [32], 'conv1')
			# conv1 = conv2d(X, params['Wc1'], params['bc1'])
		
		# Convolution Layer
		with tf.variable_scope('board_filters') as scope:
			# conv2 = conv2d(conv1, params['Wc2'], params['bc2'])
			result1 = board_filter(conv1, [3, 3, 32, 32], [32], 'conv2')
		
			# Convolution Layer, 
			# Share weights within scope
			scope.reuse_variables()
			# conv3 = conv2d(conv2, params['Wc2'], params['bc2'])
			result2 = board_filter(conv2, [3, 3, 32, 32], [32], 'conv3')
		
		# with tf.variable_scope("foo"):
		# 	v = tf.get_variable("v", [1])
		# 	tf.get_variable_scope().reuse_variables()
		# 	v1 = tf.get_variable("v", [1])
		# assert v1 is v

		# Fully connected layer
		# Reshape conv2 output to fit fully connected layer input
		fc1 = tf.reshape(conv3, [-1, 9 * 9 * 32])
		fc1 = tf.add(tf.matmul(fc1, params['Wd1']), params['bd1'])
		fc1 = tf.nn.relu(fc1)
		# Apply Dropout
		with tf.name_scope('dropout'):
			tf.scalar_summary('dropout_keep_probability', self.dropout)
			fc1 = tf.nn.dropout(fc1, self.dropout)

		# Output, class prediction
		# out = tf.add(tf.matmul(fc1, params['Wout']), params['bout'])
		out = nn_layer(fc1, 32, 2, 'out', act=tf.identity)
		return out

	@define_scope
	def train(self):
		"""
		Train
		"""
		with tf.name_scope('train'):
			optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
			minimize = optimizer.minimize(self.loss)
		return minimize

	@define_scope
	def loss(self):
		"""
		Cost
		"""
		with tf.name_scope('cross_entopy'):
			diff = tf.nn.sparse_softmax_cross_entropy_with_logits(
				logits=self.inference, labels=self.y)
			with tf.name_scope('total'):
				cross_entropy = tf.reduce_mean(diff)
		tf.summary.scalar('cross_entropy', cross_entropy)
		return cross_entropy

	@define_scope
	def predict(self):
		"""
		Predict
		"""
		with tf.name_scope('accuracy'):
			with tf.name_scope('correct_prediction'):
				correct_prediction = tf.nn.in_top_k(self.inference, self.y, 1)
			with tf.name_scope('accuracy'):
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.scalar_summary('accuracy', accuracy)
		return accuracy

