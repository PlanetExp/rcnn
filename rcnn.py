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
			'Wc1': tf.Variable(tf.random_normal([1, 1, 1, 32])),
			# 3x3 conv, 8 inputs, 16 outputs
			'Wc2': tf.Variable(tf.random_normal([3, 3, 32, 32])),  # shared
			# fully connected, 9*9*16 inputs, 512 outputs
			'Wd1': tf.Variable(tf.random_normal([9 * 9 * 32, 64])),
			# 512 inputs, 2 outputs (class prediction)
			'Wout': tf.Variable(tf.random_normal([64, output_size])),  # n_classes

			# biases
			'bc1': tf.Variable(tf.random_normal([32])),
			'bc2': tf.Variable(tf.random_normal([32])),
			'bd1': tf.Variable(tf.random_normal([64])),
			'bout': tf.Variable(tf.random_normal([output_size]))  # n_classes
		}

		# Instantiate functions once
		self.loss
		self.inference
		self.train
		self.predict
	
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

		# Unpack parameters
		X = self.X
		params = self.params

		# Convolution Layer
		conv1 = conv2d(X, params['Wc1'], params['bc1'])
		
		# Convolution Layer
		conv2 = conv2d(conv1, params['Wc2'], params['bc2'])
		
		# Convolution Layer, shared weight
		conv3 = conv2d(conv2, params['Wc2'], params['bc2'])
		
		# Fully connected layer
		# Reshape conv2 output to fit fully connected layer input
		fc1 = tf.reshape(conv3, [-1, params['Wd1'].get_shape().as_list()[0]])
		fc1 = tf.add(tf.matmul(fc1, params['Wd1']), params['bd1'])
		fc1 = tf.nn.relu(fc1)
		# Apply Dropout
		fc1 = tf.nn.dropout(fc1, self.dropout)

		# Output, class prediction
		out = tf.add(tf.matmul(fc1, params['Wout']), params['bout'])
		return out

	@define_scope
	def train(self):
		"""
		Train
		"""
		optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
		minimize = optimizer.minimize(self.loss)
		return minimize

	@define_scope
	def loss(self):
		"""
		Cost
		"""
		cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
			self.inference, self.y, name='cross_entropy')
		loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
		return loss

	@define_scope
	def predict(self):
		"""
		Predict
		"""
		correct = tf.nn.in_top_k(self.inference, self.y, 1)
		return tf.reduce_mean(tf.cast(correct, tf.float32))
