import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
	"""
	A two-layer fully connected neural network.
	The network:
		dim(input) = N
		dim(hidden) = H
		dim(output) = C

		loss-function : Softmax
		Regularization : L2
		Nonlinearilty : ReLU

	The network has the following architecture:
	Input -> FC-layer_1 -> ReLU -> FC-layer_2 -> softmax

	The output of "FC_layer_2" are the scores for each class in C

	"""

	def __init__(self, input_size, hidden_size, output_size, std=1e-4):
		"""
		Initializes the model.
		Weights(W1, W2) are initialized to small random values.
		Biases(b1, b2) are initialized to ZERO
		
		All weights and biases are stored in a 'dict' self.params with following keys:
			W1 : First-layer weights : shape = (D,H)
			b1 : First-layer biases  : shape = (H,)
			W2 : Second-layer weights: shape = (H,C)
			b2 : Second-layer biases : shape = (C,)
		
		Inputs:
		- input_size  : The dimension D of the input data
		- hidden_size : The number of neurons H in the hidden layer
		- output_size : The number of classes

		"""
		self.params = {}
		self.params['W1'] = std * np.random.randn(input_size, hidden_size)
		self.params['b1'] = np.zeros(hidden_size)
		self.params['W2'] = std * np.random.randn(hidden_size, output_size)
		self.params['b2'] = np.zeros(output_size)
