import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
	"""
	Structured SVM loss function, naive implementation (with loops).

	Inputs have dimension D, there are C classes, and we operate on minibatches
	of N examples.

	Inputs:
	- W: A numpy array of shape (D, C) containing weights.
	- X: A numpy array of shape (N, D) containing a minibatch of data.
	- y: A numpy array of shape (N,) containing training labels; y[i] = c means
		that X[i] has label c, where 0 <= c < C.
	- reg: (float) regularization strength

	Returns a tuple of:
	- loss as single float
	- gradient with respect to weights W; an array of same shape as W
	"""

	num_train = X.shape[0]			# N : num of training images
	num_classes = W.shape[1]		# C : num of classes
	dW = np.zeros(W.shape)			# gradient : same shape as W (weight matrix)
	loss = 0.0
	# print(num_classes, num_train)

	# calculating the scores (s = X.dot(W))
	# calculating the loss (loss_i = sum(max(0, s_j - s_yi + 1)))

	for i in range(num_train):
		scores = X[i].dot(W)

		for j in range(num_classes):
			if j == y[i]:
				continue
			margin = scores[j] - scores[y[i]] + 1
			if margin > 0:
				loss += margin
				dW[:, j] += X[i].T
				dW[:, y[i]] += -X[i].T
				# print(dW[:,j])
				# print(dW[:,y[i]])
	
	loss /= num_train

	# regularizationon-loss (lambda*R; using L2 regularization loss where R = sum(W*W))
	reg_loss = reg*np.sum(W*W)
	loss += reg_loss	# adding Regularization loss
	
	# taking into account the regularization factor for dW += 2*reg*W
	dW += reg*W

	return loss, dW


def svm_loss_vectorized(W, X, y, reg):
	"""
	Structured SVM loss function, vectorized implementation.

	Inputs and outputs are the same as svm_loss_naive.
	"""
	loss = 0.0
	dW = np.zeros(W.shape) # initialize the gradient as zero

	#############################################################################
	# TODO:                                                                     #
	# Implement a vectorized version of the structured SVM loss, storing the    #
	# result in loss.                                                           #
	#############################################################################
	pass
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################


	#############################################################################
	# TODO:                                                                     #
	# Implement a vectorized version of the gradient for the structured SVM     #
	# loss, storing the result in dW.                                           #
	#                                                                           #
	# Hint: Instead of computing the gradient from scratch, it may be easier    #
	# to reuse some of the intermediate values that you used to compute the     #
	# loss.                                                                     #
	#############################################################################
	pass
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################

	return loss, dW
