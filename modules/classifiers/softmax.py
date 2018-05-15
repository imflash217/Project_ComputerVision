import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
	"""
	Softmax loss function, naive implementation (with loops)

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
	#############################################################################
	# Compute the softmax loss and its gradient using explicit loops.     #
	# Store the loss in loss and the gradient in dW. If you are not careful     #
	# here, it is easy to run into numeric instability. Don't forget the        #
	# regularization!                                                           #
	#############################################################################

	# Initialize the loss and gradient to zero.
	loss = 0.0
	dW = np.zeros_like(W)
	num_train = X.shape[0]
	num_classes = W.shape[1]

	for i in range(num_train):
		# calculae the scores (i.e. un-normalized log probabilities)
		scores = X[i].dot(W)		# shape: (N, C)
		scores -= np.max(scores)	# normalization TRICK to prevent blowup due to exponentiation
		# exponentiate the scores to get the un-normalized probabilities
		scores_exp = np.exp(scores)
		# normalize the scores per image to get the normalized probabilities
		scores_exp_norm = scores_exp / np.sum(scores_exp)

		# calculating the loss (loss_i = -log(scores[y[i]]))
		loss += -1 * np.log(scores_exp_norm[y[i]])
		
		for j in range(num_classes):
			if y[i] == j:
				dW[:, j] += X[i].T * (-1 + scores_exp_norm[j])
			else:
				# dW[:, j] += (X[i].T)*scores_exp[j]/np.sum(scores_exp)
				dW[:,j] += (X[i].T) * scores_exp_norm[j]


	loss /= num_train
	loss += reg*np.sum(W*W)		# adding regularization

	dW /= num_train
	dW += 2*reg*W				# adding regularization term

	return loss, dW

#############################################################################

def softmax_loss_vectorized(W, X, y, reg):
	"""
	Softmax loss function, vectorized version.
	
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

	#############################################################################
	# Compute the softmax loss and its gradient using no explicit loops.  #
	# Store the loss in loss and the gradient in dW. If you are not careful     #
	# here, it is easy to run into numeric instability. Don't forget the        #
	# regularization!                                                           #
	#############################################################################
	#############################################################################

	# Initialize the loss and gradient to zero.
	loss = 0.0
	dW = np.zeros_like(W)
	num_train = X.shape[0]

	scores = X.dot(W)			# shape: (N, C)
	scores -= np.transpose([np.max(scores, axis=1)])		# normalization trick to stop exponentiation boom
	scores_exp = np.exp(scores)
	scores_exp_norm = scores_exp / np.sum(scores_exp, axis=1)[:,None]

	loss += np.sum(-1*np.log(scores_exp_norm[np.arange(num_train), y]))
	loss /= num_train
	loss += reg*np.sum(W*W)

	dSoft = scores_exp_norm.copy()
	dSoft[np.arange(num_train), y] += -1
	dW += X.T.dot(dSoft)  	

	# dW += X.T.dot(scores_exp_norm)			# ERROR
	# dW[:,y] += -1 * np.sum(X, axis=1).T		# ERROR

	dW /= num_train
	dW += 2*reg*W

	return loss, dW

#############################################################################





