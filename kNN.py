import numpy as np

class NearestNeighbor:
    def __init__(self):
        pass

# training phase of the classifier
def train(self, X, Y):
    """ X is 'N x D' where each row is an example image.
        Y is 'N x 1' : 1-Dimension of size N
    """

    # the Nearest Neighbor classifier simply remembers all the training data
    self.X_train = X
    self.Y_train = Y


# testing phase
def predict(self, X):
    """ X is 'N x D' where each row is an example/image we wish to PREDICT label for. """
    
    num_test = X.shape[0]  #number of test images

    # making sure that the output type matche sthe input type
    Y_pred = np.zeros(num_test, dtype = self.Y_train.dtype)

    # looping over all test images
    for i in range(num_test):

        # Finding the L1-distance (Manhattan distance) for each test image wrt. EVERY training image
        L1_distances = np.sum(np.abs(self.X_train - X[i,:]), axis=1)
        
        # Finding the index of the minimum distance
        min_index = np.argmin(L1_distances)

        # Extracting the 'LABEL' based on the min. distance index from the training data &
        # assigning the label to the current test image
        Y_pred[i] = self.Y_train[min_index]
    
    return Y_pred
