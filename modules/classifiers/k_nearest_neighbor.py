import numpy as np 

class KNearestNeighbor:
    """
    K-Nearest-Neighbor classifier with L1, L2 distance metric
    """

    def __init__(self):
        pass

    def train(self, X, Y):
        """
        TRAIN the CLASSIFIER:
        For k-Nearest Neighbor training is all bout memorizing the training data and its labels

        Inputs:
        - X: Training Data; a numpy array of shape (num_train, D) containing the training data consisting of 
            'num_train' samples each of dimemsion 'D'

        - Y: Training Data's Labels; a numpy array of shape (N,) consisting the training lables,
            where Y[i] is the LABEL of X[i]
        """

        self.X_train = X
        self.Y_train = Y

    
    def predict(self, X, k=1, distance_metric=1, num_loops=0):
        """
        PREDICT LABELS for test data using the Classifier:

        Inputs:
        - X: Test Data; a numpy array of shape (num_test, D) containing test data consisting of 'num_test' samples
            each of dimension D
        - k: The number of nearest neighbors that vote for the predicted labels
        - distance_metric: Determines which metric to use to compute the distance b/w test-point and training-point
                (values) = 1 ==> L1 (Manhattan distance)
                         = 2 ==> L2 (Eucledian distance)
        - num_loops: Determines which implementation to use to compute the distances b/w test-point and train-point
                (values) = 0, 1 or 2

        Returns: 
        - Y: a numpy array of shape (num_test,) containing predicted labels for each test data
            where Y[i] is the PREDICTED label for test-point X[i]

        """
        if num_loops == 0:
            dists = self.compute_disatances_no_loops(X, distance_metric=distance_metric)
        elif num_loops == 1:
            dists = self.compute_disatances_one_loops(X, distance_metric=distance_metric)
        elif num_loops == 2:
            dists = self.compute_disatances_two_loops(X, distance_metric=distance_metric)
        else:
            raise ValueError('Invalidvalue %d for num_loops' % num_loops)
        
        return self.predict_labels(distances=dists, k=k)

        

    def compute_disatances_two_loops(self, X, distance_metric=1):
        pass
    
    def compute_disatances_one_loops(self, X, distance_metric=1):
        pass
    
    def compute_disatances_no_loops(self, X, distance_metric=1):
        pass
    
    def predict_labels(self, distances, k=1):
        pass