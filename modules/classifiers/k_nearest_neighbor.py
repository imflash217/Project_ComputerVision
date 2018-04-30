import numpy as np 

class KNearestNeighbor:
    """
    K-Nearest-Neighbor classifier with L1, L2 distance metric
    """

    def __init__(self):
        pass

    def train(self, X, Y):
        pass
    
    def predict(self, X, k=1, distance_metric=1, num_loops=0):
        """
        distance_metric = 1 ==> L1 (Manhattan distance)
                        = 2 ==> L2 (Eucledian distance)
        """
        pass

    def compute_disatances_two_loops(self, X):
        pass
    
    def compute_disatances_one_loops(self, X):
        pass
    
    def compute_disatances_no_loops(self, X):
        pass
    
    def predict_labels(self, distances, k=1):
        pass