import numpy as np
from modules.classifiers import KNearestNeighbor

class KNNcrossValidation:
    """
    N-fold Cross Validation for k-NearestNeighbor
    """
    def __init__(self):
        pass

    def cross_validation(self, X_train, Y_train, num_folds=5, k_choices=[1, 3, 5, 8, 10, 12, 15, 20, 50, 100]):
        # num_folds = 5
        # k_choices = [1,3,5,8,10,12,15,20,50,100]
        k_to_accuracies={}

        # Step-1: Split the Training data into folds
        # Step-2: Split the Training Data-Labels into folds
        # Step-3: A disctionary holding the accuracies of different values of k that we find when running k-cross-validation
        # Step-4: For each possible value of k;
        #           * Run k-nearest-neighbor algo. n_folds times, 
        #               where in each case you will use all but one of the folds as training data and last one as validation set
        # Step-5: Store the accuracies of all folds and all values of k in 'k_to_accuracies' dictionary


        X_train_folds = np.array_split(X_train, num_folds)
        Y_train_reshaped = Y_train.reshape(-1,1)
        Y_train_folds = np.array_split(Y_train_reshaped, num_folds)

        # X_train_folds = np.array_split(X_train, num_folds)
        # Y_train_folds = np.array_split(Y_train, num_folds)

        for k in k_choices:
            # each element of k_to_accuracies dict contains the accuracies for each fold-validation on that k
            k_to_accuracies[k] = []     # setting default values

        for i in range(num_folds):
            # training data for this particular fold 
            X_train_val = np.vstack((X_train_folds[0:i] + X_train_folds[i+1:]))
            Y_train_val = np.vstack((Y_train_folds[0:i] + Y_train_folds[i+1:]))
            Y_train_val = Y_train_val[:,0]

            # now ruunning the kNN for each value of 'k' over the above training data set for this fold
            for k_ in k_choices:
                # train the data first
                classifier2 = KNearestNeighbor()
                classifier2.train(X_train_val, Y_train_val)
                predictedLabels = classifier2.predict(X_train_folds[i], k_, distance_metric=2)

                # calculating accuracy
                # print(predictedLabels.shape, Y_train_folds[i].shape[0])
                # print(Y_train_folds[i][:,0])
                num_correct = np.sum(predictedLabels == Y_train_folds[i][:,0])
                accuracy = float(num_correct) / Y_train_folds[i].shape[0]
                # print(accuracy, ' = ', num_correct, '/', Y_train_folds[i].shape[0])
                k_to_accuracies[k_].append(accuracy)

       
        return k_to_accuracies



