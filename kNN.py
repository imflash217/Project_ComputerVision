##############################------- SETUP code ---------------###################
######## k-Nearest Neighbor implementation--------by @vinaykumar2491-------###############

import random
import numpy as np
import matplotlib.pyplot as plt

from modules.data_utils import load_CIFAR10
from modules.classifiers import KNearestNeighbor

# ###### configuring matplotlib for better plots for Jupyter Notebook ONLY
# %matplotlib inline
# plt.rcParams['figure.figsize'] = (10.0,8.0)     #default size of plots
# plt.rcParams['image.interpolation'] = 'nearest'
# plt.rcParams['image.cmap'] = 'gray'

# # commanding iPython Notebook to autoreload all updated modules before executing a new line
# # very helpful when we change various modules and run our base code
# %load_ext autoreload
# %autoreload 2


##############################------- loading the raw CIFAR-10 dataset --------##############

cifar10_dir = 'problemSets/assignment1/cs231n/datasets/cifar-10-batches-py'

##### cleaning up the variables to prevent loading data multiple times(which may cause memory issues)
try:
    del X_train, Y_train
    del X_test, Y_test
    print('Cleared previously loaded data.')
except:
    pass

##### loading all batches of CIFAR-10 dataset
X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar10_dir)

###### sanity check: printing the sizes of the training and test data
print('Training DATA shape:', X_train.shape)
print('Training LABEL shape:', Y_train.shape)
print('Test DATA shape:', X_test.shape)
print('Test LABEL shape:', Y_test.shape)


####### Visualizing some images from the dataset
cifar_classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
num_classes = len(cifar_classes)
samples_per_class = 7               # no of samples we want to view per class

for y, cls in enumerate(cifar_classes):
    idxs = np.floatnonzero(Y_train == y)        # extracting the images that corrspond to each individual class
    idxs = np.random.choice(idxs, samples_per_class, replace=False)     # truncating the inds to samples per class

    for i, idx in enumerate(idxs):
        plt_idx = i*samples_per_class + y + 1       #calculating the plot index
        plt.subplot(samples_per_class, num_classes, plt_idx)
        plt.imshow(X_train[idx].astype('uint8'))
        plt.axis('off')

        if i == 0:
            plt.title(cls)

plt.show()


################## Subsampling the data  ####################
num_train = 5000
num_test = 500

mask_train = list(range(num_train))
mask_test = list(range(num_test))

X_train_sub = X_train[mask_train]
Y_train_sub = Y_train[mask_train]

X_test_sub = X_test[mask_test]
Y_test_sub = Y_test[mask_test]

del mask_train
del mask_test

################## Reshapng the image data into rows ##############

X_train_sub = np.reshape(X_train_sub, (X_train_sub.shape[0], -1))
X_test_sub = np.reshape(X_test_sub, (X_test_sub, X_test_sub.shape[0], -1))

#################################################
##### Creating a kNN classifier
##### kNN-classifier simply remebers the training data and does no further processing

classifier = KNearestNeighbor()
classifier.train(X_train_sub, Y_train_sub)
























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

            # Finding the various "distance metric" for each test image wrt. EVERY training image
            L1_distances = np.sum(np.abs(self.X_train - X[i,:]), axis=1)    # L1(Manhattan) distance
            L2_distances = np.sqrt(np.sum(np.square(self.X_train - X[i, :]), axis=1))   # L2(Euclidean) distance

            
            # Finding the index of the minimum distance
            min_index = np.argmin(L1_distances)

            # Extracting the 'LABEL' based on the min. distance index from the training data &
            # assigning the label to the current test image
            Y_pred[i] = self.Y_train[min_index]
        
        return Y_pred
