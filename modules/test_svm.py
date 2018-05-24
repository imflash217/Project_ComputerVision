import random
import numpy as np 
from modules.data_utils import load_CIFAR10
import matplotlib,pyplot as plt 

# Load the CIFAR-10 data
cifar10_dir = 'datasets/cifar-10-batches-py'

# cleaning up variables to prevent loading the data multiple times
try:
    del X_train, Y_train
    del X_test, Y_test
    print('Cleared previously loaded data')
except:
    pass

# loading the cifar-10 dataset
X_train, Y_train, X_test, Y_test = load_CIFAR10(cifar10_dir)

# sanity check
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

# splitting the data into training, validation , and test sets.
# In addition we will also create a 'development set' so that our code runs faster

num_train = 49000
num_val = 1000
num_test = 1000
num_dev = 500


# Validation set : 'num_val' data-points from the original Training set
mask_val = range(num_train, num_train + num_val)
X_val = X_train[mask_val]       # using the ORIGINAL training set
Y_val = Y_train[mask_val]

# Training set : 'num_train' data-points from the original Training set
mask_train = range(num_train)
X_train = X_train[mask_train]   # WARNING: re-assigning the X_train variable
Y_train = Y_train[mask_train]   # WARNING: re-assigning the Y_train variable

# Development Set : 'num_dev' data-points from the new-reassigned Training set
mask_dev = np.random.choice(num_train, size=num_dev, replace=False)
X_dev = X_train[mask_dev]
Y_dev = Y_train[mask_dev]

# Test Set : 'num_test' data-points drawn from ORIGINAL X_test testing set
mask_test = range(num_test)
X_test = X_test[mask_test]
Y_test = Y_test[mask_test]

# sanity check
print('X_train.shape = ', X_train.shape)
print('Y_train.shape = ', Y_train.shape)
print('X_val.shape = ', X_val.shape)
print('Y_val.shape = ', Y_val.shape)
print('X_dev.shape = ', X_dev.shape)
print('Y_dev.shape = ', Y_dev.shape)
print('X_test.shape = ', X_test.shape)
print('Y_test.shape = ', Y_test.shape)


# PRE-PROCESSING : 
# STEP-1: reshaping the image data into rows (NOTE: no need to reshape the labels bco'z they are already into vectorized form)

X_train = np.reshape(X_train, (X_train.shappe[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

# sanity check
print('\n RESHAPED the datasets into rows:')
print('X_train.shape = ', X_train.shape)
print('Y_train.shape = ', Y_train.shape)
print('X_val.shape = ', X_val.shape)
print('Y_val.shape = ', Y_val.shape)
print('X_dev.shape = ', X_dev.shape)
print('Y_dev.shape = ', Y_dev.shape)
print('X_test.shape = ', X_test.shape)
print('Y_test.shape = ', Y_test.shape)

# STEP-2: Subtracting mean-image (from training set) from every image in all datasets
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_val -= mean_image
X_dev -= mean_image
X_test -= mean_image

# STEP-3: Adding a bias of '1' to our data-points so that our SVM has to worry oly about optimizing W matrix (Wx + b == W'x)
X_train = np.hstack(X_train, np.ones(X_train.shape[0],1))
X_val = np.hstack(X_val, np.ones(X_val.shape[0], 1))
X_dev = np.hstack(X_dev, np.ones(X_dev.shape[0], 1))
X_test = np.hstack(X_test, np.ones(X_test.shape[0], 1))




