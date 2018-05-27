import numpy as np
import matplotlib.pyplot as plt
from modules.classifiers.neural_net import TwoLayerNet

def tune_hyperparams(X, y, X_val, y_val, input_size=3072, output_size=10, 
                        num_train_epochs=5, num_iters=1000,lr_range=[-5, 5], 
                        reg_range=[-3, 6], h1_size_array=[50, 75, 100, 125], verbose=False):
    
    """
    Tune the hyper parameters to obtain best network

    """

    best_net = None # store the best model into this 
    best_val_acc = -1
    results = {}

    lr_array = []
    reg_array = []
    # h1_size_array = []

    print('START...............')
    for epoch in range(num_train_epochs):
        lr_array.append(10**np.random.uniform(lr_range[0], lr_range[1]))
        reg_array.append(10**np.random.uniform(reg_range[0], reg_range[1]))
        # h1_size_array.append(np.random.randint(h1_size_range[0], h1_size_range[1]))
        print('hyper-params sampling DONE..............\nSTARTING the TRAINING procedure............')
    
    for h1_size in h1_size_array:
        for lr in lr_array:
            for reg in reg_array:
                net = TwoLayerNet(input_size=input_size, hidden_size=h1_size, output_size=output_size)
                stats = net.train(X=X, y=y, X_val=X_val, y_val=y_val, 
                                    num_iters=num_iters, learning_rate=lr, reg=reg, verbose=verbose)

                val_acc = np.mean(net.predict(X_val) == y_val)
                results[(h1_size, lr, reg)] = val_acc

                if verbose:
                    plt.subplot(1, 2, 1)
                    plt.plot(stats['loss_history'])
                    plt.title('Loss history')
                    plt.xlabel('Iteration')
                    plt.ylabel('Loss')

                    plt.subplot(1, 2, 2)
                    plt.plot(stats['train_acc_history'], label='train')
                    plt.plot(stats['val_acc_history'], label='val')
                    plt.title('Classification accuracy history')
                    plt.xlabel('Epoch')
                    plt.ylabel('Clasification accuracy')
                    plt.legend()
                    plt.show()

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_net = net
                else:
                    del net
                    del stats

    print('TRAINING procedure OVER..............\nSTOP.\n...............RESULTS.....................')
    for h1_size, lr, reg in results:
        val_acc = results[(h1_size, lr, reg)]
        print('h1_size: %d    lr: %f    reg: %f    val_acc = %f' %(h1_size, lr, reg, val_acc))
    
    return best_net

#######################################################################################################
