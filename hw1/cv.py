#coding=utf8

"""
Cross validation routine
"""
import numpy as np

def prepare_cv_dataset(x_train, y_train, nfold):
    """ Return the cv dataset: [x_train, y_train, x_test, y_test]
    nfold: number of folds
    """
    cv_dataset = []
    total = len(x_train)
    d = total / nfold # number of instances in each block
    x_list = np.vsplit(x_train[:d*nfold, :], nfold)
    # make y_train have 2 dimentions, so to use np.vsplit...
    y_train = np.column_stack((y_train, [1]*total))
    y_list = np.vsplit(y_train[:d*nfold, :], nfold)
    
    if total % nfold != 0:
        # add the remaining to the last one block
        x_list[-1] = np.concatenate((x_list[-1], x_train[d*nfold:, :]))
        y_list[-1] = np.concatenate((y_list[-1], y_train[d*nfold:, :]))
        
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    
    for i in range(nfold):
        # the ith block is the test set, while the others are training set
        train_mask = np.array([True] * nfold)
        train_mask[i] = False
        x_test = x_list[i]
        y_test = y_list[i]
        
        x_train_list = x_list[train_mask]
        x_train = np.concatenate(list(x_train_list))
        
        y_train_list = y_list[train_mask]
        y_train = np.concatenate(list(y_train_list))
        
        cv_dataset.append([x_train, y_train[:, 0], x_test, y_test[:,0]])
        
    return cv_dataset
