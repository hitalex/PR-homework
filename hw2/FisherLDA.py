#coding=utf8

"""
PCA for dimension reduction
"""
import os
import pdb

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

os.sys.path.append('/home/kqc/github/PR-homework/')
from hw1.readdata import read_dataset

"""
Three tasks:
plot samples of Iris dataset in coordinates of 
(1) the first two dimensions, 
(2) 2D subspace of PCA, 
(3) 2D subspace of LDA corresponding to the first two major eigenvalues.
"""

def FisherLDA(num_class, num_feature, x, y):
    """ Fisher LDA
    x: data set without labels
    y: labels
    """
    data = []
    count = len(x)
    for i in range(num_class):
        data.append(list())
        
    # Note: class indexes must be 0,1,2,... staring with 0
    for i in range(count):
        class_index = int(y[i])
        data[class_index].append(x[i])
        
    mean_list = []
    cov_matrix_list = []
    prior_list = []
    for i in range(num_class):
        data[i] = np.matrix(data[i], dtype=np.float64)
        mean_list.append(data[i].mean(0).T)
        prior_list.append(len(data[i]) * 1.0 / train_count)
        
    
