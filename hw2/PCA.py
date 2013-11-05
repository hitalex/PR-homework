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

def PCA(x_train, d):
    """ PCA for dimension reduction
    """      
    # caculate the covariance matrix
    cov_matrix = np.zeros((d, d), np.float64)
    N = len(x_train)
    for i in range(N):
        x = np.matrix(x_train[i], np.float64)
        cov_matrix = cov_matrix + np.matrix(x * x.T)
        
    cov_matrix = 1.0/N * cov_matrix
    # 特征值分解
    eig_values, eig_vectors = linalg.eig(cov_matrix)
    
    # 按照特征值大小排序
    idx = eig_values.argsort()
    idx = idx[::-1] # reverse the array
    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[:,idx]
    
    return eig_values, eig_vectors

def plot_subspace(x, y, num_class):
    """ Plot 2D data
    """
    colors = np.array(['red', 'green', 'blue'])
    if num_class > len(colors):
        print 'Not enough colors'
        return
        
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], marker='+', c=colors[y])
    
    plt.show()

def main(dataset_name, s):
    """
    s: the target dimension
    """
    print 'Reading dataset: ', dataset_name
    num_class, num_feature, x_train, y_train, x_test, y_test = read_dataset(dataset_name)
    
    x = np.vstack((x_train, x_test))
    y_train.shape = (len(y_train), 1)
    y_test.shape = (len(y_test), 1)
    y = np.vstack((y_train, y_test))
    y = y[:, 0]
    # plot the first 2 dimensions
    #plot_subspace(x[:, :2], y, num_class)
    
    if s > num_feature:
        print 'The target dimension must be equal or smaller than the original dim.'
    
    eig_values, eig_vectors = PCA(x_train, num_feature)
    xt = np.matrix(x) * eig_vectors[:, :s]
    plot_subspace(xt, y, num_class)

if __name__ == '__main__':
    import sys
    dataset_name = sys.argv[1]
    s = int(sys.argv[2])
    
    main(dataset_name, s)
