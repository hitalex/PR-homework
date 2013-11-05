#coding=utf8

"""
PCA for dimension reduction
"""
import os
import pdb

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

os.sys.path.append('/home/kqc/github/PR-homework/')
from hw1.readdata import read_dataset
from PCA import plot_subspace

"""
Three tasks:
plot samples of Iris dataset in coordinates of 
(1) the first two dimensions, 
(2) 2D subspace of PCA, 
(3) 2D subspace of LDA corresponding to the first two major eigenvalues.
"""

def FisherLDA(num_class, d, x, y):
    """ Fisher LDA
    d: number of features
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
    prior_list = []
    # 计算均值和先验
    for i in range(num_class):
        data[i] = np.matrix(data[i], dtype=np.float64)
        mean_list.append(data[i].mean(0).T)
        prior_list.append(len(data[i]) * 1.0 / count)
        
    # 计算每类的协方差矩阵
    cov_matrix_list = []
    for i in range(num_class):
        mean = mean_list[i]
        Ni = len(data[i]) # number of instances in class i
        cov_matrix = np.zeros((d, d), np.float64)
        for j in range(Ni):
            x = np.matrix(data[i][j], np.float64)
            x = x - mean
            cov_matrix = cov_matrix + x * x.T
        
        cov_matrix = 1.0 / Ni * cov_matrix
        cov_matrix_list.append(cov_matrix)
        
    # 计算类内散度矩阵Sw
    Sw = np.zeros((d, d), np.float64)
    for i in range(num_class):
        Sw = Sw + prior_list[i] * cov_matrix_list[i]
        
    # 计算类间散度矩阵St
    mean0 = prior_list[0] * mean_list[0] # total mean
    for i in range(1, num_class):
        mean0 = mean0 + prior_list[i] * mean_list[i]
        
    Sb = np.zeros((d, d), np.float64)
    for i in range(num_class):
        mean = mean_list[i] - mean0
        Sb = Sb + prior_list[i] * (mean * mean.T)
        
    St = Sw + Sb
    # Solve generalized eigenvalue problem of a square matrix.
    # Sb为实对称阵
    w, vr = scipy.linalg.eig(Sb, Sw, left=False, right=True)
    
    # 按照特征值排序
    idx = w.argsort()
    idx = idx[::-1] # reverse the array
    w = w[idx]
    vr = vr[:,idx]
    
    return w, vr
            
def main(dataset_name, s):
    
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
        
    eig_values, eig_vectors = FisherLDA(num_class, num_feature, x_train, y_train)
    xt = np.matrix(x) * eig_vectors[:, :s]
    plot_subspace(xt, y, num_class)
    
    
if __name__ == '__main__':
    import sys
    dataset_name = sys.argv[1]
    s = int(sys.argv[2])
    
    main(dataset_name, s)
