#coding=utf8

"""
Modified quadratic discriminant function

Ref:
F. Kimura, et al., Modified quadratic discriminant functions and the application
to Chinese character recognition, IEEE Trans. PAMI, 9(1): 149-153, 1987.
"""
import math
import pdb

import numpy as np
import numpy.matlib
from numpy import linalg
import sklearn.metrics

import readdata
from QDF import build_QDF_model

"""
Note: 
"""

def build_MQDF_model(num_class, x_train, y_train, k, delta):
    """ MQDF model
    @k and @delta are hyper-parameters
    
    Note: There are three possible ways to set @delta:
    1) @delta is a hyper-parameter, set by cross validation
    2) @delta can be estimated via ML estimation, in which case, this delta 
        is no longer a hyper-parameter
    3) @delta can be set close to $\sigma_{i,k}$ or $\sigma_{i,k+1}$, in which case, 
        this delta is also not a hyper-parameter
    """
    d = len(x_train[0]) # number of features
    assert(k<d and k>0)
    
    prior, mean, cov_matrix = build_QDF_model(num_class, x_train, y_train)
    
    eigenvalue = []    # store the first largest k eigenvalues of each class
    eigenvector = []   # the first largest k eigenvectors, column-wise of each class
    delta = [0] * num_class # deltas for each class
    for i in range(num_class):
        cov = cov_matrix[i]
        eig_values, eig_vectors = linalg.eig(cov)
        # sort the eigvalues
        idx = eig_values.argsort()
        idx = idx[::-1] # reverse the array
        eig_values = eig_values[idx]
        eig_vectors = eig_vectors[:,idx]
        
        eigenvector.append(eig_vectors[:, 0:k])
        eigenvalue.append(eig_values[:k])
        
        # delta via ML estimation
        delta[i] = (cov.trace() - sum(eigenvalue[i])) * 1.0 / (d-k)
        
        # delta close to $\sigma_{i,k}$ or $\sigma_{i,k+1}$
        
    return prior, mean, eigenvalue, eigenvector, delta
    
def MQDF_predict(x_test, num_class, k, mean, eigenvalue, eigenvector, delta):
    """ MQDF predict
    """
    d = len(x_test[0])
    y_pred = []
    for row in x_test:
        x = np.matrix(row, np.float64).T
        max_posteriori = -float('inf')
        prediction = -1
        for i in range(num_class):
            dis = (x.T * mean[i])[0,0] # the Euclidean distance
            # Mahalanobis distance
            ma_dis = [0] * k
            for j in range(k):
                ma_dis[j] = ((x - mean[i]).T * eigenvector[i][:, j])[0,0]
            
            p = 0
            for j in range(k):
                p += (ma_dis[j] * 1.0 / eigenvalue[i][j])
            
            p += (dis - sum(ma_dis)) / delta[i]
            for j in range(k):
                p += math.log(eigenvalue[i][j])
                
            p += (d-k) * math.log(delta[i])
            p = -p
                
            if p > max_posteriori:
                max_posteriori = p
                prediction = i
                
        y_pred.append(prediction)
        
    return y_pred

if __name__ == '__main__':
    import sys
    dataset_name = sys.argv[1]
    num_class, num_feature, x_train, y_train, x_test, y_test = readdata.read_dataset(dataset_name)
    
    k = 10
    delta = 1
    prior, mean, eigenvalue, eigenvector, delta = build_MQDF_model(num_class, x_train, y_train, k, delta)

    y_pred = MQDF_predict(x_test, num_class, k, mean, eigenvalue, eigenvector, delta)
    #pdb.set_trace()
    print sklearn.metrics.classification_report(y_test, y_pred)
