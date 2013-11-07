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
from cv import prepare_cv_dataset

"""
Note: 
"""

def build_MQDF_model(num_class, x_train, y_train, k, delta0):
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
        eig_values, eig_vectors = linalg.eigh(cov)
        # sort the eigvalues
        idx = eig_values.argsort()
        idx = idx[::-1] # reverse the array
        eig_values = eig_values[idx]
        eig_vectors = eig_vectors[:,idx]
        
        eigenvector.append(eig_vectors[:, 0:k])
        eigenvalue.append(eig_values[:k])
        
        # delta via ML estimation
        #delta[i] = (cov.trace() - sum(eigenvalue[i])) * 1.0 / (d-k)
        
        # delta close to $\sigma_{i,k-1}$ or $\sigma_{i,k}$
        #delta[i] = (eig_values[k-1] + eig_values[k])/2
        #print 'Suggestd delta[%d]: %f' % (i, (eig_values[k] + eig_values[k+1])/2)
        
        # delta as the mean of minor values
        delta[i] = sum(eig_values[k:]) / len(eig_values[k:])
    
    #delta = [delta0] * num_class
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
            dis = np.linalg.norm(x.reshape((d,)) - mean[i].reshape((d,))) ** 2 # 2-norm
            # Mahalanobis distance
            ma_dis = [0] * k
            for j in range(k):
                ma_dis[j] = (((x - mean[i]).T * eigenvector[i][:, j])[0,0])**2
            
            p = 0
            for j in range(k):
                p += (ma_dis[j] * 1.0 / eigenvalue[i][j])
            
            p += ((dis - sum(ma_dis)) / delta[i])
            
            for j in range(k):
                p += math.log(eigenvalue[i][j])
                
            p += ((d-k) * math.log(delta[i]))
            p = -p
                
            if p > max_posteriori:
                max_posteriori = p
                prediction = i
                
        y_pred.append(prediction)
        
    return y_pred
    
def cross_validation(cv_dataset, num_class, k, delta0):
    """cross validation for beta and gamma based on average precision
    hyper-parameters: k, delta0
    """
    nfold = len(cv_dataset)
    score = 0.0
    for (x_train, y_train, x_test, y_test) in cv_dataset:
        prior, mean, eigenvalue, eigenvector, delta = build_MQDF_model(num_class, x_train, y_train, k, delta0)
        y_pred = MQDF_predict(x_test, num_class, k, mean, eigenvalue, eigenvector, delta)
        score += sklearn.metrics.accuracy_score(y_test, y_pred)
        
    score /= nfold
    
    return score

def main(dataset_name):
    num_class, num_feature, x_train, y_train, x_test, y_test = readdata.read_dataset(dataset_name)
    
    print 'Number of folds:'
    nfold = int(input())
    print 'Preparing cv dataset...'
    cv_dataset = prepare_cv_dataset(x_train, y_train, nfold)
    
    k = num_feature / 2
    bestk = k
    bestdelta0 = 0
    highest_prec = 0
    
    """
    while 1:
        print 'Input delta:'
        s = raw_input().strip()
        if s == '':
            break
        delta0 = float(s)
        
        avg_precision = cross_validation(cv_dataset, num_class, k, delta0)
        print 'cross valiation: k=%d, delta=%f, avg precision=%f\n' % (k, delta0, avg_precision)
        
        if avg_precision > highest_prec:
            highest_prec = avg_precision
            bestk = k
            bestdelta0 = delta0
            
    print 'Best k and delta0: ', bestk, bestdelta0
    print 'Best avg precision: ', highest_prec
    
    k = bestk
    delta0 = bestdelta0
    """
    prior, mean, eigenvalue, eigenvector, delta = build_MQDF_model(num_class, x_train, y_train, k, 0)

    y_pred = MQDF_predict(x_test, num_class, k, mean, eigenvalue, eigenvector, delta)
    
    #pdb.set_trace()
    print sklearn.metrics.classification_report(y_test, y_pred)
    
    print 'Average accuracy: ', sklearn.metrics.accuracy_score(y_test, y_pred)

if __name__ == '__main__':
    import sys
    dataset_name = sys.argv[1]
    main(dataset_name)
