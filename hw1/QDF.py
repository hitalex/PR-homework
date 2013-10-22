#coding=utf8

"""
Gaussian Classifiers

QDF: Quadratic discriminant function
"""

import pdb

import numpy as np
import math
import sklearn.metrics

import readdata

def build_QDF_model(num_class, x_train, y_train):
    """ Cacualte prior prob., means and covariance matrix for each class
    """
    data = []
    train_count = len(x_train)
    for i in range(num_class):
        data.append(list())
        
    # Note: class indexes must be 0,1,2,... staring with 0
    for i in range(train_count):
        class_index = int(y_train[i])
        data[class_index].append(x_train[i])
        
    mean = []
    cov_matrix = []
    prior = []
    for i in range(num_class):
        data[i] = np.matrix(data[i], dtype=np.float64)
        mean.append(data[i].mean(0).T)
        # np.cov treat each row as one feature, so data[i].T has to be transposed
        cov_matrix.append(np.matrix(np.cov(data[i].T)))
        prior.append(len(data[i] * 1.0 / train_count))
        
    return prior, mean, cov_matrix
    
def QDF_predict(x_test, num_class, prior, mean, cov_matrix):
    """ Predict class labels
    Find the class lable that maximize the prob
    """
    inverse_cov = []
    log_det_cov = []
    for i in range(num_class):
        inverse_cov.append(cov_matrix[i].getI())
        det = np.linalg.det(cov_matrix[i])
        log_det_cov.append(math.log(det))
        
    predicted_labels = []
    for row in x_test:
        x = np.matrix(row, np.float64).T
        max_posteriori = -float('inf')
        prediction = -1
        for i in range(num_class):
            diff = x - mean[i]
            p = 2 * math.log(prior[i]) # we do not ignore priors here
            p = p - (diff.T * inverse_cov[i] * diff)[0,0] - log_det_cov[i]
            if p > max_posteriori:
                max_posteriori = p
                prediction = i
                
        predicted_labels.append(prediction)
        
    return predicted_labels

if __name__ == "__main__":
    import sys
    dataset_name = sys.argv[1]
    
    num_class, num_feature, x_train, y_train, x_test, y_test = readdata.read_dataset(dataset_name)
    
    prior, mean, cov_matrix = build_QDF_model(num_class, x_train, y_train)
    #print mean
    
    y_pred = QDF_predict(x_test, num_class, prior, mean, cov_matrix)
    #print predicted_labels
    print sklearn.metrics.classification_report(y_test, y_pred)
