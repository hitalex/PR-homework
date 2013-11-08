#coding=utf8

"""
Gaussian Classifiers

LDF: Linear discriminant function
Case: All classes share the same covariance matrix
"""
import math
import pdb
import sys

import numpy as np
import numpy.matlib
import sklearn.metrics

import readdata
from QDF import build_QDF_model, print_cov_matrix

def build_LDF_model(num_class, x_train, y_train):
    """ LDF model
    First call QDF model to caculate the mean, cov_matirx
    """
    prior, mean, cov_matrix = build_QDF_model(num_class, x_train, y_train)
    #print_cov_matrix(cov_matrix)
    
    # cacualte the shared covariance matirx
    avg_cov = np.matlib.zeros(cov_matrix[0].shape)
    for i in range(num_class):
        avg_cov += (prior[i] * cov_matrix[i])
        
    inverse_cov = avg_cov.getI() # get the inverse covariance matrix
    
    num_feature = x_train.shape[1]
    # each column for weight[i]
    weight = np.matrix([0] * num_feature).T
    w0 = []
    for i in range(num_class):
        wi = 2 * inverse_cov.T * mean[i]
        weight = np.hstack((weight, wi))
        
        wi0 = 2 * math.log(prior[i]) - (mean[i].T * inverse_cov * mean[i])[0,0]
        w0.append(wi0)
        
    return inverse_cov, weight[:, 1:], w0
    
def LDF_predict(x_test, num_class, inverse_cov, weight, w0):
    predicted_labels = []
    for row in x_test:
        x = np.matrix(row, np.float64).T
        max_posteriori = -float('inf')
        prediction = -1
        for i in range(num_class):
            p = (-1 * (x.T * inverse_cov * x) + weight[:, i].T * x + w0[i])[0,0]
            #p = (weight[:, i].T * x + w0[i])[0,0]
            if p > max_posteriori:
                max_posteriori = p
                prediction = i
                
        predicted_labels.append(prediction)
        
    return predicted_labels
    
def main(dataset_name):
    num_class, num_feature, x_train, y_train, x_test, y_test = readdata.read_dataset(dataset_name)
    
    inverse_cov, weight, w0 = build_LDF_model(num_class, x_train, y_train)

    y_pred = LDF_predict(x_test, num_class, inverse_cov, weight, w0)
    #pdb.set_trace()
    print sklearn.metrics.classification_report(y_test, y_pred)
    
    print 'Average accuracy: ', sklearn.metrics.accuracy_score(y_test, y_pred)

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    main(dataset_name)    
