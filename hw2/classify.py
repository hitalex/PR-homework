#coding=utf8

"""
Classification via 1) PCA or LDA dimension reduction and 2) QDF, LDF, MQDF, nearest mean
"""
import os
import pdb

import numpy as np
import sklearn.metrics

os.sys.path.append('/home/kqc/github/PR-homework/')
from hw1.readdata import read_dataset
from hw1.QDF import build_QDF_model, QDF_predict
from hw1.LDF import build_LDF_model, LDF_predict
from hw1.MQDF import build_MQDF_model, MQDF_predict

from PCA import PCA
from FisherLDA import FisherLDA

"""
Classification accuracies of 
(1) PCA subspace of variable dimensionality, 
(2) LDA subspace of variable dimensionality. 
Dimensions are varied from 1 to the last. 
Both tables and figures are required.
Five-fold evaluation for Iris.
"""

def classify_QDF(num_class, num_feature, x_train, y_train, x_test):
    """classification using QDF
    """
    prior, mean, cov_matrix = build_QDF_model(num_class, x_train, y_train)
    y_pred = QDF_predict(x_test, num_class, prior, mean, cov_matrix)
    
    return y_pred
    
def classify_LDF(num_class, num_feature, x_train, y_train, x_test):
    """classification using LDF
    """
    inverse_cov, weight, w0 = build_LDF_model(num_class, x_train, y_train)
    y_pred = LDF_predict(x_test, num_class, inverse_cov, weight, w0)
    
    return y_pred
    
def classify_MQDF(num_class, num_feature, x_train, y_train, x_test):
    """classification using MQDF
    """
    k = num_feature / 2
    prior, mean, eigenvalue, eigenvector, delta = build_MQDF_model(num_class, x_train, y_train, k, 0)
    y_pred = MQDF_predict(x_test, num_class, k, mean, eigenvalue, eigenvector, delta)
    
    return y_pred


def main(dataset_name):
    print 'Reading dataset: ', dataset_name
    num_class, num_feature, x_train, y_train, x_test, y_test = read_dataset(dataset_name)
    
    rng = range(5, num_feature, 5)
    rng.append(1)
    for s in [36]:
        # PCA
        #eig_values, eig_vectors = PCA(x_train, num_feature)
        #DR_method = 'PCA'
        # FisherLDA
        eig_values, eig_vectors = FisherLDA(num_class, num_feature, x_train, y_train)
        DR_method = 'FisherLDA'
        
        # reduce the dimension of training set and test set
        x_train_dr = np.matrix(x_train) * eig_vectors[:, :s]
        x_test_dr = np.matrix(x_test) * eig_vectors[:, :s]
        
        x_train_dr = np.array(x_train_dr)
        x_test_dr = np.array(x_test_dr)
        """
        y_pred = classify_QDF(num_class, s, x_train_dr, y_train, x_test_dr)
        classifier = 'QDF'
        print '%s and %s reports with dim: %d, accuracy: %f' % (DR_method, classifier, s, sklearn.metrics.accuracy_score(y_test, y_pred))
        
        y_pred = classify_LDF(num_class, s, x_train_dr, y_train, x_test_dr)
        classifier = 'LDF'
        print '%s and %s reports with dim: %d, accuracy: %f' % (DR_method, classifier, s, sklearn.metrics.accuracy_score(y_test, y_pred))
        """
        y_pred = classify_MQDF(num_class, s, x_train_dr, y_train, x_test_dr)
        classifier = 'MQDF'
        print '%s and %s reports with dim: %d, accuracy: %f' % (DR_method, classifier, s, sklearn.metrics.accuracy_score(y_test, y_pred))
        
        print ''

if __name__ == '__main__':
    import sys
    dataset_name = sys.argv[1]
    
    main(dataset_name)
