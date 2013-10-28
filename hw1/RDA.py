#coding=utf8

"""
Regularized discriminant analysis

"""
import numpy as np
import numpy.matlib
import sklearn.metrics

import readdata
from QDF import build_QDF_model, QDF_predict
from cv import prepare_cv_dataset

def build_RDA_model(num_class, x_train, y_train, beta, gamma):
    """ First call QDF model to caculate the mean, cov_matirx
    beta, gamma: hyper-parameters, range: 0~1
    """
    prior, mean, cov_matrix = build_QDF_model(num_class, x_train, y_train)
    
    # cacualte the shared covariance matirx
    avg_cov = np.matlib.zeros(cov_matrix[0].shape)
    for i in range(num_class):
        avg_cov += (prior[i] * cov_matrix[i])
        
    num_feature = len(x_train[0])
    for i in range(num_class):
        
        # the following formula is from PPT
        tmp = cov_matrix[i]
        dev = tmp.trace()[0,0] * 1.0 / num_feature
        cov_matrix[i] = (1-gamma) * ((1-beta)*tmp + beta*avg_cov) + gamma * dev * np.matlib.eye(num_feature)
        """
        # refer to Statistical Pattern Recognition, page 43
        tmp = cov_matrix[i]
        tmp = (1-beta) * prior[i] * tmp + beta * avg_cov
        dev = tmp.trace()[0,0] * 1.0 / num_feature
        cov_matrix[i] = (1-gamma) * tmp + gamma * dev * np.matlib.eye(num_feature)
        """
        
    return prior, mean, cov_matrix
    
def cross_validation(cv_dataset, num_class, beta, gamma):
    """cross validation for beta and gamma based on average precision
    hyper-parameters: beta, gamma
    """
    nfold = len(cv_dataset)
    score = 0.0
    for (x_train, y_train, x_test, y_test) in cv_dataset:
        prior, mean, cov_matrix = build_RDA_model(num_class, x_train, y_train, beta, gamma)
        y_pred = QDF_predict(x_test, num_class, prior, mean, cov_matrix)
        score += sklearn.metrics.accuracy_score(y_test, y_pred)
        
    score /= nfold
    
    return score
    
def main(dataset_name):
    num_class, num_feature, x_train, y_train, x_test, y_test = \
        readdata.read_dataset(dataset_name)
        
    #prepare_cv_dataset(x_train, y_train, 3)
    
    print 'Number of folds:'
    nfold = int(input())
    print 'Preparing cv dataset...'
    cv_dataset = prepare_cv_dataset(x_train, y_train, nfold)
    
    best = [0, 0, 0] # beta, gamma, highest precision
    while 1:
        print 'Input beta, gamma:'
        s = raw_input().strip()
        if s == '':
            break
        beta, gamma = s.split()
        beta = float(beta)
        gamma = float(gamma)
        
        avg_precision = cross_validation(cv_dataset, num_class, beta, gamma)
        print 'cross valiation: beta=%f, gamma=%f, avg precision=%f' % (beta, gamma, avg_precision)
        
        if avg_precision > best[2]:
            best[2] = avg_precision
            best[0] = beta
            best[1] = gamma
            
    print 'Best beta and gamma: ', best[0], best[1]
    print 'Best avg precision: ', best[2]
    
    beta = best[0]
    gamma = best[1]
    prior, mean, cov_matrix = build_RDA_model(num_class, x_train, y_train, beta, gamma)
    # predict like QDF
    y_pred = QDF_predict(x_test, num_class, prior, mean, cov_matrix)
    print sklearn.metrics.classification_report(y_test, y_pred)
    
    print 'Average accuracy: ', sklearn.metrics.accuracy_score(y_test, y_pred)

if __name__ == '__main__':
    import sys
    dataset_name = sys.argv[1]
    main(dataset_name)
