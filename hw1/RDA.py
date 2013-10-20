#coding=utf8

"""
Regularized discriminant analysis

"""
import numpy as np
import numpy.matlib
import sklearn.metrics

import readdata
from QDF import build_QDF_model, QDF_predict

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
        tmp = cov_matrix[i]
        dev = tmp.trace()[0,0] * 1.0 / num_feature
        cov_matrix[i] = (1-gamma) * ((1-beta)*tmp + beta*avg_cov) + gamma * dev * np.matlib.eye(num_feature)
        
    return prior, mean, cov_matrix
    
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
    
def cross_validation(cv_dataset, num_class, beta, gamma):
    """cross validation for beta and gamma based on average precision
    hyper-parameters: beta, gamma
    """
    nfold = len(cv_dataset)
    score = 0.0
    for (x_train, y_train, x_test, y_test) in cv_dataset:
        prior, mean, cov_matrix = build_RDA_model(num_class, x_train, y_train, beta, gamma)
        y_pred = QDF_predict(x_test, num_class, mean, cov_matrix)
        score += sklearn.metrics.accuracy_score(y_test, y_pred)
        
    score /= nfold
    
    return score

if __name__ == '__main__':
    import sys
    dataset_name = sys.argv[1]
    num_class, num_feature, x_train, y_train, x_test, y_test = \
        readdata.read_dataset(dataset_name)
        
    #prepare_cv_dataset(x_train, y_train, 3)
    
    print 'Number of folds:'
    nfold = int(input())
    print 'Preparing cv dataset...'
    cv_dataset = prepare_cv_dataset(x_train, y_train, nfold)
    
    best = [0, 0, -float('inf')] # beta, gamma, highest precision
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
    y_pred = QDF_predict(x_test, num_class, mean, cov_matrix)
    print sklearn.metrics.classification_report(y_test, y_pred)
