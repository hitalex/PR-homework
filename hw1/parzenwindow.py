#coding=utf8

"""
Parzen window method for classification

Kernel: Gaussian kernel
"""

import math

import sklearn.metrics
import numpy as np

import readdata
from cv import prepare_cv_dataset

def parzen_predict(num_class, x_train, y_train, x_test, h):
    """
    @h, hyper-parameter
    """
    # split training dataset according class label
    data = []
    train_count = len(x_train)
    for i in range(num_class):
        data.append(list())
        
    # Note: class indexes must be 0,1,2,... staring with 0
    for i in range(train_count):
        class_index = int(y_train[i])
        data[class_index].append(x_train[i])
        
    prior = [0] * num_class
    total = len(x_train)
    for i in range(num_class):
        prior[i] = len(data[i]) *1.0 / total
    
    y_pred = []
    for x in x_test:
        maxp = 0
        prediction = -1
        for i in range(num_class):
            Ni = len(data[i]) # number of instances in class i
            p = 0
            for j in range(Ni):
                tmp = - sum((data[i][j] - x)**2) * 1.0 / (2*(h**2))
                tmp = math.exp(tmp)
                p += tmp
            #p = p / Ni * prior[i]
            p = p / Ni
            
            if p > maxp:
                maxp = p
                prediction = i
                
        y_pred.append(prediction)
        
    return y_pred
    
def cross_validation(cv_dataset, num_class, h):
    """cross validation for beta and gamma based on average precision
    hyper-parameters: beta, gamma
    """
    nfold = len(cv_dataset)
    score = 0.0
    print 'Cross validataion process...'
    for (x_train, y_train, x_test, y_test) in cv_dataset:
        y_pred = parzen_predict(num_class, x_train, y_train, x_test, h)
        tmp = sklearn.metrics.accuracy_score(y_test, y_pred)
        print '%d fold precision: %f' % (nfold, tmp)
        score += tmp
        
    score /= nfold
    
    return score
    
def make_small_dataset(x, y, num):
    """ Make small dataset out of large dataset
    x: the data matrix
    y: the corresponding lables
    num: the number of instances
    """
    y.shape = (len(y), 1)
    # combine x and y
    data = np.hstack((x, y))
    
    import numpy.random
    np.random.shuffle(data)
    
    if len(data) > num:
        x = data[:num, :-1]
        y = data[:num, -1]
        
    return x, y
    
def accuracy_score(y_true, y_pred):
    """ Average accuracy score
    """
    assert(len(y_true) == len(y_pred))
    
    correct = 0
    for i in range(len(y_true)):
        if int(y_pred[i]) == int(y_true[i]):
            correct = correct + 1
        
    return correct * 1.0 / len(y_true)
    
if __name__ == '__main__':
    import sys
    dataset_name = sys.argv[1]
    num_class, num_feature, x_train, y_train, x_test, y_test = readdata.read_dataset(dataset_name)
    
    x_train, y_train = make_small_dataset(x_train, y_train, 500)
    x_test, y_test = make_small_dataset(x_test, y_test, 200)
    
    print 'Number of training data:', len(x_train)
    print 'Number of test data:', len(x_test)
    
    print 'Number of folds:'
    nfold = int(input())
    print 'Preparing cv dataset...'
    cv_dataset = prepare_cv_dataset(x_train, y_train, nfold)
    
    best = [0, 0] # h, highest precision
    while 1:
        print 'Input window width:'
        s = raw_input().strip()
        if s == '':
            break
        h = float(s)
        
        avg_precision = cross_validation(cv_dataset, num_class, h)
        print 'cross valiation: h=%f, avg precision=%f' % (h, avg_precision)
        
        if avg_precision > best[1]:
            best[1] = avg_precision
            best[0] = h
            
    print 'Best h: ', best[0]
    print 'Best avg precision: ', best[1]
    
    h = best[0]
    y_pred = parzen_predict(num_class, x_train, y_train, x_test, h)
    print sklearn.metrics.classification_report(y_test, y_pred)
    
    print 'Average acc: ', sklearn.metrics.accuracy_score(y_test, y_pred)
    print 'My avg acc: ', accuracy_score(y_test, y_pred)
    
