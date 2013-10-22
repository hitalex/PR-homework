#coding=utf8

from random import choice

import sklearn.metrics

import readdata
from cv import prepare_cv_dataset

def knn_predict(num_class, x_train, y_train, x_test, k):
    """
    @k: hyper-parameter
    """
    y_pred = []
    for x in x_test:
        # (distance, classlabel)
        knn_neighbor =  [(float('inf'), -1)] * k
        for t, l in zip(x_train, y_train):
            dis = sum((x - t)**2)
            # insert to knn neighbors
            i = 0
            while i<k:
                if knn_neighbor[i][0] > dis:
                    knn_neighbor.insert(i, (dis, l))
                    knn_neighbor.pop() # remove the last item
                    break
                i = i + 1
        # find votes
        votes = [0] * num_class
        for i in range(k):
            index = knn_neighbor[i][1]
            votes[index] = votes[index] + 1
        # find the dominating label in knn neighbor
        prediction = votes.index(max(votes))
        
        y_pred.append(prediction)
        
    return y_pred
    
def cross_validation(cv_dataset, num_class, k):
    nfold = len(cv_dataset)
    score = 0.0
    for (x_train, y_train, x_test, y_test) in cv_dataset:
        y_pred = knn_predict(num_class, x_train, y_train, x_test, k)
        score += sklearn.metrics.accuracy_score(y_test, y_pred)
        
    score /= nfold
    
    return score
    
if __name__ == '__main__':
    import sys
    dataset_name = sys.argv[1]
    num_class, num_feature, x_train, y_train, x_test, y_test = readdata.read_dataset(dataset_name)
    
    print 'Number of folds:'
    nfold = int(input())
    print 'Preparing cv dataset...'
    cv_dataset = prepare_cv_dataset(x_train, y_train, nfold)
    
    bestk = 0
    highest_prec = 0
    while 1:
        print 'Input number of nearest neighbor:'
        s = raw_input().strip()
        if s == '':
            break
        k = int(s)
        
        avg_precision = cross_validation(cv_dataset, num_class, k)
        print 'cross valiation: k=%d, avg precision=%f' % (k, avg_precision)
        
        if avg_precision > highest_prec:
            highest_prec = avg_precision
            bestk = k
                        
    print 'Best k: ', bestk
    print 'Best %d-fold avg precision: %f' % (nfold, highest_prec)

    y_pred = knn_predict(num_class, x_train, y_train, x_test, bestk)
    print sklearn.metrics.classification_report(y_test, y_pred)
