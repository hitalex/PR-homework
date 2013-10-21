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
        knn_neighbor =  [(flot('inf'), -1)] * k
        for t, l in x_train, y_train:
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
        prediction = choice(prediction)
        
        y_pred.append(prediction)
    
if __name__ == '__main__':
    import sys
    dataset_name = sys.argv[1]
    num_class, num_feature, x_train, y_train, x_test, y_test = readdata.read_dataset(dataset_name)
