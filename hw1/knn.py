#coding=utf8

import sklearn.metrics

import readdata
from cv import prepare_cv_dataset

def knn_predict(num_class, x_train, y_train, x_test, k):
    """
    @k: hyper-parameter
    """
    y_pred = []
    
    
if __name__ == '__main__':
    import sys
    dataset_name = sys.argv[1]
    num_class, num_feature, x_train, y_train, x_test, y_test = readdata.read_dataset(dataset_name)
