#coding=utf-8

import csv
import random

import numpy as np

"""
Read multiple dataset from ./dataset

Routines usually return the following infomration:
1, Number of classes
2, number of features
3, training data, ndarray format
4, tranining data labels
5, test data, ndarray format
6, test data labels
"""
def read_dataset(dataset_name):
    """ Read a dataset with a specified name
    """
    print 'Reading dataset: ', dataset_name, ' ...'
    class_set = set()
    x_train = []
    y_train = []
    with open('./dataset/' + dataset_name + '.train', 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            if row == []:
                continue
            y_train.append(row[-1])
            class_set.add(int(row[-1]))
            x_train.append(row[:-1])
    
    x_test = []
    y_test = []
    with open('./dataset/' + dataset_name + '.test', 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            if row == []:
                continue
            y_test.append(row[-1])
            class_set.add(int(row[-1]))
            x_test.append(row[:-1])
    
    num_class = len(class_set)
    num_feature = len(x_train[0])
    
    print 'Number of classes           :', num_class
    print 'Number of features          :', num_feature
    print 'Number of training instance :', len(y_train)
    print 'Number of testing instance  :', len(y_test)
    
    return num_class, num_feature, np.array(x_train, np.float64), \
        np.array(y_train, np.int), np.array(x_test, np.float64), np.array(y_test, np.int)
    
def prepare_iris():
    """ read iris data set
    """
    num_class = 3
    num_feature = 4
    
    class_map = {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
    
    data = []
    with open('./dataset/iris.data', 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            if row == []:
                continue
            row[-1] = class_map[row[-1]] # use numbers to indicate classes
            data.append(row)
    
    random.shuffle(data)
    
    #import pdb
    #pdb.set_trace()

    data = np.array(data, dtype='S')
    y = data[:, -1]
    
    num_test = len(data) / 3
    x_test = data[0:num_test, 0:-1]
    y_test = y[0:num_test]
    
    x_train = data[num_test:, 0:-1]
    y_train = y[num_test:]
    
    f = open('iris.train', 'w')
    for i in range(len(x_train)):
        f.write(','.join(x_train[i]))
        f.write(',' + str(y_train[i]) + '\n')
    f.close()
    
    f = open('iris.test', 'w')
    for i in range(len(x_test)):
        f.write(','.join(x_test[i]))
        f.write(',' + str(y_test[i]) + '\n')
    f.close()
    
    return num_class, num_feature, x_train, y_train, x_test, y_test
    
def prepare_letter_recognition():
    """ read the letter recognition dataset
    """
    num_class = 26
    num_feature = 16
    
    data = []
    labels = []
    with open('./dataset/letter-recognition.data', 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            if row == []:
                continue
            labels.append(row[0])
            data.append(row[1:])
    
    for i in range(len(labels)):
        labels[i] = ord(labels[i]) - 65
        
    num_test = len(data) / 3
    x_test = np.array(data[0:num_test], dtype='S')
    y_test = labels[0:num_test]
    
    x_train = np.array(data[num_test:], dtype='S')
    y_train = labels[num_test:]
    
    f = open('letter.train', 'w')
    for i in range(len(x_train)):
        f.write(','.join(x_train[i]))
        f.write(',' + str(y_train[i]) + '\n')
    f.close()
    
    f = open('letter.test', 'w')
    for i in range(len(x_test)):
        f.write(','.join(x_test[i]))
        f.write(',' + str(y_test[i]) + '\n')
    f.close()
    
    return num_class, num_feature, x_train, y_train, x_test, y_test
    
def prepare_sat():
    """ Make class distribution is 0..6, instead of 1..7
    """
    ft = open('./dataset/sat.train1', 'w')
    f = open('./dataset/sat.train', 'rb')
    for line in f:
        line = line.strip()
        if line == '':
            continue
        label = int(line[-1]) - 1
        ft.write(line[:-1] + ' ' + str(label) + '\n')
        
    ft.close(); f.close()
    
    ft = open('./dataset/sat.test1', 'w')
    f = open('./dataset/sat.test', 'rb')
    for line in f:
        line = line.strip()
        if line == '':
            continue
        label = int(line[-1]) - 1
        ft.write(line[:-1] + ' ' + str(label) + '\n')
        
    ft.close(); f.close()
    
if __name__ == '__main__':
    #prepare_iris()
    #prepare_letter_recognition()
    #prepare_sat()
    import sys
    read_dataset(sys.argv[1])
