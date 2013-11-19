#coding=utf8

"""
RBF network training
"""
import pickle
from math import exp
import os
import random
import sys
from select import select

import numpy as np
import numpy.linalg
import numpy.random
import ipdb

os.sys.path.append('/home/kqc/github/PR-homework/')
from hw1.readdata import read_dataset

def get_initial_weight():
    """ Set the initial weight and the shape is (num_class, num_hidden+1)
    weight[:, 0] are biases
    Note: set them to small random values chosen from a zero-mean Gaussian with a standard deviation of about 0.01
    """
    weight = np.zeros((num_class, num_hidden+1)) # add biases
    
    weight[:, :-1] = np.random.normal(0, 0.01, size=(num_class, num_hidden))
    weight[:, -1] = 0 # set biases to 0
    
    return weight
    
def get_hidden_layer_parameter():
    """ Set the hidden layer prameters: mean and \sigma^2
    """
    mean = np.zeros((num_feature, num_hidden)) # shape: num_feature * num_hidden

    # all \sigma are the same
    sigma2 = np.array([1000] * num_hidden)
    # get the mean of the whole training dataset
    m = np.mean(x_train, axis=0)
    # the covariance matrix
    # assuming the features are independent
    cov = 0.05 * np.eye(num_feature)
    mean = np.random.multivariate_normal(m, cov, size=num_hidden) # shape: num_hidden * num_feature
    mean = mean.T
    
    return mean, sigma2
    
def sigmoid(x):
    """ the Sigmoid function
    x: a scala
    """
    return 1.0 / (1 + exp(-x))
    
def get_output(mean, sigma2, weight, x):
    """ Get the output for the single input instance, i.e. y_k(x_i)
    Return: the probability of x belonging to each class, list of length num_class
    """
    u = [0] * (num_hidden+1) # the hidden layer output
    u[-1] = 1    # weight of the biase
    
    for j in range(0, num_hidden):
        u[j] = exp(-1 * numpy.linalg.norm(x - mean[:, j])**2 / (2 * sigma2[j]))
        
    v = [0] * num_class
    y = [0] * num_class
    for k in range(num_class):
        v[k] = 0
        for j in range(num_hidden+1):
            v[k] += (weight[k, j] * u[j])
        
        y[k] = sigmoid(v[k])
        
    return y, v, u
    
def get_delta(y, target):
    delta = [0] * num_class
    for k in range(num_class):
        delta[k] = (y[k] - target[k]) * y[k] * (1 - y[k])
    
    return delta
    
def compute_weight_gradient(delta, u):
    """ Compute weight gradient
    Return: 
    Note: This update routine changes the original weight
    """
    delta = np.matrix(delta).T # shape: num_class * 1
    u = np.matrix(u) # shape: 1 * (num_hidden+1)
    
    gradient = delta * u
    
    return np.array(gradient)

def compute_gaussian_gradient(mean, sigma2, weight, x, delta, u):
    """ Compute the gradient of Gaussian parameters using the old weight, instead of the updated weight
    Return: the gradient
    """
    # update the mean
    weight_delta_coff = [0] * num_hidden
    gradient_mean = np.zeros(mean.shape)        # shape: num_feature * num_hidden
    gradient_sigma2 = np.zeros(sigma2.shape)
    
    for j in range(num_hidden):
        # update each mean
        weight_delta_coff[j] = np.dot(weight[:, j], delta)
        
        pgradient_mean = 1.0 * u[j] / (2 * sigma2[j]) * (x - mean[:, j]) # shape: num_feature * 1
        gradient_mean[:, j] =  weight_delta_coff[j] * pgradient_mean
        
        pgradient_sigma2 = 1.0 * u[j] / (2 * sigma2[j]) * np.linalg.norm(x-mean[:, j])**2
        gradient_sigma2[j] = weight_delta_coff[j] * pgradient_sigma2
        
    return gradient_mean, gradient_sigma2

def classify_validation_set(mean, sigma2, weight):
    """ Method: check the classification acc of the validation set
    """
    total = len(x_validation)
    correct = 0
    for i in range(total):
        x = x_validation[i]
        y_true = y_validation[i]
        
        output, v, u = get_output(mean, sigma2, weight, x)
        maxp = 0
        y_pred = -1
        for k in range(num_class):
            if output[k] > maxp:
                maxp = output[k]
                y_pred = k
        
        if y_true == y_pred:
            correct += 1
            
    print 'Classification report on validation set: Total=%d, Correct=%d, Acc=%f' % (total, correct, correct*1.0/total)
    
    return correct*1.0/total
    
def check_convergence(val_acc):
    """ Check if the results has converged
    """
    return False

def train_RBF(learning_rate, max_iter, is_gaussian_unknown = False):
    """ Train RBF network: assume m_j and sigma_j^2 are known for j=1..num_hidden
    max_iter: max iterations
    learning_rate: learning rate
    is_gaussian_unknown: if True, assume mean and sigma^2 are unknown; if False, assume they are known
    
    Note: Consider how to set learning rate for different parameters for different iterations
    """
    mean, sigma2 = get_hidden_layer_parameter()
    weight = get_initial_weight()
    
    train_count = len(x_train)
    val_acc = [-1] * max_iter    # validation set acc list
    for t in range(max_iter):
        print 'Iteration: ', t+1
        for i in range(train_count):
            x = x_train[i]
            y = y_train[i]
            # the target for instance x
            target = [0] * num_class
            target[y] = 1
            # get the output given an input
            output, v, u = get_output(mean, sigma2, weight, x)
            delta = get_delta(output, target)

            # update weight using the gradient descent algorithm
            gradient_weight = compute_weight_gradient(delta, u)
            
            if is_gaussian_unknown:
                gradient_mean, gradient_sigma2 = compute_gaussian_gradient(mean, sigma2, weight, x, delta, u)
                mean -= (learning_rate * gradient_mean)
                sigma2 -= (learning_rate * gradient_sigma2)
                
            # update weight
            weight -= (learning_rate * gradient_weight)
        
        acc = classify_validation_set(mean, sigma2, weight)
        val_acc[t] = acc
        
        if check_convergence(val_acc):
            print 'Convergence criterion has been met and training is over!'
            break
        # waiting for human intervention
        
            
    return mean, sigma2, weight, t
            
def main():
    """
    RBF network structure:
        input layer: num_feature
        hidden layer: num_hidden
        output layer: num_class
    """
    learning_rate = 50.0
    max_iter = 100
    # assume m_j and sigma_j^2 are known for j=1..num_hidden
    mean, sigma2, weight, num_iter = train_RBF(learning_rate, max_iter, True)
    
    print 'Dumping into file...'
    f = open(dataset_name + '_RBF_pickle', 'w')
    pickle.dump([dataset_name, num_feature, num_hidden, num_class, mean, sigma2, weight, num_iter], f)
    #pickle.dump([cov, Sw, Sb], f) # version 1
    f.close()

if __name__ == '__main__':
    """ Load dataset and start to train the network
    """
    import sys
    dataset_name = sys.argv[1]
    num_hidden = int(sys.argv[2]) # number of hidden nodes
    
    print 'Reading dataset: ', dataset_name
    num_class, num_feature, x_train, y_train, x_test, y_test = read_dataset(dataset_name)
    
    # set a validation set to check convergence
    count = len(x_train) / 3
    # Note: the dataset has been shuffled
    x_validation = np.array(x_train[:count])
    y_validation = np.array(y_train[:count])
    
    x_train = x_train[count:]
    y_train = y_train[count:]
    
    main()
