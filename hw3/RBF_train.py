#coding=utf8

"""
RBF network training
"""
import pickle
from math import exp
import os

import numpy as np
import numpy.linalg
import numpy.random

os.sys.path.append('/home/kqc/github/PR-homework/')
from hw1.readdata import read_dataset

def get_initial_weights():
    """ Set the initial weights and the shape is (num_class, num_hidden+1)
    weights[:, 0] are biases
    Note: set them to small random values chosen from a zero-mean Gaussian with a standard deviation of about 0.01
    """
    weights = np.zeros((num_class, num_hidden+1)) # add biases
    # set biases to 0
    weights[:, 0] = 0
    weights[:, 1:] = np.random.normal(0, 0.01, size=(num_class, num_hidden))
    
    return weights
    
def get_hidden_layer_parameter():
    """ Set the hidden layer prameters: means and \sigma^2
    """
    means = np.zeros((num_feature, num_hidden)) # shape: num_feature * num_hidden
    sigma2 = np.array([0] * num_hidden) # \sigma^2

    # all \sigma are the same
    sigma2 = [1] * num_hidden
    # get the mean of the whole training dataset
    m = np.mean(x_train, axis=0)
    # the covariance matrix
    # assuming the features are independent
    cov = 0.5 * np.eye(num_feature)
    means = np.random.multivariate_normal(m, cov, size=num_hidden) # shape: num_feature * num_hidden
    
    return means, sigma2
    
def sigmoid(x):
    """ the Sigmoid function
    x: a scala
    """
    return 1.0 / (1 + exp(-x))
    
def get_output(means, sigma2, weights, x):
    """ Get the output for the single input instance, i.e. y_k(x_i)
    Return: the probability of x belonging to each class, list of length num_class
    """
    u = [0] * (num_hidden+1) # the hidden layer output
    u[0] = 1    # weight of the biase
    
    for j in range(0, num_hidden):
        u[j+1] = exp(-1 * numpy.linalg.norm(x - means[j, :])**2 / (2 * sigma2[j]))
        
    v = [0] * num_class
    y = [0] * num_class
    for k in range(num_class):
        for j in range(num_hidden+1):
            v[k] += (weights[k, j] * u[j])
        
        y[k] = sigmoid(v[k])
        
    return y, u
    
def update_weight(weights, target, y, u, learning_rate):
    """ Update network weights
    """
    delta = [0] * num_class
    for k in range(num_class):
        delta[k] = (y[k] - target[k]) * y[k] * (1 - y[k])
        for j in range(num_hidden+1):
            gradient = delta[k] * u[j]
            weights[k, j] -= (learning_rate * gradient)
            
    return weights
            
def check_convergence(previous_weights, weights):
    """ Check if the results has converged
    """
    return False

def train_RBF_simplified(learning_rate, max_iter):
    """ Train simple RBF network: assume m_j and sigma_j^2 are known for j=1..num_hidden
    max_iter: max iterations
    learning_rate: learning rate
    """
    means, sigma2 = get_hidden_layer_parameter()
    weights = get_initial_weights()
    
    train_count = len(x_train)
    for t in range(max_iter):
        print 'Iteration: ', t+1
        previous_weights = np.array(weights)
        for i in range(train_count):
            x = x_train[i]
            y = y_train[i]
            # the target for instance x
            target = [0] * num_class
            target[y] = 1
            # get the output given an input
            output, u = get_output(means, sigma2, weights, x)
            # update weights using the gradient descent algorithm
            weights = update_weight(weights, target, output, u, learning_rate)
            
        if check_convergence(previous_weights, weights):
            print 'Convergence criterion has been met and training is over!'
            break
            
    return weights, means, sigma2, t
            
def main():
    """
    RBF network structure:
        input layer: num_feature
        hidden layer: num_hidden
        output layer: num_class
    """
    learning_rate = 1.0
    max_iter = 10
    # assume m_j and sigma_j^2 are known for j=1..num_hidden
    weights, weights, means, sigma2 = train_RBF_simplified(learning_rate, max_iter)

if __name__ == '__main__':
    """ Load dataset and start to train the network
    """
    import sys
    dataset_name = sys.argv[1]
    num_hidden = int(sys.argv[2]) # number of hidden nodes
    
    print 'Reading dataset: ', dataset_name
    num_class, num_feature, x_train, y_train, x_test, y_test = read_dataset(dataset_name)
    
    main()
