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
    
    weights[:, :-1] = np.random.normal(0, 0.01, size=(num_class, num_hidden))
    weights[:, -1] = 0 # set biases to 0
    
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
    means = np.random.multivariate_normal(m, cov, size=num_hidden) # shape: num_hidden * num_feature
    means = means.T
    
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
    u[-1] = 1    # weight of the biase
    
    for j in range(0, num_hidden):
        u[j] = exp(-1 * numpy.linalg.norm(x - means[:, j])**2 / (2 * sigma2[j]))
        
    v = [0] * num_class
    y = [0] * num_class
    for k in range(num_class):
        v[k] = 0
        for j in range(num_hidden+1):
            v[k] += (weights[k, j] * u[j])
        
        y[k] = sigmoid(v[k])
        
    return y, u
    
def update_weight(weights, target, y, u, learning_rate):
    """ Update network weights
    Return: the updated weights and delta
    Note: This update routine changes the original weights
    """
    delta = [0] * num_class
    for k in range(num_class):
        delta[k] = (y[k] - target[k]) * y[k] * (1 - y[k])
        for j in range(num_hidden+1):
            gradient = delta[k] * u[j]
            weights[k, j] -= (learning_rate * gradient)
            
    return delta

def update_gaussian_parameter(means, sigma2, weights, x, target, output, u, delta, learning_rate):
    """ Update Gaussian parameters using the old weights, instead of the updated weights
    Return: the updated means and sigma2
    """
    # update the means
    weights_delta_coff = [0] * num_hidden
    
    for j in range(num_hidden):
        # update each mean
        weights_delta_coff[j] = np.dot(weights[:, j], delta)
        
        pgradient_mean = u[j] / (2 * sigma2[j]) * (x - means[:, j]) # shape: num_feature * 1
        gradient_mean =  weights_delta_coff[j] * pgradient_mean
        
        pgradient_sigma2 = u[j] / (2 * sigma2[j]) * np.linalg.norm(x-means[:, j])**2
        gradient_sigma2 = weights_delta_coff[j] * pgradient_sigma2
        
        # update means and sigma together
        means[:, j] -= (learning_rate * gradient_mean)
        sigma2[j] -= (learning_rate * gradient_sigma2)

def check_convergence(previous_weights, weights):
    """ Check if the results has converged
    """
    return False

def train_RBF(learning_rate, max_iter, is_gaussian_unknown = False):
    """ Train RBF network: assume m_j and sigma_j^2 are known for j=1..num_hidden
    max_iter: max iterations
    learning_rate: learning rate
    is_gaussian_unknown: if True, assume means and sigma^2 are unknown; if False, assume they are known
    
    Note: Consider how to set learning rate for different parameters for different iterations
    """
    means, sigma2 = get_hidden_layer_parameter()
    weights = get_initial_weights()
    
    train_count = len(x_train)
    for t in range(max_iter):
        print 'Iteration: ', t+1
        for i in range(train_count):
            x = x_train[i]
            y = y_train[i]
            # the target for instance x
            target = [0] * num_class
            target[y] = 1
            # get the output given an input
            output, u = get_output(means, sigma2, weights, x)
            previous_weights = np.array(weights) # weights befor updating
            # update weights using the gradient descent algorithm
            delta = update_weight(weights, target, output, u, learning_rate)
            
            if is_gaussian_unknown:
                update_gaussian_parameter(means, sigma2, previous_weights, x, target, output, u, delta, learning_rate)
            
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
    weights, means, sigma2, num_iter = train_RBF(learning_rate, max_iter, True)

if __name__ == '__main__':
    """ Load dataset and start to train the network
    """
    import sys
    dataset_name = sys.argv[1]
    num_hidden = int(sys.argv[2]) # number of hidden nodes
    
    print 'Reading dataset: ', dataset_name
    num_class, num_feature, x_train, y_train, x_test, y_test = read_dataset(dataset_name)
    
    main()
