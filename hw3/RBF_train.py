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
import time

import numpy as np
import numpy.linalg
import numpy.random
#import ipdb
import sklearn.metrics
from sklearn.cluster import KMeans

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
    
def get_hidden_layer_parameter_gaussian():
    """ Set the hidden layer prameters: mean and \sigma^2
    initial_sigma2: para to tuned
    Method: use Gaussian
    """
    mean = np.zeros((num_feature, num_hidden)) # shape: num_feature * num_hidden

    # all \sigma are the same
    sigma2 = np.array([10] * num_hidden, float)
    # get the mean of the whole training dataset
    m = np.mean(x_train, axis=0)
    # the covariance matrix
    # assuming the features are independent
    cov = 0.05 * np.eye(num_feature)
    mean = np.random.multivariate_normal(m, cov, size=num_hidden) # shape: num_hidden * num_feature
    mean = mean.T
    
    return mean, sigma2

def get_hidden_layer_parameter_kmeans(initial_sigma2 = 10):
    """ Set the hidden layer prameters: mean and \sigma^2
    Method: Use kmeans algorithm to find num_hidden clustering centers as means
    """
    #mean = np.zeros((num_feature, num_hidden)) # shape: num_feature * num_hidden
    # all \sigma are the same
    sigma2 = np.array([initial_sigma2] * num_hidden, float)
    # get the mean of the whole training dataset
    classifier = KMeans(n_clusters=num_hidden).fit(x_train)
    mean = (classifier.cluster_centers_).T # shape: num_feature * num_hidden
    
    return mean, sigma2
        
def sigmoid(x):
    """ the Sigmoid function
    x: a scala
    """
    # 1/exp(20) = 2.061153622438558e-09
    if x >= 30:
        return 1
    elif x <= -30:
        return 0
        
    return 1.0 / (1 + exp(-x))
    
def get_output(mean, sigma2, weight, x):
    """ Get the output for the single input instance, i.e. y_k(x_i)
    Return: the probability of x belonging to each class, list of length num_class
    """
    u = [0] * (num_hidden+1) # the hidden layer output
    tmp = [0] * (num_hidden+1) # the hidden layer output
    u[-1] = 1    # weight of the biase
    
    for j in range(0, num_hidden):
        tmp[j] = -1 * numpy.linalg.norm(x - mean[:, j])**2 / (2 * sigma2[j])
        u[j] = exp(tmp[j])
        
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
        
        pgradient_sigma2 = 1.0 * u[j] / (2 * sigma2[j] ** 2) * np.linalg.norm(x-mean[:, j])**2
        gradient_sigma2[j] = weight_delta_coff[j] * pgradient_sigma2
        
    return gradient_mean, gradient_sigma2

def classify(mean, sigma2, weight, test_set):
    """ classify instances using the network
    Return: the predicted labels
    """
    total = len(test_set)
    y_pred = [0] * total
    for i in range(total):
        x = test_set[i]
        output, v, u = get_output(mean, sigma2, weight, x)
        maxp = 0
        prediction = -1
        for k in range(num_class):
            if output[k] > maxp:
                maxp = output[k]
                prediction = k
        
        assert(prediction != -1) # this should never happen
        y_pred[i] = prediction
    
    return y_pred
    
def classify_with_energy(mean, sigma2, weight, test_set, y_true):
    """ classify instances using the network
    Return: the predicted labels and average energy
    """
    total = len(test_set)
    y_pred = [0] * total
    avg_energy = 0
    for i in range(total):
        x = test_set[i]
        output, v, u = get_output(mean, sigma2, weight, x)
        Ei = compute_energy(output, y_true[i])
        avg_energy += Ei
        maxp = 0
        prediction = -1
        for k in range(num_class):
            if output[k] > maxp:
                maxp = output[k]
                prediction = k
        
        assert(prediction != -1) # this should never happen
        y_pred[i] = prediction
    
    avg_energy /= total
    
    return y_pred, avg_energy
    
def compute_energy(output, label):
    """ Compute energy for each sample
    """
    t = [0] * num_class
    t[label] = 1
    t = np.array(t, float)
    y = np.array(output, float)
    E = sum((y-t) ** 2) / 2.0
    
    return E
    
def check_convergence(val_acc, t):
    """ Check if the results has converged
    """
    if abs(val_acc[t] - val_acc[t-1]) < 1e-6:
        return True
    else:
        return False

def train_RBF(lr_weight, lr_mean, lr_sigma2, max_iter, min_iter, is_gaussian_unknown = False, initial_sigma2 = 100):
    """ Train RBF network: assume m_j and sigma_j^2 are known for j=1..num_hidden
    max_iter: max iterations
    min_iter: min iterations
    learning_rate: learning rate
    is_gaussian_unknown: if True, assume mean and sigma^2 are unknown; if False, assume they are known
    
    Note: Consider how to set learning rate for different parameters for different iterations
    """
    #mean, sigma2 = get_hidden_layer_parameter_gaussian()
    mean, sigma2 = get_hidden_layer_parameter_kmeans(initial_sigma2) # for experiment
    #mean, sigma2 = get_hidden_layer_parameter_kmeans()
    weight = get_initial_weight()
    
    train_count = len(x_train)
    val_acc = [-1] * max_iter    # validation set acc list
    avg_energy_list = [0] * max_iter
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
            #print 'Class prob. output :', output
            #print 'Output layer output(v):', v
            #print 'Hidden layer output(u):', u
            #print ''
            #time.sleep(1)
            
            delta = get_delta(output, target)

            # update weight using the gradient descent algorithm
            gradient_weight = compute_weight_gradient(delta, u)
            
            if is_gaussian_unknown:
                gradient_mean, gradient_sigma2 = compute_gaussian_gradient(mean, sigma2, weight, x, delta, u)
                mean -= (lr_mean * gradient_mean)
                sigma2 -= (lr_sigma2 * gradient_sigma2)
                
                #print 'Gradient mean:', gradient_mean
                #print 'Gradient sigma2:', gradient_sigma2
                
            # update weight
            weight -= (lr_weight * gradient_weight)
        
        # 输出最后一次的hidden layer输出
        #print 'Hidden layer output:', u
        #print 'Current mean:', mean
        #print 'Current sigma2:', sigma2
        #time.sleep(1)
        
        #y_pred = classify(mean, sigma2, weight, x_validation)
        y_pred, avg_energy = classify_with_energy(mean, sigma2, weight, x_validation, y_validation)
        avg_energy_list[t] = avg_energy
        
        val_acc[t] = sklearn.metrics.accuracy_score(y_validation, y_pred)
        print 'Acc score:', val_acc[t]
        print 'Avg energy: ', avg_energy
        
        if t > min_iter and check_convergence(val_acc, t):
            print 'Convergence criterion has been met and training is over!'
            break
        # waiting for human intervention
        
        print ''
        #time.sleep(1)
    
    # write data to file
    avg_energy_list = avg_energy_list[:t]
    val_acc = val_acc[:t]
    
    return mean, sigma2, weight, t, avg_energy_list, val_acc
            
def main():
    """
    RBF network structure:
        input layer: num_feature
        hidden layer: num_hidden
        output layer: num_class
    """
    f = open('sigma2.csv', 'w')
    for initial_sigma2 in [1, 10, 100]:
        # learning rate for 3 variabls
        times = 1
        lr_weight = 1.0 * times
        lr_mean = 1.0 * times
        lr_sigma2 = 1.0 * times
        
        max_iter = 200
        min_iter = 50
        # assume m_j and sigma_j^2 are known for j=1..num_hidden
        mean, sigma2, weight, num_iter, avg_energy_list, val_acc = train_RBF(lr_weight, lr_mean, lr_sigma2, max_iter, min_iter, False, initial_sigma2)
        
        # write to file
        tmp = [str(item) for item in avg_energy_list]
        f.write(','.join(tmp) + '\n')
    f.close()
    
    print 'Dumping into file...'
    f = open(dataset_name + '_RBF_pickle', 'w')
    pickle.dump([dataset_name, num_feature, num_hidden, num_class, mean, sigma2, weight, num_iter], f)
    #pickle.dump([cov, Sw, Sb], f) # version 1
    f.close()
    
    y_pred = classify(mean, sigma2, weight, x_train)
    print 'Training acc:', sklearn.metrics.accuracy_score(y_train, y_pred)
    
    y_pred = classify(mean, sigma2, weight, x_test)
    print 'Test acc:', sklearn.metrics.accuracy_score(y_test, y_pred)
    
    
    
if __name__ == '__main__':
    """ Load dataset and start to train the network
    """
    import sys
    dataset_name = sys.argv[1]
    num_hidden = int(sys.argv[2]) # number of hidden nodes
    
    print 'Reading dataset: ', dataset_name
    num_class, num_feature, x_train, y_train, x_test, y_test = read_dataset(dataset_name, scaled=True)
    
    # set a validation set to check convergence
    count = len(x_train) / 3
    # Note: the dataset has been shuffled
    x_validation = np.array(x_train[:count])
    y_validation = np.array(y_train[:count])
    
    x_train = x_train[count:]
    y_train = y_train[count:]
    
    main()
