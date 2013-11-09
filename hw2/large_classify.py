#coding=utf8

"""
Do PCA and Fisher LDA dimension reduction using the pre-computed co-variance matrix
"""
import pickle
import math

import numpy as np
import numpy.linalg
import scipy.linalg

def classify_LDF(num_class, num_feature, class_prior, class_mean, avg_cov, W, testpath):
    """classification using LDF
    """
    d = num_feature
    
    Wt = W.T
    new_avg_cov = Wt * avg_cov * W
    avg_cov_inv = new_avg_cov.getI()
    
    weight = [0] * num_class
    weight0 = [0] * num_class
    
    for i in range(num_class):
        weight[i] = 2 * avg_cov_inv.T * Wt * class_mean[i]
        weight0[i] = 2 * math.log(class_prior[i]) - class_mean[i].T * W * avg_cov_inv * Wt * class_mean[i]
        
    f = open(testpath, 'r')
    y_true = []
    y_pred = []
    
    correct = 0
    tmp = np.matrix([0] * d).T
    
    for line in f:
        line = line.strip()
        seg_list = line.split(',')
        
        for i in range(d):
            tmp[i, 0] = int(seg_list[i])
        
        x = Wt * tmp # transform to low dim
        
        label = int(seg_list[-1])
        y_true.append(label)
        
        max_posteriori = -float('inf')
        prediction = -1
        for i in range(num_class):
            p = (-1 * (x.T * avg_cov_inv * x) + weight[i].T * x + weight0[i])[0,0]
            if p > max_posteriori:
                max_posteriori = p
                prediction = i
                
        y_pred.append(prediction)
        
        print 'Label: %d, prediction: %d' % (label, prediction)
        
        if label == prediction:
            correct = correct + 1
            print 'Correct: ', correct
        
    f.close()
    total = len(y_pred)
    print 'Total: %d, Correct: %d, acc: %f' % (total, correct, correct*1.0 / total)
    
    return y_true, y_pred
    
def classify_nearest_mean(num_class, num_feature, class_mean, W, testpath):
    """classification using nearest mean
    """
    Wt = W
    d = num_feature
    
    f = open(testpath, 'r')
    y_true = []
    y_pred = []
    
    correct = 0
    for line in f:
        line = line.strip()
        seg_list = line.split()
        
        x = [0] * d
        for i in range(d):
            x[i] = int(seg_list[i])
        x = np.matrix(x).T
        label = int(seg_list[-1])
        y_true.append(label)
        
        min_dis = float('inf')
        prediction = -1
        for i in range(num_class):
            x = np.array(Wt * x)
            dis = np.linalg.norm(x - class_mean[i])
            
            if dis < min_dis:
                min_dis = dis
                prediction = i
            
            y_pred.append(prediction)
            
            if prediction == label:
                correct = correct + 1
                
    f.close()
    
    total = len(y_pred)
    print 'Total: %d, Correct: %d, acc: %f' % (total, correct, correct*1.0 / total)
    
    return y_true, y_pred

def main():
    # laod pre-computed variabls
    
    print 'Loading variables from pickle...'
    f = open('matrixpickle', 'r')
    total, char_list, class_count, class_total, mean, cov, class_mean, \
        class_prior, Sb = pickle.load(f)
    f.close()
    
    char_map = dict()
    index = 0
    for char in char_list:
        char_map[char] = index
        index = index + 1
    
    d = 512
    dr_method = 'PCA'
    #dr_method = 'FisherLDA'
    
    for s in [20]:
        print 'Dimension reduction: %d' % s
        
        St = cov
        Sw = St - Sb
        
        if dr_method == 'PCA':
            eig_values, eig_vectors = np.linalg.eig(cov)
            
        elif dr_method == 'FisherLDA':
            eig_values, eig_vectors = scipy.linalg.eigh(Sb, Sw)
        else:
            print 'Unknown dimention reduction method!'
            break
        
        # 按照特征值大小排序
        idx = eig_values.argsort()
        idx = idx[::-1] # reverse the array
        eig_values = eig_values[idx]
        eig_vectors = eig_vectors[:,idx]
        # 转换矩阵
        W = eig_vectors[:, :s]
        
        testpath = '/home/kqc/dataset/HWDB1.1/test.txt'
        print 'LDF prediction:'
        y_true, y_pred = classify_LDF(class_count, d, class_prior, class_mean, Sw, W, testpath)
        
        return 
        
        print 'Nearest mean prediction:'
        y_true, y_pred = classify_nearest_mean(class_count, d, class_mean, W, testpath)
        
        
if __name__ == '__main__':
    main()
