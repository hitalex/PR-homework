#coding=utf8

"""
Do PCA and Fisher LDA dimension reduction using the pre-computed co-variance matrix

Note：这是一个经过了优化的版本，它能够一次性计算多个降维的结果。
"""
import pickle
import math

import numpy as np
import numpy.linalg
import scipy.linalg

def classify_LDF(num_class, num_feature, class_prior, class_mean, avg_cov, eig_vectors, srange, testpath):
    """classification using LDF
    
    W: the 512*512 eignvector matrix
    srange: the target dimension list
    """
    d = num_feature
        
    log_class_prior = [0] * num_class
    for i in range(num_class):
        log_class_prior[i] = math.log(class_prior[i])
        
    # 将所有的class_mean整理到一个矩阵中
    class_mean_matrix = [0] * num_class
    for i in range(num_class):
        class_mean_matrix[i] = class_mean[i]
        
    # size: d * k, where k is the number of classes
    class_mean_matrix = np.hstack(class_mean_matrix)
    
    maxs = max(srange)
    W_all = np.matrix(eig_vectors)
    W_all = W_all[:, :maxs] # get the largest transform matrix
    Wt_all = W_all.T
    
    class_mean_matrix = Wt_all * class_mean_matrix # shape: maxs * k
    
    total = 0
    correct = [0] * len(srange) # one correct count for each s in srange
    tmp = np.matrix([0] * d).T
    f = open(testpath, 'r')
    
    for line in f:
        line = line.strip()
        seg_list = line.split(',')
        
        for i in range(d):
            tmp[i, 0] = int(seg_list[i])
            
        total += 1
            
        # for each target dimension
        sindex = 0
        for s in srange:
            W = W_all[:, :s]        # shape: d * s
            Wt = Wt_all[:s, :]      # shape: s * d
            
            new_avg_cov = Wt * avg_cov * W      # shape: s * s
            avg_cov_inv = new_avg_cov.getI()
            
            weight = 2 * avg_cov_inv.T * class_mean_matrix[:s, :]       # shape: s * k
            
            weight0 = [0] * num_class
            for i in range(num_class):
                weight0[i] = 2 * log_class_prior[i] - class_mean_matrix[:s, i].T * avg_cov_inv * class_mean_matrix[:s, i]
                        
            #  transform to low dim
            x = Wt * tmp # shape: s * 1
            
            label = int(seg_list[-1])
            
            max_posteriori = -float('inf')
            prediction = -1
            for i in range(num_class):
                p = (-1 * (x.T * avg_cov_inv * x) + weight[:, i].T * x + weight0[i])[0,0]
                if p > max_posteriori:
                    max_posteriori = p
                    prediction = i
            
            print 'Label: %d, prediction: %d' % (label, prediction)
            
            if label == prediction:
                correct[sindex] += 1
                print '[s=%d]Total: %d, Correct: %d, acc: %f' % (srange[sindex], total, correct[sindex], correct[sindex]*1.0 / total)
                
            sindex += 1
        
    f.close()
    
    for sindex in len(srange):
        s = srange[sindex]
        print '[s=%d]Total: %d, Correct: %d, acc: %f' % (s, total, correct[sindex], correct[sindex]*1.0 / total)
    
def classify_nearest_mean(num_class, num_feature, class_prior, class_mean, delta, W, srange, testpath):
    """classification using nearest mean
    """
    Wt = W.T
    d = num_feature
    
    f = open(testpath, 'r')
    y_true = []
    y_pred = []
    
    correct = 0
    total = 0
    tmpx = np.matrix([0] * d).T
    
    log_class_prior = [0] * num_class
    for i in range(num_class):
        log_class_prior[i] = math.log(class_prior[i])
    
    for line in f:
        line = line.strip()
        seg_list = line.split(',')
        
        for i in range(d):
            tmpx[i, 0] = int(seg_list[i])
        
        x = Wt * tmpx # transform to low dim
        
        label = int(seg_list[-1])
        y_true.append(label)
        
        total = total + 1
        
        maxp = - float('inf')
        prediction = -1
        for i in range(num_class):
            #tmp = x - Wt * class_mean[i]
            #tmp = sum( np.array(tmp) ** 2) * 1.0
            tmp = np.linalg.norm(x - Wt * class_mean[i]) ** 2
            tmp = -1.0 * tmp / delta**2 + 2 * log_class_prior[i]
            
            if tmp > maxp:
                maxp = tmp
                prediction = i
            
        y_pred.append(prediction)
            
        print 'Label: %d, prediction: %d' % (label, prediction)
            
        if prediction == label:
            correct = correct + 1
            print 'Total: %d, Correct: %d, acc: %f' % (total, correct, correct*1.0 / total)
                
    f.close()
    
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
    
    St = cov
    Sw = St - Sb
    
    if dr_method == 'PCA':
        eig_values, eig_vectors = np.linalg.eig(cov)
        
    elif dr_method == 'FisherLDA':
        eig_values, eig_vectors = scipy.linalg.eigh(Sb, Sw)
    else:
        print 'Unknown dimention reduction method!'
    
    # 按照特征值大小排序
    idx = eig_values.argsort()
    idx = idx[::-1] # reverse the array
    eig_values = eig_values[idx]
    eig_vectors = eig_vectors[:,idx]

    # 转换矩阵，这里保存所有的512*512维的结果
    srange = range(10, 100, 10)
    
    testpath = '/home/kqc/dataset/HWDB1.1/test.txt'
    print 'LDF prediction:'
    classify_LDF(class_count, d, class_prior, class_mean, Sw, eig_vectors, srange, testpath)
    
    #prof.runcall(classify_LDF, class_count, d, class_prior, class_mean, Sw, eig_vectors, srange, testpath)
    
    print 'Nearest mean prediction:'
    # hyper paramter
    delta = 1
    #y_true, y_pred = classify_nearest_mean(class_count, d, class_prior, class_mean, delta, eig_vectors, srange, testpath)
        
        
if __name__ == '__main__':
    import hotshot
    import hotshot.stats
    prof = hotshot.Profile("prof.txt", 1)
    
    from datetime import datetime
    start = datetime.now()
    main()
    end = datetime.now()
    
    print 'Time taken: ', end-start
    
    prof.close()
    p = hotshot.stats.load("prof.txt")
    p.print_stats()
