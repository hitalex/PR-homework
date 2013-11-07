#coding=utf8

"""
HWDB1.1 large database PCA and Fisher LDA dimension reduction
Requirement: Just one scan through the database
"""
import resource
import gc

import numpy as np


def main():
    train_path = '/home/kqc/dataset/HWDB1.1/train/'
    
    char_list = [] # character list
    char_map = dict() # char_code ==> index
    
    d = 512 # number of features
    
    mean = np.matrix([0] * d).T # mean for all dataset
    moment = np.zeros((d,d), np.float64) #  second order origin moment for all dataset
    
    class_mean = []
    
    total = 897758 # total number of samples
    class_total = [] # total number of samples for each class
    
    total_inverse = 1.0 / total
    
    x = np.matrix([0] * d).T
    for i in range(1, 241):
        # scan through the whole database
        path = train_path + str(1000+i) + '.txt'
        f = open(path, 'r')
        print 'Reading file: %s ' % path
        print 'Current mem usage: %s KB.' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        
        for line in f:
            #total = total + 1
            line = line.strip()
            seg_list = line.split(' ')
            char = seg_list[0]
            
            if not char in char_map:
                char_class = len(char_list)
                char_map[char] = char_class
                
                char_list.append(char)
                class_mean.append(np.array([0] * d).T)
                
                class_total.append(0)
                
                #print 'Add new char: %s with index: %d' % (char, char_class)
            else:
                char_class = char_map[char]
                
            class_total[char_class] = class_total[char_class] + 1
            
            for i in range(0, d):
                x[i, 0] = int(seg_list[i+1])
            
            mean = mean + x
            moment = moment + (x * x.T)
            
            class_mean[char_class] = class_mean[char_class] + x
            
            #print 'Current mem usage: %s KB.' % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            
            
        f.close()
    
    class_count = len(char_list) #number of classes
    
    mean = 1.0 / total * mean
    cov = 1.0 / total * moment - mean * mean.T
    #cov = moment - mean * mean.T
    
    class_cov = [0] * class_count
    class_prior = [0] * class_count
    Sb = np.zeros((d,d), np.float64)
    for i in range(class_count):
        Ni = class_total[i]
        class_prior[i] = Ni * 1.0 / total
        class_mean[i] = 1.0 / Ni * class_mean[i]
        
        tmp = class_mean[i] - mean
        Sb = Sb + class_prior[i] * (tmp * tmp.T)
    
    St = cov
    Sw = St - Sb
    
    # 序列化已经计算完成的对象
    import pickle
    f = open('matrixpickle', 'w')
    pickle.dump([cov, Sw, Sb], f)
    f.close()

if __name__ == '__main__':
    main()
