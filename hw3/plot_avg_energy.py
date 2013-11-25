#coding=utf8

import matplotlib.pyplot as plt

if __name__ == '__main__':
    f = open('sigma2.csv', 'r')
    data = []
    for i in range(3):
        line = f.readline().strip()
        seg_list = line.split(',')
        
        row = [0] * len(seg_list)
        for k in range(len(seg_list)):
            row[k] = float(seg_list[k])

        row = row[:50]
        data.append(row)
        
    colors = ['red', 'green', 'blue', 'k']
    lines = []
    texts = ['$\sigma^2=0.1$', '$\sigma^2=1$', '$\sigma^2=10$']
    plt.figure()
    ax = plt.subplot(111)
    for i in range(len(data)):
        p = ax.plot(data[i], ls='-', c=colors[i], label=texts[i])
        #lines.append(p)
    
    #plt.legend(lines, text)
    ax.legend()
    plt.show()
