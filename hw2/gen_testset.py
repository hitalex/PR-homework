#coding=utf8

"""
Generate test set using the char list into ONE file
"""

def load_id_list(file_path):
    """ 从文件内导入所有的id，每行一个，返回这些id的list
    """
    f = open(file_path, 'r')
    id_list = []
    for line in f:
        line = line.strip()
        if line != '':
            id_list.append(line)
    f.close()
    
    return id_list

def main():
    test_path = '/home/kqc/dataset/HWDB1.1/test/'
    
    # laod char list and build the char map
    char_map = dict()
    char_list = load_id_list('charlist.txt')
    index = 0
    for char in char_list:
        char_map[char] = index
        index = index + 1
    
    # test file
    dst_path = '/home/kqc/dataset/HWDB1.1/test.txt'
    ftest = open(dst_path, 'w')
    
    for index in range(241, 301):
        path = test_path + str(1000+index) + '.txt'
        print 'Reading file: ', path
        f = open(path, 'r')
        for line in f:
            line = line.strip()
            seg_list = line.split(' ')
            char = seg_list[0]

            if char in char_map:
                label = char_map[char]
                ftest.write(','.join(seg_list[1:]) + ',' + str(label) + '\n')
            else:
                print 'Error: %s not found in charset.' % char
                
        f.close()
        
    ftest.close()

if __name__ == '__main__':
    main()
