# -*- coding: utf-8 -*-
import os
import re
import matplotlib.pyplot as plt

path = "D:\\document\\毕业论文\\file\\03\\equilibrium-propagation\\result"

def find_word(word, text):
    search_w = r'(^|[^\w]){}([^\w]|$)'.format(word)
    search_w = re.compile(search_w, re.IGNORECASE)
    search_result = re.search(search_w, text)    
    return bool(search_result)

def read_log():
    files = os.listdir(path)
    acc = []

    for file in files:
        position = path+'\\'+file
        with open(position,'r',encoding = 'utf-8') as f:
            data = f.readlines()
            pattern = re.compile('test_acc:(.+)\t')
            data = str(data)
            result = pattern.findall(data)
            '''a = re.findall(r'(?<=\test_acc: )\d+\.\d*', data)'''
            print(result)
        f.close()
    
        
def draw():
    
    plt.subplot() 
    plt.legend()
    plt.savefig()
    plt.show()
    
if __name__ == '__main__':
    read_log()
    