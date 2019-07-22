from __future__ import print_function
import os

def opennewfile(path):
    result = [] 
    files = os.listdir(path)
    for name in files:
        result.append(name)
    result.sort(reverse = True)

    return(path+'/'+result[0])
