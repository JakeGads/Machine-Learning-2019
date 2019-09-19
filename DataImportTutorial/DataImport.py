import pandas as pd
import numpy as np
import os
print(os.chdir(r"/home/jakegadaleta/Documents/code"))

filename = '/home/jakegadaleta/Documents/code/Machine-Learning-2019/DataImportTutorial/AnyTextFile.txt'

os.getcwd()

file = open('AnyTextFile.txt', mode='r')
text = file.read()
file.close()
print(text)

with open('AnyTextFile.txt', 'r') as f:
    print(f.readline)

f =  'SampleData.txt' 
data = np.loadtxt(f, delimiter=',',skiprows=2)