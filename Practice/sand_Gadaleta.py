import warnings
from collections import Counter
from itertools import permutations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

"""
1. Classification (RI)
2. Regression
"""

def knn(file, y):
    # open the df and prep it
    dataset = pd.read_csv(file)
    dataset = dataset.dropna()
    dataset = dataset.reset_index(drop=True)
    
    # adds scoring lists
    scoringList = []
    columnList1 = []
    columnList2 = []
    knn_scores = []
    log_loss_score = []

    # preparing the data for conversion
    a = np.arange(0,len(dataset.columns),1)
    #list out all perms
    perm = list(permutations(a))

    y = convertIndex(dataset.columns, y)

    y = dataset.iloc[:, y].values

    for i in perm:
        X = dataset.iloc[:, [i[0], i[1]]].values
        
def convertIndex(data, y):
    if not isinstance(y, int):
        for i in range(len(data)):
            if data.columns[i] == y:
                y = i
                return y
        if not isinstance(y, int):
            print("Failed to generate a index for y")
            exit()
    

if __name__ == "__main__":
    knn("Data/glass.csv", "RI")
