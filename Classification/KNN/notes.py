import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd

dataset = pd.read_csv("")
dataset = dataset.dropna()
dataset = dataset.reset_index(drop = True)

X = dataset.iloc[:, [0,1]].values
y = dataset.iloc[:, 9].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 0)

#scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# fotting knn to the trainging set
from sklearn.neighbors import KNeighborsClassifier
