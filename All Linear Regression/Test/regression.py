import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder, PolynomialFeatures)

class Data():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.value = 0

    def value(self, value=0):
        if value != 0:
            self.value = value
        return value


def linear(file, X, y, save = False):
    """
    Applys Linear Regression to the model
    Shows the PLT and the train values

    Parameters:
    file (str): The location of the .csv it will read
    X (str): The name of the indenpent col
    y (str): The name of the depenent colum 

   """
    returner = Data(X, y)

    df = pd.read_csv("file")
    df = df.dropna()
    df = df.reset_index(drop=True)

    plt.title(f"{X} vs {y}")
    plt.xlabel(X)
    plt.ylabel(y)
    plt.plot(df[X], y, 'c.')
    plt.grid(True)

    line_X = df[X].values
    line_y = df[y].values

    model = LR()

    model.fit(line_X, line_y)
    if save:   
        plt.plot(line_X, model.predict(line_X), color = 'r')

        plt.savefig(plt.title(f"Linear {X} vs {y}"))
    
    returner.value(value=model.score(X))
    return returner

def multiple(file,  y, stringCols = [], save = False) :
    returner = Data("All", y)
    
    data = pd.read_csv(file)

    X = data.loc[:,  data.colums != y].values
    y = data.loc[: y].values

    for i in stringCols:
        le = LabelEncoder()
        X[:, i] = le.fit_transform(X[:, i])
        onehotencoder = OneHotEncoder(categorical_features = [i])
        X = onehotencoder.fit_transform(X).toarray()

    X_train, X_test, y_train, y_test = train_test_split(
    X, y,  test_size = .25, random_state = 0

    )

    regressor = LR()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    
    if save:
        plt.plot(X_test, y_test, color='g')
        plt.plot(X_test, y_pred, color='b')


        regression_model_mse = mean_squared_error(y_pred, y_test)
        regression_model_mse_sq = math.sqrt(regression_model_mse)
        plt.savefig()

    returner.value(regressor.score(X_test, y_test) )

    return returner

def polynomial(file, X, y, save = False):
    dataset = pd.read_csv(file)

    returner = Data(X, y)

    X = dataset[:, X].values
    y = dataset[:, y].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = .2, random_state = 0
    )

    
    reg_tests = []
    print("We are calculating the poly regression, this may take a while")
    
    for i in range(100):
        poly_reg = PolynomialFeatures(degree=i)
        X_ = poly_reg.fit_transform(X)
        X_test = poly_reg.fit_transform(X_test)
        
        lin_reg = LR()
        lin_reg.fit(X_,y)
        
        reg_tests.append(lin_reg.score(X_, y))
        
        
    highest = 0
    for i in range(len(reg_tests)):
        if reg_tests[i] > reg_tests[highest]:
            highest = i
        

    if save:
        plt.scatter(X, y, color = 'red')
        
        poly_reg = PolynomialFeatures(degree=highest)
        X_ = poly_reg.fit_transform(X)
        

        line_reg = LR()
        line_reg.fit(X_,y)

        plt.plot(
            X, 
            line_reg.predict(poly_reg.fit_transform(X)), 
            color = 'blue'
        )

        plt.title(f'{X} vs {y}')
        plt.xlabel(f'{X}')
        plt.ylabel(f'{y}')
        plt.savefig(f"{X} vs {y}") 

    returner.value(reg_tests[highest])
    return returner                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
