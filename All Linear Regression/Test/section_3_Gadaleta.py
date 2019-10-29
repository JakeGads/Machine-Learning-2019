"""
Apply the regression models to the following data - Employee_Compensation_SF.csv.
Questions to answer:
1. Are “Health/Dental” benefits dependent upon “Salaries”, “Overtime” and “Retirement”?
2. If the answer is yes, then which type of regression best fits this analysis – Linear, Multiple or Polynomial?
3. This question requires submission of a python script that contains the implementation of the model that best fits
the data.
4. Include one graph using linear regression with a discussion on what that graph represents.
5. Include one graph using polynomial regression with a discussion on what that graph represents.
"""

import math

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import (LabelEncoder, OneHotEncoder, PolynomialFeatures)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def q1():
    def linear():
        col = [
        "Year Type","Year","Organization Group Code","Organization Group","Department Code","Department",
        "Union Code","Union","Job Family Code","Job Family","Job Code","Job,Employee Identifier","Salaries","Overtime",
        "Other Salaries","Total Salary","Retirement","Health/Dental","Other Benefits","Total Benefits","Total Compensation"
        ]
        file = "Data/Employee_Compensation_SF.csv"
        X_ = ["Salaries", "Overtime", "Retirement"]
        X_loc = []
        for i in X_:
            for h in range(len(col)):
                if i == col[h]:
                    X_loc.append(h)
                if col[h] == "Health/Dental":
                    y_loc = h

        # y = "Health/Dental"
        data = pd.read_csv(file)
        lineY = data.iloc[y_loc].values
        scores = []

        for X__ in X_loc:
            lineX = data.iloc[X__].values
            if not (isinstance(lineX, float) or isinstance(lineX, int)):
                labelencoder_X = LabelEncoder()
                lineX[X__] = labelencoder_X.fit_transform(lineX[X__])
                onehotencoder = OneHotEncoder(categorical_features=[X__])
                lineX = onehotencoder.fit_transform(X).toarray()

            model = LR()
            model.fit(lineX, lineY)
            scores.append(model.score(lineX, lineY))

    linear()





    # Linear Calculations


if __name__ == "__main__":
    q1()
