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

import regression  # see regression.py for further function defs I use throughout this script
from regression import Data  # this is for simple reading and writing of data


def q1():
    file = "Data/Employee_Compensation_SF.csv"

    X = ["Salaries", "Overtime", "Retirement"]
    y = "Health/Dental"

    linear_scores = []
    polynomial_scores = []
    # Linear
    for X_ in X:
        print("Testing Linear", X_)
        linear_scores.append(regression.linear(file, X_, y).value)
        print("Testing Poly", X_)
        polynomial_scores.append(regression.polynomial(file, X_, y).value)
        print("done")

    # creates list combos for evey possible combinations of the list
    X.append([X[0], X[1]])
    X.append([X[0], X[2]])
    X.append([X[1], X[2]])
    X.append([X[0], X[1], X[2]])

    # Multiple
    multiple_scores = []
    for X_ in X:
        print("Testing Multiple", X_)
        multiple_scores.append(regression.multiple(file, X_, y))

    for i in range(len(linear_scores)):
        print(f"""
        {X[i]}:
            Linear:\t\t{linear_scores[i]}
            Polynomial (highest):\t\t{polynomial_scores[i]} 
            
        """)
    # Multiple:\t\t{multiple_scores[i]}
    # for i in range(len(multiple_scores) - len(linear_scores)): 
    #     i += len(linear_scores)
    #     print(f"""
    #     {X[i]}:
    #         Multiple:\t\t{multiple_scores[i]}
    #     """)


if __name__ == "__main__":
    q1()
