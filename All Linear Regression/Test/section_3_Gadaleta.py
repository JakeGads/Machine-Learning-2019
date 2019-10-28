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

    # Linear Calculations


if __name__ == "__main__":
    q1()
