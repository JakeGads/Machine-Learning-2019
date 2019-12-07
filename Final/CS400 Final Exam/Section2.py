"""
Dataset: insurance.csv
Dataset description: It is a list of beneficiaries for an insurance company.
a. age: age of primary beneficiary
b. sex: insurance contractor gender, female, male
c. bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height,
    objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
d. children: Number of children covered by health insurance / Number of dependents
e. smoker: Smoking
f. region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest.
g. charges: Individual medical costs billed by health insurance

Problem description: We have charges available reported for each insured person. Our objective is to forecast the
charges billed to an individual. The task is to find which columns and which model can best predict the charges for
an individual. Be mindful of overfitting and underfitting.
"""
import pandas as pd
import numpy as np
import data_help as dh


if __name__ == "__main__":
    data_set = dh.clean_data("Data/insurance.csv")

    y = data_set['charges']

    x = dh.gen_permutations(data_set)
    for i in x:
        print(i)

