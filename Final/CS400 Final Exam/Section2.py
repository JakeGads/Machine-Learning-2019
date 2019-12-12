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

Jake -
This must be a classification problem as you will use data to predict future data, I will call all of my classification
code
"""
import warnings

import data_help as dh
import classification_algorithms as ca

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    data_set = dh.clean_data("Data/insurance.csv")

    y = data_set['charges']

    x = dh.gen_permutations(data_set)

    print(f"KNN:\t\t{ca.knn(data_set, x, 'charges', 100, 'out_files/2_knn.csv')} ")
    print(f"LogReg:\t\t\t{ca.logistic_regression(data_set, x, 'charges', 'out_files/2_log.csv')}")
    print(f"DecTree:\t\t{ca.decision_tree(data_set, x, 'charges', 'out_files/2_DecTree.csv')}")
    print(f"RandFor:\t\t{ca.random_forest(data_set, x, 'charges', 'out_files/2_RandFor.csv')}")



