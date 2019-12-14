"""
Dataset: mushrooms.csv
Dataset Description: This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled
mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North
American Mushrooms (1981). Each species is identified as edible (e) or poisonous (p).

Problem description: Which of the columns best predicts the “classification” of a mushroom? Be mindful of
overfitting and underfitting.
"""

import warnings
import data_help as dh
import classification_algorithms as ca

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    data_set = dh.clean_data("Data/pdb_data_no_dups.csv")

    y = 'classification'

    x = data_set.columns

    scores = []

    scores.append(ca.knn(data_set, x, y, 25, "None"))
    scores.append(ca.decision_tree(data_set, x, y, "None"))
    scores.append(ca.logistic_regression(data_set, x, y, "None"))
    scores.append(ca.random_forest(data_set, x, y, "None"))

    print(f"""
    {scores[0]},\tKNN
    {scores[1]},\tDecision Tree
    {scores[2]}\tLogistic Regression
    {scores[3]},\tRandom Forrest
    """)
