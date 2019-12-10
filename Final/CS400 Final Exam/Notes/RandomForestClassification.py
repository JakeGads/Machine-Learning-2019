import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\guptap\Downloads\breast_cancer_dataset.csv',header=0)

#independent variable
X = dataset.iloc[:,:-1].values 

#dependent variable
y = dataset.iloc[:, 9].values 

#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    
# Fitting Decision Tree Classification to the Training set
classifier = RandomForestClassifier(n_estimators=600, max_depth=300, max_features='sqrt')
classifier.fit(X_train, y_train)
    
#predicting the test set results
y_pred = classifier.predict(X_test)

print('Accuracy Score: ', metrics.accuracy_score(y_test, y_pred))
