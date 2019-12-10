import pandas as pd
from sklearn.tree import DecisionTreeClassifier
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
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0) #Entropy is nothing but the measure of disorder.
classifier.fit(X_train, y_train)
    
#predicting the test set results
y_pred = classifier.predict(X_test)

print('Accuracy Score: ', metrics.accuracy_score(y_test, y_pred))
