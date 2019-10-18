import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as LR

# loads in the csv and cleans the failed the load data
df = pd.read_csv("housing.csv")
df = df.dropna()
df = df.reset_index(drop=True)


Xs = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
#    "ocean_proximity",
]

y = df["median_house_value"]
counter = 0

plt.style.use('ggplot')

for X in Xs:

    if counter == 8:
        counter += 1
        continue
    
    #builds the graph itself
    plt.title(f"{X} vs median_house_value")
    plt.xlabel(X)
    plt.ylabel("median_house_value")
    plt.plot(df[X], y, 'c.')
    plt.grid(True)
    
    #Calculates the Linear Regression

    '''
    longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,
    households,median_income,median_house_value,ocean_proximity'''
    
    try:
        lineX = df.iloc[:,counter:counter + 1].values
        lineY = df.iloc[:,8:9].values

        model = LR()
        model.fit(lineX, lineY)
    except:
        lineX = df.iloc[:,:-1].values
        lineY = df.iloc[:,8:9].values

        model = LR()
        model.fit(lineX, lineY)
    plt.plot(lineX, model.predict(lineX), color ='r')

    plt.savefig(f"{X} vs median_house_value")
    plt.close()
    counter += 1
    




# For the sake of my sanity we will be starting with a data set
dataset = pd.read_csv('housing.csv')
dataset = dataset.dropna()
dataset = dataset.reset_index(drop = 2)

X=dataset.iloc[:, :-2].values
y=dataset.iloc[:, 8:9].values

# Spliting into a training and a test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,  test_size = .25, random_state = 0
)

from sklearn.linear_model import LinearRegression as LR

regressor = LR()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

regressor.score(X_test, y_test)

plt.plot(X_test, y_test, color='g')
plt.plot(X_test, y_pred, color='b')

from sklearn.metrics import mean_squared_error
import math
regression_model_mse = mean_squared_error(y_pred, y_test)
regression_model_mse_sq = math.sqrt(regression_model_mse)

print(regression_model_mse_sq)
plt.savefig('MultiRegression')
