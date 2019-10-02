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

for X in Xs:
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
    lineX = df.iloc[:,:-1].values
    lineY = df.iloc[:,8:9].values

    model = LR()
    model.fit(lineX, lineY)

    plt.plot(lineX, model.predict(lineX), color ='b')

    plt.savefig(f"{X} vs median_house_value")
    counter += 1