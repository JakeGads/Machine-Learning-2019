import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'E:\CS400 Fall 2019\titanic.csv',header=0)
dataset = dataset.fillna(0)
dataset.columns

# creating a dict file  
dataset.sex.unique()
#converting categorical data
sex = {'male': 1,'female': 2} 

dataset.embarked.unique()
embarked = {'S': 1,'C': 2,'Q': 3, 0: 0} 

dataset.who.unique()
who = {'man': 1,'woman': 2,'child': 3} 

dataset.deck.unique()
deck = {'C': 1,'E': 2,'G': 3,'D': 4,'A': 5,'B': 6,'F': 7, 0: 0} 

dataset.embark_town.unique()
embark_town = {'Southampton': 1,'Cherbourg': 2,'Queenstown': 3, 0: 0} 

dataset.alive.unique()
alive = {'no': 1,'yes': 2} 

classV = {'First': 1,'Second': 2, 'Third': 3} 

dataset.sex = [sex[item] for item in dataset.sex] 
dataset.embarked = [embarked[item] for item in dataset.embarked] 
dataset.who = [who[item] for item in dataset.who] 
dataset.deck = [deck[item] for item in dataset.deck] 
dataset.embark_town = [embark_town[item] for item in dataset.embark_town] 
dataset.alive = [alive[item] for item in dataset.alive] 
dataset.classV = [classV[item] for item in dataset.classV] 