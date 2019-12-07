"""
Dataset: Commodity1_price.csv, Commodity2_price.csv, Commodity3_price.csv
Dataset description: It is a list of prices of a perishable, limited consumption good or commodity reported in
    markets of a country.
a. Date: It’s the date commodity was reported in the respective market.
b. Market: Market in which commodity was reported.
c. State: State in which the corresponding market is situated.
d. Variety: Variety of commodity reported.
e. Grade: Grade of commodity reported.
f. Tonnage (Arrival): Tonnage of a crop that arrives at the market
g. Prices: MinimumPrice, ModalPrice, and MaximumPrice columns are the corresponding prices of commodity for the
    date-state-market-variety-grade combination.

Problem description: We have prices available reported for commodity in different state and markets of the country.
Our objective is to forecast the minimum and maximum price of a commodity for a given state, market, variety, and grade.
The dataset includes three csv files for three different commodities. Find which model best fits the forecasting of the
price for each commodity. Be mindful of overfitting and underfitting
"""