"""
Day 25 Activity: Mini-Project (Feature Engineering)
Tasks:
1) Load dataset
2) Apply domain features, interactions, and transformations
3) Apply feature selection
4) Save engineered dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# TODO: Load data from data/day25_project.csv
df25 = pd.read_csv("data2/day25_project.csv")

# TODO: Build engineered feature set
df25["price_per_rooms"]=df25["price"]/df25["rooms"].replace(0,np.nan)
df25["price_per_sqft"]=df25["price"]/df25["sqft"].replace(0,np.nan)


df25["price_*_rooms"]=df25["price"] * df25["rooms"]
df25["price_+_sqft"]=df25["price"] + df25["sqft"]

df25["log_price"]=np.log(df25["price"])
df25["log_sqft"]=np.log(df25["sqft"])



print(df25)




# TODO: Save engineered dataset to data/day25_engineered.csv
