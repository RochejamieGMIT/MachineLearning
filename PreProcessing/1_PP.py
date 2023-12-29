import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
import random
import sklearn as sk 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.inspection import DecisionBoundaryDisplay

#read url https://bobbyhadz.com/blog/read-csv-file-from-url-using-python
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"

df = pd.read_csv(url,sep=',',encoding='utf-8',)

df = df.dropna()

x = df[["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]]

#print(x)

# eucliean distance 
#Step 1 - subtract 
out = (x.loc[0] - x.loc[342])

#step 2 - square
out = out**2

#3 - sum 
out = out.sum()

# 4 - square root
out = out**.5

# convert body mass to kg
df["body_mass_kg"] = df["body_mass_g"]/1000
#print(df)

x = df[["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_kg"]]

#print(x)

# eucliean distance 
#Step 1 - subtract 
out = (x.loc[0] - x.loc[342])

#step 2 - square
out = out**2

#3 - sum 
out = out.sum()

# 4 - square root
out = out**.5

print(out)