import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
import random
import sklearn as sk 
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.inspection import DecisionBoundaryDisplay

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"

df = pd.read_csv(url,sep=',',encoding='utf-8',)

df = df.dropna()

scaler = StandardScaler()
x = df[["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]]
scaler.fit(x)

#print(scaler.mean_) 
#print(scaler.var_) 

x_transform = scaler.transform(x)
#print(x_transform)

out = x_transform.mean(axis=0)
print(out)

#recreat dataframe from transformed data

TDF = pd.DataFrame(x_transform, columns=x.columns)
print(TDF)