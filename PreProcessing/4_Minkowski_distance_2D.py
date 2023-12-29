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
import seaborn as sns

#read url https://bobbyhadz.com/blog/read-csv-file-from-url-using-python
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"

df = pd.read_csv(url,sep=',',encoding='utf-8',)

df = df.dropna()

x = df[["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]]

#sns.pairplot(df,hue = "species")
#plt.show()
df_gentoos = df[df["species"]=="Gentoo"]
fig, ax = plt.subplots()



df_gentoos[df_gentoos["flipper_length_mm"] == 208]

ax.plot(df_gentoos["bill_length_mm"],df_gentoos["flipper_length_mm"],"k.")
ax.plot(df_gentoos.loc[259]["bill_length_mm"],df_gentoos.loc[259]["flipper_length_mm"],"bx")
ax.plot(df_gentoos.loc[327]["bill_length_mm"],df_gentoos.loc[327]["flipper_length_mm"],"yx")


ax.hlines(df_gentoos.loc[259]["flipper_length_mm"],df_gentoos.loc[259]["bill_length_mm"],df_gentoos.loc[327]["bill_length_mm"])
ax.vlines(df_gentoos.loc[327]["bill_length_mm"],df_gentoos.loc[327]["flipper_length_mm"],df_gentoos.loc[259]["flipper_length_mm"])
plt.show()