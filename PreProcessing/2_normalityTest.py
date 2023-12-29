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

df["body_mass_g"].hist()
plt.title('Histogram of Body Mass')
#plt.show()

# shapiro test 

out = ss.shapiro(df["body_mass_g"])
#print(out) 

#seperate out gentoos
df_gentoos = df[df["species"]=="Gentoo"]

df_gentoos["body_mass_g"].hist()
plt.title('Histogram of Body Mass')

df_Adelie = df[df["species"]=="Adelie"]

df_Adelie["body_mass_g"].hist()


df_Chinstrap = df[df["species"]=="Chinstrap"]

df_Chinstrap["body_mass_g"].hist()
plt.title('Histogram of Body Mass')
plt.show()