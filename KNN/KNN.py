import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"

df = pd.read_csv(url,sep=',',encoding='utf-8',)
df_NoNA = df.dropna()

print(df)

# new instance if classifer 
clf = KNeighborsClassifier()
X = df_NoNA[['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']]
print(X)
Y=df_NoNA['sex']
print(Y)

#fit the data
fitted = clf.fit(X,Y)

print(df_NoNA.iloc[0])

X.iloc[0]

# quick check classifer preficts right
print(clf.predict(X.iloc[:1]))

