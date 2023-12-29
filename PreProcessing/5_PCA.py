import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
import random
import sklearn as sk 
from sklearn.preprocessing import StandardScaler
import sklearn.decomposition as dec
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.inspection import DecisionBoundaryDisplay
import seaborn as sns

# new pca 
pca = dec.PCA(n_components=2)

# fit data 
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"

df = pd.read_csv(url,sep=',',encoding='utf-8',)

df = df.dropna()

x = df[["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]]

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

TDF = pd.DataFrame(x_transform, columns=["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"])

pca.fit(TDF)
#print(pca.explained_variance_ratio_)
x_pca = pca.transform(TDF)

#fig,ax = plt.subplots()
#ax.plot(x_pca[:,0],x_pca[:,1],"k.")
#plt.show()



df_pca = pd.DataFrame(df[["species","sex"]])

#include oca variables
df_pca["pca0"] = x_pca[:,0]
df_pca["pca1"] = x_pca[:,1]

print(df_pca)

print(pca.explained_variance_ratio_)

#sns.pairplot(df_pca, hue='species')
#plt.show()

# before scaling

# new instance if classifer 
clf = KNeighborsClassifier()

y = df["species"].to_numpy()

print(cross_val_score(clf,x,y,cv=5))
#fit the data
fitted = clf.fit(x,y)
eval = (clf.predict(x) == y).sum()

# new instance if classifer 

#fit the data
#fitted = clf.fit(df_pca,df["species"])

clf = KNeighborsClassifier()
X = TDF
Y  = df["species"].to_numpy()

print(cross_val_score(clf,X,Y,cv=5))
#fit the data
fitted = clf.fit(X,Y)
eval = (clf.predict(X) == Y).sum()

# new instance if classifer 
