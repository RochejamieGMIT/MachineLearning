import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

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

# Ask classifier to predict outputs for training set inputs.

# count values of true
eval = (clf.predict(X) == Y).sum()

# total number of cases - first column
count = X.shape[0]

outcome = (eval/count)*100 

print(outcome)


# keep some samples for testing
X_train,X_test,y_train,y_test = train_test_split(X,Y,random_state=35)

clf.fit(X_train,y_train)

# predict based on test test. 
#proportion of correct classification
print((clf.predict(X_test) == y_test).sum() / X_test.shape[0])


#Run cross validation with 5 folds 
print(cross_val_score(clf,X,Y,cv=5))


