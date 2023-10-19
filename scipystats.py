# Laerd Stats - Chi Square test

import pandas as pd
import scipy.stats as ss
import random

male_books = [['Male', 'Books']] * 16

female_books = [['Female', 'Books']] * 13

male_online = [['Male', 'Online']] * 24

female_online = [['Female', 'Online']] * 27

# raw data join 4 lists

raw_date = male_books + male_online + female_books + female_online

random.shuffle(raw_date)

# zip - turns data rows in to columns and columns in to rows to make data processing easier 
# interchanges outer and inner lists

gender, medium =list(zip(*raw_date))

# create data frame

df = pd.DataFrame({'gender': gender, 'medium': medium})

#show
#print(df)

# preform cross tab contingency
ct = ss.contingency.crosstab(df['gender'],df['medium'])

first,second = ct.elements
#print(first)
#print(second)
#print(ct.count)

# df where all gender = first gender 
df[df['gender']==first[0]]

# df where gender = first gender and medium = first medium 
#print(df[df['gender']==first[0]][df['medium']==second[0]])

# Doing stats bah

results = ss.chi2_contingency(ct.count, correction=False)

print(results.expected_freq)

# pref books irrespective of gender 


# if no relationship between gender and medium then, we should has sam proportion of males liking books as we have in the data set (overall)

print (40 * (29/80)) # this proportion of people prefer books, gender does not matter, 40 is number of people 29/80 is the proportion 
print (40 * (51/80)) # this proportion of people prefer online, gender does not matter