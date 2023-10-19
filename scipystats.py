# Laerd Stats - Chi Square test

import pandas as pd
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
print(df)


