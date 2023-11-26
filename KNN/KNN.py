import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
import random

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"

df = pd.read_csv(url,sep=',',encoding='utf-8',)

