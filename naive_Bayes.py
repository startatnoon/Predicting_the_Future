#Naive_Bayes

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.cross_validation import KFold

weights = pd.read_csv('./ideal_weight.csv')

weights.columns = weights.columns.map(lambda x: x.rstrip('\'')) #why did it only remove one?
weights.columns = [w[1:] for w in weights.columns]
weights.sex = [a[1:len(a)-1] for a in weights.sex]
