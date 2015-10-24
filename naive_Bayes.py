#Naive_Bayes

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

weights = pd.read_csv('./ideal_weight.csv')

weights.columns = weights.columns.map(lambda x: x.rstrip('\'')) #why did it only remove one?
weights.columns = [w[1:] for w in weights.columns]
weights.sex = [a[1:len(a)-1] for a in weights.sex]

plt.figure()
plt.hist(weights.actual, width = 5, label = 'Actual')
plt.hist(weights.ideal, width = 5, label = 'Ideal')
plt.legend()
plt.show()

plt.figure()
plt.hist(weights['diff'], width = 5)
plt.show()

pd.Categorical(weights.sex).labels
men = sum(pd.Categorical(weights.sex).labels)
women = len(weights.sex) - men
if men > women:
	print "More men"
else:
	print "More women"

#using sex to predict actual, ideal, and diff
g = GaussianNB()
X = weights[['actual','ideal','diff']]
y = pd.Categorical(weights.sex).labels
g.fit(X,y)

x_pred = [[145,160,-15], [160,145,15]]
y_pred = g.predict(x_pred)
for b in y_pred:
	if b == 0:
		print "Predict woman"
	else:
		print "Predict man"
