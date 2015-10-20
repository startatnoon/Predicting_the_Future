#cross_validation.py
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.cross_validation import KFold

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

loansData['Interest.Rate'] = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%')) / 100, 4))
cleanLoanLength = loansData['Loan.Length'].map(lambda x: int(x.rstrip(' months')))
cleanFICO = loansData['FICO.Range'].map(lambda x: int(x[0:3]))
loansData['FICO.Score'] = cleanFICO


intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']


kf = KFold(len(loansData),n_folds=10, shuffle=True)

output = []
for train, test in kf:
	y = np.matrix(intrate[test]).transpose()
	x1 = np.matrix(fico[train]).transpose()
	x2 = np.matrix(loanamt[train]).transpose()
	x = np.column_stack([x1,x2])
	X = sm.add_constant(x)
	model = sm.OLS(y,X)
	f = model.fit()

print output