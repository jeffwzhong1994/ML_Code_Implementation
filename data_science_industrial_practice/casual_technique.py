import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

df = pd.read_csv('basque.csv', sep=',', header=0)
data = df[((df.regionname == 'Baseque Counry') | (df.regionname== 'Argon'))]
def is_test_region(x):
	if x == 'Baseque Counry':
		return 1
	else:
		return 0

def is_test_year(x):
	if x >= 1975:
		return 1
	else:
		return 0

data['is_test_region'] = data['regionname'].apply(lambda x: is_test_region(x))
data['is_test_year'] = data['year'].apply(lambda x: is_test_year(x))
mod_fit = smf.ols(formula="gdpcap ~ year+ is_test_region + is_test_year + is_test_region:is_test_year", data=data).fit()
mod_fit.summary()

