from collections import OrderedDict
import pylogit as pl

long_data_path = u'long_data.csv'
long_df.to_csv(long_data_path, sep=',', index=False)

individual_specific_variables = ["HINC", "PSIZE"]
alternative_specific_variables = ['TTIME', 'INVC', 'INVT', 'GC']

subset_specific_variables = {}
observation_id_column = "OBS_ID"
alternative_id_column = "ALT_ID"
choice_column = "MODE"

alternative_name_dict = {0: "AIR",
						1: "TRAIN",
						2: "BUS",
						3: "CAR"}


wide_df = pl.convert_long_to_wide(long_df, individual_specific_variables, alternative_specific_variables,	
							subset_specific_variables, observation_id_column, alternative_id_column,
							choice_column, alternative_name_dict)


ind_variables = ["HINC", "PSIZE"]
availabilty_variables = {0: 'availabilty_AIR', 1: 'availabilty_TRAIN', 2: 'availabilty_BUS', 3: 'availabilty_CAR'}

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import scipy

data_path = 'wide_data.csv'
raw_data = pd.read_table(data_path, sep=',', header=0)
model_data.info()

model_data = model_data.dropna()
model_data = model_data.fillna(0)

import re
float_pattern = '^(-?\\d+)(\\.\\d+)?$'
float_re = re.compile(float_pattern)
model_data['HINC'].apply(lambda x: 'not_float' if float.re(match(str(x))) == None else 'float')

crosstab = pd.cross_tab(model_data['y'], model_data['PSIZE'])
p = scipy.stats.chi2.contingency(crosstab)[1]
print("p size:", p)

# Build up the Logistic Regression Model:
X = model_data[['HINC', 'PSIZE', 'TIME_TRAIN', 'INVC_CAR']]
y = raw_data['y']

dummies = pd.get_dummies(X['PSIZE'], drop_first = False)
dummies.columns = ['PSIZE' + '_' + str(x) for x in dummies.columns.values]
X = pd.concat([X, dummies], axis = 1)
X = X.drop('PSIZE', axis = 1)
X = X.drop('PSIZE_4', axis = 1)
X = X.drop('PSIZE_5', axis = 1)
X = X.drop('PSIZE_6', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1234)

logistic = sm.Logit(y_train, X_train).fit()
print(logistic.summary2())

