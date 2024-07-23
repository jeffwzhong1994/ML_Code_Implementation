"""
instant: Id
dteday: date
season: 1(spring), 2(summer), 3(autumn), 4(winter)
yr: year
mnth: month
hr
holiday: 1,0
weekday: 0~6
weathersit: weather situation
temp
atemp
hum
windspeed
casual
registered
cnt
"""

#Begin data mining: Use XGBoost as Blackbox model
%matplotlib inline
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC, SVR, SVC
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals.six import StringIO
from sklearn import tree
import pydotplus
from IPython.core.display import Image
from sklearn import metrics
from xgboost.sklearn import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt

data = pd.read_csv('day.csv')
data.head()

#Onehot encoding:
data_xgb = data
data_xgb = data[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday',
				'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'cnt']].astype(object)

data_xgb_use = pd.get_dummies(data_xgb)
data_xgb_use.head()

#Split train/test data:
X = data_xgb_use.drop('cnt', axis=1)
y = data_xgb_use[['cnt']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

#Train XGBoost model:
clf = XGBRegressor(n_estimators=100, max_depth=4, subsample=0.9, learning_rate=0.3)
clf.fit(X_train, y_train)
y_test_pre = clf.predict(X_test)
print("r-square: %f" % metrics.r2_score(y_test, y_test_pre))
MAPE = np.mean(abs(y_test.reset_index(drop=True) - pd.DataFrame(y_test_pre, columns=['cnt'])) / y_test.reset_index(drop=True))

#Build decision tre model:
data_xgb_use['cnt_pre_xgb'] = clf.predict(X)
data_white_tree = data_xgb_use[['temp', 'atemp', 'hum', 'windspeed',
			'season_1', 'season_2', 'season_3', 'season_4', 'yr_0', 'yr_1',
			'mnth_1', 'mnth_2', 'mnth_3', 'mnth_4', 'mnth_5', 'mnth_6', 'mnth_7',
			'mnth_8', 'mnth_9', 'mnth_10', 'mnth_11', 'mnth_12', 'holiday_0', 
			'holiday_1', 'weekday_0', 'weekday_1', 'weekday_2', 'weekday_3', 
			'weekday_4', 'weekday_5', 'weekday_6', 'workingday_0', 'workingday_1',
			'weathersit_1', 'weathersit_2', 'weathersit_3', 'cnt', 'cnt_pre_xgb']]

X_white_tree = data_white_tree.drop(['cnt', 'cnt_pre_xgb'], axis = 1)
y_white_tree = data_white_tree[['cnt_pre_xgb']]
X_white_tree = data_white_tree[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday',
				'weathersit', 'temp', 'atemp', 'hum', 'windspeed']]
y_white_tree = data_white_tree[['cnt_pre_xgb']]

X_train, X_test, y_train, y_test = train_test_split(X_white_tree, y_white_tree, test_size=0.2, random_state=101)
white_tree = DecisionTreeRegressor(max_depth=3)
white_tree.fit(X_train, y_train)

# Visualize it:
dot_data = StringIO()
tree.export_graphviz(white_tree, out_file=dot_data,
			feature_names=X_white_tree.columns, filled=True, rounded=True,
			proportion=True, special_characters=True, node_ids=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
