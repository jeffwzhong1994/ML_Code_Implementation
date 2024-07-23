#Data columns
"""
Publish_period -> time for sale
is_sold -> (0,1)
Distance_travelled -> mileage
Color -> of the car
N_photos -> photos available
N_inquries -> number of inquries
Price

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import NelsonAalenFitter
from lifelines import CoxPHFitter

data_survival = pd.read_csv('input_survival_v2.csv', sep=',', encoding='GBK', index_col=False)
data_survival = data_survival[(data_survival['Publish_period'] <= 90)].reset_index(drop=True)
data_survival = data_survival.drop(['Departure_Date', 'End_Date', 'Car_id'], axis = 1)
data_survival.head()

#deal with discrete variables:
df = pd.get_dummies(data_survival, drop_first=True, columns=['Color'])
df.head()

kmf = KaplanMeierFitter()
T = df['Publish_period']
E = df['is_sold']
kmf.fit(T, event_observed = E, label= 'Survival Curve')
kmf.plot()
plt.xlim()
plt.ylabel("est. probability of survival $\hat{S}(t)$")
plt.xlabel("day $t$")

# Then plot subplots of survival curve:
avg_inquries = np.mean(df['N_Inquiries'])
df_less_inquires = df[(df['N_Inquiries'] < avg_inquries)]
df_more_inquires = df[(df['N_Inquiries'] > avg_inquries)]

ax = plt.subplot(111)
kmf = KaplanMeierFitter()

kmf.fit(df_less_inquires['Publish_period'], event_observed = df_less_inquires['is_sold'], label="less_inquires")
ax = kmf.plot(ax=ax)

kmf.fit(df_more_inquires['Publish_period'], event_observed = df_more_inquires['is_sold'], label="more_inquires")
ax = kmf.plot(ax=ax)

plt.title('survival curves of two different types')

#do they have significant differences?
results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
results.print_summary() # P-value is less than 0.05

# Feature Explaination:
cph = CoxPHFitter()
cph.fit(df, duration_col = 'Publish_period', event_col = 'is_sold', show_progress=True, step_size=0.5)
cph.print_summary()
cph.plot()

X = df.drop(['Publish_period', 'is_sold'], axis = 1)
surv_hat = cph.predict_survival_function(X)
surv_hat[24].plot(label='24')
surv_hat[11166].plot(label='11166')
plt.legend()

# Finding the best pricing strategies using Predictive Life Curves:
def predict_day(surv_hat):
	days = np.zeros(surv_hat.shape[1])
	prob = np.zeros(surv_hat.shape[1])
	j = surv_hat.shape[1]
	for i in range(1, surv_hat.shape[1]):
		prob[i-1] = surv_hat[surv_hat[i-1] >= 0.5][i-1].min()
		prob[j-1] = surv_hat[surv_hat[j-1] >= 0.5][j-1].min()
		days[i-1] = surv_hat[surv_hat[i-1] == prob[i-1]].index.values.min()
		days[j-1] = surv_hat[surv_hat[j-1] == prob[j-1]].index.values.min()

	return prob, days

# 2nd step:
def is_sold(data):
	y = np.zeros(data.shape[0])
	for i in range(1, data.shape[0]):
		if data[i-1] >= 0.6:
			y[i-1] = 0
		else:
			y[i-1] = 1
	return y

# 3rd step:
def profit(data, predict_days, sold_tag):
	d = list(predict_days)
	y = list(sold_tag)
	revenue = np.sum(data['Price'] * y)
	cost = np.sum(1000*d)
	profit = revenue - cost
	return profit

# 4th step:
min_price = df['Price'].values.min()
max_price = df['Price'].values.max()

sp_price = np.linspace(min_price, max_price, 10)
X = df.drop(['Publish_period', 'is_sold'], axis = 1)

profit_list = []
price_list = list(sp_price)

for p in price_list:
	X['Price'] = p
	surv_hat = cph.predict_survival_function(X)
	prob_result, day_result = predict_day(surv_hat)
	sold_result = is_sold(prob_result)
	profit_result = profit(X, day_result, sold_result)
	profit_list.append(profit_result)

profit_res = pd.DataFrame({'price': price_list, 'profit': profit_list})
