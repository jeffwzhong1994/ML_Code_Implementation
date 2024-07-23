import numpy as np
import pandas as pd

OnlRt = pd.read_csv('OnlineRetail.csv', usecols = ['CustomerID', 'InvoiceDate', 'UnitPrice', 'Quantity', 'Country'],
				encoding='ISO-8859-1', parse_dates=['InvoiceDate'], dtypes=
				{'CustomerID': np.str, 'UnitPrice':np.float32, 'Quantity':np.int32, 'Country': np.str})

OnlRt.head()

neg_id = OnlRt[(OnlRt['Quantity'] <= 0) | (OnlRt['UnitPrice'] <=0)].loc[:, 'CustomerID']
data0 = OnlRt[(OnlRt['CustomerID'].notnull()) &
			(~OnlRt['CustomerID'].isin(neg_id)) &
			(OnlRt['Country'] == 'United Kingdom')].drop('Country', axis = 1)

data1 = data0.assign(amount=data0['UnitPrice'].multiply(data0['Quantity']))

first_time = data1['InvoiceDate'].sort_values(ascending=True).groupby(data1['CustomerID']).nth(0)\
			.apply(lambda x: x.date()).reset_index().rename(columns={'InvoiceDate': 'first_time'})
data2 = pd.merge(data1, first_time, how='left', on=['CustomerID'])

dayth = (data2['InvoiceDate'].apply(lambda x: x.date()) - data2['first_time']).apply(lambda x: x.days)
 
month = data2['InvoiceDate'].apply(lambda x: x.month)
weekday = data2['InvoiceDate'].apply(lambda x: x.weekday())
hour = data2['InvoiceDate'].apply(lambda x: x.hour)
minute = data2['InvoiceDate'].apply(lambda x: x.minute)
second = data2['InvoiceDate'].apply(lambda x: x.second)
hour_preci = (second/60+minute)/60+hour

data3 = data2.assign(dayth=dayth).assign(hour=hour_preci).\	
		assign(weekday=weekday).drop(['first_time', 'InvoiceDate'], axis=1)\
		.sort_values(by=['CustomerID', 'dayth', 'hour'])

X = data3[data3['dayth']<28].set_index('CustomerID').drop('amount',axis=1).sort_index()
data180=data3[(data3['dayth']<180)&(data3['CustomerID'].isin(X.index))]
y = data180['amount'].groupby(data180['CustomerID']).sum().sort_index()
X.to_csv('bookdata_X.csv',index=True,header=True)
y.to_csv('bookdata_y.csv',index=True,header=True)

###
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from Keras.layers import Input, Conv1D, Dropout, LSTM, TimeDistributed, Bidirectional, Dense
from Keras.models import Model
from Keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

columns_picked = ['CustomerID','Quantity','UnitPrice','dayth','hour','weekday']
y = pd.read_csv('bookdata_y.csv').rename(columns={'CustomerID':'id'}).set_index('id')['amount']
X=pd.read_csv('bookdata_X.csv', usecols=columns_picked).rename(columns={'CustomerID': 'id'}).set_index('id')
columns_picked.remove('CustomerID')

indices = y.index.tolist()
ind_train, int_test = map(sorted, train_test_split(indices, test_size=0.25, random_state=42))
X_train = X.loc[ind_train,:]
y_train = y[ind_train]

X_test = X.loc[ind_test,:]
y_test = y[ind_test]

scaler = MinMaxScaler(feature_range=(-0.5,0.5))
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train),
							columns=X_train.columns,
							index=X_train.index)


week_train=X_train['dayth'].apply(lambda x: int(x/7)).rename('week')
inner_length=32
outer_length=4
feature_len=len(columns_picked)

def cut_pad(x,maxl):
	head=np.array(x)[0:maxl]
	head_padding=head if len(head)==maxl else np.pad(head,(0, maxl-len(head)),mode='constant')
	return head_padding

#Feature Array Engineering:
def feature_array(df, col_n, week, len_outer, len_inner):
	col =df[[col_n]].assign(week=week).reset_index()

	ids=col['id'].drop_duplicates().values.tolist()
	weeks=np.arange(0, len_outer).tolist()

	id_week=pd.DataFrame([(id, week) for id in ids for week in weeks]).rename(columns={0:'id',1:'week'}).sort_values(by=['id','week'])

	arr_base = pd.merge(id_week, col, how='left', on=['id','week']).fillna(0)
	arr_frame = arr_base[col_n].groupby([arr_base['id'], arr_base['week']]).\
				apply(lambda x: cut_pad(x, len_inner)).reset_index().drop('week',axis=1).set_index('id')[col_n]
	userarray=arr_frame.groupby(arr_frame.index).apply(np.vstack).\
			apply(lambda x: x.reshape([1,x.shape[0],x.shape[1]])).sort_index()
	userarray_var = np.vstack(userarray.values.tolist())

	return userarray.index.tolist(), userarray_var

def make_data_array(df, columns, week, len_outer, len_inner):
	ids_num = len(set(df.index))

	df_ready = np.zeros([ids_num, len_outer, len_inner, len(columns)])
	for i, item in enumerate(columns):
		the_ind, df_ready[:,:,:,i] = feature_array(df, item, week, len_outer, len_inner)

	return the_ind, df_ready

X_train_ind, X_train_data = make_data_array(X_train_scaled, columns_picked, week_train, outer_length, inner_length)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
y_test_log = y_test.apply(np.log)
week_test=X_test['dayth'].apply(lambda x: int(x/7)).rename('week')
X_test_ind, X_test_data= make_data_array(X_test_scaled, columns_picked, week_test, outer_length, inner_length)


def build_model(len_outer, len_inner, len_fea):
	filters = [64,32]
	kernel_size = [2,2]
	dropout_rate = [0.1,0]

	inner_input = Input(shape=(len_inner, len_fea), dtype='float32')
	cnn1d=inner_input

	for i in range(len(filters)):
		cnn1d = Conv1D(filters=filters[i],
					kernel_size=kernel_size[i],
					padding='valid',
					activation='relu',
					strides=1)(cnn1d)
		cnn1d = Dropout(dropout_rate[i])(cnn1d)

	lstm = LSTM(32, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)(cnn1d)
	inner_output = lSTM(16, return_sequences=False)(lstm)
	inner_model = Model(inputs=inner_input, outputs=inner_output)

	outer_input = Input(shape=(len_outer, len_inner, len_fea), dtype='float32')
	innered = TimeDistributed(inner_model)(outer_input)
	outered = Bidirectional(LSTM(16, return_sequences=False))(innered)
	outered = Dense(8, activation='relu')(outered)
	outer_output = Dense(1)(outered)

	model = Model(inputs=outer_input, outputs=outer_output)
	model.compile(loss='mape', optimizer='adam')

	return model, inner_model


LTV_model, LTV_inner_model = build_model(outer_length, inner_length, feature_len)
LTV_model.summary()
LTV_inner_model.summary()

cb = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
history = LTV_model.fit(x=X_train_data,
						y=y_train_log,
						validation_data=(X_test_data, y_test_log),
						epochs=200,
						batch_size=128,
						callbacks=[cb],
						verbose=2)

LTV_model.save('LTV_model.h5')



