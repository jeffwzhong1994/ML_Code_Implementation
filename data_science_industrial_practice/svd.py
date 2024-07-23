import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as la
from PIL import Image
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

raw_data = raw_data.fillna(0)

raw_data = raw_data[raw_data['Quantity'] >= 0]
raw_data = raw_data[~raw_data['StockCode'].isnull()]
raw_data['year_week'] = (raw_data['InvoiceDate'].dt.year) * 100 + raw_data['InvoiceDate'].dt.week
raw_data = raw_data[~((raw_data['year_week'] == 201048) | (raw_data['year_week'] == 201049))]
data = raw_data.groupby(['year_week', 'StockCode']).agg({'Quantity': 'sum'}).reset_index()
data = data.pivot_table(index=['StockCode'], columns=['year_week'], values=['Quantity'])

data.columns = data.columns.droplevel(0)
data = data.reset_index().fillna(0)

data = data.sample(frac = 1)
x = data.iloc[:,1:].values
u,s,v = np.linalg.svd(x, full_matrices=False)

cum_var = []
for i in range(len(s)):
	if i == 0:
		cum_var.append(s[i] ** 2)
	else:
		cum_var.append(cum_var[-1] + s[i] ** 2)

cum_var_percentage = (cum_var / cum_var[-1])
plt.plot(range(len(cum_var_percentage)), cum_var_percentage, 'bx-')
min_explantory_ratio = 0.9

feature_nums = 0
for i in cum_var_percentage:
	if i < min_explantory_ratio:
		feature_nums += 1
print('feature_nums:', feature_nums)

#Calculate V matrix's correlation coefficient:
corr = [[] for i in range(feature_nums)]
t = data['StockCode']
for i in range(len(x)):
	for j in range(feature_nums):
		corr[j].append(np.corrcoef(x[i], v[j,:])[0][1])

for i in range(feature_nums):
	t = pd.concat([t, pd.Series(corr[i], index=data.index)], axis=1)
t.columns = ['StockCode'] + ['v{}'.format(i) for i in range(feature_nums)]


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib as mpl

x_cluster = t.iloc[:, 1:].values
k = range(1,10)
meandistortions = []
for k in K:
	kmeans=KMeans(n_clusters=k)
	kmeans.fit(x_cluster)

meandistortions.append(sum(np.min(cdist(x_cluster, kmeans.cluster_centers_, "euclidean"), axis=1)) / x_cluster.shape[0])
plt.plot(K.meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('meandistortions')
plt.title('figture_k')
plt.show()

#K-means Clusters:
class_number = 4
x_cluster[np.isnan(x_cluster)] = 0
kmeans = KMeans(n_clusters = class_number, random_state=100).fit(x_cluster)
kmeans_label = kmeans.labels_

data_res = pd.concat([data[['StockCode']], pd.Series(kmeans_label, index=data.index)], axis=1)
data_res.columns = ['StockCode','group_id']

idx_sort = np.argsort(data_res.group_id)
group_nums = data_res[['StockCode', 'group_id']].groupby(['group_id']).count().cumsum()

vmax=50
plt.figure(figsize=(24,5))
plt.subplot(131)
cmap = mpl.cm.viridis
im = plt.imshow(x[:,:], aspect='auto',cmap=cmap, vmax=vmax)
plt.xlim(0,data.shape[1])
plt.colorbar(im,cmap=cmap)
plt.title('Original image')

