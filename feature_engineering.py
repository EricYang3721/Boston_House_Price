# import libraries and dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats

train = pd.read_csv('train.csv').drop('Id', axis=1)
test = pd.read_csv('test.csv').drop('Id', axis=1)

all_data = pd.concat([train.iloc[:,:80], test.iloc[:, :80]])
y_train = train['SalePrice']

# 1 Processing label
print("There are {} NAs in the label '{}'".format(y_train.isnull().sum(), 'SalePrice'))
print("The skewness of the label '{0:s}' is: {1:0.5f}".
      format('SalePrice', y_train.skew()))
print("Get the log price")
y_train = np.log(y_train)
print("The skewness of the logged label '{0:s}' is: {1:0.5f}".
     format('logSalePrice', y_train.skew()))
# The distributio of logSalePrice is 
sns.distplot(y_train, axlabel='logSalePrice')

# 2 Feature processing
# 2.1 Correlation matrix of the features and label
train_cormat = train.corr() 
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(train_cormat, vmax=0.8, square=True)
f.tight_layout()
# Correlation matrix only evaluate numeric variables
cols = train_cormat.nlargest(10, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
cm2 = train[cols].corr()
# Heat map of features having top 10 correlations with SalePrice
cor_sort=train_cormat.sort_values(by='SalePrice', ascending = False).index
cm_max = train[cor_sort[0:10]].corr()
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(cm_max, vmax=0.8, square=True, annot=True,fmt='.2f',
            annot_kws={'size':10})
f.tight_layout()

# Bar plot of sorted correlation
f, ax = plt.subplots(figsize=(8, 12))
sorted_train_cormat = train_cormat.sort_values(by='SalePrice', ascending = False)
sns.barplot(y='index', x='SalePrice', data=sorted_train_cormat.reset_index())
f.tight_layout()

# Pairplot of the top 10 correlated features
sns.set()
f, ax = plt.subplots(figsize=(15, 15))
sns.pairplot(data=train[cor_sort[0:7]], size=2.5)
plt.show()


