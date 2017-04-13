# scale num features
# Boston house price
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Import dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
all_data = pd.concat([train_data.loc[:,'MSSubClass':'SaleCondition'],
                     test_data.loc[:,'MSSubClass':'SaleCondition']])

# Convert the SalePrice into log unit
train_data.iloc[:,-1] = np.log(train_data.iloc[:,-1])

num_feature = all_data.columns[all_data.dtypes != 'object']
num_feature_all = all_data[num_feature]
num_feature_all = num_feature_all.fillna(num_feature_all.mean())

X_train = num_feature_all.iloc[0:train_data.shape[0],:].values
X_test = num_feature_all.iloc[train_data.shape[0]:,:].values
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)

cat_feature = all_data.columns[all_data.dtypes == 'object']
cat_feature_all = all_data[cat_feature]
cat_feature_all = pd.get_dummies(cat_feature_all)
X_train_cat = cat_feature_all.iloc[0:train_data.shape[0],:].values

train_data_ready = np.concatenate((X_train, X_train_cat), axis=1)
y_train = train_data.iloc[:,-1].values

# 1. setup LinearRegression models and do cross validations
#  1.1 before feature scaling on numeric features
linregressor = LinearRegression()
scores = np.sqrt(-1*cross_val_score(estimator=linregressor, X = train_data_ready, 
                                    y=y_train, 
                                    scoring='neg_mean_squared_error', cv=10))
print("RSME of Linear regression is {0:5.5f}".format(scores.mean()))