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

# Extract only numeric features
num_feature = all_data.columns[all_data.dtypes != 'object']
num_feature_all = all_data[num_feature]
num_feature_all = num_feature_all.fillna(num_feature_all.mean())

# Splitting train and test set
X_train = num_feature_all.iloc[0:train_data.shape[0],:].values
X_test = num_feature_all.iloc[train_data.shape[0]:,:].values
y_train = train_data.iloc[:,-1].values
                         
# Fit the model and cross validation after scaling the feature
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
linregressor = LinearRegression()
scores = np.sqrt(-1*cross_val_score(estimator=linregressor, X = X_train, 
                                    y=y_train, 
                                    scoring='neg_mean_squared_error', cv=10))
print("RSME of Linear regression is {0:5.5f}".format(scores.mean()))

