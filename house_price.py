# Boston house price
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Import dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
all_data = pd.concat([train_data.loc[:,'MSSubClass':'SaleCondition'],
                     test_data.loc[:,'MSSubClass':'SaleCondition']])

# Convert the SalePrice into log unit
train_data.iloc[:,-1] = np.log(train_data.iloc[:,-1])

# Check the data with plots
#distribution of 'SalePrice' in train_data
sns.distplot(train_data['SalePrice'], bins=30, kde=False)
 
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())

X_train = all_data.iloc[0:train_data.shape[0],:].values
X_test = all_data.iloc[train_data.shape[0]:,:].values
y_train = train_data.iloc[:,-1].values

# 1. Model comparision
#  1.1 Linear Regression
from sklearn.linear_model import LinearRegression
linregressor = LinearRegression()
scores = np.sqrt(-1*cross_val_score(estimator=linregressor, X = X_train, 
                                    y=y_train, 
                                    scoring='neg_mean_squared_error', cv=10))
print("RMSE of Linear regression is {0:5.5f}".format(scores.mean()))
# RMSE for Linear Regression is 0.15338

#  1.2 Ridge regression
from sklearn.linear_model import Ridge
alphas = list(range(30))[1:]
rge_scores = []
for item in alphas:
    rge_regressor = Ridge(alpha=item)
    rge_scores.append(np.sqrt(-1*cross_val_score(estimator=rge_regressor, X = X_train, 
                                    y=y_train, 
                                    scoring='neg_mean_squared_error', cv=10)).mean())
plt.plot(alphas, rge_scores)
plt.xlabel('Alpha')
plt.ylabel('RMSE')
plt.title('Ridge Regression RMSE')
plt.show()
min(rge_scores)
# Lowest RMSE for Ridge Regression is 0.13666

#  1.3 Lasso Regression
from sklearn.linear_model import Lasso
alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]
lasso_scores = []
for item in alphas:
    lasso_regressor = Lasso(alpha=item)
    lasso_scores.append(np.sqrt(-1*cross_val_score(estimator=lasso_regressor, X = X_train, 
                                    y=y_train, 
                                    scoring='neg_mean_squared_error', cv=10)).mean())
plt.plot(alphas, lasso_scores)
plt.xlabel('Alpha')
plt.ylabel('RMSE')
plt.title('Lasso Regression RMSE')
plt.show()
min(lasso_scores)
# Lowest RMSE for Ridge Regression is 0.13378

#  1.3 Supported Vector Rgression
from sklearn.svm import SVR
Cs = [0.0001, 0.001, 0.01, 0.1, 1, 10, 50]
svr_scores =[]
for item in Cs:
    svr_regressor = SVR(C=item)
    svr_scores.append(np.sqrt(-1*cross_val_score(estimator=svr_regressor, X = X_train, 
                                    y=y_train, 
                                    scoring='neg_mean_squared_error', cv=10)).mean())
plt.plot(Cs, svr_scores)
plt.xlabel('C')
plt.ylabel('RMSE')
plt.title('SVR Regression RMSE')
plt.show()
min(svr_scores)
# Bad performance 

# 1.4 Random forest regression
from sklearn.ensemble import RandomForestRegressor
rfg_scores =[]
rfg_regressor = RandomForestRegressor(n_estimators=10, random_state=0)
rfg_scores.append(np.sqrt(-1*cross_val_score(estimator=rfg_regressor, X = X_train, 
                                    y=y_train, 
                                    scoring='neg_mean_squared_error', cv=10)).mean())
# Lowest RMSE for Ridge Regression is 0.13378

# 1.5 KNearestNeighbors Regression 
from sklearn.neighbors import KNeighborsRegressor
neighbors = [1,2,3,4,5,6,7,8,9,10]
knn_scores=[]
for item in neighbors:
    knn_regressor = KNeighborsRegressor(n_neighbors=item)
    knn_scores.append(np.sqrt(-1*cross_val_score(estimator=knn_regressor, X = X_train, 
                                    y=y_train, 
                                    scoring='neg_mean_squared_error', cv=10)).mean())
plt.plot(neighbors, knn_scores)
plt.xlabel('C')
plt.ylabel('RMSE')
plt.title('KNN Regression RMSE')
plt.show()
min(knn_scores)
# Bad performance


#2 Backward elimination of Linear Regression for feature selection
