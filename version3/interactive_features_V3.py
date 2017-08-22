# modeing with interactive features

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
import xgboost as xgb
warnings.filterwarnings('ignore')

# import data
X_train = pd.read_csv('X_train_2degree_full.csv')
X_test = pd.read_csv('X_test_2degree_full.csv')
y_train = pd.read_csv('y_train_2degree_full.csv')
features = pd.read_csv('feature_correlation_to_label_2degree_full.csv')
y_train = np.expm1(y_train)
y_train = np.log(y_train)


#  1.1 Lasso Regression with different figures
from sklearn.linear_model import Lasso
feature_num = list(range(10, X_train.shape[1], 10))
lasso_scores = []

for topN in feature_num:    
    X_train_lasso = X_train[features.loc[0:topN-1,'features']].copy()
    X_train_lasso = X_train_lasso.values
    y_train_lasso = y_train.values.copy()
    lasso_regressor = Lasso(alpha=0.0004)
    score = np.sqrt(-1*cross_val_score(estimator=lasso_regressor, X = X_train_lasso, 
                                    y=y_train_lasso, 
                                    scoring='neg_mean_squared_error', cv=10)).mean()
    lasso_scores.append(score)
    print('number of features selected: {0:3d}, the RMSE score is: {1:5.5f}'.format(
            topN, score))
    
    
    
# Summarized the data, min score is 0.11180
lasso_score_feature = pd.DataFrame({'feature_num':feature_num, 'RSME':lasso_scores})
print('minimum score is {0:5.5f} with {1:3d} features'.format(min(lasso_scores), 
             int(lasso_score_feature[lasso_score_feature.RSME==min(lasso_scores)]['feature_num'])))

plt.plot(feature_num, lasso_scores)
plt.xlabel('feature_num')
plt.ylabel('RMSE')
plt.title('Lasso Regression RMSE')
plt.show()

# 1.1.2 select alphas using 500 features
alphas = [0.0010, 0.0012, 0.0014, 0.0015, 0.0016]
lasso_scores = []
X_train_lasso = X_train[features.loc[0:1000,'features']].copy()
X_train_lasso = X_train_lasso.values
y_train_lasso = y_train.values.copy()
X_test_lasso =  X_test[features.loc[0:1000,'features']].copy()
for item in alphas:
    lasso_regressor = Lasso(alpha=item)
    lasso_scores.append(np.sqrt(-1*cross_val_score(estimator=lasso_regressor, X = X_train_lasso, 
                                    y=y_train_lasso, 
                                    scoring='neg_mean_squared_error', cv=10)).mean())
plt.plot(alphas, lasso_scores)
plt.xlabel('Alpha')
plt.ylabel('RMSE')
plt.title('Lasso Regression RMSE')
plt.show()
min(lasso_scores)

# Best RMSE 0.1196

# 1.1.3 Generate output data with 1470 features and alphas = 
#y_pred_train_real = np.expm1(y_pred_train)
#real_log = np.log1p(y_pred_train_real)
from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
selected_lasso = Lasso(alpha=0.00095)
selected_lasso.fit(X_train_lasso, y_train_lasso)
coe = np.sort(selected_lasso.coef_)

y_pred_lasso = selected_lasso.predict(X_test_lasso)
y_train_pred = selected_lasso.predict(X_train_lasso)

rmse(y_train, y_train_pred)

y_pred_lasso=np.expm1(y_pred_lasso)

ID = list(range(1461,2920))
output_result = pd.DataFrame({'Id': ID, 'SalePrice':y_pred_lasso})
output_result.to_csv('output_feature_selected_lasso_new.csv', header=True, index=False)

# Best classifiers are Lasso, Gradient, XGBoosting and Ridge
# Try PCA with those classifiers



#  1.2 Ridge Regression with different figures
from sklearn.linear_model import Ridge
ridge_scores = []
for topN in feature_num:    
    X_train_ridge = X_train[features.loc[0:topN-1,'features']].copy()
    X_train_ridge = X_train_ridge.values
    y_train_ridge = y_train.values.copy()
    rge_regressor = Ridge(alpha=5)
    score = np.sqrt(-1*cross_val_score(estimator=rge_regressor, X = X_train_ridge, 
                                    y=y_train_ridge, 
                                    scoring='neg_mean_squared_error', cv=10)).mean()
    ridge_scores.append(score)
    print('number of features selected: {0:3d}, the RMSE score is: {1:5.5f}'.format(
            topN, score))
# Summarized the data, min score is 0.11197
ridge_score_feature = pd.DataFrame({'feature_num':feature_num, 'RSME':ridge_scores})
print('minimum score is {0:5.5f} with {1:3d} features'.format(min(ridge_scores), 
             int(ridge_score_feature[ridge_score_feature.RSME==min(ridge_scores)]['feature_num'])))
    
plt.plot(feature_num, ridge_scores)
plt.xlabel('feature_num')
plt.ylabel('RMSE')
plt.title('Ridge Regression RMSE')
plt.show()

# 1.63 Gradient boosting with different features 0.11449
'''from sklearn.preprocessing import Normalizer
nl_X = Normalizer()
X_train = nl_X.fit_transform(X_train)'''
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
GBest_scores = []
feature_num = list(range(10, X_train.shape[1]))
for topN in feature_num:    
    X_train_gbst = X_train[features.loc[0:topN-1,'features']].copy()
    X_train_gbst = X_train_gbst.values
    y_train_gbst = y_train.values.copy()

    GBest = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=10, loss='huber')
    
    score = np.sqrt(-1*cross_val_score(estimator=GBest, X = X_train_gbst, 
                                    y=y_train_gbst, 
                                    scoring='neg_mean_squared_error', cv=5)).mean()
    GBest_scores.append(score)     
    print('number of features selected: {0:3d}, the RMSE score is: {1:5.5f}'.format(
            topN, score))
    
# Summarized the data, min score is 
GBest_score_feature = pd.DataFrame({'feature_num':feature_num, 'RSME':ridge_scores})
print('minimum score is {0:5.5f} with {1:3d} features'.format(min(GBest_scores), 
             int(GBest_score_feature[GBest_score_feature.RSME==min(GBest_scores)]['feature_num'])))
    
plt.plot(feature_num, GBest_scores)
plt.xlabel('feature_num')
plt.ylabel('RMSE')
plt.title('Gradient Boosting Regression RMSE')
plt.show()

# GridSearch
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators':[100, 1000, 5000], 'max_depth':[3,6,9],
              'max_features':['sqrt', None], 'min_samples_leaf':[2, 5, 10],
              'min_samples_split':[3, 9, 15], 'loss':['huber']}]
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
GBest = ensemble.GradientBoostingRegressor(learning_rate=0.05)
grid_GBest = GridSearchCV(estimator=GBest, param_grid=parameters, scoring='neg_mean_squared_error',
                          cv=5)
grid_GBest.fit(X_train_GBest, y_train_GBest)
y_pred_train = GBest.predict(X_train)

# 1.1.3 Generate output data with 400 features
#y_pred_train_real = np.expm1(y_pred_train)
#real_log = np.log1p(y_pred_train_real)
from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

X_train_GBest = X_train[features.loc[0:1500,'features']].copy()
X_train_GBest = X_train_GBest.values
y_train_GBest = y_train.values.copy()
X_test_GBest =  X_test[features.loc[0:1500,'features']].copy()

selected_GBest = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',
                                               min_samples_leaf=5, min_samples_split=15, loss='huber')
selected_GBest.fit(X_train_GBest, y_train_GBest)
y_pred_GBest = selected_GBest.predict(X_test_GBest)
y_train_pred_GBest = selected_GBest.predict(X_train_GBest)

rmse(y_train, y_train_pred_GBest)

y_pred_GBest=np.expm1(y_pred_GBest)

ID = list(range(1461,2920))
output_result = pd.DataFrame({'Id': ID, 'SalePrice':y_pred_GBest})
output_result.to_csv('output_feature_selected_GBest_2degree_full_new_23.csv', header=True, index=False)
