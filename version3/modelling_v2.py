# Price prediction
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
X_train = pd.read_csv('X_train_v2.csv')
X_test = pd.read_csv('X_test_v2.csv')
y_train = pd.read_csv('y_train_v2.csv')
y_train = np.expm1(y_train)
y_train = np.log(y_train)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values

# 1. Model comparison
#  1.1 Linear Regression
from sklearn.linear_model import LinearRegression
linregressor = LinearRegression()
scores = np.sqrt(-1*cross_val_score(estimator=linregressor, X = X_train, 
                                    y=y_train, 
                                    scoring='neg_mean_squared_error', cv=10))
print("RMSE of Linear regression is {0:5.5f}".format(scores.mean()))
# Very bad performance

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
# Lowest RMSE for Ridge Regression is 0.11144

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
# Lowest RMSE for Ridge Regression is 0.11026
# OUTPUT DATA
lasso_choose = Lasso(alpha=0.0004)
lasso_choose.fit(X_train, y_train)
y_pred_train = lasso_choose.predict(X_train)
#y_pred_train_real = np.expm1(y_pred_train)
#real_log = np.log1p(y_pred_train_real)
from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse(y_train, y_pred_train)
y_pred_test=lasso_choose.predict(X_test)
y_pred_test=np.expm1(y_pred_test)

ID = list(range(1461,2920))
output_result = pd.DataFrame({'Id': ID, 'SalePrice':y_pred_test})
output_result.to_csv('output_lasso_0.0004_2.csv', header=True, index=False)



# 1.3 Supported Vector Rgression
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
# Lowest RMSE for SVR is 0.3544

# 1.4 Random forest regression
from sklearn.ensemble import RandomForestRegressor
rfg_scores =[]
num_esti = [10, 50, 100, 500, 1000, 3000, 5000]
for item in num_esti:
    rfg_regressor = RandomForestRegressor(n_estimators=item, random_state=0)
    rfg_scores.append(np.sqrt(-1*cross_val_score(estimator=rfg_regressor, X = X_train, 
                                    y=y_train, 
                                    scoring='neg_mean_squared_error', cv=10))
plt.plot(num_esti, rfg_scores)
plt.xlabel('num_esti')
plt.ylabel('RMSE')
plt.title('Random Forest Regression RMSE')
plt.show()
min(rfg_scores)
# Lowest RMSE for Ridge Regression is 0.13265

# 1.5 XGBoosting
xgb_reg = xgb.XGBRegressor(colsample_bytree=0.2,
                           gamma=0.0,
                           learning_rate=0.01,
                           max_depth=4,
                           min_child_weight=1.5,
                           n_estimators=7200,
                           reg_alpha=0.9,
                           reg_lambda=0.6,
                           subsample=0.2,
                           seed=42,
                           silent=1)
xgfg_scores =[]
xgfg_scores = np.sqrt(-1*cross_val_score(estimator=xgb_reg, X = X_train, 
                                    y=y_train, 
                                    scoring='neg_mean_squared_error', cv=10))
print("RMSE of XGBosting is {0:5.5f}".format(xgfg_scores.mean()))
# RMSE of XGBosting is 0.11157

# 1.6 Gradient boosting
'''from sklearn.preprocessing import Normalizer
nl_X = Normalizer()
X_train = nl_X.fit_transform(X_train)'''
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
GBest = ensemble.GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=3, max_features='sqrt',
                                               min_samples_leaf=2, min_samples_split=3, loss='ls')

GBest_scores = np.sqrt(-1*cross_val_score(estimator=GBest, X = X_train, 
                                    y=y_train, 
                                    scoring='neg_mean_squared_error', cv=5))
print("RMSE of GradientBosting is {0:5.5f}".format(GBest_scores.mean()))
# RMSE of GradientBosting is 0.11153


# GridSearch on data
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
grid_GBest.fit(X_train, y_train)
y_pred_train = GBest.predict(X_train)
print("Gradient boosting score on training set: ", rmse(y_train, y_pred_train))
grid_GBest.best_score_
grid_GBest.cv_results_
# OUTPUT DATA

GBest.fit(X_train, y_train)
y_pred_train = GBest.predict(X_train)
#y_pred_train_real = np.expm1(y_pred_train)
#real_log = np.log1p(y_pred_train_real)
from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse(y_train, y_pred_train)
y_pred_test=GBest.predict(X_test)
y_pred_test=np.expm1(y_pred_test)

ID = list(range(1461,2920))
output_result = pd.DataFrame({'Id': ID, 'SalePrice':y_pred_test})
output_result.to_csv('output_GBest_v2.csv', header=True, index=False)

# Best classifiers are Lasso, Gradient, XGBoosting and Ridge
# Try PCA with those classifiers