# Scaling features

# Price prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
import xgboost as xgb
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')

# import data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values

# Scaling features
from sklearn.preprocessing import StandardScaler
sc_X = StandardScale()
X_train = sc_X.fit_transform(X_train)

#  1.1 Lasso Regression
from sklearn.linear_model import Lasso
#alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]
alphas = [0.0001, 0.001, 0.003, 0.005, 0.007, 0.009, 0.011, 0.13]
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
# Lowest RMSE for Ridge Regression is 0.11147

# 1.2 XGBoosting
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
# RMSE of XGBosting is 0.11155

# 1.3 Gradient boosting
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
GBest = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=10, loss='huber')

GBest_scores = np.sqrt(-1*cross_val_score(estimator=GBest, X = X_train, 
                                    y=y_train, 
                                    scoring='neg_mean_squared_error', cv=10))
print("RMSE of GradientBosting is {0:5.5f}".format(GBest_scores.mean()))
# RMSE of GradientBosting is 0.11327