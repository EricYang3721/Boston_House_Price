# Modeling with PCA
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
from sklearn.linear_model import Lasso

# import data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values

# 1. PCA on X_train without scaling
lasso_scores = []
for n_comp in list(range(100, 365, 5)):
    pca = PCA(n_components=n_comp)
    X_train_PCA = pca.fit_transform(X_train).copy()
    
    lasso_regressor = Lasso(alpha=0.0003)
    lasso_scores.append(np.sqrt(-1*cross_val_score(estimator=lasso_regressor, X = X_train_PCA, 
                                    y=y_train, 
                                    scoring='neg_mean_squared_error', cv=5)).mean())
    

plt.plot( list(range(100, 365, 5)), lasso_scores)
plt.xlabel('PCA_n')
plt.ylabel('RMSE')
plt.title('Lasso Regression RMSE')
plt.show()
min(lasso_scores)
#RSME 0.11529

# 2. PCA on X_train with scaling
lasso_scores = []
for n_comp in list(range(100, 365, 5)):
    pca = PCA(n_components=n_comp)
    X_train_PCA = pca.fit_transform(X_train).copy()
    
    
    lasso_regressor = Lasso(alpha=0.0005)
    lasso_scores.append(np.sqrt(-1*cross_val_score(estimator=lasso_regressor, X = X_train_PCA, 
                                    y=y_train, 
                                    scoring='neg_mean_squared_error', cv=5)).mean())
    

plt.plot( list(range(100, 365, 5)), lasso_scores)
plt.xlabel('PCA_n')
plt.ylabel('RMSE')
plt.title('Lasso Regression RMSE')
plt.show()
min(lasso_scores)
#RSME 0.11635

# 3 Standardize then PCA
# Feature Scaling
x_Sc = StandardScaler()
X_train = x_Sc.fit_transform(X_train)

# PCA on X_train
lasso_scores = []
for n_comp in list(range(100, 365, 5)):
    pca = PCA(n_components=n_comp)
    X_train_PCA = pca.fit_transform(X_train).copy()
    
    lasso_regressor = Lasso(alpha=0.0003)
    lasso_scores.append(np.sqrt(-1*cross_val_score(estimator=lasso_regressor, X = X_train_PCA, 
                                    y=y_train, 
                                    scoring='neg_mean_squared_error', cv=5)).mean())
    

plt.plot( list(range(100, 365, 5)), lasso_scores)
plt.xlabel('PCA_n')
plt.ylabel('RMSE')
plt.title('Lasso Regression RMSE')
plt.show()
min(lasso_scores)
#RSME 0.12805




# 4 PCA on X_train for Gradient boosting

from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
GBest_scores = []
for n_comp in list(range(100, 365, 20)):
    pca = PCA(n_components=n_comp)
    X_train_PCA = pca.fit_transform(X_train).copy()
    GBest = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=10, loss='huber')

    GBest_scores.append(np.sqrt(-1*cross_val_score(estimator=GBest, X = X_train_PCA, 
                                    y=y_train, 
                                    scoring='neg_mean_squared_error', cv=10)).mean())
    
plt.plot( list(range(100, 365, 20)), lasso_scores)
plt.xlabel('PCA_n')
plt.ylabel('RMSE')
plt.title('GBest RMSE')
plt.show()
min(Best_scores)