# Modeling with feature selection based on correlation
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

#  1.1 Lasso Regression with different figures
from sklearn.linear_model import Lasso
feature_num = list(range(10, 266))
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

# 1.63 Gradient boosting with different features
'''from sklearn.preprocessing import Normalizer
nl_X = Normalizer()
X_train = nl_X.fit_transform(X_train)'''
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
GBest_scores = []
feature_num = list(range(10, 266, 5))
for topN in feature_num:    
    X_train_gbst = X_train[features.loc[0:topN-1,'features']].copy()
    X_train_gbst = X_train_gbst.values
    y_train_gbst = y_train.values.copy()

    GBest = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=10, loss='huber')
    GBest_scores = np.sqrt(-1*cross_val_score(estimator=GBest, X = X_train, 
                                    y=y_train, 
                                    scoring='neg_mean_squared_error', cv=5))
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