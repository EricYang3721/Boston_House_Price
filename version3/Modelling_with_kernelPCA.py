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
from sklearn.decomposition import KernelPCA
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
for n_comp in list(range(100, 360, 20)):
    pca = KernelPCA(n_components=n_comp, kernel='rbf')
    X_train_PCA = pca.fit_transform(X_train).copy()
    
    lasso_regressor = Lasso(alpha=0.0004)
    lasso_scores.append(np.sqrt(-1*cross_val_score(estimator=lasso_regressor, X = X_train_PCA, 
                                    y=y_train, 
                                    scoring='neg_mean_squared_error', cv=5)).mean())
    

plt.plot( list(range(100, 360, 20)), lasso_scores)
plt.xlabel('PCA_n')
plt.ylabel('RMSE')
plt.title('Lasso Regression RMSE')
plt.show()
min(lasso_scores)
#RSME

# 1.1 Lasso fine tune
pca = KernelPCA(n_components=150, kernel='poly')
X_train_PCA = pca.fit_transform(X_train).copy()
alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]
lasso_scores = []
for item in alphas:
    lasso_regressor = Lasso(alpha=item, random_state=0)
    lasso_scores.append(np.sqrt(-1*cross_val_score(estimator=lasso_regressor, X = X_train, 
                                    y=y_train, 
                                    scoring='neg_mean_squared_error', cv=10)).mean())
plt.plot(alphas, lasso_scores)
plt.xlabel('Alpha')
plt.ylabel('RMSE')
plt.title('Lasso Regression RMSE')
plt.show()
min(lasso_scores)
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
output_result.to_csv('output_lasso_0.0004_2_poly_PCA.csv', header=True, index=False)


# 2. PCA on_train, GBest
# 4 PCA on X_train for Gradient boosting
'''from sklearn.preprocessing import Normalizer
nl_X = Normalizer()
X_train = nl_X.fit_transform(X_train)'''
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
GBest_scores = []
for n_comp in list(range(10, 365, 10)):
    pca = KernelPCA(n_components=n_comp, kernel='rbf')
    X_train_PCA = pca.fit_transform(X_train).copy()
    GBest = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=10, loss='huber')
    score = np.sqrt(-1*cross_val_score(estimator=GBest, X = X_train_PCA, 
                                    y=y_train, 
                                    scoring='neg_mean_squared_error', cv=5)).mean()
    GBest_scores.append(score)
    print("number of features: {0:3d}, and the score is: {1:5.5f}".format(n_comp, score))
    
plt.plot( list(range(10, 365, 10)), lasso_scores)
plt.xlabel('PCA_n')
plt.ylabel('RMSE')
plt.title('GBest RMSE')
plt.show()
min(GBest_scores)