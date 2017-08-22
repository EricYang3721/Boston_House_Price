# Test
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
X_train = pd.read_csv('X_test_2degree_full.csv')
#X_test = pd.read_csv('X_test_2degree_full.csv')
y_train = pd.read_csv('GBest_sol.csv')
y_train = np.log(y_train)
features = pd.read_csv('feature_correlation_to_label_2degree_full.csv')


from sklearn.linear_model import Lasso
alphas = [0.001, 0.005, 0.01, 0.15]
lasso_scores = []
X_train_lasso = X_train[features.loc[0:1500,'features']].copy()
X_train_lasso = X_train_lasso.values
y_train_lasso = y_train.values.copy()

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

from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
selected_lasso = Lasso(alpha=0.25)
selected_lasso.fit(X_train_lasso, y_train_lasso)
coe = np.sort(selected_lasso.coef_)

y_train_pred = selected_lasso.predict(X_train_lasso)

rmse(y_train, y_train_pred)
