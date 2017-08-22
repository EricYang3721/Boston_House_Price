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


# 1.5 XGBoosting


from sklearn.linear_model import Ridge
feature_num = list(range(10, X_train.shape[1]))
xgb_scores = []
for topN in feature_num:    
    X_train_xgb = X_train[features.loc[0:topN-1,'features']].copy()
    X_train_xgb = X_train_xgb.values
    y_train_xgb = y_train.values.copy()
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
    score = np.sqrt(-1*cross_val_score(estimator=xgb_reg, X = X_train_xgb, 
                                    y=y_train_xgb, 
                                    scoring='neg_mean_squared_error', cv=10)).mean()
    xgb_scores.append(score)
    print('number of features selected: {0:3d}, the RMSE score is: {1:5.5f}'.format(
            topN, score))
# Summarized the data, min score is 0.11197
xgb_score_feature = pd.DataFrame({'feature_num':feature_num, 'RSME':xgb_scores})
print('minimum score is {0:5.5f} with {1:3d} features'.format(min(xgb_scores), 
             int(xgb_score_feature[xgb_score_feature.RSME==min(xgb_scores)]['feature_num'])))
    
plt.plot(feature_num, xgb_scores)
plt.xlabel('feature_num')
plt.ylabel('RMSE')
plt.title('Ridge Regression RMSE')
plt.show()

X_train_xgb = X_train[features.loc[0:2400,'features']].copy()
X_train_xgb = X_train_xgb.values
y_train_xgb = y_train.values.copy()
X_test_xgb =  X_test[features.loc[0:2400,'features']].copy()
xgb_reg = xgb.XGBRegressor(colsample_bytree=0.2,
                           gamma=0.0,
                           learning_rate=0.01,
                           max_depth=4,
                           min_child_weight=1.5,
                           n_estimators=30000,
                           reg_alpha=0.9,
                           reg_lambda=0.6,
                           subsample=0.2,
                           seed=42,
                           silent=1)    
xgb_reg.fit(X_train_xgb, y_train_xgb)

y_pred_xgb = xgb_reg.predict(X_test_xgb)
y_train_pred_xgb = xgb_reg.predict(X_train_xgb)

from sklearn.metrics import mean_squared_error
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse(y_train, y_train_pred_xgb)

y_pred_xgb=np.expm1(y_pred_xgb)

ID = list(range(1461,2920))
output_result = pd.DataFrame({'Id': ID, 'SalePrice':y_pred_xgb})
output_result.to_csv('output_feature_selected_GBest_2degree_full.csv', header=True, index=False)





