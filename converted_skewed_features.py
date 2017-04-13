# Boston house price
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Import dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
all_data = pd.concat([train_data.loc[:,'MSSubClass':'SaleCondition'],
                     test_data.loc[:,'MSSubClass':'SaleCondition']])

# Convert the SalePrice into log unit
train_data.iloc[:,-1] = np.log(train_data.iloc[:,-1])
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

# Check the data with plots
#distribution of 'SalePrice' in train_data
sns.distplot(train_data['SalePrice'], bins=30, kde=False)
 
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())

X_train = all_data.iloc[0:train_data.shape[0],:].values
X_test = all_data.iloc[train_data.shape[0]:,:].values
y_train = train_data.iloc[:,-1].values



#  1.2 Ridge regression
from sklearn.linear_model import Ridge
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
rge_scores = []
for item in alphas:
    rge_regressor = Ridge(alpha=item)
    rge_scores.append(np.sqrt(-1*cross_val_score(estimator=rge_regressor, X = X_train, 
                                    y=y_train, 
                                    scoring='neg_mean_squared_error', cv=10)).mean())

plt.plot(alphas, rge_scores)



