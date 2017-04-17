import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer

# 1.1 Import data
train = pd.read_csv('train.csv').drop('Id', axis=1)
test = pd.read_csv('test.csv').drop('Id', axis=1)
all_data = pd.concat([train.iloc[:, :79], 
                      test.iloc[:, :79]], ignore_index = True)
y_train = train['SalePrice']

# 1.2 Preprocessing the labels
print("There are {} NAs in the label '{}'".format(y_train.isnull().sum(), 'SalePrice'))
print("The skewness of the label '{0:s}' is: {1:0.5f}".
      format('SalePrice', y_train.skew()))
print("Get the log price")
y_train = np.log(y_train)
print("The skewness of the logged label '{0:s}' is: {1:0.5f}".
     format('logSalePrice', y_train.skew()))

# 1.3 Correlation matrix of the features and labels
# The distributio of logSalePrice is 
sns.distplot(y_train, axlabel='logSalePrice')
train_cormat = train.corr() 
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(train_cormat, vmax=0.8, square=True)
f.tight_layout()
# Correlation matrix only evaluate numeric variables
cols = train_cormat.nlargest(10, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
cm2 = train[cols].corr()
# Heat map of features having top 10 correlations with SalePrice
cor_sort=train_cormat.sort_values(by='SalePrice', ascending = False).index
cm_max = train[cor_sort[0:10]].corr()
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(cm_max, vmax=0.8, square=True, annot=True,fmt='.2f',
            annot_kws={'size':10})
f.tight_layout()
# Correlation of each variable on lables
sorted_train_cormat = train_cormat.sort_values(by='SalePrice', ascending = False)
f, ax = plt.subplots(figsize=(8, 12))
sns.barplot(y='index', x='SalePrice', data=sorted_train_cormat.reset_index())
f.tight_layout()
# Pairplot of the numerical variables
sns.set()
sns.pairplot(data=train[cor_sort[0:7]], size=2.5)
plt.show()

# 2.1 Missing values distribution
all_NAs = all_data.isnull().sum().rename('numbers').to_frame()
all_NAs['percent'] = all_NAs['numbers']/all_data.shape[0]*100
all_NAs = all_NAs.sort_values(by='percent', ascending=False)
f, ax = plt.subplots(figsize=(8, 12))
sns.barplot(x=all_NAs[all_NAs['percent']>0]['percent'], y=all_NAs[all_NAs['percent']>0].index)
plt.title('Missing values percent', fontsize=18)
f.tight_layout()

#2.2 'PoolQC' and 'PoolArea'
print("The Pool Quality vs Pool area")
all_data[all_data['PoolQC'].notnull() | all_data['PoolArea']>0][['PoolQC','PoolArea']]
all_data.groupby(['PoolQC'])['PoolArea'].mean()
all_data[all_data['PoolQC'].notnull() | all_data['PoolArea']>0][['PoolQC','PoolArea']]
all_data.loc[2420,'PoolQC'] = 'Ex'
all_data.loc[2503,'PoolQC'] = 'Ex'
all_data.loc[2599,'PoolQC'] ='Fa'
all_data.ix[all_data['PoolQC'].isnull(),'PoolQC']='None'
all_data.loc[(all_data.MiscFeature.isnull()) & (all_data.MiscVal!=0)][['MiscFeature','MiscVal']]
all_data.groupby(['MiscFeature'])['MiscVal'].mean()
all_data.loc[2549, 'MiscFeature'] = 'Othr'
all_data.ix[all_data['MiscFeature'].isnull(),'MiscFeature']='None'

# 2.3 'Alley' 
# Replace NA in Alley with 'None'
all_data.ix[all_data['Alley'].isnull(),'Alley']='None'

# 2.4 'Fence'
# Replace NA in Fence with 'None'
all_data.ix[all_data['Fence'].isnull(),'Fence']='None'

# 2.5 'FirelaceQu'
# a. samples: FireplaceQu == nan and Fireplaces > 0
all_data.loc[(all_data['FireplaceQu'].isnull()) & (all_data['Fireplaces']>0)].shape[0]
# b. samples: FireplaceQu != nan and Fireplaces ==0
all_data.loc[(all_data['FireplaceQu'].notnull()) & (all_data['Fireplaces']==0)].shape[0]
# Replace NA in Alley with 'None'
all_data.ix[all_data['FireplaceQu'].isnull(),'FireplaceQu']='None'

# 2.6 'LotFrontage' and 'LotArea'
all_data[['LotFrontage','LotArea']].isnull().sum()
all_data['LotFrontage'].hist(bins=50)
sns.lmplot(x='LotFrontage',y='LotArea', data=all_data)
plt.title("LotArea vs LotFrontage", fontsize=18)
# Fit a linear model between 'LotFrontage' and 'LotArea'
# and fill the NAs with the prnited values
lot_data = all_data[['LotFrontage','LotArea']].copy(deep=True)
lot_noNa = lot_data.loc[lot_data['LotFrontage'].notnull()]
lot_Na = lot_data.loc[lot_data['LotFrontage'].isnull()]
area = lot_noNa['LotArea'].values.reshape(-1,1)
frontage = lot_noNa['LotFrontage'].values.reshape(-1,1)
lot_regressor = LinearRegression()
lot_regressor.fit(area, frontage)
lot_data.insert(2, 'predict',lot_regressor.predict(lot_data['LotArea'].values.reshape(-1,1)))
lot_data['LotFrontage']=lot_data['LotFrontage'].fillna(lot_data['predict'])
all_data['LotFrontage'] = lot_data['LotFrontage']
all_data['LotFrontage'].isnull().sum()

# 2.7 Garage features
garage = ['GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'GarageType',
          'GarageArea', 'GarageCars']
all_data[garage].isnull().sum()
garage_Nas = all_data[all_data[garage].isnull().any(axis=1)][garage]
all_data[(all_data['GarageType'].notnull()) & (all_data['GarageYrBlt'].
         isnull())][garage]
all_data.loc[2576, 'GarageType'] = 'None'
all_data.loc[2576, ['GarageArea','GarageCars']] = 0
common_values = [all_data[x].value_counts().index[0] for x in garage]
common_values
all_data.loc[2126, garage[0:4]] = common_values[0:4]
all_data[garage[0:5]] = all_data[garage[0:5]].fillna('None')


# 2.8 Basement Features
bsmt = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType1',
       'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
       'BsmtFullBath', 'BsmtHalfBath']
all_data[bsmt].isnull().sum()
bsmt_Nas = all_data[all_data[bsmt].isnull().any(axis=1)][bsmt]
sns.barplot(y=all_data['BsmtFinSF2'], x=all_data['BsmtFinType2'])
# Replace NA in 332 as ALQ
all_data.loc[332, 'BsmtFinType2'] = 'ALQ'

sns.barplot(x='BsmtExposure', y='TotalBsmtSF', data=all_data)
all_data.loc[948, 'BsmtExposure'] = 'No'
all_data.loc[1487, 'BsmtExposure'] = 'Gd'
all_data.loc[2348, 'BsmtExposure'] = 'No'

sns.factorplot(x='BsmtCond', data=all_data, kind='count')
plt.title("Summary of BsmtCond")
all_data.loc[2040, 'BsmtCond']='TA'
all_data.loc[2185, 'BsmtCond']='TA'
all_data.loc[2524, 'BsmtCond']='TA'
sns.factorplot(x='BsmtQual', data=all_data, kind='count')
all_data.loc[2217, 'BsmtQual'] = 'TA'
all_data.loc[2218, 'BsmtQual'] = 'TA'
all_data.loc[2188, ['BsmtHalfBath', 'BsmtFullBath']]=0
all_data.loc[2120, ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                    'BsmtFullBath','BsmtHalfBath']]=0
all_data[bsmt]=all_data[bsmt].fillna('None')
all_data[bsmt].isnull().sum()

# 2.9 MasVnrType, MasVnrArea
all_data[['MasVnrType','MasVnrArea']].isnull().sum()
all_data[(all_data['MasVnrType'].isnull()) 
        | (all_data['MasVnrArea'].isnull())][['MasVnrType','MasVnrArea']]
sns.barplot(x='MasVnrType', y='MasVnrArea', data=all_data)
plt.title('MasVnrType vs MasVnrArea')
all_data.loc[2610, 'MasVnrType'] = 'BrkFace'
all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)

# 2.10 MSSubclass and MSZoning
all_data[(all_data['MSSubClass'].isnull()) 
       | (all_data['MSZoning'].isnull())][['MSSubClass','MSZoning']]
sns.factorplot(x='MSZoning', data=all_data[all_data['MSSubClass'].isin([20,30,70])],
                kind='count', hue='MSSubClass')
plt.title('MSZoning distribution for MSSubClass == 20, 30, 70')
all_data.loc[1915, 'MSZoning'] = 'RM'
all_data.loc[2250, 'MSZoning'] = 'RM'
all_data.loc[[2216,2904],'MSZoning'] = 'RL'
all_data[['MSZoning','MSSubClass']].isnull().sum()

# 2.11 Functional
all_data[['Functional']].isnull().sum()
sns.factorplot(x='Functional', data=all_data, kind='count')
plt.title('Functional distribution')
all_data.loc[all_data['Functional'].isnull(), 'Functional'] = 'Typ'

# 2.12 Utilities
all_data[['Utilities']].isnull().sum()
sns.factorplot(x='Utilities', data=all_data, kind='count')
plt.title('Utilities distribution')
all_data.loc[all_data['Utilities'].isnull(), 'Utilities'] = 'AllPub'

# 2.13 SaleType
all_data.loc[all_data['SaleType'].isnull(), ['SaleType','SaleCondition']]
sns.factorplot(x='SaleType', hue='SaleCondition', data=all_data, kind='count')
# For normal sale, the WD dominates the SaleType
all_data.loc[all_data['SaleType'].isnull(), 'SaleType'] = 'WD'

# 2.14 KitchenQual
all_data[['KitchenQual']].isnull().sum()
sns.factorplot(x='KitchenQual', data=all_data, kind='count')
plt.title('KitchenQual distribution')
all_data.loc[all_data['KitchenQual'].isnull(), 'KitchenQual'] = 'TA'

# 2.15 Exterior1st and Exterior2nd
all_data[['Exterior1st', 'Exterior2nd']].isnull().sum()
all_data[(all_data['Exterior1st'].isnull()) 
       | (all_data['Exterior2nd'].isnull())][['RoofStyle', 'RoofMatl', 
         'Exterior1st','Exterior2nd']]
exterior = all_data[(all_data['RoofStyle']=='Flat') &
                    (all_data['RoofMatl']=='Tar&Grv')][['RoofStyle', 'RoofMatl', 
         'Exterior1st','Exterior2nd']]
print("Exterior1st used for Flat RoofStyle and Tar&Grv RoofMatl is:",
      exterior.Exterior1st.value_counts())
print("Exterior2nd used for Flat RoofStyle and Tar&Grv RoofMatl is:",
      exterior.Exterior2nd.value_counts())
all_data.loc[2151, 'Exterior1st'] = 'Wd Sdng'
all_data.loc[2151, 'Exterior2nd'] = 'Wd Sdng'

# 2.16 Electrical
sns.factorplot(x='Electrical', data=all_data, kind='count')
plt.title('Electrical distribution')
all_data.loc[all_data['Electrical'].isnull(), 'Electrical'] = 'SBrkr'
# Checki if any missing value 
NAs = all_data.isnull().sum()

# 3 Change some Categorical Features into Numerical features

quality_dict = {'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':5, 'Ex':5}
quality_fea_names =['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                    'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 
                    'PoolQC']
# 3.1 MSSubClass
# newer building and older building has different values
all_data['newMSSubClass'] = all_data['MSSubClass'].replace({20:1, 30:0, 40:0,
        45:0, 50:0, 60:1, 70:10, 75:0, 80:0, 85:0, 90:0, 120:1, 150:0, 160:1,
        180:0, 190:0})
# converting numbers into categorical MSC = MSSubClass
Subclass_cat = {x:'MSC_'+str(x) for x in all_data['MSSubClass'].unique()}
all_data['MSSubClass'] = all_data['MSSubClass'].replace(Subclass_cat)

# 3.2 Street
all_data['Street'] = all_data['Street'].replace({'Grvl':0, 'Pave':1})

# 3.3 Alley
all_data['Alley'] = all_data['Alley'].replace({'None':0, 'Grvl':1, 'Pave':2})

# 3.4 BsmtExposure
all_data['BsmtExposure'] = all_data['BsmtExposure'].replace({'Gd':4, 'Av':3, 'Mn':2,
        'No':1, 'None':0})

# 3.5 PavedDrive
all_data['PavedDrive'] = all_data['PavedDrive'].replace({'Y':2, 'P':1, 'N':0})

# 3.6 Fence
all_data['Fence'] = all_data['Fence'].replace({'None':0, 'MnWw':1, 'GdWo':2,
        'MnPrv':3, 'GdPrv':4})

# 3.7 Quality features
all_data[quality_fea_names] = all_data[quality_fea_names].replace(quality_dict)

# 4 



