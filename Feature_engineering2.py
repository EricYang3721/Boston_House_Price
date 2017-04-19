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
y_train = np.log1p(y_train)
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

# 3.8 LotShape
all_data['LotShape'] = all_data['LotShape'].replace({'Reg':2, 'IR1':1, 'IR2':0, 'IR3':0})

# 3.9 LandContour
all_data['LandContour'] = all_data['LandContour'].replace({'Lvl':1, 'Bnk':0, 'HLS':0, 'Low':0})

# 3.10 Utilities
all_data['Utilities'] = all_data['Utilities'].replace({'AllPub':1, 'NoSewr':0, 'NoSeWa':0, 'ELO':0})

# 3.11 Neighborhood
nb_price = train.copy().groupby(by='Neighborhood')['SalePrice'].median().reset_index().sort_values(by='SalePrice')
sns.factorplot(y='Neighborhood', x = 'SalePrice', data=nb_price, kind='bar')
# Add a column of NbPriceCat labeling the neighborhood price category
nb_price['NbPriceCat'] = 0
nb_price.loc[nb_price['SalePrice'] > 250000, 'NbPriceCat'] = 2
nb_price.loc[(nb_price['SalePrice'] <= 250000) & (nb_price['SalePrice'] > 170000), 'NbPriceCat'] = 1

nb_dict = dict(zip(nb_price['Neighborhood'], nb_price['NbPriceCat']))
all_data['NeighborhoodCat'] = all_data['Neighborhood']
all_data['NeighborhoodCat'] = all_data['NeighborhoodCat'].replace(nb_dict)

# 3.12 BldgType:
sns.factorplot(x='BldgType', y='SalePrice', data=train)
all_data['BldgType'] = all_data['BldgType'].replace({'1Fam':1, '2FmCon':0, 'Duplx':0, 'TwnhsE':1, 'TwnhsI':0})

# 3.13 HouseStype
# Create a house style price category
all_data['HouseStyleCat'] = all_data['HouseStyle']
all_data['HouseStyleCat'] = all_data['HouseStyleCat'].replace({'2Story': 1, '1Story':1, 
        '1.5Fin':0, '1.5Unf':0, 'SFoyer':0, 'SLvl':0, '2.5Unf':0, '2.5Fin':1})
    
# 3.14 Central Air
all_data['CentralAir'] = all_data['CentralAir'].replace({'N':0, 'Y':1})

# 3.15 Electrical 
all_data['Electrical'] = all_data['Electrical'].replace({'Sbrkr':1, 'FuseA':0, 'FuseF':0,
        'FuseP':0, 'Mix':0})
    
# 3.16 Functional
all_data['Functional'] = all_data['Functional'].replace({'Typ':5, 'Min1':4, 'Min2':4,
        'Mod':3, 'Maj1':2, 'Maj2':2, 'Sev':1, 'Sal':0})
    
# 3.17 GarageFinish
all_data['GarageFinish'] = all_data['GarageFinish'].replace({'Fin':2, 'RFn':1, 'Unf':0})

# 3.18 LandSlope
all_data['LandSlope'] = all_data['LandSlope'].replace({'Gtl':1, 'Mod':0, 'Sev':0})

# 3.19 MoSold --> Season
all_data['HotSeason'] = 0
all_data.loc[all_data['MoSold'].isin([4,5,6,7]), 'HotSeason'] = 1
# Set Month to 'Jan', 'Feb' ...
all_data['MoSold'] = all_data['MoSold'].replace({1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr',
        5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'})

# 3.20 SaleType and SaleCondition
# add column of newSale
all_data['NewSale'] = 0
all_data.loc[all_data['SaleCondition']=='Partial', 'NewSale'] = 1

# 3.21 Add Recon feature
all_data['Recon'] = 0
all_data.loc[all_data.YearBuilt < all_data.YearRemodAdd, 'Recon'] = 1 

# 3.22 Add feature YrSoBu for years from built to sold
all_data['YrSoBu'] = all_data['YrSold'] - all_data['YearBuilt']

# 3.22 Add feature YrSoRecon for years from built to Reconstructed
all_data['YrSoRecon'] = all_data['YrSold'] - all_data['YearRemodAdd']

# 4 Outliers
sns.jointplot(x=train.GrLivArea, y = train.SalePrice)
# 2 points with extremely large GrLivArea, but very low SalePrice
outlier_index = train[(train['GrLivArea'] > 4000) & (train['SalePrice']<300000)].index
all_data = all_data.drop(outlier_index, axis=0)
y_train = y_train.drop(outlier_index, axis=0)

# 5 Reduce the Skewness of the numeric features
num_features = all_data.dtypes[all_data.dtypes != 'object'].index
fea_skewness = all_data[num_features].apply(lambda x: stats.skew(x.dropna()))
skewed_fea = list(fea_skewness[fea_skewness > 0.75].index)
all_data[skewed_fea] = np.log1p(all_data[skewed_fea])

# 6 Get the dummy variables
all_data_new = all_data.copy()
all_data_new = pd.get_dummies(all_data_new)

# 7 Separate into training and test set
X_train = all_data_new.iloc[0:1458, :]
X_test = all_data_new.iloc[1458:,:]

X_train.to_csv('X_train.csv', index = False)
X_test.to_csv('X_test.csv', index = False)
y_train.to_csv('y_train.csv', header = ['lgSalePrice'], index=False)
