import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer
import warnings
warnings.filterwarnings('ignore')
# 1 PreProcessing
# 1.1 DataLoading and pre processing
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
all_data = pd.concat([train.loc[:,'MSSubClass':'SaleCondition'], 
                      test.loc[:,'MSSubClass':'SaleCondition']], ignore_index=True)
y_train = train.SalePrice
# 1.2 separate into object and numeric features
all_data.dtypes.value_counts()
obj_features = list(all_data.dtypes[all_data.dtypes == 'object'].index)
num_features = list(all_data.dtypes[all_data.dtypes != 'object'].index)
processed_NAfeats = []
# all_data shape is (2917, 79) ##################################################

# 2 Missing values
# 2.1 Missing values summary
NA_counts = all_data.isnull().sum()
NA_obj=NA_counts[obj_features]
NA_obj = NA_obj[NA_obj!=0]
NA_num = NA_counts[num_features]
NA_num = NA_num[NA_num!=0]

# 2.2 LotFrontage
# LotFrontage correlates to LotArea, LotShape, BldgType, Use randomForest for imputation
# Extract data without NA at LotFrontage
LotFront_feats = ['LotArea', 'LotShape', 'BldgType', 'LotFrontage']
LotData = all_data[LotFront_feats].copy()
Lot_X = LotData[['LotArea', 'LotShape', 'BldgType']]
Lot_X_dummy = pd.get_dummies(Lot_X)
Lot_non_NAindex = LotData[LotData.LotFrontage.notnull()].index
Lot_train_X = Lot_X_dummy.loc[Lot_non_NAindex]
Lot_label = LotData.loc[Lot_non_NAindex,'LotFrontage']
processed_NAfeats = processed_NAfeats + ['LotFrontage']

from sklearn.ensemble import RandomForestRegressor
rfr_NA = RandomForestRegressor(n_estimators=50, random_state=0)
rfr_NA.fit(Lot_train_X, Lot_label)
for itr in range(0, 2919):
    if(np.isnan(LotData.loc[itr, 'LotFrontage'])):
        LotData.loc[itr, 'LotFrontage'] = rfr_NA.predict(Lot_X_dummy.loc[itr,:])
for item in LotFront_feats:
    all_data[item] = LotData[item]
    
# 2.3 MasVnrArea and MasVnrType
MasVnr_feats = ['MasVnrType', 'MasVnrArea']
all_data[MasVnr_feats].isnull().sum()
all_data.loc[all_data.MasVnrType.isnull(), MasVnr_feats]
all_data.MasVnrType.value_counts(dropna=False)
all_data.loc[all_data['MasVnrArea'].isnull(),'MasVnrType'] = 'None'
all_data.loc[all_data['MasVnrArea'].isnull(),'MasVnrArea'] = 0
all_data.loc[all_data['MasVnrType'].isnull(),'MasVnrType'] = 'BrkFace'
processed_NAfeats = processed_NAfeats + MasVnr_feats 

# 2.4 Basement related feature
bsmt_featsNA_num = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
              'BsmtFullBath', 'BsmtHalfBath']
all_data[bsmt_featsNA_num].isnull().sum()
bsmt_featsNA_cat = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                   'BsmtFinType2']
all_data[bsmt_featsNA_cat].isnull().sum()
all_data.loc[all_data.BsmtFullBath.isnull(), bsmt_featsNA_num+bsmt_featsNA_cat]
# These 2 has not bsmt
all_data.loc[all_data.BsmtFullBath.isnull(), bsmt_featsNA_num] = 0.0
all_data.loc[all_data.BsmtFullBath.isnull(), bsmt_featsNA_cat] = 'None'
all_data.loc[all_data.BsmtExposure.isnull(), bsmt_featsNA_num+bsmt_featsNA_cat]
all_data.loc[332, 'BsmtFinType2'] = 'ALQ'
sns.barplot(x='BsmtExposure', y='TotalBsmtSF', data=all_data)
all_data.loc[948, 'BsmtExposure'] = 'No'
all_data.loc[1487, 'BsmtExposure'] = 'Gd'
all_data.loc[2348, 'BsmtExposure'] = 'No'
# plt.title("Summary of BsmtCond")
all_data.loc[2040, 'BsmtCond']='TA'
all_data.loc[2185, 'BsmtCond']='TA'
all_data.loc[2524, 'BsmtCond']='TA'
sns.factorplot(x='BsmtQual', data=all_data, kind='count')
all_data.loc[2217, 'BsmtQual'] = 'TA'
all_data.loc[2218, 'BsmtQual'] = 'TA'
all_data.loc[all_data.BsmtQual.isnull(),bsmt_featsNA_cat] = 'None'
processed_NAfeats = processed_NAfeats + bsmt_featsNA_cat +bsmt_featsNA_num

# 2.5 Garage features
garage_feats_num = ['GarageYrBlt', 'GarageCars', 'GarageArea'] 
garage_feats_cat = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
all_data[garage_feats_num+garage_feats_cat].isnull().sum()
all_data.loc[all_data.GarageArea.isnull(), garage_feats_num+garage_feats_cat]
all_data.loc[all_data.GarageArea.isnull(), garage_feats_num] = 0.0
all_data.loc[all_data.GarageArea.isnull(), garage_feats_cat] = 'None'
all_data.loc[all_data.GarageQual.isnull(), garage_feats_cat+ garage_feats_num]['GarageCars'].unique()
all_data.loc[all_data.GarageCars==1, garage_feats_cat].mode()
all_data.loc[2576, ['GarageType', 'GarageFinish','GarageQual', 'GarageCond']] = 'None'
all_data.loc[(all_data.GarageType.isnull()) & (all_data.GarageCars==1.0), 'GarageType'] = 'Detchd'
all_data.loc[(all_data.GarageFinish.isnull()) & (all_data.GarageCars==1.0), 'GarageFinish'] = 'Unf'
all_data.loc[(all_data.GarageQual.isnull()) & (all_data.GarageCars==1.0), 'GarageQual'] = 'TA'
all_data.loc[(all_data.GarageCond.isnull()) & (all_data.GarageCars==1.0), 'GarageCond'] = 'TA'
all_data.loc[(all_data.GarageCond.isnull()) & (all_data.GarageCars==0.0), garage_feats_cat+['GarageYrBlt']] = 'None'
all_data.loc[all_data.GarageYrBlt.isnull(), 'GarageYrBlt']=0.0
processed_NAfeats = processed_NAfeats + garage_feats_cat+garage_feats_num

# 2.6 MSZoning and MSSubClass
all_data[(all_data['MSSubClass'].isnull()) 
       | (all_data['MSZoning'].isnull())][['MSSubClass','MSZoning']]
sns.factorplot(x='MSZoning', data=all_data[all_data['MSSubClass'].isin([20,30,70])],
                kind='count', hue='MSSubClass')
plt.title('MSZoning distribution for MSSubClass == 20, 30, 70')
all_data.loc[1915, 'MSZoning'] = 'RM'
all_data.loc[2250, 'MSZoning'] = 'RM'
all_data.loc[[2216,2904],'MSZoning'] = 'RL'
all_data[['MSZoning','MSSubClass']].isnull().sum()
processed_NAfeats = processed_NAfeats + ['MSZoning','MSSubClass']

# 2.7 Alley
all_data.loc[all_data['Alley'].isnull(),'Alley']='None'

# 2.8 Utilities
all_data[['Utilities']].isnull().sum()
#sns.factorplot(x='Utilities', data=all_data, kind='count')
#plt.title('Utilities distribution')
all_data.loc[all_data['Utilities'].isnull(), 'Utilities'] = 'AllPub'

# 2.9 Exterior1st and Exterior2nd
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
all_data.loc[2151, 'Exterior1st'] = 'Polywood'
all_data.loc[2151, 'Exterior2nd'] = 'Polywood'

# 2.10 Electrical
sns.factorplot(x='Electrical', data=all_data, kind='count')
plt.title('Electrical distribution')
all_data.loc[all_data['Electrical'].isnull(), 'Electrical'] = 'SBrkr'

# 2.11 KitchenQual
all_data[['KitchenQual']].isnull().sum()
sns.factorplot(x='KitchenQual', data=all_data, kind='count')
plt.title('KitchenQual distribution')
all_data.loc[all_data['KitchenQual'].isnull(), 'KitchenQual'] = 'TA'

# 2.12 Functional
all_data[['Functional']].isnull().sum()
sns.factorplot(x='Functional', data=all_data, kind='count')
plt.title('Functional distribution')
all_data.loc[all_data['Functional'].isnull(), 'Functional'] = 'Typ'

# 2.13 FireplaceQu
# a. samples: FireplaceQu == nan and Fireplaces > 0
all_data.loc[(all_data['FireplaceQu'].isnull()) & (all_data['Fireplaces']>0)].shape[0]
# b. samples: FireplaceQu != nan and Fireplaces ==0
all_data.loc[(all_data['FireplaceQu'].notnull()) & (all_data['Fireplaces']==0)].shape[0]
# Replace NA in FireplaceQue with 'None'
all_data.loc[all_data['FireplaceQu'].isnull(),'FireplaceQu']='None'

# 2.14 PoolQC
print("The Pool Quality vs Pool area")
all_data[all_data['PoolQC'].notnull() | all_data['PoolArea']>0][['PoolQC','PoolArea']]
all_data.groupby(['PoolQC'])['PoolArea'].mean()
all_data.loc[2420,'PoolQC'] = 'Ex'
all_data.loc[2503,'PoolQC'] = 'Ex'
all_data.loc[2599,'PoolQC'] ='Fa'
all_data.ix[all_data['PoolQC'].isnull(),'PoolQC']='None'

# 2.15 Fence 
# Replace NA in Fence with 'None'
all_data.ix[all_data['Fence'].isnull(),'Fence']='None'

# 2.16 MiscFeature
all_data.loc[(all_data.MiscFeature.isnull()) & (all_data.MiscVal!=0)][['MiscFeature','MiscVal']]
all_data.groupby(['MiscFeature'])['MiscVal'].mean()
all_data.loc[2549, 'MiscFeature'] = 'Gar2'
all_data.loc[all_data['MiscFeature'].isnull(),'MiscFeature']='None'

# 2.17 SaleType
all_data.loc[all_data['SaleType'].isnull(), ['SaleType','SaleCondition']]
sns.factorplot(x='SaleType', hue='SaleCondition', data=all_data, kind='count')
# For normal sale, the WD dominates the SaleType
all_data.loc[all_data['SaleType'].isnull(), 'SaleType'] = 'WD'

# 3. Change some Categorical Features into Numerical features
quality_dict = {'None':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}
quality_fea_names =['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                    'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 
                    'PoolQC'] # OverallQual, OverallCond
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


# 3.14 Central Air
all_data['CentralAir'] = all_data['CentralAir'].replace({'N':0, 'Y':1})

# 3.15 Electrical 
all_data['Electrical'] = all_data['Electrical'].replace({'SBrkr':1, 'FuseA':0, 'FuseF':0,
        'FuseP':0, 'Mix':0})
    
# 3.16 Functional
all_data['Functional'] = all_data['Functional'].replace({'Typ':5, 'Min1':4, 'Min2':4,
        'Mod':3, 'Maj1':2, 'Maj2':2, 'Sev':1, 'Sal':0})
    
# 3.17 GarageFinish
all_data['GarageFinish'] = all_data['GarageFinish'].replace({'Fin':2, 'RFn':1, 'Unf':0, 'None':0})

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

# 3.23  Add feature YrSoRecon for years from built to Reconstructed
all_data['YrSoRecon'] = all_data['YrSold'] - all_data['YearRemodAdd']

# 3.24 Set years to be in category
# YearBuilt, YearRemodAdd, 20
all_data.YearBuilt = -(all_data.YearBuilt - 2010)/20
all_data.GarageYrBlt = all_data.GarageYrBlt.replace({'None': 2030})
all_data.GarageYrBlt = -(all_data.GarageYrBlt - 2010)/20
all_data.YearRemodAdd = (all_data.YearRemodAdd - 2010)/20
# YrSold  5 categories
all_data['YrSold']=all_data['YrSold'].replace({
        2006:'sold'+str(2006), 2007:'sold'+str(2007),
        2008:'sold'+str(2008), 2009:'sold'+str(2009),
                       2010:'sold'+str(2010)})
# YrSoBu, YrSoRecon /10
all_data.YrSoBu = all_data.YrSoBu/10
all_data.YrSoRecon = all_data.YrSoRecon/10


# 4. get dummy variables
all_data = pd.get_dummies(all_data)

# 5. Scale teh data
t = all_data.quantile(.95)
use_max_scater = t[t == 0].index
use_95_scater = t[t != 0].index
all_data[use_max_scater] = all_data[use_max_scater]/all_data[use_max_scater].max()
all_data[use_95_scater] = all_data[use_95_scater]/all_data[use_95_scater].quantile(.95)


# 5 Reduce the Skewness of the numeric features and labels
num_features = all_data.dtypes[all_data.dtypes != 'object'].index
fea_skewness = all_data[num_features].apply(lambda x: stats.skew(x.dropna()))
skewed_fea = list(fea_skewness[fea_skewness > 0.75].index)
all_data[skewed_fea] = np.log1p(all_data[skewed_fea])

print("There are {} NAs in the label '{}'".format(y_train.isnull().sum(), 'SalePrice'))
print("The skewness of the label '{0:s}' is: {1:0.5f}".
      format('SalePrice', y_train.skew()))
print("Get the log price")
y_train = np.log1p(y_train)
print("The skewness of the logged label '{0:s}' is: {1:0.5f}".
     format('logSalePrice', y_train.skew()))



# 7. Feature interactions
from itertools import product

def poly(X):
    areas = ['LotArea', 'LowQualFinSF', 'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'BsmtUnfSF', 'KitchenAbvGr',
             'TotRmsAbvGrd', 'MiscVal']
    t = ['OverallQual_num', 'OverallCond_num', 'ExterQual_num', 'ExterCond_num',
               'BsmtCond_num', 'GarageQual_num', 'GarageCond_num', 'KitchenQual_num', 
               'HeatingQC_num', 'MiscFeature']
    for a, t in product(areas, t):
        x = X.loc[:, [a, t]].prod(1)
        x.name = a + '_' + t
        yield x

XP = pd.concat(poly(all_data.copy()), axis=1)
df = pd.concat((all_data.copy(), XP), axis=1)

df.shape

# 8 Outliers
sns.jointplot(x=train.GrLivArea, y = train.SalePrice)
# 2 points with extremely large GrLivArea, but very low SalePrice
outlier_index = train[(train['GrLivArea'] > 4000) & (train['SalePrice']<300000)].index
all_data = all_data.drop(outlier_index, axis=0)
y_train = y_train.drop(outlier_index, axis=0)
indices = list(range(0, 2917))
all_data.index = indices
y_train.index = indices[0:len(y_train)]

# 9. Separate into training and test set
X_train = df.iloc[0:1458, :]
X_test = df.iloc[1458:,:]

X_train.to_csv('X_train.csv', index = False)
X_test.to_csv('X_test.csv', index = False)
y_train.to_csv('y_train.csv', header = ['lgSalePrice'], index=False)
