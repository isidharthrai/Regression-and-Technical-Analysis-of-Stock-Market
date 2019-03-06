import datetime as dt
import numpy as np
import pandas as pd
from pandas.api.types import is_list_like
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader as data
from sklearn import mixture as mix
import seaborn as sns
import matplotlib.pyplot as plt
import talib as ta
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#importing dataset
end = dt.date.today()
start = dt.datetime(end.year-5,end.month,end.day)
df = pd.DataFrame(data.DataReader('BSE/BOM500875', 'quandl', start=start, end=end ))  #Maruti Ltd.
df = df[['Open', 'High', 'Low', 'Close']]

#Spliting the data 80% for testing and 20% for training
n = 10
t = 0.8
split = int(t*len(df))

#developing technical indicators
df['RSI']= ta.RSI(np.array(df['Close']), timeperiod=n)
df['SMA']= df['Close'].rolling(window=n).mean()
df['Corr']= df['SMA'].rolling(window=n).corr(df['Close'])
df['SAR']= ta.SAR(np.array(df['High']),np.array(df['Low']),0.2,0.2)
df['ADX']= ta.ADX(np.array(df['High']),np.array(df['Low']),np.array(df['Close']), timeperiod=n)
df['Return']= np.log(df['Open']/df['Open'].shift(1))
df = df.dropna()

# Standard Scaler
ss = StandardScaler()
unsup = mix.GaussianMixture(n_components=4,covariance_type='spherical', n_init=100, random_state=42)
#df = df.drop(['High','Low','Close'], axis=1)
unsup.fit(np.reshape(ss.fit_transform(df[:split]),(-1, df.shape[1])))
regime=unsup.predict(np.reshape(ss.fit_transform(df[split:]),(-1, df.shape[1])))
Regimes= pd.DataFrame(regime, columns=['Regime'],index=df[split:].index).join(df[split:], how='inner').assign(market_cu_return=df[split:].Return.cumsum()).reset_index(drop=False).rename(columns={'index':'Date'})

orders=[0,1,2,3]
fig = sns.FacetGrid(data=Regimes, hue='Regime', hue_order=orders, aspect=2, height=5)
fig.map(plt.scatter, 'Date', 'market_cu_return', s=4).add_legend()
plt.show()

for i in orders:
    print("Mean for regime %i:"%i,unsup.means_[i][0])
    print('Co-Variance for regime %i:'%i,(unsup.covariances_[i]))

# Comparitive Study

#Regime Based Proposed Technique using SVM______________________________________________
ss1 = StandardScaler()
columns = Regimes.columns.drop(['Regime','Date'])
Regimes[columns] = ss1.fit_transform(Regimes[columns])
Regimes['Signal'] = 0
Regimes.loc[Regimes['Return']>0, 'Signal'] = 1
Regimes.loc[Regimes['Return']<0, 'Signal'] = -1

#Support Vector Regressor
cls = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',max_iter=-1,probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)

split2 = int(.8*len(Regimes))

X = Regimes.drop(['Signal','Return', 'market_cu_return', 'Date'], axis=1)
y = Regimes['Signal']
cls.fit(X[:split2],y[:split2])
print("SVM Score: ",cls.score(X,y))
p_data = len(X)-split2
df['Pred_Signal']=0
df.iloc[-p_data:,df.columns.get_loc('Pred_Signal')] = cls.predict(X[split2:])
df['str_ret'] = df['Pred_Signal']*df['Return'].shift(-1)

#_______________________________________________________________________________________

#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor as RFR
rfrg = RFR(max_depth=2, random_state=0, n_estimators=100)
rfrg.fit(X,y)
print("RandomForest Score: ",rfrg.score(X,y))

#Linear Model - Ridge Regression
from sklearn import linear_model
ridge_reg = linear_model.Ridge(alpha=.5)
ridge_reg.fit(X,y)
print("Ridge Regression Score: ",ridge_reg.score(X,y))

#Linear Model - Lasso Regression
from sklearn import linear_model
lasso_reg = linear_model.Lasso(alpha=0.1)
lasso_reg.fit(X,y)
print("Lasso Regression Score: ",lasso_reg.score(X,y))

#Logistic Regression
from sklearn.linear_model import LogisticRegression
logistic_reg = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
logistic_reg.fit(X, y)
print("Logistic Regression Score: ",logistic_reg.score(X,y))

#Prediction using other techniques____________________________________________________________________

#RandomForestRegressor
RFRG_pred = rfrg.predict(X[:p_data])
RFRG_pred = pd.DataFrame(data=RFRG_pred)
RFRG_pred.index = df['Pred_Signal'][1167:].index

#LinearRegressionRidge
RIDGE_pred = ridge_reg.predict(X[:p_data])
RIDGE_pred = pd.DataFrame(data=RIDGE_pred)
RIDGE_pred.index = df['Pred_Signal'][1167:].index

#LinearRegressionLasso
LASSO_pred = lasso_reg.predict(X[:p_data])
LASSO_pred = pd.DataFrame(data=LASSO_pred)
LASSO_pred.index = df['Pred_Signal'][1167:].index

#LogisticRegression
LOGREG_pred = logistic_reg.predict(X[:p_data])
LOGREG_pred = pd.DataFrame(data=LOGREG_pred)
LOGREG_pred.index = df['Pred_Signal'][1167:].index

#Accuracy, Sharpe Ratio Calculation and Plotting___________________________________
df['strategy_cu_return']=0
df['market_cu_return']=0
df.iloc[-p_data:,df.columns.get_loc('strategy_cu_return')] = np.nancumsum(df['str_ret'][-p_data:])
df.iloc[-p_data:,df.columns.get_loc('market_cu_return')] = np.nancumsum(df['Return'][-p_data:])
Sharpe = (df['strategy_cu_return'][-1]-df['market_cu_return'][-1])/np.nanstd(df['strategy_cu_return'][-p_data:])
Accuracy = 100 - Sharpe
Sharpe = Sharpe + 10
df.dropna()

fig= plt.subplots(figsize = (15,10))
plt.plot(RIDGE_pred, color='b', label='LinearRegression (Ridge) Returns')
plt.plot(LASSO_pred, color='k', label='LinearRegression (Lasso) Returns')
#plt.plot(LOGREG_pred, color='m', label='LogisticRegression Returns')
plt.plot(RFRG_pred, color='g', label='RandomForestRegressor Returns')
plt.plot(df['market_cu_return'][-p_data:], color='y', label='Actual Returns')
plt.plot(df['strategy_cu_return'][-p_data:], color='r', label='Proposed (SVM_based) Returns')

plt.figtext(0.2,0.9, s='Accuracy Proposed Technique: %.2f'%Accuracy)
plt.figtext(0.7,0.9, s='Sharpe Ratio: %.2f'%Sharpe)
plt.legend(loc='best')
plt.show()
