import math
import random
import numpy as np
import pandas as pd
import yfinance as yf

from scipy.stats import norm
from scipy.stats import uniform
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

random.seed(0)

tick = yf.Ticker('NQ=F')
data = tick.history(period="max")
data.reset_index(inplace=True)
data.Date = data.Date.dt.date


# functions for option prices
N = norm.cdf
def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T)* N(d2)


def BS_PUT(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    return K*np.exp(-r*T)*N(-d2) - S*N(-d1)


# add daily HPR and 250 day volatility
data['dailyR'] = data.Close/data.Close.shift(1)-1
data['dailyV'] = data.dailyR.rolling(250).std()
data.dropna(inplace=True)
data.reset_index(inplace=True, drop=True)

# define parameter ranges to simulate inputs
# risk_free_range = np.arange(.01,.06,0.005)
# expiration_range = np.arange(25,150,25)
# moneyness_range = np.arange(-.10,.10,0.004)

risk_free_range = [.03]
expiration_range = [20]
moneyness_range = [0]

# compute option prices
output = []
for i in data.index:
    risk_free = random.choice(risk_free_range)/250
    expiration = random.choice(expiration_range)
    moneyness = 1 + random.choice(moneyness_range)

    volatility = data.dailyV[i]
    close = data.Close[i]
    strike = close*moneyness

    call_price = BS_CALL(close, strike, expiration, risk_free, volatility)
    put_price = BS_PUT(close, close, expiration, risk_free, volatility)

    output.append([risk_free,expiration,moneyness,strike,call_price,put_price])

output_df = pd.DataFrame(output)
output_df.columns = ['risk_free','expiration','moneyness','strike','call_price','put_price']

# create final dataset for training and testing
full_data = pd.concat([data[['Date','Close','dailyV']],output_df], axis=1)

test_size = .05
train_end = int(len(full_data)*(1-test_size))

train_data = full_data.loc[:train_end].copy()
test_data = full_data.loc[train_end+1:].copy()

# define x y data sets
X_train = train_data[['Close','strike','moneyness','expiration','risk_free','dailyV']].to_numpy()
Y_train_call = train_data['call_price'].to_numpy()
Y_train_put = train_data['put_price'].to_numpy()

X_test = test_data[['Close','strike','moneyness','expiration','risk_free','dailyV']].to_numpy()
Y_test_call = test_data['call_price'].to_numpy()
Y_test_put = test_data['put_price'].to_numpy()

# scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## fit predict with ML models
# K-nearest neighbors
knn_search = RandomizedSearchCV(KNeighborsRegressor(), {'leaf_size':np.arange(5,100,5),'n_neighbors':np.arange(5,100,5)})
knn_search.fit(X_train,Y_train_call)
call_predict = knn_search.predict(X_test)
knn_call_df = pd.DataFrame(call_predict-Y_test_call, columns=['knn_call'])

knn_search.fit(X_train,Y_train_put)
put_predict = knn_search.predict(X_test)
knn_put_df = pd.DataFrame(put_predict-Y_test_put, columns=['knn_put'])

# Multi Layer Perceptron
mlp_search = RandomizedSearchCV(MLPRegressor(random_state=0,max_iter=500,solver='lbfgs',tol=1e-8), 
                                {'alpha':[0.01,0.001,0.0001],'hidden_layer_sizes':[(5,5,5,),(5,),(5,5,)]})
mlp_search.fit(X_train,Y_train_call)
call_predict = mlp_search.predict(X_test)
mlp_call_df = pd.DataFrame(call_predict-Y_test_call, columns=['mlp_call'])

mlp_search.fit(X_train,Y_train_put)
put_predict = mlp_search.predict(X_test)
mlp_put_df = pd.DataFrame(put_predict-Y_test_put, columns=['mlp_put'])

# LightGBM
lgb_search = RandomizedSearchCV(LGBMRegressor(boosting_type='goss',subsample=0.8,colsample_bytree=0.8),
                                {'n_estimators':np.arange(50,1000,50),'max_depth':np.arange(3,10,1)})
lgb_search.fit(X_train,Y_train_call)
call_predict = lgb_search.predict(X_test)
lgb_call_df = pd.DataFrame(call_predict-Y_test_call, columns=['lgb_call'])

lgb_search.fit(X_train,Y_train_put)
put_predict = lgb_search.predict(X_test)
lgb_put_df = pd.DataFrame(put_predict-Y_test_put, columns=['lgb_put'])

# Support Vector Machines
svr_search = RandomizedSearchCV(SVR(),{'C':np.arange(.01,1,.01),'gamma':np.arange(.01,1,.01)})
svr_search.fit(X_train,Y_train_call)
call_predict = svr_search.predict(X_test)
svr_call_df = pd.DataFrame(call_predict-Y_test_call, columns=['svr_call'])

svr_search.fit(X_train,Y_train_put)
put_predict = svr_search.predict(X_test)
svr_put_df = pd.DataFrame(put_predict-Y_test_put, columns=['svr_put'])

# save ML model results in excel file
test_df = test_data[['Date','call_price','put_price']].copy()
test_df.reset_index(inplace=True, drop=True)
ml_results = pd.concat([test_df,
                        knn_call_df,mlp_call_df,lgb_call_df,svr_call_df,
                        knn_put_df,mlp_put_df,lgb_put_df,svr_put_df], axis=1)
ml_results.to_excel('ml_results.xlsx', index=False)


# function for option to expand
def option_to_expand(T,S,sig,r,N,expand,cost):

    dt=T/N
    dxu=math.exp(sig*math.sqrt(dt))
    dxd=1/dxu
    pu=((math.exp(r*dt))-dxd)/(dxu-dxd)
    pd=1-pu
    disc=math.exp(-r*dt)

    St = [0] * (N+1)
    V = [0] * (N+1)

    St[0]=S*dxd**N

    for j in range(1, N+1): 
        St[j] = St[j-1] * dxu/dxd

    for j in range(1, N+1):
        V[j] = max(St[j]*expand-cost,St[j])

    for i in range(N, 0, -1):
        for j in range(0, i):
            V[j] = disc*(pu*V[j+1]+pd*V[j])

    return V[0]


r=0.05
T=3
N=3
expand=1.5
cost=.5
S = np.arange(10,1000,10)
sig = np.arange(.2,.5,.01)

expand_options = []
for i in range(10000):
    i_S = random.choice(S)
    i_sig = random.choice(sig)
    expansion = option_to_expand(T=T,S=i_S,sig=i_sig,r=r,N=N,expand=expand,cost=i_S*cost)
    expand_options.append([i_S,i_sig,expansion])

expand_df = pd.DataFrame(expand_options)
expand_df.columns = ['S','sig','expand']

# create final dataset for training and testing
test_size = .05
train_end = int(len(expand_df)*(1-test_size))

train_data = expand_df.loc[:train_end].copy()
test_data = expand_df.loc[train_end+1:].copy()

# define x y data sets
X_train = train_data[['S','sig']].to_numpy()
Y_train = train_data['expand'].to_numpy()
X_test = test_data[['S','sig']].to_numpy()
Y_test = test_data['expand'].to_numpy()

# scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## fit predict with ML models
# K-nearest neighbors
knn_search = RandomizedSearchCV(KNeighborsRegressor(), {'leaf_size':np.arange(5,100,5),'n_neighbors':np.arange(5,100,5)})
knn_search.fit(X_train,Y_train)
predict = knn_search.predict(X_test)
knn_expand_df = pd.DataFrame(predict-Y_test, columns=['knn_expand'])

# Multi Layer Perceptron
mlp_search = RandomizedSearchCV(MLPRegressor(random_state=0,max_iter=500,solver='lbfgs',tol=1e-8), 
                                {'alpha':[0.01,0.001,0.0001],'hidden_layer_sizes':[(5,5,5,),(5,),(5,5,)]})
mlp_search.fit(X_train,Y_train)
predict = mlp_search.predict(X_test)
mlp_expand_df = pd.DataFrame(predict-Y_test, columns=['mlp_expand'])

# LightGBM
lgb_search = RandomizedSearchCV(LGBMRegressor(boosting_type='goss',subsample=0.8,colsample_bytree=0.8),
                                {'n_estimators':np.arange(50,1000,50),'max_depth':np.arange(3,10,1)})
lgb_search.fit(X_train,Y_train)
predict = lgb_search.predict(X_test)
lgb_expand_df = pd.DataFrame(predict-Y_test, columns=['lgb_expand'])

# Support Vector Machines
svr_search = RandomizedSearchCV(SVR(),{'C':np.arange(.01,1,.01),'gamma':np.arange(.01,1,.01)})
svr_search.fit(X_train,Y_train)
predict = svr_search.predict(X_test)
svr_expand_df = pd.DataFrame(predict-Y_test, columns=['svr_expand'])

# save ML model results in excel file
test_df = test_data.copy()
test_df.reset_index(inplace=True, drop=True)
test_df.insert(loc=0,value=cost,column='expand cost')
test_df.insert(loc=0,value=expand,column='expand factor')
test_df.insert(loc=0,value=N,column='N steps')
test_df.insert(loc=0,value=T,column='T years')
test_df.insert(loc=0,value=r,column='risk-free')

expand_results = pd.concat([test_df,knn_expand_df,mlp_expand_df,lgb_expand_df,svr_expand_df,], axis=1)
expand_results.to_excel('expansion_results.xlsx', index=False)



print('DONE')
# references for borrowed code:
#     - https://www.codearmo.com/python-tutorial/options-trading-black-scholes-model
#     - https://github.com/joseprupi/pyOptionPricing/blob/master/binomialTree.py