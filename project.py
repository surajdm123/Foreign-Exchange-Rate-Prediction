#!/usr/bin/env python
# coding: utf-8

# # Project Group: 36
# ## Names:
# ### 1. Suraj Mallikarjuna Devatha (sdevath)
# ### 2. Sruthi Vandhana
# ### 3. Akhil

# In[1]:


#Data Visualisation and Manipulation Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

#Preprocessing
from sklearn.model_selection import train_test_split

#Libraries needed for modeling
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn import metrics

#Libraries needed for metrics
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

#Libraries needed for stats
import scipy
import statsmodels
from scipy.stats import boxcox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.seasonal import seasonal_decompose

import warnings 
warnings.filterwarnings('ignore')


# #### READ THE DATA

# In[2]:


dataframe = pd.read_csv('data/USDINR_series.csv')
print('Shape of the dataset: ', dataframe.shape)
dataframe.head()


# In[3]:


dataframe = dataframe[['Date', 'USDINR_Adj Close']]
dataframe.columns = ['date',	'rate']
dataframe.head()
dataframe = dataframe.dropna()

data = dataframe


# In[4]:


#Converting data to timestamp -> time series data
data.date = pd.to_datetime(data.date)


# ## Exploratory Data Analysis
# ---

# In[5]:


#Yahoo Finance will contain null values, they have to be filtered.
data = data.drop(data[data['rate']=='null'].index)

data['rate'] = pd.to_numeric(data.rate)
#Since this is time series data, the data should be arranged by date
data = data.sort_values('date', ascending=True)


# In[6]:


#show basic stats
data.rate.describe()


# In[7]:


#Data Distribution
plt.figure(figsize=(11,6))
sns.distplot(data.rate, bins=10, color='red');
plt.show()


# In[8]:


#Time Series Plot
figure = go.Figure()

figure.add_trace(go.Scatter(x=data.date, y=data.rate, marker_color='blue'))

figure.update_layout(title='Time Series plot of INR and USD Rate', 
                  height=450, width=1000, template='plotly_dark', font_color='green', 
                  font=dict(family="sans serif",
                            size=14,
                            color="white"
                            ))

figure.update_xaxes(title='Year')
figure.update_yaxes(title='Rate / INR')
figure.show()


# In[9]:


# This is the autocorrelation functional plot
figure, axes = plt.subplots(1,2,figsize=(14,4))
plot_acf(data.rate, lags=20, ax=axes[0]);
plot_pacf(data.rate, lags=20, ax=axes[1]);
plt.show()


# ## Modeling
# ---

# In[10]:


#This divides the dataset to training data and its target values
X_train, X_val = data[:-400], data[-400:]


# ### Auto Regressive Integrated Moving Average (ARIMA) Model

# In[11]:


pred = []

# Building the Arima model
arima = sm.tsa.statespace.SARIMAX(X_train.rate,order=(0,0,0),seasonal_order=(1,1,1,6),
                                  enforce_stationarity=False, enforce_invertibility=False,).fit()

pred.append(arima.forecast(10))

pred = np.array(pred).reshape((10,))


# In[12]:


# Prints the summary of the ARIMA model
arima.summary()


# #### COMPARISON OF TRUE VALUE AND ARIMA PREDICTIONS

# In[13]:


yValue = data.rate[-10:]
plt.figure(figsize=(14,5))
plt.plot(np.arange(len(yValue)), yValue, color='steelblue');
plt.plot(np.arange(len(yValue)), pred, color='salmon');
plt.legend(['True Value', 'Prediction']);
plt.show()


# ### Metrics

# In[14]:


mae = mean_absolute_error(yValue, pred)
mse = mean_squared_error(yValue, pred)
rmse = np.sqrt(mean_squared_error(yValue, pred))

print('Mean Absolute Error:   ', mae)
print('Mean Squared Error:   ', mse)
print('Root Mean Squared Error:   ', rmse)


# In[15]:


error_rate = abs(((yValue - pred) / yValue).mean()) * 100
print('Mean Absolute Percentage Error:', round(error_rate,2), '%')


# In[16]:


print('R2 Score: ', r2_score(yValue, pred))


# ### XGBOOST

# In[17]:


data['day'] = data.date.dt.day
data['dayofweek'] = data.date.dt.dayofweek
data['dayofyear'] = data.date.dt.dayofyear
data['week'] = data.date.dt.week
data['month'] = data.date.dt.month
data['year'] = data.date.dt.year


# In[18]:


for i in range(1,8):
    data['lag'+str(i)] = data.rate.shift(i).fillna(0)


# In[19]:


data.drop('date', axis=1, inplace=True)
data.head(10)


# In[20]:


X = data.drop('rate', axis=1)
y = data.rate

X_train, X_test = X[:-10], X[-10:]
y_train, y_test = y[:-10], y[-10:]


# In[21]:


d_train_matrix = xgb.DMatrix(X_train,label=y_train)
d_test_matrix = xgb.DMatrix(X_test)


# In[22]:


def xgbEvaluate(max_depth, gamma, colsample_bytree):
    parameters = {'eval_metric': 'rmse',
              'max_depth': int(max_depth),
              'subsample': 0.8,
              'eta': 0.1,
              'gamma': gamma,
              'colsample_bytree': colsample_bytree}
    
    cvResult = xgb.cv(parameters, d_train_matrix, num_boost_round=250, nfold=3)    
    return -1.0 * cvResult['test-rmse-mean'].iloc[-1]


# In[23]:


xgb_boost_model = BayesianOptimization(xgbEvaluate, {'max_depth': (3, 7), 
                                             'gamma': (0, 1),
                                             'colsample_bytree': (0.3, 0.9)})
xgb_boost_model.maximize(init_points=10, n_iter=15, acq='ei')


# In[24]:


paramaters = xgb_boost_model.max['params']
paramaters['max_depth'] = int(round(paramaters['max_depth']))


#training the model with data
model = xgb.train(paramaters, d_train_matrix, num_boost_round=200)


# In[25]:


#predicting the test data 
preds = model.predict(d_test_matrix)


# In[26]:


yValue = data.rate[-10:]
plt.figure(figsize=(15,6))
plt.plot(np.arange(len(yValue)), yValue, color='blue');
plt.plot(np.arange(len(yValue)), preds, color='red');
plt.legend(['True Value', 'Prediction']);
plt.show()


# In[27]:


mae = mean_absolute_error(yValue, preds)
mse = mean_squared_error(yValue, preds)
rmse = np.sqrt(mean_squared_error(yValue, preds))

print('Mean Absolute Error:   ', mae)
print('Mean Squared Error:   ', mse)
print('Root Mean Squared Error:   ', rmse)


# In[28]:


error_rate = abs(((yValue - preds) / yValue).mean()) * 100
print('Mean Absolute Percentage Error:', round(error_rate,2), '%')


# In[29]:


print('R2 Score: ', r2_score(yValue, preds))

