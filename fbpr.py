import pandas as pd
import numpy as np
import warnings
import scipy
from datetime import timedelta
from fbprophet import Prophet

# Forceasting with decompasable model
from pylab import rcParams
# import statsmodels.api as sm
# from statsmodels.tsa.stattools import adfuller

# For marchine Learning Approach
# from statsmodels.tsa.tsatools import lagmat
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Visualisation
import matplotlib.pyplot as plt
# import seaborn as sns

plt.style.use('fivethirtyeight')

warnings.filterwarnings('ignore')
# Load the data
train = pd.read_excel('ShenyangEcxel.xlsx')

# print(train.tail())

# train_flattened = pd.melt(train[list(train.columns[-42000:])+['No']], id_vars='No', var_name='date', value_name='PM_Taiyuanjie')
# train_flattened['date'] = train_flattened['date'].to_datetime(train_flattened['date'])
plt.figure(figsize=(50, 8))
mean_group = train[['PM_Taiyuanjie','date']].groupby(['date'])['PM_Taiyuanjie'].mean()
plt.plot(mean_group)
plt.title('Time Series - Average')
# plt.show()

times_series_means = pd.DataFrame(mean_group).reset_index(drop=False)
print(times_series_means)

# sns.set(font_scale=1)
df_date_index = times_series_means[['date','PM_Taiyuanjie']]
df_date_index = df_date_index.set_index('date')
df_prophet = df_date_index.copy()
df_prophet.reset_index(drop=False,inplace=True)
df_prophet.columns = ['ds','y']

m = Prophet()
m.fit(df_prophet)
future = m.make_future_dataframe(periods=300,freq='H')
forecast = m.predict(future)

m.plot(forecast)

plt.show()