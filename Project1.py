#!/usr/bin/env python
# coding: utf-8

# # Project 1 - Energy Services
# Electricity forecasting for the Central Building of the IST Campus. <br>
# Alfred Ernest Reinier Arnold - 98023 <br>
# email: rein.arnold@quicknet.nl

# ## Data Collection and Data Cleaning

# #### Import libraries

# In[1481]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


# #### Import data files , set date columns, join power consumption data

# In[1482]:


raw_data_cons_2017 = pd.read_csv('IST_Central_Pav_2017_Ene_Cons.csv')
raw_data_cons_2017.rename(columns={'Date_start':'date_time'}, inplace=True)
raw_data_cons_2017['date_time'] = pd.to_datetime(raw_data_cons_2017['date_time']) 

raw_data_cons_2018 = pd.read_csv('IST_Central_Pav_2018_Ene_Cons.csv')
raw_data_cons_2018.rename(columns={'Date_start':'date_time'}, inplace=True)
raw_data_cons_2018['date_time'] = pd.to_datetime(raw_data_cons_2018['date_time'])

raw_data_cons = pd.concat([raw_data_cons_2017,raw_data_cons_2018])

raw_data_weather = pd.read_csv('IST_meteo_data_2017_2018_2019.csv')
raw_data_weather.rename(columns={'yyyy-mm-dd hh:mm:ss':'date_time'}, inplace=True)
raw_data_weather['date_time'] = pd.to_datetime(raw_data_weather['date_time']) 

raw_data_holidays = pd.read_csv('holiday_17_18_19.csv')
raw_data_holidays['Date'] = pd.to_datetime(raw_data_holidays['Date'],dayfirst=True) 


# ##### Create column for Power last hour (Power-1), Power last day (Power_ld) and Power last week (Power_lw)

# In[1483]:


raw_data_cons['Power-1'] = raw_data_cons['Power_kW'].shift(1)
raw_data_cons['Power_ld'] = raw_data_cons['Power_kW'].shift(24)
raw_data_cons['Power_lw'] = raw_data_cons['Power_kW'].shift(168)
raw_data_cons = raw_data_cons.iloc[168:]


# ##### Create columns for holiday, hour , day of week and work day

# In[1484]:


raw_data_holidays.set_index('Date', inplace=True)
raw_data_holidays_hourly = raw_data_holidays.resample('H',origin='1/1/2017',convention='start').fillna(method='ffill',limit=23)
data_days = raw_data_holidays_hourly.fillna(value=0)
data_days['Hour'] = (data_days.index.hour)
data_days['Day_of_week'] = pd.to_datetime(data_days.index).dayofweek

#create function to check Work_day
def workday(df):
    if (df['Day_of_week'] >= 5) or (df['Holiday'] == 1):
        return 0
    else:
        return 1
    
data_days['Work_day'] = data_days.apply(workday, axis=1)


# #### Add column for daily period (sine wave), as an alternative for hour of the day

# In[1485]:


period = np.linspace(0,2*math.pi - math.pi/12,24)
period_wave_day = - np.cos(period) 
#extend to 2 years
period_wave = np.tile(period_wave_day,365*2)

dates = pd.date_range(start='1-1-2017',periods=17520,freq='H')

period_df = pd.DataFrame(period_wave)
period_df.index = dates
period_df.columns=['Period']


# #### Resample weather data to hourly data (take mean of data points) and remove missing values

# In[1486]:


raw_data_weather_hourly = raw_data_weather.resample('H', on='date_time').mean()
data_weather_hourly = raw_data_weather_hourly.dropna()


# ##### Create DataFrame with all data

# In[1487]:


all_data = pd.merge(raw_data_cons,data_weather_hourly,on='date_time')
all_data.set_index('date_time', inplace=True)
all_data = pd.merge(all_data,data_days,left_index=True,right_index=True)
all_data = pd.merge(all_data,period_df,left_index=True,right_index=True)


# #### Check for infeasible values

# In[1488]:


all_data.describe()


# All values in table are realistic.

# ## Clustering

# In[1489]:


from sklearn.cluster import KMeans 


# In[1490]:


Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(all_data).score(all_data) for i in range(len(kmeans))]
score
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.savefig('assets/elbow.png', bbox_inches='tight', dpi=200)
plt.clf()


# On the basis of above graph the number of clusters is decided at 3, as the relative gain in score is low for n>3. For the clustering purpose not all variables are regarded as this results in too much noise in the clustering.  

# In[1491]:


model = KMeans(n_clusters=3).fit(all_data[['Power_kW','Day_of_week','Hour','Holiday','temp_C']])
pred = model.labels_


# In[1492]:


all_data['Cluster']=pred
all_data


# ### Cluster Analysis

# In[1493]:

colormap = np.array(['g', 'r', 'b'])

ax1=all_data.plot.scatter(x='Power_kW',y='solarRad_W/m2',c=colormap[all_data['Cluster']],)
plt.savefig('assets/cluster_a.png', bbox_inches='tight', dpi=200)
plt.clf()

# In[1494]:


ax1=all_data.plot.scatter(x='Power_kW',y='Day_of_week',c=colormap[all_data['Cluster']],)
plt.savefig('assets/cluster_b.png', bbox_inches='tight', dpi=200)
plt.clf()

# In[1495]:


ax3=all_data.plot.scatter(x='Hour',y='Power_kW',c=colormap[all_data['Cluster']],)
plt.savefig('assets/cluster_c.png', bbox_inches='tight', dpi=200)
plt.clf()

# In[1496]:


ax4=all_data.plot.scatter(x='Hour',y='Day_of_week',c=colormap[all_data['Cluster']],)
plt.savefig('assets/cluster_d.png', bbox_inches='tight', dpi=200)
plt.clf()

# In[1497]:


fig = plt.figure()
ax = plt.axes(projection="3d")


cluster_0=all_data[pred==0]
cluster_1=all_data[pred==1]
cluster_2=all_data[pred==2]


cluster_0
ax.scatter3D(cluster_0['Hour'], cluster_0['Day_of_week'],cluster_0['Power_kW'],c='green');
ax.scatter3D(cluster_1['Hour'], cluster_1['Day_of_week'],cluster_1['Power_kW'],c='red');
ax.scatter3D(cluster_2['Hour'], cluster_2['Day_of_week'],cluster_2['Power_kW'],c='blue');

plt.savefig('assets/cluster_3d.png', bbox_inches='tight', dpi=200)
plt.clf()


# In the 3D Graph, the base load and peak power consumption is clearly visible. 

# In[1]:


df=all_data
df=df[['Power_kW','Hour']]
df.rename(columns = {'Power_kW': 'Power'}, inplace = True)
df.index = df.index.normalize()

#Create a pivot table
df_pivot = df.pivot(columns='Hour')
df_pivot = df_pivot.dropna()

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

sillhoute_scores = []
n_cluster_list = np.arange(2,10).astype(int)

X = df_pivot.values.copy()

sc = MinMaxScaler()
X = sc.fit_transform(X)

for n_cluster in n_cluster_list:
    
    kmeans = KMeans(n_clusters=n_cluster)
    cluster_found = kmeans.fit_predict(X)
    sillhoute_scores.append(silhouette_score(X, kmeans.labels_))
    


# In[1499]:


kmeans = KMeans(n_clusters=3)
cluster_found = kmeans.fit_predict(X)
cluster_found_sr = pd.Series(cluster_found, name='cluster')
print(cluster_found_sr.value_counts())
df_pivot = df_pivot.set_index(cluster_found_sr, append=True )


# In[1500]:


fig, ax= plt.subplots(1,1, figsize=(18,10))
color_list = ['green','red','blue']
cluster_values = sorted(df_pivot.index.get_level_values('cluster').unique())
for cluster, color in zip(cluster_values, color_list):
    df_pivot.xs(cluster, level=1).T.plot(
        ax=ax, legend=False, alpha=0.01, color=color, label= f'Cluster {cluster}'
        )
    df_pivot.xs(cluster, level=1).median().plot(
        ax=ax, color=color, alpha=0.9, ls='--'
    )

ax.set_xticks(np.arange(1,25))
ax.set_ylabel('Kilowatt')
ax.set_xlabel('Hour')
plt.savefig('assets/cluster_set.png', bbox_inches='tight', dpi=200)
plt.clf()

# ## Feature selection

# ##### Import required packages

# In[1501]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


# ##### Convert DataFrame to Array

# In[1502]:


X = all_data.values
Y = X[:,0]
X = X[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]


# ### Check features with various methods

# In[1503]:


all_data.describe()


# ##### Random Forest

# In[1504]:


model = RandomForestRegressor()
model.fit(X,Y)
print(model.feature_importances_)


# ##### kBest

# In[1505]:


features=SelectKBest(k=5,score_func=f_regression) # Test different k number of features, uses f-test ANOVA
fit=features.fit(X,Y) #calculates the f_regression of the features
print(fit.scores_)
#Power-1	Power_ld	Power_lw	temp_C	
#HR	windSpeed_m/s	windGust_m/s	pres_mbar	
#solarRad_W/m2	rain_mm/h	rain_day	Holiday	
#Hour	Day_of_week	Work_day	Period


# Random Forest and kBest have similar results. 

# ### Feature selection

# Power-1 is the most correlated variable, according to Random Forest and kBest. However, this limits the scope of the forecasting model to only one hour, as the Power from the last hour is needed as input. 
# 
# Therefore, a second model is made, the daily model. which uses the Power consumption in the same hour in the last day and in same day of the previous week (Power_ld and Power_lw). This increases the scope of the model to a complete day. 
# 
# For both the hourly and the daily model, the rain and wind data is not considered, as the correlation is not sufficient. The pressure, relative humidity, temperature and solar radiation data are used in both models.
# 
# <b>Engineered functions</b> 
# The daily model was slightly improved by the addition of various engineered functions: the Work_day function (combination of Holiday and Day_of_week), and the Period function, which is a generated sine wave going from -1 at midnight to 1 at noon. These engineered functions unfortunately did not improve the hourly model, and are therefore not used in this model. 
# 

# In[1506]:


#0 Power-1	1 Power_ld	2 Power_lw	3 temp_C	4 HR	5 windSpeed_m/s	6 windGust_m/s	7 pres_mbar	8 solarRad_W/m2	9 rain_mm/h	10 rain_day	11 Holiday	12 Hour	13 Day_of_week	14 Work_day	15 Perio

#For Model 1(Hourly Model)
X = X[:,[0,3,4,7,8,11,12,13]]

#For Model 2(Daily Model)
#X = X[:,[1,2,3,4,7,8,11,13,14,15]]



# ## Regression

# #### Import required packages

# In[1507]:


from sklearn.model_selection import train_test_split
from sklearn import  metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor


# #### Split training and test data

# In[1508]:


#by default, it chooses randomly 75% of the data for training and 25% for testing
X_train, X_test, y_train, y_test = train_test_split(X,Y)


# ### Linear Regression

# In[1509]:


from sklearn import  linear_model

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train,y_train)

# Make predictions using the testing set
y_pred_LR = regr.predict(X_test)


# In[1511]:


#Evaluate errors
MAE_LR=metrics.mean_absolute_error(y_test,y_pred_LR) 
MSE_LR=metrics.mean_squared_error(y_test,y_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y_test)


# ### Random Forest

# In[1512]:


parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 200, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 20,
              'max_leaf_nodes': None}
RF_model = RandomForestRegressor(**parameters)
#RF_model = RandomForestRegressor()
RF_model.fit(X_train, y_train)
y_pred_RF = RF_model.predict(X_test)


# In[1514]:


#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF) 
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)


# ### Random Forest (uniformized data)

# In[1515]:


scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 100, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 10,
              'max_leaf_nodes': None}

RF_model = RandomForestRegressor(**parameters)
RF_model.fit(X_train_scaled, y_train.reshape(-1,1))
y_pred_RF = RF_model.predict(X_test_scaled)


# In[1517]:


MAE_RFu=metrics.mean_absolute_error(y_test,y_pred_RF) 
MSE_RFu=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RFu= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RFu=RMSE_RFu/np.mean(y_test)



# ### Support Vector Regressor

# In[1518]:


sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_SVR = sc_X.fit_transform(X_train)
y_train_SVR = sc_y.fit_transform(y_train.reshape(-1,1))


# In[1519]:


regr = SVR(kernel='rbf')
regr.fit(X_train_SVR,y_train_SVR)


# In[1520]:


y_pred_SVR = regr.predict(sc_X.fit_transform(X_test))
y_test_SVR=sc_y.fit_transform(y_test.reshape(-1,1))
y_pred_SVR2=sc_y.inverse_transform(y_pred_SVR)


# In[1521]:


MAE_SVR=metrics.mean_absolute_error(y_test,y_pred_SVR2) 
MSE_SVR=metrics.mean_squared_error(y_test,y_pred_SVR2)  
RMSE_SVR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_SVR2))
cvRMSE_SVR=RMSE_SVR/np.mean(y_test)

# ### Decision Tree Regressor

# In[1522]:


from sklearn.tree import DecisionTreeRegressor

# Create Regression Decision Tree object
DT_regr_model = DecisionTreeRegressor()

# Train the model using the training sets
DT_regr_model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred_DT = DT_regr_model.predict(X_test)



# In[1524]:


#Evaluate errors
MAE_DT=metrics.mean_absolute_error(y_test,y_pred_DT) 
MSE_DT=metrics.mean_squared_error(y_test,y_pred_DT)  
RMSE_DT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_DT))
cvRMSE_DT=RMSE_DT/np.mean(y_test)


# ### Gradient Boosting

# In[1525]:


GB_model = GradientBoostingRegressor()
GB_model.fit(X_train, y_train)
y_pred_GB =GB_model.predict(X_test)

# In[1527]:


MAE_GB=metrics.mean_absolute_error(y_test,y_pred_GB) 
MSE_GB=metrics.mean_squared_error(y_test,y_pred_GB)  
RMSE_GB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_GB))
cvRMSE_GB=RMSE_GB/np.mean(y_test)

# ### Extreme Gradient Boosting

# In[1528]:


XGB_model = XGBRegressor()
XGB_model.fit(X_train, y_train)
y_pred_XGB =XGB_model.predict(X_test)



# In[1530]:


MAE_XGB=metrics.mean_absolute_error(y_test,y_pred_XGB) 
MSE_XGB=metrics.mean_squared_error(y_test,y_pred_XGB)  
RMSE_XGB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_XGB))
cvRMSE_XGB=RMSE_XGB/np.mean(y_test)


# ### Bootstrapping

# In[1531]:


BT_model = BaggingRegressor()
BT_model.fit(X_train, y_train)
y_pred_BT =BT_model.predict(X_test)


# In[1533]:


MAE_BT=metrics.mean_absolute_error(y_test,y_pred_BT) 
MSE_BT=metrics.mean_squared_error(y_test,y_pred_BT)  
RMSE_BT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_BT))
cvRMSE_BT=RMSE_BT/np.mean(y_test)


# ### Neural Networks

# In[1534]:


NN_model = MLPRegressor(hidden_layer_sizes=(10,10,10,10))
NN_model.fit(X_train,y_train)
y_pred_NN = NN_model.predict(X_test)


# In[1536]:


MAE_NN=metrics.mean_absolute_error(y_test,y_pred_NN) 
MSE_NN=metrics.mean_squared_error(y_test,y_pred_NN)  
RMSE_NN= np.sqrt(metrics.mean_squared_error(y_test,y_pred_NN))
cvRMSE_NN=RMSE_NN/np.mean(y_test)


