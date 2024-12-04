import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from plotly import graph_objs as go

import gzip
import warnings
warnings.filterwarnings('ignore')


# import streamlit as st
# import pandas as pd

# st.write("Namaste")
# st.write("Srinandana")
# st.text_input("Hegidya??")
st.title("Ola ride prediction")
df = pd.read_csv('train.csv')
#st.write(df)
shape = df.shape
#st.write(shape)#returns the shapr of the dataframe

#seperating the date and time
parts = df["datetime"].str.split(" ", n=2, expand=True)
df["date"] = parts[0]
df["time"] = parts[1].str[:2].astype('int')
df.head()
#st.write(df)

#seperating the date into day, month and year
parts = df["date"].str.split("-", n=3, expand=True)
df["day"] = parts[2].astype('int')
df["month"] = parts[1].astype('int')
df["year"] = parts[0].astype('int')
#st.write(df)

#function for checking whether it's a holiday or note
from datetime import datetime
import calendar


def weekend_or_weekday(year, month, day):


	d = datetime(year, month, day)
	if d.weekday() > 4:
		return 0
	else:
		return 1


df['weekday'] = df.apply(lambda x: weekend_or_weekday(x['year'], x['month'], x['day']), axis=1)

#checking whether AM or PM
def am_or_pm(x):
	if x > 11:
		return 1
	else:
		return 0


df['am_or_pm'] = df['time'].apply(am_or_pm)

#Function for assigning Indian holidays as 1
from datetime import date
import holidays


def is_holiday(x):

	india_holidays = holidays.country_holidays('IN')

	if india_holidays.get(x):
		return 1
	else:
		return 0


df['holidays'] = df['date'].apply(is_holiday)

#st.write(df)

#dropping unimportant coloumns
df.drop(['datetime', 'date'], axis=1, inplace=True)
#st.write(df)

#null elements
df.isnull().sum()

#EDA
features = ['day', 'time', 'month']

#line graphs
plt.subplots(figsize=(15, 10))
for i, col in enumerate(features):
	plt.subplot(2, 2, i + 1)
	df.groupby(col).mean()['count'].plot()
plt.show()
#st.pyplot(plt)

#Bar graphs
features = ['season', 'weather', 'holidays','am_or_pm', 'year', 'weekday']

plt.subplots(figsize=(20, 10))
for i, col in enumerate(features):
	plt.subplot(2, 3, i + 1)
	df.groupby(col).mean()['count'].plot.bar()
plt.show()
#st.pyplot(plt)

#distribution graphs
features = ['temp', 'windspeed']

plt.subplots(figsize=(15, 5))
for i, col in enumerate(features):
  plt.subplot(1, 2, i + 1)
  sb.distplot(df[col])
plt.show()
#st.pyplot(plt)

#another outlier graphs
features = ['temp', 'windspeed']

plt.subplots(figsize=(15, 5))
for i, col in enumerate(features):
  plt.subplot(1, 2, i + 1)
  sb.boxplot(df[col])

plt.show()
#st.pyplot(plt)

#dropping outliers
num_rows = df.shape[0] - df[df['windspeed']<32].shape[0]
#print(f'Number of rows that will be lost if we remove outliers is equal to {num_rows}.')
#st.write(f"Number of rows that will be lost if we remove outliers is equal to {num_rows}")

#dropping outliers in windspeed and humidity and removing registerd & time coloms
df.drop(['registered', 'time'], axis=1, inplace=True)
df = df[(df['windspeed'] < 32) & (df['humidity'] > 0)]

#Dropping the count
features = df.drop(['count'], axis=1)
target = df['count'].values

#Splitting the dataset and training
X_train, X_val, Y_train, Y_val = train_test_split(features,target,test_size = 0.1,random_state=22)
t_shape = X_train.shape
val_shape = X_val.shape
#st.write(t_shape, val_shape)

#Normalising
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

#model_selection
regr = RandomForestRegressor(n_estimators=100)
regr.fit(X_train,Y_train)
train_pred = regr.predict(X_train)

#val_pred = regr.predict(X_test)
#st.write(train_pred)
#st.write(df)

st.header('Input new data for prediction')

am_or_pm = 0
holidays =0
weekday =0
year = 2011
month = 4
day = 1
casual = 1
windspeed = st.slider("windspeed",min_value=int(df['windspeed'].min()),max_value=int(df['windspeed'].max()),value=int(df['windspeed'].mean()),disabled=False, label_visibility="visible",step=1)
humidity = st.slider("Humidity",min_value=int(df['humidity'].min()),max_value=int(df['humidity'].max()),value=int(df['humidity'].mean()),disabled=False, label_visibility="visible",step=1)
weather = 1
workingday = 1
holiday = 0
season = 1
temp = st.slider("temp",min_value=int(df['temp'].min()),max_value=int(df['temp'].max()),value=int(df['temp'].mean()),disabled=False, label_visibility="visible",step=1)
atemp = df['atemp'].mean()
# time = st.number_input('Time (in hours)', min_value=0, max_value=24, value=12)
# # month = st.number_input('Month', min_value=1, max_value=12, value=1)
#
# # Predict the count based on user input
if st.button('Predict'):
    new_data = np.array([[am_or_pm,temp,atemp,holidays,weekday,year,month,day,casual,windspeed,humidity,weather,workingday,holiday,season]])
    prediction = regr.predict(new_data)

    # Display the prediction
    st.write(f'The predicted count is: {int(prediction[0])}')
