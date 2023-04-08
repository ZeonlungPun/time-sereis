import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
from pandas.api.types import CategoricalDtype
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
def mean_absolute_percentage_error(y_true,y_pred):
    y_true,y_pred=np.array(y_true),np.array(y_pred)
    return np.mean(np.abs((y_true-y_pred)/y_true))*100

pjme=pd.read_csv('AEP_hourly.csv',index_col=[0],parse_dates=[0])

#time features
cat_type=CategoricalDtype(categories=['Monday','Tuesday','Wednesday','Thursday',
                                      'Friday','Saturday','Sunday'])
def create_features(df,label=None):
    df=df.copy()
    df['date']=df.index
    df['hour']=df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekday'] = df['date'].dt.day_name()
    df['weekday'] =df['weekday'].astype(cat_type)
    df['quarter']=df['date'].dt.quarter
    df['month']=df['date'].dt.month
    df['year']=df['date'].dt.year
    df['dayofyear']=df['date'].dt.dayofyear
    df['dayofmonth']=df['date'].dt.day
    df['weekofyear']=df['date'].dt.weekofyear
    df['date_offset']=(df.date.dt.month*100+df.date.dt.day-320)%1300
    df['season']=pd.cut(df['date_offset'],[0,300,602,900,1300],labels=['spring','summer','fall','winter'])
    x=df[ ['date','hour','dayofweek','quarter','month','year','dayofyear','dayofmonth',
           'weekofyear','weekday','season'] ]
    if label:
        y=df[label]
        return x,y
    return x
x,y=create_features(pjme,label='AEP_MW')
feature_and_target=pd.concat([x,y],axis=1)

print(feature_and_target)
split_date='1-Jan-2015'
train=pjme.loc[pjme.index<=split_date].copy()
test=pjme.loc[pjme.index>split_date].copy()
train_prohet=train.reset_index().rename(columns={'Datetime':'ds','AEP_MW':'y'})
test_prohet=test.reset_index().rename(columns={'Datetime':'ds','AEP_MW':'y'})

#adding holiday
cal=calendar()
holidays=cal.holidays(start=pjme.index.min(),end=pjme.index.max(),return_name=True)
ho_df=pd.DataFrame(data=holidays,columns=['holiday_name']).assign(holiday='USFederalHoliday')
ho_df=ho_df.reset_index().rename(columns={'index':'ds'})
print(ho_df)
model=Prophet(holidays=ho_df)
model.fit(train_prohet)
pred=model.predict(df=test_prohet)
mae=mean_absolute_percentage_error(y_true=test['AEP_MW'],y_pred=pred['yhat'])
print('mae:',mae)
#compare forecast with actuals
f,ax=plt.subplots(figsize=(5,5))
ax.scatter(test.index,test['AEP_MW'],color=['r'])
fig=model.plot(pred,ax=ax)
plt.show()

fig2=model.plot_components(pred)
plt.show()

future=model.make_future_dataframe(periods=365*24,freq='h',include_history=False)
forecast=model.predict(future)
print(forecast)