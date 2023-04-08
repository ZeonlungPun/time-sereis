import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

df=pd.read_csv('AEP_hourly.csv',index_col=[0],parse_dates=[0])

df.index=pd.to_datetime(df.index)

train=df.loc[df.index<'1-Jan-2015'].copy()
test=df.loc[df.index>='1-Jan-2015'].copy()

def create_features(df):

    df['hour']=df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter']=df.index.quarter
    df['month']=df.index.month
    df['year']=df.index.year
    df['dayofyear']=df.index.dayofyear
    return df

train=create_features(train)
test=create_features(test)

features=['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear']
target=['AEP_MW']

x_train=train[features]
y_train=train[target]
x_test=test[features]
y_test=test[target]

reg=xgb.XGBRegressor(early_stopping_rounds=50)
reg.fit(x_train,y_train,eval_set=[(x_train,y_train),(x_test,y_test)],
        verbose=True)

#feature importance
f1=pd.DataFrame(data=reg.feature_importances_,index=reg.feature_names_in_,
                columns=['importance'])
f1.sort_values('importance').plot(kind='barh',title='feature_importances')
plt.show()





#forecast
test['prediction']=reg.predict(x_test)
df=df.merge(test[['prediction']],how='left',left_index=True,right_index=True)
print(df)
ax=df[['AEP_MW']].plot(figsize=(15,5))
df['prediction'].plot(ax=ax,style='.')
plt.legend(['true','prediction'])
ax.set_title('raw data and prediction')
plt.show()