import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

def create_features(df):
    df=df.copy()
    df['hour']=df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter']=df.index.quarter
    df['month']=df.index.month
    df['year']=df.index.year
    df['dayofyear']=df.index.dayofyear
    return df

df=pd.read_csv('AEP_hourly.csv',index_col=[0],parse_dates=[0])
df.index=pd.to_datetime(df.index)
df=create_features(df)




#lag features
def add_lags(df):
    target_map=df['AEP_MW'].to_dict()
    df['lag1']=(df.index-pd.Timedelta('364 days')).map(target_map)
    df['lag2'] = (df.index - pd.Timedelta('728 days')).map(target_map)
    df['lag3'] = (df.index - pd.Timedelta('1092 days')).map(target_map)
    return df

#training use cross validation
tss=TimeSeriesSplit(n_splits=5,test_size=24*365*1,gap=24)
df=df.sort_index()
df=add_lags(df)
fold=0
preds=[]
scores=[]
for train_idx,val_idx in tss.split(df):
    train=df.iloc[train_idx]
    test=df.iloc[val_idx]

    train=create_features(train)
    test=create_features(test)

    features = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear','lag1','lag2','lag3']
    target='AEP_MW'

    x_train = train[features]
    y_train = train[target]
    x_test = test[features]
    y_test = test[target]

    reg = xgb.XGBRegressor(early_stopping_rounds=50)
    reg.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)],
            verbose=True)

    y_pred=reg.predict(x_test)
    preds.append(y_pred)
    score=np.sqrt((mean_squared_error(y_test,y_pred)))
    scores.append(score)

#create future dataframe
future=pd.date_range('2018-08-03','2019-08-01',freq='1h')
future_df=pd.DataFrame(index=future)
future_df['isFuture']=True
df['isFuture']=False
df_and_future=pd.concat([df,future_df])
df_and_future=create_features(df_and_future)
df_and_future=add_lags(df_and_future)
print(df_and_future)
future_w_features=df_and_future.query('isFuture').copy()

future_w_features['pred']=reg.predict(future_w_features[features])

future_w_features['pred'].plot(ms=1,lw=1,title='Future Prediction')
plt.show()
