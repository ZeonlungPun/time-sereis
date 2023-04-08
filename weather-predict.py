import pandas as pd
from neuralprophet import NeuralProphet
from matplotlib import pyplot as plt
import pickle

#load data
df=pd.read_csv('weatherAUS.csv')
melb=df[df['Location']=='Melbourne']
melb['Date']=pd.to_datetime(melb['Date'])
print('head:')
print(melb.head())
#plt.plot(melb['Date'],melb['Temp3pm'])
#plt.show()
melb['Year']=melb['Date'].apply(lambda x:x.year)
melb=melb[melb['Year']<=2015]
print(melb.columns)
#data=melb[['Date','Rainfall','WindGustSpeed','Humidity3pm','Pressure3pm','Temp3pm']]
data=melb[['Date','Temp3pm']]
#data.columns=['ds','Rainfall','WindGustSpeed','Humidity3pm','Pressure3pm','y']
data.columns=['ds','y']
data.dropna(inplace=True)
print(data.head())
#train model
m=NeuralProphet()
#point out the X
#m=m.add_lagged_regressor(names=list(data)[1:-1])
m.fit(data,freq='D',epochs=1000 )

#forecast away
future=m.make_future_dataframe(data,periods=900)
forecast=m.predict(future)
print(forecast)
#plot1=m.plot(forecast)
plt.plot(forecast['ds'],forecast['yhat1'])
plt.show()
plot2=m.plot_components(forecast )
plt.show()

#save model
with open('forecast.pkl','wb') as f:
    pickle.dump(m,f)