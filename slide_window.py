import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input, Dense,Dropout
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D,BatchNormalization
from tensorflow.keras import regularizers,initializers
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
#normalize the time_step data: give back the ratio of the first step
def normalise_window(window_data,single_window=True):
    if single_window:#keep the shape
        window_data=window_data.reshape((-1,1))
    else:
        window_data=window_data
    # window_data:[seq_len, dim]  col:
    normalize_window=np.zeros_like(window_data)
    # every coloums normalize seperately
    for col_i in range(window_data.shape[1]):
        # [seq_len,1]
        normalize_col=np.zeros_like(window_data[:,col_i])
        for id,p in enumerate(window_data[:, col_i]):
            temp= float(p) /float(window_data[0,col_i])-1
            normalize_col[id]=temp
        normalize_window[:,col_i]=normalize_col
    # [seq_len, dim]
    return normalize_window


#generate slide window datasets
# [1--100]  1-50 --> 51 2-51-->52  3-52-->53  4-53-->54
def next_window(i,seq_len,normalise,single_window,data,predicted_cols):
    """
    data : [time_step,dim]  predicted_cols: the variable need to be predicted
    """
    window=data[i:i+seq_len] #[seq_len, dim] eg: [1,2,3,4,5] or [[1,2],[3,4],[5,6] ]
    if normalise:
        window=normalise_window(window,single_window)
    else:
        window=window
    x=window[:-1]  #[seq_len-1, dim]
    if single_window:
        y=window[-1]
    else:
        y=window[-1,[predicted_cols]]  # [, 1]
    return x,y


def get_train_data(seq_len,normalise,single,data,predicted_cols):
    """
    :param seq_len: the length of x and y
    :param normalise:   true means normalize the data in every sequence
    :param single:   dim=1 or not
    :param data:    [seq_len,dim]
    :param predicted_cols:  the dimension need to be predicted
    """
    data_x=[]
    data_y=[]
    train_len=data.shape[0]
    # the last sequence cannot cross over
    for i in range(train_len-seq_len):
        x,y= next_window(i,seq_len,normalise,single,data,predicted_cols)
        data_x.append(x) #[seq_len-1, dim]
        data_y.append(y) # [, 1]

    return np.array(data_x),np.array(data_y) #[bs,seq_len-1, dim] , [bs, 1]


"""
data_all len 150 predict len 15    data :  [0,^,8--> 9 ,……, 141,^,149-->150]
predict:
0,^,8 ---> 9, ^,23
15,^23---> 24,^,38
135,^,143-->144,^,158 


"""

def predict_sequence_multiple(model,data_x,window_size,prediction_len):
    #data_len- seq_len   lack begin :  0,^,seq_len-1  (first: 0,^,seq_len-1 ---> seq_len  last: data_len-seq_len,^,data_len-2---> data_len-1  )
    raw_predict=model.predict(data_x)
    print(raw_predict.shape)
    # seq_len-1
    cur_frame=data_x[-1]
    new_result=[]
    for i in range(prediction_len):
        pred_result=model.predict(cur_frame[np.newaxis,:,:])[0,0]
        new_result.append(pred_result)
        cur_frame=cur_frame[1:]
        cur_frame=np.insert(cur_frame,[window_size-2],new_result[-1],axis=0)
    print(new_result)
    #(prediction_len,1)
    new_result=np.array(new_result).reshape((-1,1))
    predict=np.concatenate([raw_predict,new_result],axis=0)
    return predict

'''自定义R^2度量函数'''
@tf.function
def R_2(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))



def data_get_back(predict_data,raw_data):
    anchor=raw_data[0]
    back_data=[]
    for i in predict_data:
        back_data.append((i+1)*anchor)
    return np.array(back_data)


def plot_results_multiple(predicted_data,true_data,prediced_len,seq_len):
    data_len=len(true_data)
    fig=plt.figure(facecolor='white')
    ax=fig.add_subplot(111)
    ax.plot(true_data,label='true_data')
    plt.legend()
    plt.plot(predicted_data,label='prediction')
    index=np.arange(seq_len,data_len+prediced_len)
    plt.plot(predicted_data)
    plt.show()


def build_model():
    inputs = Input(shape=(49,1))
    x = Conv1D(filters=128, kernel_size=5, strides=1, padding='valid', activation='relu', kernel_initializer='uniform')(
        inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    # Conv2
    x = Conv1D(256, 5, 1, padding='valid', activation='relu', kernel_initializer='uniform')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    y0 = Flatten()(x)
    y0 = Dense(128, activation='relu')(y0)
    y0 = BatchNormalization()(y0)
    y0 = Dropout(0.5)(y0)
    y0 = Dense(128, activation='relu')(y0)
    outputs0 = Dense(1, activation='sigmoid')(y0)
    model = Model(inputs=inputs, outputs=outputs0)
    model.summary()

    return model


cnn_model=build_model()
i=np.arange(100,350).reshape((-1,1))

raw_data=50*np.sin(i)
#plt.plot(x)
#plt.show()
print(raw_data.shape)
xx,yy=get_train_data(data=raw_data,seq_len=50,normalise=False,single=True,predicted_cols=0)
print(xx)

xtrain,xtest,ytrain,ytest=train_test_split(xx,yy,test_size=0.1)
scaler = MinMaxScaler(feature_range=(0, 1))
ytrain=scaler.fit_transform(ytrain)
ytest=scaler.transform(ytest)



cnn_model.compile(optimizer='adam', loss='mse', metrics =R_2)
es = EarlyStopping(monitor='val_mse', mode='min', verbose=1, patience=10)
mc = ModelCheckpoint('best_time.h5', monitor='val_mse', mode='min', verbose=1, save_best_only=True)

history = cnn_model.fit(xtrain,ytrain,batch_size =50, epochs = 100, verbose = 1,validation_data=(xtest,ytest), callbacks=[es, mc])


pred_seqs=predict_sequence_multiple(cnn_model,xx,window_size=50,prediction_len=30)



#pred_seqs=data_get_back(pred_seqs,raw_data)

pred_seqs=scaler.inverse_transform(np.array(pred_seqs).reshape((-1,1)))
print(len(pred_seqs))
plot_results_multiple(pred_seqs,yy,prediced_len=30,seq_len=30)