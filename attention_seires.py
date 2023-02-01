import keras.optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D,BatchNormalization,Bidirectional,GRU,add,Multiply,Activation,RepeatVector,Lambda,Permute
from tensorflow.keras import regularizers,initializers
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

#gpu
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

'''自定义R^2度量函数'''
@tf.function
def R_2(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))




# 残差块
def ResBlock(x, filters, kernel_size, dilation_rate):
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu',kernel_initializer=initializers.RandomNormal(),kernel_regularizer=regularizers.l2(0.0002))(x)  # 第一卷积
    r=BatchNormalization()(r)
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate,activation='relu')(r)  # 第二卷积
    if x.shape[-1] == filters:
        shortcut = x
    else:
        shortcut = Conv1D(filters, kernel_size, padding='same')(x)  # shortcut（捷径）
    o = add([r, shortcut])
    o = Activation('relu')(o)  # **函数
    return o
##注意力机制
def attention_3d_block(inputs,SINGLE_ATTENTION_VECTOR):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)##相当于转置:(bs,dim,time_steps)
    a = Dense(inputs.shape[1], activation='softmax')(a)###给每个时间步计算权重:(bs,dim,time_stpes)
    if SINGLE_ATTENTION_VECTOR==1:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)###每个维度取平均:(bs,time_steps)
        a = RepeatVector(input_dim)(a)####复制到相同维度：（bs,input_dim,time_stpes）
    a_probs = Permute((2, 1))(a)###转置:(batch_size, time_steps, input_dim)
    output_attention_mul = Multiply()([inputs, a_probs])####乘权重
    return output_attention_mul
# 序列模型
def ATT_TCN_BiGRU():
    inputs = Input(shape=(128,9))
    x = ResBlock(inputs, filters=24, kernel_size=10, dilation_rate=1)###TCN
    x = ResBlock(x, filters=36, kernel_size=3, dilation_rate=3)###TCN
    x = ResBlock(x, filters=36, kernel_size=3, dilation_rate=9)###TCN
    x = ResBlock(x, filters=36, kernel_size=3, dilation_rate=27)###TCN
    x = MaxPooling1D(10)(x)
    # x=LSTM(units=30,dropout=0.1,return_sequences=True)(x)
    x = Bidirectional(GRU(units=100,activation='relu', recurrent_initializer='orthogonal',return_sequences=True,dropout=0.1),merge_mode='concat')(x)  # -------
    x = attention_3d_block(x, 1)
    x = Flatten()(x)
    x =Dense(1500,activation='relu')(x)
    x =Dense(50,activation='relu')(x)
    output = Dense(6,activation='softmax')(x)
    model = Model(inputs, output)
    # 查看网络结构
    model.summary()

    return model




#读取txt文件数据:x:[datanums,timesteps,feature numbers]  y:[datanums,1]
def get_x_data(x_train_signals_paths,data_num=7352):
    xall = np.zeros((data_num, 128, 9))
    for j, signal_type_path in enumerate(x_train_signals_paths):
        x_signals = []
        with open(signal_type_path, 'r', encoding='utf-8') as r:
            for row in r:
                transient1 = []
                transient1.append([float(i) for i in row.split()])
                x_signals.append(np.array(transient1).reshape(1, -1))
        x_signals = np.squeeze(np.array(x_signals), axis=1)
        xall[:, :, j] = x_signals
    return xall

def get_y_data(y_path):
    with open(y_path,'r',encoding='utf-8') as r:
        transient1=[]
        for row in r:
            transient1.append(row.replace(" ","").strip().split(" "))
        y=[]
        for i in transient1:
            y.append(i)
        y_=np.array(y,dtype=np.int32)

        return y_-1

def get_train_data(dataset_path,train_path):
    inertial_signals=['body_acc_x_','body_acc_y_','body_acc_z_','body_gyro_x_','body_gyro_y_','body_gyro_z_','total_acc_x_','total_acc_y_','total_acc_z_']
    x_train_signals_paths=[dataset_path+train_path+'Inertial Signals\\'+signal+'.txt' for signal in inertial_signals]
    xtrain=get_x_data(x_train_signals_paths)
    y_train_path=dataset_path+train_path+'y_train.txt'
    ytrain=get_y_data(y_train_path)
    return xtrain,ytrain


def get_test_data(dataset_path,test_path):
    inertial_signals=['body_acc_x_','body_acc_y_','body_acc_z_','body_gyro_x_','body_gyro_y_','body_gyro_z_','total_acc_x_','total_acc_y_','total_acc_z_']
    x_test_signals_paths=[dataset_path+test_path+'Inertial Signals\\'+signal+'.txt' for signal in inertial_signals]
    xtest=get_x_data(x_test_signals_paths,data_num=2947)
    y_test_path=dataset_path+test_path+'y_test.txt'
    ytest=get_y_data(y_test_path)
    return xtest,ytest



xtrain,ytrain=get_train_data(dataset_path='E:\\time_series\\UCI HAR Dataset\\',train_path='train\\')
xtest,ytest=get_test_data(dataset_path='E:\\time_series\\UCI HAR Dataset\\',test_path='test\\')
ytrain=tf.one_hot(ytrain,depth=6)

ytest=tf.one_hot(ytest,depth=6)
ytrain=tf.squeeze(ytrain,axis=1)
ytest=tf.squeeze(ytest,axis=1)


model=ATT_TCN_BiGRU()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics ='accuracy')
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=6)
mc = ModelCheckpoint('best_model_TCN_new.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
history = model.fit(xtrain,ytrain,batch_size =32, epochs = 25, verbose = 1, callbacks=[es, mc],validation_data=(xtest,ytest))
