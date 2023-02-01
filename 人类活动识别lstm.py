import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers,optimizers
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#gpu
gpus=tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

tf.random.set_seed(1)
np.random.seed(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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

print('train:',xtrain.shape,ytrain.shape)
print('test:',xtest.shape,ytest.shape)

print(xtrain)


batchsz = 128
train_db = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
train_db = train_db.shuffle(10000).batch(batchsz, drop_remainder=True)  # 小于batchsize的批次放弃
test_db = tf.data.Dataset.from_tensor_slices((xtest, ytest))
test_db = test_db.batch(batchsz, drop_remainder=True)

class MyRNN(keras.Model):

    def __init__(self, units):
        super(MyRNN, self).__init__()

        self.state0 = [tf.zeros([batchsz, units]),tf.zeros([batchsz,units])]  # initialise h and c
        self.state1 = [tf.zeros([batchsz, units]),tf.zeros([batchsz,units])]
        self.state2 = [tf.zeros([batchsz, units]),tf.zeros([batchsz,units])]

        # [b,128,9] =>[b,hidden dim(64)]
        self.rnn_cell0 = layers.LSTMCell(units, dropout=0.2)
        self.rnn_cell1 = layers.LSTMCell(units, dropout=0.2)
        self.rnn_cell2 = layers.LSTMCell(units,dropout=0.2)

        # [b,64]=>[b,1]

        self.fc1 = layers.Dense(64,activation='relu')
        self.out_layers = layers.Dense(6)

    def call(self, inputs, training=None):
        # traing mode net(x);testing mode:net(x,traing=False)
        x = inputs  # [b,128,9]


        state0 = self.state0
        state1 = self.state1
        state2 = self.state2
        for word in tf.unstack(x, axis=1):  # word:[b,9],每个时间序列拆分开
            out0, state0 = self.rnn_cell0(word, state0, training)
            out1, state1 = self.rnn_cell1(out0, state1, training)
            out2, state2 = self.rnn_cell1(out1, state2, training)
        # [b,units]=>[b,6]

        x = self.fc1(out2)
        x = self.out_layers(x)

        return x



units = 64
epochs =15

optimizer=optimizers.Adam(learning_rate=0.001)
model = MyRNN(units)
for epoch in range(epochs):
    for step,(x,y) in enumerate(train_db):
        with tf.GradientTape() as tape:

            out=model(x)


            loss=tf.losses.categorical_crossentropy(y,out,from_logits=True)
            loss=tf.reduce_mean(loss)

        grads=tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))

        if step %100 ==0:
            print(epoch,step,'loss:',loss)

    total_num=0
    total_correct=0
    for x,y in test_db:

        out=model(x)

        prob=tf.nn.softmax(out,axis=1)
        pred=tf.cast(tf.argmax(prob,axis=1),dtype=tf.int32)
        y=tf.cast(tf.argmax(y,axis=1),dtype=tf.int32)

        correct=tf.cast(tf.equal(pred,y),dtype=tf.int32)
        correct=tf.reduce_sum(correct)

        total_num+=x.shape[0]
        total_correct+=int(correct)
    acc=total_correct/total_num
    print(epoch,acc)

#最终预测训练集和测试集

out1=model(xtrain[0:batchsz,:,:])
prob1=tf.nn.softmax(out1,axis=1)
pred1=tf.cast(tf.argmax(prob1,axis=1),dtype=tf.int32)
ytrain=tf.cast(tf.argmax(ytrain[0:batchsz,:],axis=1),dtype=tf.int32)

correct=tf.cast(tf.equal(pred1,ytrain),dtype=tf.int32)
correct=tf.reduce_sum(correct)

acc=correct/batchsz
print('train accuracy:',acc)

out1=model(xtest[0:batchsz,:,:])
prob1=tf.nn.softmax(out1,axis=1)
pred1=tf.cast(tf.argmax(prob1,axis=1),dtype=tf.int32)
ytest=tf.cast(tf.argmax(ytest[0:batchsz,:],axis=1),dtype=tf.int32)

correct=tf.cast(tf.equal(pred1,ytest),dtype=tf.int32)
correct=tf.reduce_sum(correct)

acc=correct/batchsz
print('test accuracy:',acc)

