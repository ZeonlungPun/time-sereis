
import numpy as np

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