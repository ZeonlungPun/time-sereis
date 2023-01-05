#2021华为杯B题时间序列预测
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import time
from minepy import MINE
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os



os.chdir('D:\Desktop\数学建模\華為\\2021年B题')

data=pd.read_excel('附件1 监测点A空气质量预报基础数据.xlsx',sheet_name='监测点A逐小时污染物浓度与气象实测数据',engine='openpyxl')

#统计数据的缺失值，按日期的时间点进行
date_time=data.loc[:,'监测时间']

#将缺失值进行标记 ： - ->  np.nan

data=data.replace('—',np.nan)
#删去有很多缺失值的样本:非缺失值少于3
#data=data.dropna(axis=0,thresh=3)
#删除无关列
del data['地点']

#检验缺失值情况
print(data.isnull().sum(axis=0))

#提取数据标签
names=data.columns.values[1::]


#提取数据子集
sub_data=np.array(data.iloc[:,1:7])
time_=data.iloc[:,0]


#填补缺失值:KNN
knn_imputer=KNNImputer(n_neighbors=10)
sub_data=knn_imputer.fit_transform(sub_data)
#再次检查缺失情况
print(pd.DataFrame(sub_data).isnull().sum(axis=0))


#处理异常值
#浓度不可能低于0
#大于等于0的值保持不变，否则替换为0
sub_data=np.where(sub_data>=0,sub_data,0)

#3σ原则
#data为矩阵
def check_abnormal_3sigmal(data):
    std=np.std(data,axis=0)
    ave=np.mean(data,axis=0)
    up_value=ave+3*std
    low_value=ave-3*std

    data=np.where(data<=up_value,data,up_value)
    data=np.where(data>=low_value,data,low_value)

    return data

sub_data=check_abnormal_3sigmal(sub_data)


#合并数据
newdata=np.concatenate([sub_data,np.array(data.iloc[:,7::])],axis=1)
newdata=pd.DataFrame(newdata,columns=names)

data_=pd.concat([time_,newdata],axis=1)
#排除2021-7-13
data_=data_.iloc[:-8,:]

# #检查日期缺失:sheet2 缺失 2020-4-25  2021-7-4/5/6
# F=data_['监测时间'].value_counts()
# #输出结果：每个日期下均有72条数据（略）
# #分析：72条时数据即预测的三天。说明只要进行了模拟，就会一次性模型三天的，不存在中间某个小时的预测缺失
#
# #通过交叉表查看是否有“模型运行日期”的缺失
# data['年份'] = data_['监测时间'].dt.year   #提取“模型运行日期”的年份
# data['月份'] = data_['监测时间'].dt.month  #提取“模拟运行日期”的月份
# cross_ym = pd.crosstab(index=data['年份'],columns=data['月份'],values=data['月份'],aggfunc='count')
# #由于每个“监测时间”对应24条数据，所以还需要除以24
# cross_ym2 = cross_ym/24





#从2020年8月25日到8月28日,共4天
sub_time1='2020-08-25 00:00:00'
sub_time2='2020-08-28 23:00:00'

#在时间戳中查找对应的索引以在后续中方便取出对应的数据
def find_corrosponding_index_timestamp(time_array,target):
    for index,i in enumerate(time_array):
        if  str(time_array.iloc[index])==target:
            return index
        else:
            continue


time_index1=find_corrosponding_index_timestamp(time_,sub_time1)
time_index2=find_corrosponding_index_timestamp(time_,sub_time2)


#time_index1=time_[time_==sub_time1].index.tolist()[0]
#time_index2=time_[time_==sub_time2].index.tolist()[0]
#print(time_index1,time_index2)

target_data1=sub_data[time_index1:time_index2+1,:]
print('提取出的目标数据对应时间')
print(time_[time_index1:time_index2+1])


#算各指标的每个自然日的最大8小时滑动平均
#一天一般24条数据
#Ci=max{1/8 sum_{i=t-7}^{t} Ct}

Ci=np.zeros((4,target_data1.shape[1]))
#遍历每一个指标
for i in range(target_data1.shape[1]):
    sub_target_data1=target_data1[:,i]
    #遍历每一天
    for day in range(4):
        sub_sub_target_data1=sub_target_data1[day*24:day*24+24]
    #遍历每一天的8时至24时
        C_temp=np.zeros((1,17))
        for t in range(8,25):
            C_temp[0,t-8]=np.mean(sub_sub_target_data1[t-7:t+1])
        ci=np.max(C_temp)
        Ci[day,i]=ci

print("最大8小时滑动平均：",Ci)

IAQI_limit=np.array([0,50,100,150,200,300,400,500])
limits=np.array([[0,50,150,475,800,1600,2100,2620],
                [0,40,80,180,280,565,750,940],
                 [0, 50, 150, 250, 350, 420, 500, 600],
                [0,35,75,115,150,250,350,500],
                 [0, 100, 160, 215, 265, 800, 0, 0],
                [0,2,4,14,24,36,48,60]])

#遍历天数
range_index=np.zeros((4,6))
for day in range(0,4):
    data1=Ci[day]
    #遍历污染物
    for i in range(0,6):
        pollute=data1[i]
        limit=limits[i]
    #查找数值落在哪一个区间
        for j in range(0,8):
            if pollute >= limit[j] and pollute <=limit[j+1]:

                range_index[day,i]=j
                break
            else:
                continue
print(range_index)

#计算AQI

AQI=np.zeros((4,6))
#遍历天数
for x,i in enumerate(range_index):
    #遍历污染物种类
    for y,j in enumerate(i):
        j=int(j)
        result=((Ci[x,y] - limits[y,j]) * (IAQI_limit[j + 1] - IAQI_limit[j]) / (limits[y,j + 1] - limits[y,j])) + IAQI_limit[j]
        AQI[x,y]=result
AQI_index=np.argmax(AQI,axis=1)
AQI=np.max(AQI,axis=1)

print('AQI:',AQI,AQI_index)


#根据对污染物浓度的影响程度，对气象条件进行合理分类
#聚类分析步骤：识别相关特征  --》  确定聚类个数  --》 T-sne降维可视化结果




# #利用离散化互信息识别变量重要性
# def calculate_IZX_i(input):
#     mine = MINE(alpha=0.6, c=15)
#     score = np.zeros((input.shape))
#     for i in range(input.shape[1]):
#         for j in range(input.shape[1]):
#             mine.compute_score(input[:, i], input[:, j])
#             score_ = mine.mic()
#             score[i,j]=score_
#
#     return score
#
# cor_score=calculate_IZX_i(newdata)
# sns.heatmap(cor_score, annot=True, fmt="g", cmap='viridis',vmin=0,yticklabels=names,xticklabels=names)
# plt.show()


#kmeans 聚类：通过评估类之间方差和类内方差来计算得分
#确定聚类数后，利用tsne可视化
# for index, kk in enumerate((3,4,5,6)):
#     y_pred = KMeans(n_clusters=kk).fit_predict(newdata)
#     score= calinski_harabasz_score(newdata,y_pred)
#     print('聚类数为：',kk)
#     print('得分：',score)
#     plt.subplot(2, 2, index + 1)
#     tsne = TSNE(n_components=2)
#     x_embeded=tsne.fit_transform(newdata)
#     plt.scatter(x_embeded[:, 0], x_embeded[:, 1], c=y_pred)
#     plt.text(.99, .01, ('k=%d, score: %.2f' % (kk, score)),
#              transform=plt.gca().transAxes, size=10,
#              horizontalalignment='right')
#
# plt.show()







#sheet3中的数据几乎都是sheet2中对应日期的24h平均；O3的计算方式特别些，是取“8小时滑动平均的最大值
#读入附件1sheet3
# day_data=pd.read_excel('附件1 监测点A空气质量预报基础数据.xlsx',sheet_name='监测点A逐日污染物浓度实测数据',engine='openpyxl')
# day_data=day_data.iloc[:-3,:]
# del day_data['地点']
#
#
#
#
# # 定义一个函数，可以将df2A的“时数据”转换成对应日期的“日数据”
# def change_htod(df):
#
#         #适用于df2A数据
#         #将小时实测数据转化为日数据
#         #df为完整数据列表[监测时间,SO2,NO2,PM10,PM2.5,O3,CO,温度,湿度,气压,风速,风向]
#
#     # 第一步：识别出日期
#     df['日期'] = df['监测时间'].dt.date
#     df['时间'] = df['监测时间'].dt.time
#
#     # 第二步：按日期计算平均值
#     df_mean_day = pd.DataFrame({'SO2平均浓度(μg/m³)': df.groupby('日期')['SO2监测浓度(μg/m³)'].mean(),
#                                 'NO2平均浓度(μg/m³)': df.groupby('日期')['NO2监测浓度(μg/m³)'].mean(),
#                                 'PM10平均浓度(μg/m³)': df.groupby('日期')['PM10监测浓度(μg/m³)'].mean(),
#                                 'PM2.5平均浓度(μg/m³)': df.groupby('日期')['PM2.5监测浓度(μg/m³)'].mean(),
#                                 'O3平均浓度(μg/m³)': df.groupby('日期')['O3监测浓度(μg/m³)'].mean(),
#                                 'CO平均浓度(mg/m³)': df.groupby('日期')['CO监测浓度(mg/m³)'].mean(),
#                                 '平均温度(℃)': df.groupby('日期')['温度(℃)'].mean(),
#                                 '平均湿度(%)': df.groupby('日期')['湿度(%)'].mean(),
#                                 '平均气压(MBar)': df.groupby('日期')['气压(MBar)'].mean(),
#                                 '平均风速(m/s)': df.groupby('日期')['风速(m/s)'].mean(),
#                                 '风向(°)': df.groupby('日期')['风向(°)'].mean()})
#
#     # 第三步：臭氧问题重处理
#     # o3数据重排：使用O3的“日期-时间”交叉表
#     cross_o3 = pd.crosstab(index=df['时间'], columns=df['日期'], values=df['O3监测浓度(μg/m³)'], aggfunc='sum')
#     # 根据cross_o3，计算8小时滑动平均值最大值，得到序列o3
#     adj = []
#     o3 = []
#     for j in range(cross_o3.shape[1]):  # [cross_o3.shape[1]]为cross_o3的列数，即日期数
#         for i in range(7, 24):  # 与o3“8小时滑动平均”定义有关，早上八点才产生第一个滑动平均值
#             adj.append(cross_o3.iloc[i - 7:i + 1, j].mean())  # 八小时滑动平均
#         o3.append(max(adj))  # 将每天滑动平均的最大值存储到序列o3中
#         adj = []
#     # 将日期与o3对应
#     dfo3 = pd.DataFrame(o3, index=cross_o3.columns, columns=["O3最大八小时滑动平均监测浓度(μg/m³)"])
#
#     # 第四步：替换掉步骤二中由简单平均产生的臭氧数据
#     df_mean_day.iloc[:, 4] = dfo3.iloc[:, 0]
#
#     # 第五步：返回根据“时数据”转化的“日数据”
#     return df_mean_day
#
# day_calculate=change_htod(data_)
#
# day_new=day_calculate.iloc[:,6::]
# day_cal=day_calculate.iloc[:,0:6]
#
# #找出sheet3中的缺失值，用day_calculate中的值进行填补
# miss=day_data.isnull()
# del miss['监测日期']
# del day_data['监测日期']
#
# day_data,day_calculate,miss=np.array(day_data),np.array(day_calculate),np.array(miss)
# day_data=np.where(miss==False,day_data,day_cal)
#
# labels_name=['SO2监测浓度(μg/m³)','NO2监测浓度(μg/m³)','PM10监测浓度(μg/m³)','PM2.5监测浓度(μg/m³)','O3监测浓度(μg/m³)','CO监测浓度(mg/m³)','温度(℃)','湿度(%)','气压(MBar)','风速(m/s)','风向(°)']
#
# day_data=np.concatenate([day_data,np.array(day_new)],axis=1)
#
# #提取出最后六个数据，制作预测集的X
# x_pred=day_data[-6::,:]
# x_pred=np.expand_dims(x_pred,axis=0)
# print(x_pred)
#
#
#
# #预测2021年7月13日至7月15日6种常规污染物的单日浓度
#
#
# #拆分训练集、验证集、测试集：0.8 0.1 0.1
# index=np.random.permutation(day_data.shape[0])
# train_num=round(0.8*day_data.shape[0])
# test_num=round(0.1*day_data.shape[0])
#
# train_index=index[0:train_num]
# test_index=index[train_num:train_num+test_num]
# val_index=index[train_num+test_num::]
#
# train_df,test_df,val_df=day_data[train_index,:],day_data[test_index,:],day_data[val_index,:]
#
# #print(train.shape,test.shape,val.shape)
#
# #数据标准化,标准化后格式为array
# sd=StandardScaler()
# sd.fit(train_df)
# train=sd.transform(train_df)
# test=sd.transform(test_df)
# val=sd.transform(val_df)
# train_df, test_df, val_df = pd.DataFrame(train_df, columns=labels_name), pd.DataFrame(test_df, columns=labels_name), pd.DataFrame(val_df, columns=labels_name)
#
#
# class WindowGenerator():
#     def __init__(self, input_width, label_width, shift,
#                  train_df=train_df, val_df=val_df, test_df=test_df,
#                  label_columns=None):
#         # 储存原始数据
#
#         self.train_df = train_df
#         self.val_df = val_df
#         self.test_df = test_df
#
#         # 处理标签页索引
#         self.label_columns = label_columns
#         if label_columns is not None:
#             self.label_columns_indices = {name: i for i, name in
#                                           enumerate(label_columns)}
#         self.column_indices = {name: i for i, name in
#                                enumerate(train_df.columns)}
#
#         # 处理窗口参数
#         self.input_width = input_width
#         self.label_width = label_width
#         self.shift = shift
#
#         self.total_window_size = input_width + shift
#
#         self.input_slice = slice(0, input_width)
#         self.input_indices = np.arange(self.total_window_size)[self.input_slice]
#
#         self.label_start = self.total_window_size - self.label_width
#         self.labels_slice = slice(self.label_start, None)
#         self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
#
#     def __repr__(self):
#         return '\n'.join([
#             f'Total window size: {self.total_window_size}',
#             f'Input indices: {self.input_indices}',
#             f'Label indices: {self.label_indices}',
#             f'Label column name(s): {self.label_columns}'])
#
#
# def split_window(self, features):
#     #将其扩展为（batch,time_series,dim）
#     #features=np.expand_dims(features,axis=0)
#     inputs = features[:, self.input_slice, :]
#     labels = features[:, self.labels_slice, :]
#     if self.label_columns is not None:
#         labels = tf.stack(
#             [labels[:, :, self.column_indices[name]] for name in self.label_columns],
#             axis=-1)
#
#     # slicing不会保留固定形状信息，因此需要手动设置形状
#     inputs.set_shape([None, self.input_width, None])
#     labels.set_shape([None, self.label_width, None])
#
#     return inputs, labels
#
# WindowGenerator.split_window = split_window
#
# def make_dataset(self, data):
#     data = np.array(data, dtype=np.float32)
#     #data: 表示x数据，里面的每个叫做一个timestep。
#     #targets: 表示y标签。如果不处理标签只处理数据，传入targets = None。
#     #sequence_length: 一个输出序列sequence的长度，即有多少个timestep。
#     ds = tf.keras.preprocessing.timeseries_dataset_from_array(
#         data=data,
#         targets=None,
#         sequence_length=self.total_window_size,
#         sequence_stride=1,
#         shuffle=True,
#         batch_size=32,)
#
#     ds = ds.map(self.split_window)
#
#     return ds
#
# WindowGenerator.make_dataset = make_dataset
#
# @property
# def train(self):
#     return self.make_dataset(self.train_df)
#
# @property
# def val(self):
#     return self.make_dataset(self.val_df)
#
# @property
# def test(self):
#     return self.make_dataset(self.test_df)
#
# @property
# def example(self):
#     """Get and cache an example batch of `inputs, labels` for plotting."""
#     result = getattr(self, '_example', None)
#     if result is None:
#         # No example batch was found, so get one from the `.train` dataset
#         result = next(iter(self.train))
#         # And cache it for next time
#         self._example = result
#     return result
#
# WindowGenerator.train = train
# WindowGenerator.val = val
# WindowGenerator.test = test
# WindowGenerator.example = example
# def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
#     inputs, labels = self.example
#     plt.figure(figsize=(12, 8))
#     plot_col_index = self.column_indices[plot_col]
#     max_n = min(max_subplots, len(inputs))
#     for n in range(max_n):
#         plt.subplot(max_n, 1, n+1)
#         plt.ylabel(f'{plot_col} [normed]')
#         plt.plot(self.input_indices, inputs[n, :, plot_col_index],
#                 label='输入', marker='.', zorder=-10)
#
#         if self.label_columns:
#             label_col_index = self.label_columns_indices.get(plot_col, None)
#         else:
#             label_col_index = plot_col_index
#
#         if label_col_index is None:
#             continue
#
#         plt.scatter(self.label_indices, labels[n, :, label_col_index],
#                     edgecolors='k', label='标签', c='#2ca02c', s=64)
#         if model is not None:
#             predictions = model(inputs)
#             plt.scatter(self.label_indices, predictions[n, :, label_col_index],
#                         marker='X', edgecolors='k', label='预测',
#                         c='#ff7f0e', s=64)
#
#         if n == 0:
#             plt.legend()
#
#     plt.xlabel('时间 [h]')
#
# WindowGenerator.plot = plot
#
# #测试代码
# w2 = WindowGenerator(input_width=6, label_width=1, shift=1)
# for example_inputs, example_labels in w2.train.take(1):
#     print(f'输入形状 (批量数, 时间步长, 特征): {example_inputs.shape}')
#     print(f'标签形状 (批量数, 时间步长, 特征): {example_labels.shape}')
#
#
#
# #基本训练函数
# MAX_EPOCHS = 50
#
# def compile_and_fit(model, window, patience=2):
#     early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
#                                                       patience=patience,
#                                                       mode='min')
#
#     model.compile(loss=tf.losses.MeanSquaredError(),
#                   optimizer=tf.optimizers.Adam(),
#                   metrics=[tf.metrics.MeanAbsoluteError()])
#
#     history = model.fit(window.train, epochs=MAX_EPOCHS,
#                         validation_data=window.val,
#                         callbacks=[early_stopping])
#     return history
#
#
#
#
#
# #多输出，只有data没有target，多时间步预测
# #根据前6天预测后3天的结果
#
# OUT_STEPS = 3
# num_features=11
# multi_window = WindowGenerator(input_width=6,
#                                label_width=OUT_STEPS,
#                                shift=OUT_STEPS)
#
# """
# 实现多时间步预测的一种高级方法是使用单发模型，该模型可以一次完成整个序列的预测。
# 这可以通过设置layers.Dense的输出单元数为OUT_STEPS*features高效实现。
# 模型只需要将输出的形状重塑为(OUT_STEPS, features)即可。
# """
# CONV_WIDTH = 3
# multi_conv_model = tf.keras.Sequential([
#     # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
#     tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
#     # Shape => [batch, 1, conv_units]
#     tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
#     #tf.keras.layers.Conv1D(128, activation='relu', kernel_size=(CONV_WIDTH)),
#     tf.keras.layers.Dense(128),
#     tf.keras.layers.Dense(100),
#     # Shape => [batch, 1,  out_steps*features]
#     tf.keras.layers.Dense(OUT_STEPS*num_features,
#                           kernel_initializer=tf.initializers.zeros()),
#     # Shape => [batch, out_steps, features]
#     tf.keras.layers.Reshape([OUT_STEPS, num_features])
# ])
#
# #用于储存各个模型的最终表现
# multi_performance={}
# multi_val_performance={}
#
# history1 = compile_and_fit(multi_conv_model, multi_window)
# multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
# multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)
#
# pred_conv=multi_conv_model.predict(x_pred)
#
# #循环神经网络
# #LSTM只需要在最后一个时间步产生输出，因此需要设置return_sequences = False。
# multi_lstm_model = tf.keras.Sequential([
#     # Shape [batch, time, features] => [batch, lstm_units]
#     # Adding more `lstm_units` just overfits more quickly.
#     tf.keras.layers.LSTM(80, return_sequences=True),
#     tf.keras.layers.LSTM(80, return_sequences=False),
#     tf.keras.layers.Dense(128),
#     tf.keras.layers.Dense(100),
#     # Shape => [batch, out_steps*features]
#     tf.keras.layers.Dense(OUT_STEPS*num_features,
#                           kernel_initializer=tf.initializers.zeros()),
#     # Shape => [batch, out_steps, features]
#     tf.keras.layers.Reshape([OUT_STEPS, num_features])
# ])
# history2 = compile_and_fit(multi_lstm_model, multi_window)
# multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
# multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
#
# lstm_pred=multi_lstm_model.predict(x_pred)
#
# #自回归LSTM
# class FeedBack(tf.keras.Model):
#     def __init__(self, units, out_steps):
#         super().__init__()
#         self.out_steps = out_steps
#         self.units = units
#         self.lstm_cell = tf.keras.layers.LSTMCell(units)
#         # 为了简化`warmip`方法，将LSTMCell包裹入RNN
#         self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
#         self.dense1 =tf.keras.layers.Dense(128)
#         self.dense2 = tf.keras.layers.Dense(100)
#         self.dense3 = tf.keras.layers.Dense(num_features)
#
# feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)
#
# def warmup(self, inputs):
#     # inputs.shape => (batch, time, features)
#     # x.shape => (batch, lstm_units)
#     x, *state = self.lstm_rnn(inputs)
#
#     # predictions.shape => (batch, features)
#     x=self.dense1(x)
#     x=self.dense2(x)
#     prediction = self.dense3(x)
#     return prediction, state
#
# FeedBack.warmup = warmup
#
# def call(self, inputs, training=None):
#     # 使用TensorArray捕获动态输出
#     predictions = []
#     # 初始化LSTM状态
#     prediction, state = self.warmup(inputs)
#
#     # 插入第一个预测
#     predictions.append(prediction)
#
#     # 运行其余的预测步骤
#     for n in range(1, self.out_steps):
#         # 使用上次预测作为输入
#         x = prediction
#         # 执行一个LSTM步骤
#         x, state = self.lstm_cell(x, states=state,
#                               training=training)
#         # 将LSTM的输出转换为预测
#         x=self.dense1(x)
#         x=self.dense2(x)
#         prediction = self.dense3(x)
#         # 将预测加入到输出列表
#         predictions.append(prediction)
#
#     # predictions.shape => (time, batch, features)
#     predictions = tf.stack(predictions)
#     # predictions.shape => (batch, time, features)
#     predictions = tf.transpose(predictions, [1, 0, 2])
#     return predictions
#
# FeedBack.call = call
#
# history3 = compile_and_fit(feedback_model, multi_window)
# multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
# multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0)
#
# ar_lstm_pred=feedback_model.predict(x_pred)
#
# x = np.arange(len(multi_performance))
# width = 0.3
#
#
# lstm_model = tf.keras.models.Sequential([
#     # 形状 [批量数, 时间步, 特征数] => [批量数, 时间步, 单元数]
#     tf.keras.layers.LSTM(32, return_sequences=True),
#     # 形状 => [批量数, 时间步, 特征数]
#     tf.keras.layers.Dense(units=num_features)
# ])
#
# print('误差：')
# print(multi_performance)
# print("预测结果：")
# #反归一化
# pred_conv,lstm_pred,ar_lstm_pred=sd.inverse_transform(pred_conv),sd.inverse_transform(lstm_pred),sd.inverse_transform(ar_lstm_pred)
# print(pred_conv,lstm_pred,ar_lstm_pred)


