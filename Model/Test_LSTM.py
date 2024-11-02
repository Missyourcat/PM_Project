import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 设置Matplotlib的默认字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
# 读取数据
data = pd.read_excel('../Data/商丘PM2.5研究-逐月(2013-2023)(完整数据).xlsx', engine='openpyxl')
Y = data.iloc[:, 3].values  # 假设目标值在第3列

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaled = scaler.fit_transform(Y.reshape(-1, 1))

# 创建数据集
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 12  # 假设我们使用过去的12个月的数据来预测下一个月
X, y = create_dataset(Y_scaled, look_back)

# 划分训练集和测试集
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 调整输入形状以适应LSTM [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# 构建LSTM模型
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(1, look_back)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# 训练模型
early_stop = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train, y_train, epochs=100, batch_size=1, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=2)

# 预测
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 反归一化
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# 预测未来
last_data = Y_scaled[-look_back:]
future_steps = 12
predictions = []
for _ in range(future_steps):
    last_data_reshaped = np.reshape(last_data, (1, 1, look_back))
    next_prediction = model.predict(last_data_reshaped)
    predictions.append(next_prediction[0, 0])
    last_data = np.append(last_data[1:], next_prediction, axis=0)

predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# 可视化预测结果
plt.figure(figsize=(12, 6))

# 只显示最后100条数据及其预测
start_index = max(0, len(Y) - 100)
plt.plot(range(start_index, len(Y)), Y[start_index:], label='实际值', color='blue')
plt.plot(range(len(Y), len(Y) + future_steps), predictions, label='预测值', color='red', linestyle='--')

plt.legend()
plt.xlabel('时间步')
plt.ylabel('PM2.5值')
plt.title('LSTM实际值与预测值对比（只显示最后100条）')
plt.grid(True)
plt.show()