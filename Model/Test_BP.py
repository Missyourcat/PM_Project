# @Time: 2024/10/19 00:06
# @Author: Shen Hao
# @File: Test_BP.py
# @system: Win10
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
# import matplotlib.pyplot as plt
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

# 构建BP神经网络模型
model = MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', solver='adam', max_iter=500, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 预测
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 反归一化
train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 计算R²值
train_r2 = r2_score(y_train_actual, train_predict)
test_r2 = r2_score(y_test_actual, test_predict)

print(f"Training R-squared: {train_r2:.4f}")
print(f"Test R-squared: {test_r2:.4f}")
# 预测未来
last_data = Y_scaled[-look_back:]
future_steps = 12
predictions = []
for _ in range(future_steps):
    last_data_reshaped = last_data.reshape(1, -1)
    next_prediction = model.predict(last_data_reshaped)
    predictions.append(next_prediction[0])
    last_data = np.append(last_data[1:], next_prediction)

predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# 可视化预测结果
plt.figure(figsize=(12, 6))

# 只显示最后100条数据及其预测
start_index = max(0, len(Y) - 100)
plt.plot(range(start_index, len(Y)), Y[start_index:], label='实际值', color='blue')
plt.plot(range(len(Y), len(Y) + future_steps), predictions, label='预测值', color='red', linestyle='--')

# 添加R²值作为图例的一部分
plt.legend(title=f'Train R²: {train_r2:.2f}, Test R²: {test_r2:.2f}')
plt.xlabel('时间步')
plt.ylabel('PM2.5值')
plt.title('BP实际值与预测值对比（只显示最后100条）')
plt.grid(True)
plt.show()