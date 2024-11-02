import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from matplotlib.font_manager import FontProperties

# 设置Matplotlib的默认字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 读取数据
data = pd.read_excel('../Data/商丘PM2.5研究-逐月(2013-2023)(完整数据).xlsx', engine='openpyxl')
Y = data.iloc[:, 2].values  # 假设目标值在第3列

# 检查平稳性
result = adfuller(Y)
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# 如果数据不平稳，进行一次差分
if result[1] > 0.05:  # p值大于0.05，数据不平稳
    Y_diff = np.diff(Y)
    print("数据不平稳，进行了差分。")
else:
    Y_diff = Y

# 绘制 ACF 和 PACF 图
plt.figure(figsize=(12, 6))
plt.subplot(211)
plot_acf(Y_diff, lags=20, ax=plt.gca())
plt.title('ACF 图')

plt.subplot(212)
plot_pacf(Y_diff, lags=20, ax=plt.gca())
plt.title('PACF 图')
plt.tight_layout()
plt.show()

# 拟合 ARIMA 模型
model = ARIMA(Y, order=(2, 1, 2))  # 根据 ACF 和 PACF 图来选择适当的 p 和 q
model_fit = model.fit()

# 计算R²
y_mean = np.mean(Y)
residuals = model_fit.resid
rss = np.sum(residuals**2)
tss = np.sum((Y - y_mean)**2)
r_squared = 1 - (rss / tss)
print(f"模型的R²值: {r_squared:.4f}")

# 如果您想在训练集上计算R²，可以使用模型的预测值
train_forecast = model_fit.predict(start=0, end=len(Y)-1)
train_rss = np.sum((Y - train_forecast)**2)
train_tss = np.sum((Y - y_mean)**2)
train_r_squared = 1 - (train_rss / train_tss)
print(f"训练集上的R²值: {train_r_squared:.4f}")

# 进行预测
future_steps = 12
forecast = model_fit.forecast(steps=future_steps)

# 可视化预测结果
plt.figure(figsize=(12, 6))

# 只显示最后100条数据及其预测
if len(Y) > 100:
    plt.plot(range(len(Y) - 100, len(Y)), Y[-100:], label='实际值', color='blue')
    plt.plot(range(len(Y), len(Y) + future_steps), forecast, label='预测值', color='red', linestyle='--')
else:
    plt.plot(range(len(Y)), Y, label='实际值', color='blue')
    plt.plot(range(len(Y), len(Y) + future_steps), forecast, label='预测值', color='red', linestyle='--')

plt.legend()
plt.xlabel('时间步')
plt.ylabel('值')
plt.title(f'ARIMA实际值与预测值对比（只显示最后100条），R²={train_r_squared:.4f}')
plt.grid(True)
plt.show()