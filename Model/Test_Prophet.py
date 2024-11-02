# @Time: 2024/10/19 00:10
# @Author: Shen Hao
# @File: Test_Prophet.py
# @system: Win10
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# 设置Matplotlib的默认字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
# 读取数据
data = pd.read_excel('../Data/商丘PM2.5研究-逐月(2013-2023)(完整数据).xlsx', engine='openpyxl')
Y = data.iloc[:, 1].values  # 假设目标值在第3列

# 准备数据
df = pd.DataFrame({'ds': data['日期'], 'y': Y})  # 假设日期在第1列

# 初始化Prophet模型
model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)

# 拟合模型
model.fit(df)

# 创建未来日期的数据框
future = model.make_future_dataframe(periods=12, freq='ME')  # 预测未来12个月

# 进行预测
forecast = model.predict(future)

# 提取预测结果
predicted_values = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12)

# 可视化预测结果
plt.figure(figsize=(12, 6))

# 只显示最后100条数据及其预测
start_index = max(0, len(Y) - 100)
plt.plot(range(start_index, len(Y)), Y[start_index:], label='实际值', color='blue')
plt.plot(range(len(Y), len(Y) + 12), predicted_values['yhat'].values, label='预测值', color='red', linestyle='--')

plt.fill_between(range(len(Y), len(Y) + 12), predicted_values['yhat_lower'], predicted_values['yhat_upper'], color='pink', alpha=0.5)

plt.legend()
plt.xlabel('时间步')
plt.ylabel('PM2.5值')
plt.title('Prophet实际值与预测值对比（只显示最后100条）')
plt.grid(True)
plt.show()