# @Time: 2024/10/20 22:05
# @Author: Shen Hao
# @File: Test_NeuralProphet.py
# @system: Win10
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from neuralprophet import NeuralProphet

# 设置Matplotlib的默认字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 读取数据
data = pd.read_excel('../Data/商丘PM2.5研究-逐月(2013-2023)(完整数据).xlsx', engine='openpyxl')
data['ds'] = pd.to_datetime(data.iloc[:, 0])  # 假设日期在第1列
data['y'] = data.iloc[:, 3]  # 假设目标值在第3列

# 初始化NeuralProphet模型
m = NeuralProphet(
    n_changepoints=10,  # 更改点的数量
    yearly_seasonality=True,  # 是否考虑年季节性
    weekly_seasonality=False,  # 是否考虑周季节性
    daily_seasonality=False,  # 是否考虑日季节性
    seasonality_mode='multiplicative'  # 季节性的模式
)

# 拟合模型
metrics = m.fit(data, freq='MS')  # 'MS' 表示每月的第一天

# 预测未来12个月
future = m.make_future_dataframe(data, periods=12, n_historic_predictions=len(data))
forecast = m.predict(future)

# 可视化预测结果
fig = m.plot(forecast)
plt.title('NeuralProphet 实际值与预测值对比')
plt.show()

# 显示模型组件
fig_comp = m.plot_components(forecast)
plt.show()