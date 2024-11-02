# @Time: 2024/10/2 22:43
# @Author: Shen Hao
# @File: Main_Project.py
# @system: Win10
import warnings

import pandas as pd

from Main.Correlation_Analysis import statistical_description_set, linear_test_set, pearson_correlation_set, \
    spearman_correlation_set
from Main.Reduced_Dimensional_Model import factor_analysis_set
from Main.Regression_Model import stepwise_linear_regression_set, linear_regression_set
from PyChartsTool.liner_aqi import liner_aqi

# 忽略openpyxl警告
warnings.filterwarnings("ignore")
# 读取数据
# SQ_TenYears_Data = pd.read_excel('../Data/商丘PM2.5研究-逐日(2013-2023)(剔除异常值).xlsx', engine='openpyxl')
SQ_TenYears_Data = pd.read_excel('../Data/商丘PM2.5研究-逐月(2013-2023)(完整数据).xlsx', engine='openpyxl')
# 日期数据提取
Year_Data = SQ_TenYears_Data.iloc[:, 0].tolist()
# AQI质量数据提取
AQI_Data = SQ_TenYears_Data.iloc[:, 1].tolist()
# PM2.5,PM10,SO2,CO,NO2,O3_8H六个数据提取
Pollutants_Data = SQ_TenYears_Data.iloc[:, 3:9]
# print(Pollutants_Data)
# 对气象数据的提取
Weather_Data = SQ_TenYears_Data.iloc[:, 9:13]
# print(Weather_Data)
# 研究对象总和
Research_Target = SQ_TenYears_Data.iloc[:, 3:13]
# print(Research_Target)
if __name__ == '__main__':
    # 0.Echarts
    liner_aqi(Year_Data, AQI_Data)
    # 1.相关性分析
    statistical_description_set(Research_Target)
    linear_test_set(Research_Target)
    pearson_correlation_set(Research_Target)
    spearman_correlation_set(Research_Target)
    # 2.多元线性回归方程
    stepwise_linear_regression_set(Research_Target, Pollutants_Data.iloc[:, 0])
    linear_regression_set(Research_Target, Pollutants_Data.iloc[:, 0])

  
    # # 3.主成分分析
    # factor_num = int(input("请输入主成分个数："))
    # factor_analysis_set(Research_Target, factor_num)
    # # 4.BP神经网络预测
