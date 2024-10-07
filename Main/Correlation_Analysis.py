# @Time: 2024/10/2 22:44
# @Author: Shen Hao
# @File: Correlation_Analysis.py
# @system: Win10
from Model.Pearson import pearson_correlation
from Model.QQ_Plot import qq_plot
from Model.Statistical_Description import statistical_description

# 描述描述
"""
对AQI各项指标数进行描述描述
"""
def statistical_description_set(element_parameters):
    statistical_description(element_parameters)
# 线性关系检验
"""
1.正态分布JB检验
2.QQ图
"""
def linear_test_set(element_parameters):
    # JB_test(element_parameters)
    qq_plot(element_parameters)
"""
Person相关系数
"""
def pearson_correlation_set(element_parameters):
    pearson_correlation(element_parameters)