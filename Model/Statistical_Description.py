# @Time: 2024/9/25 09:04
# @Author: Shen Hao
# @File: Statistical_Description.py
# @system: Win10
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def statistical_description(element_parameters):
    # 计算描述性统计量
    MIN = np.min(element_parameters, axis=0)
    MAX = np.max(element_parameters, axis=0)
    MEAN = np.mean(element_parameters, axis=0)
    MEDIAN = np.median(element_parameters, axis=0)
    SKEWNESS = skew(element_parameters, axis=0)
    KURTOSIS = kurtosis(element_parameters, axis=0)
    STD = np.std(element_parameters, axis=0)

    # 将这些统计量放到一个矩阵中
    RESULT = pd.DataFrame({
        '最小值': MIN,
        '最大值': MAX,
        '均值': MEAN,
        '中位数': MEDIAN,
        '偏度': SKEWNESS,
        '峰度': KURTOSIS,
        '标准差': STD
    })

    print(RESULT)
    return RESULT
