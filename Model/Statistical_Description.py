# @Time: 2024/9/25 09:04
# @Author: Shen Hao
# @File: Statistical_Description.py
# @system: Win10
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
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
    # 创建一个图像和子图
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(15, 20))  # 调整nrows和ncols以适应你的数据集
    axes = axes.flatten()  # 将二维数组转换成一维

    # 对每个特征单独绘制箱型图
    for i, column in enumerate(element_parameters.columns):
        ax = axes[i]
        ax.boxplot(element_parameters[column], showmeans=True)
        ax.set_title(f'箱型图 - {column}')
        ax.set_ylabel('Value')

    # 调整布局，避免重叠
    plt.tight_layout()

    # 保存图像
    plt.savefig('../Chart/boxplots.png', bbox_inches='tight', dpi=300)

    # 显示图像
    plt.show()

    print(RESULT.round(2))
    return RESULT
