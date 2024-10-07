# @Time: 2024/10/3 16:49
# @Author: Shen Hao
# @File: Pearson.py
# @system: Win10

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def pearson_correlation(element_parameters):
    # 计算皮尔逊相关系数矩阵
    correlation_matrix = element_parameters.corr(method='pearson')

    # 打印相关系数矩阵
    # print(correlation_matrix)

    # 绘制相关性矩阵热图
    plt.rcParams['font.family'] = ['SimSun']  # 设置字体为宋体
    plt.figure(figsize=(16, 12), dpi=300)  # 增加图像尺寸和 DPI

    """
    设置参数和标题
    """
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('皮尔逊相关性矩阵热图', fontsize=16)

    # 生成热力图
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 12})

    # 保存图像
    plt.savefig('../Chart/pearson_heatmap.png', bbox_inches='tight', dpi=300)

    plt.show()

    # 输出每个变量对之间的皮尔逊相关系数和 p 值，并标记显著性水平
    for col1 in element_parameters.columns:
        for col2 in element_parameters.columns:
            if col1 != col2:
                R, P = pearsonr(element_parameters[col1], element_parameters[col2])
                significance = ''
                if P < 0.01:
                    significance = '***'
                elif P < 0.05:
                    significance = '**'
                elif P < 0.1:
                    significance = '*'
                if col1 == 'PM2.5(μg/m³)':
                    print(f"变量 {col1} 和 {col2} 的皮尔逊相关系数: {R:.2f}, p 值: {P:.4f}, {significance}")
