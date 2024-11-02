# @Time: 2024/11/2 09:39
# @Author: Shen Hao
# @File: Shapiro_wilk.py
# @system: Win10
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import shapiro, norm

def shapiro_wilk_test(element_parameters):
    """
        对DataFrame中的每个列执行Shapiro-Wilk检验，并绘制直方图与正态分布拟合曲线。

        :param df: 包含需要测试的数据的pandas DataFrame。
    """
    num_plots = len(element_parameters.columns)
    rows = (num_plots + 1) // 2  # 计算所需的行数
    fig, axs = plt.subplots(rows, 2, figsize=(15, 5 * rows))
    axs = axs.ravel()  # 将axs展平以便于循环访问

    for idx, col in enumerate(element_parameters.columns):
        data = element_parameters[col].dropna()  # 删除缺失值
        # Shapiro-Wilk检验
        sw_stat, p_val = shapiro(data)
        # 绘制直方图
        axs[idx].hist(data, bins=30, density=True, alpha=0.6, color='b', edgecolor='black')
        # 绘制正态分布拟合曲线
        mu, std = norm.fit(data)  # 计算均值和标准差
        xmin, xmax = axs[idx].get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        axs[idx].plot(x, p, 'k', linewidth=2)
        title = f"{col}\nSW 检验: {sw_stat:.2f}, P-值: {p_val:.4f}"
        axs[idx].set_title(title)
        axs[idx].grid(True)

    # 如果最后一行只有一个图，则删除空的ax
    if num_plots % 2 == 1:
        fig.delaxes(axs[-1])

    plt.tight_layout()
    plt.savefig('../Chart/shapiro_wilk_plots.png', bbox_inches='tight', dpi=300)  # 保存图像
    plt.show()