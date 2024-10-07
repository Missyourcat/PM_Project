# @Time: 2024/10/3 15:32
# @Author: Shen Hao
# @File: QQ_Plot.py
# @system: Win10
import matplotlib.pyplot as plt
import scipy.stats as stats

# 设置 Matplotlib 使用支持中文的字体
plt.rcParams['font.family'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def qq_plot(element_parameters):
    # 获取列的数量
    num_columns = len(element_parameters.columns)

    # 计算所需的子图数量
    num_subplots = min(num_columns, 12)
    num_rows = 3
    num_cols = 4

    # 创建子图布局
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(24, 18), dpi=300)  # 增加图像尺寸和 DPI

    # 将axes转换为一维数组以便遍历
    axes = axes.flatten()

    # 遍历每一列参数并绘制QQ图
    for i, column in enumerate(element_parameters.columns):
        if i < num_subplots:
            data = element_parameters[column]

            # 绘制QQ图
            stats.probplot(data, dist="norm", plot=axes[i])

            # 添加标题和标签
            axes[i].set_title(f"{column}-QQ图正态分布", fontsize=16)
            axes[i].set_xlabel("理论分位数", fontsize=14)
            axes[i].set_ylabel("样本分位数", fontsize=14)
            axes[i].tick_params(axis='both', which='major', labelsize=12)

    # 删除多余的子图
    for i in range(num_subplots, num_rows * num_cols):
        fig.delaxes(axes[i])

    # 调整子图间距
    plt.tight_layout()

    # 保存图像
    plt.savefig('../Chart/qq_plots_4k.png', bbox_inches='tight', dpi=300)

    # 显示图形
    plt.show()