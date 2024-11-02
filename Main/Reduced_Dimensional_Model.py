# @Time: 2024/10/2 22:47
# @Author: Shen Hao
# @File: Reduced_Dimensional_Model.py
# @system: Win10
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity
from mpl_toolkits.mplot3d import Axes3D

def factor_analysis_set(df, n_factors):
    # 数据标准化
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # KMO 和 Bartlett's 检验
    kmo_all, kmo_model = calculate_kmo(df_scaled)
    chi_square_value, p_value = calculate_bartlett_sphericity(df_scaled)

    print(f'KMO 值: {kmo_model}')
    print(f'Bartlett\'s 球形检验卡方值: {chi_square_value}, P-值: {p_value}')

    if kmo_model < 0.6 or p_value > 0.05:
        print("数据可能不适合进行因子分析。")
        return

    # 进行因子分析
    fa = FactorAnalyzer(rotation='varimax', n_factors=n_factors)
    fa.fit(df_scaled)

    # 获取因子载荷
    loadings = fa.loadings_
    print('因子载荷：')
    print(loadings)

    # 绘制因子载荷图
    plt.figure(figsize=(10, 8))
    sns.heatmap(loadings, annot=True, cmap='coolwarm')
    plt.title('因子载荷热力图')
    plt.show()

    # 解释总方差
    ev, v = fa.get_eigenvalues()
    plt.figure(figsize=(8, 6))
    plt.scatter(range(1, df.shape[1] + 1), ev)
    plt.plot(range(1, df.shape[1] + 1), ev)
    plt.title('碎石图')
    plt.xlabel('因子')
    plt.ylabel('特征值')
    plt.grid()
    plt.show()

    # 计算旋转前后的方差解释率
    variance_explained = fa.get_factor_variance()
    total_variance_explained = pd.DataFrame(variance_explained,
                                            index=['旋转前方差解释率', '旋转后方差解释率', '累积方差解释率']).T
    total_variance_explained['特征根'] = ev[:n_factors]

    # 输出方差解释表格
    print('总方差解释表：')
    print(total_variance_explained)

    # 转换原始数据到因子空间
    transformed_data = fa.transform(df_scaled)

    # 输出因子分析法综合得分
    scores = pd.DataFrame(transformed_data, columns=[f'因子{i + 1}' for i in range(n_factors)])
    print('因子分析得分：')
    print(scores.head())

    # 因子载荷象限分析
    if n_factors >= 2:
        plt.figure(figsize=(10, 8))
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.axvline(x=0, color='gray', linestyle='--')
        plt.scatter(loadings[:, 0], loadings[:, 1])

        for i, (x, y) in enumerate(zip(loadings[:, 0], loadings[:, 1])):
            plt.text(x, y, df.columns[i], ha='center', va='center')

        plt.xlabel('因子1')
        plt.ylabel('因子2')
        plt.title('因子载荷象限图')
        plt.grid(True)
        plt.show()
    else:
        print("因子数量少于2，无法绘制因子载荷象限图。")

    if n_factors >= 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制坐标轴
        ax.plot([-1, 1], [0, 0], [0, 0], color='gray', linestyle='--')  # X 轴上的线
        ax.plot([0, 0], [-1, 1], [0, 0], color='gray', linestyle='--')  # Y 轴上的线
        ax.plot([0, 0], [0, 0], [-1, 1], color='gray', linestyle='--')  # Z 轴上的线

        # 绘制散点图
        ax.scatter(loadings[:, 0], loadings[:, 1], loadings[:, 2])

        # 添加标签
        for i, (x, y, z) in enumerate(zip(loadings[:, 0], loadings[:, 1], loadings[:, 2])):
            ax.text(x, y, z, df.columns[i], zdir=None, ha='center', va='center')

        ax.set_xlabel('因子1')
        ax.set_ylabel('因子2')
        ax.set_zlabel('因子3')
        ax.set_title('因子载荷三维图')
        plt.grid(True)
        plt.show()
    else:
        print("因子数量少于3，无法绘制三维因子载荷图。")

    # 成分矩阵表
    component_matrix = pd.DataFrame(loadings, columns=[f'因子{i + 1}' for i in range(n_factors)], index=df.columns)
    print('成分矩阵表：')
    print(component_matrix)

    # 计算模型公式
    model_formula = []
    for i in range(n_factors):
        factor_formula = f'F{i + 1} = '
        for j, col in enumerate(df.columns):
            factor_formula += f'{loadings[j, i]:.3f} * {col} + '
        factor_formula = factor_formula.rstrip(' + ')
        model_formula.append(factor_formula)

    # 计算综合得分公式
    cumulative_variance = total_variance_explained['累积方差解释率'].iloc[-1]
    weights = total_variance_explained['旋转后方差解释率'] / cumulative_variance
    composite_score_formula = 'F = '
    for i in range(n_factors):
        composite_score_formula += f'{weights[i]:.3f} * F{i + 1} + '
    composite_score_formula = composite_score_formula.rstrip(' + ')

    print('模型公式：')
    for formula in model_formula:
        print(formula)
    print('综合得分公式：')
    print(composite_score_formula)

    return scores, total_variance_explained