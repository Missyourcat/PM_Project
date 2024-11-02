# @Time: 2024/11/2 11:04
# @Author: Shen Hao
# @File: Linear_Regresssion.py
# @system: Win10
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import networkx as nx


# 纯线性回归函数
def linear_regression(element_parameters):
    X = element_parameters.iloc[:, 1:6]
    y = element_parameters.iloc[:, 0]

    # 添加常数项
    X = sm.add_constant(X)

    # 拟合线性回归模型
    model = sm.OLS(y, X).fit()

    # 计算VIF
    vif_data = calculate_vif(X)

    # 计算标准化系数
    standardized_coefficients = calculate_standardized_coefficients(model, X, y)
    standardized_coefficients_without_const = standardized_coefficients.drop('const')

    print("\n模型拟合情况:")
    print(f"R²: {model.rsquared:.3f}")
    print(f"调整R²: {model.rsquared_adj:.3f}")
    print("\n共线性分析 (VIF):")
    print(vif_data)
    print("\n最终模型的系数:")
    print(model.summary())
    print("\n标准化系数:")
    print(standardized_coefficients)
    print("\n标准化系数（不含常数项）:")
    print(standardized_coefficients_without_const)

    # 绘制模型路径图
    plot_model_path(standardized_coefficients_without_const, y.name)

    formula = build_model_formula(model)
    print(f"模型公式为: {formula}")
    return model


# 计算VIF
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["变量"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data


# 计算标准化系数
def calculate_standardized_coefficients(model, X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.drop(columns='const'))
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
    X_scaled_with_const = sm.add_constant(X_scaled)
    model_scaled = sm.OLS(y_scaled, X_scaled_with_const).fit()
    standardized_coefficients = pd.Series(model_scaled.params, index=['const'] + list(X.columns[1:]))
    return standardized_coefficients


# 绘制模型路径图
def plot_model_path(coefficients, target_variable):
    # 创建有向图
    G = nx.DiGraph()

    # 添加目标变量节点
    G.add_node(target_variable)

    # 添加其他变量节点及其与目标变量的边
    for variable in coefficients.index:
        G.add_node(variable)
        G.add_edge(target_variable, variable, weight=coefficients[variable])

    # 使用 spring_layout 自动调整节点位置
    pos = nx.spring_layout(G, k=0.5, iterations=50)  # k 控制节点间的距离，iterations 是迭代次数

    # 绘图
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=12, font_weight='bold',
            arrows=True)

    # 添加边的权重标签
    edge_labels = {(target_variable, variable): f"{coefficients[variable]:.2f}" for variable in coefficients.index}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    # 设置标题
    plt.title('模型路径图')

    # 保存图表
    plt.savefig('../Chart/Linear_Regression_4k.png', bbox_inches='tight', dpi=100)

    # 显示图表
    plt.show()


# 构建模型公式
def build_model_formula(model):
    params = model.params
    formula_parts = [f"{params[name]:.2f} * {name}" for name in params.index if name != 'const']
    formula_parts.insert(0, f"{params['const']:.2f}")
    formula = " + ".join(formula_parts)
    return formula