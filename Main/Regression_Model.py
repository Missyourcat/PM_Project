import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from Model.Linear_Regresssion import linear_regression
from Model.Stepwise_Linear_Regression import stepwise_linear_regression


def stepwise_linear_regression_set(element_parameters, PM25_parameters):
    # 调用逐步回归函数，获取最终模型
    final_model = stepwise_linear_regression(element_parameters)

    # 提取最终模型的自变量
    included_features = final_model.model.exog_names[1:]  # 排除常数项

    # 使用最终模型预测新数据
    X_final = element_parameters[included_features]
    X_final_with_const = sm.add_constant(X_final)
    y_pred = final_model.predict(X_final_with_const)

    def plot_fitted_vs_actual(actual, predicted):
        plt.figure(figsize=(10, 6))

        # 创建一个索引用于排序
        index = np.arange(len(actual))

        # 绘制实际值折线
        plt.plot(index, actual, label='PM2.5 真实值', marker='o', linestyle='-', color='blue')

        # 绘制预测值折线
        plt.plot(index, predicted, label='PM2.5 预测值', marker='x', linestyle='--', color='red')

        # 添加标题和标签
        plt.title('逐步回归PM2.5拟合效果图')
        plt.xlabel('')
        plt.ylabel('PM2.5')
        plt.legend()

        # 保存图表
        plt.savefig('../Chart/stepwise_Fitted_vs_Actual_PM2.5_Line.png', bbox_inches='tight', dpi=100)

        # 显示图表
        plt.show()
    # 绘制实际值与预测值的对比图
    plot_fitted_vs_actual(PM25_parameters, y_pred)

def linear_regression_set(element_parameters, PM25_parameters):
    final_model = linear_regression(element_parameters)
    # 提取最终模型的自变量
    included_features = final_model.model.exog_names[1:]  # 排除常数项

    # 使用最终模型预测新数据
    X_final = element_parameters[included_features]
    X_final_with_const = sm.add_constant(X_final)
    y_pred = final_model.predict(X_final_with_const)

    def plot_fitted_vs_actual(actual, predicted):
        plt.figure(figsize=(10, 6))

        # 创建一个索引用于排序
        index = np.arange(len(actual))

        # 绘制实际值折线
        plt.plot(index, actual, label='PM2.5 真实值', marker='o', linestyle='-', color='blue')

        # 绘制预测值折线
        plt.plot(index, predicted, label='PM2.5 预测值', marker='x', linestyle='--', color='red')

        # 添加标题和标签
        plt.title('线性回归PM2.5拟合效果图')
        plt.xlabel('')
        plt.ylabel('PM2.5')
        plt.legend()

        # 保存图表
        plt.savefig('../Chart/linear_Fitted_vs_Actual_PM2.5_Line.png', bbox_inches='tight', dpi=100)

        # 显示图表
        plt.show()
    # 绘制实际值与预测值的对比图
    plot_fitted_vs_actual(PM25_parameters, y_pred)



