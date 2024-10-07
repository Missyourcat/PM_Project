# @Time: 2024/10/3 23:16
# @Author: Shen Hao
# @File: Stepwise_Linear_Regression.py
# @system: Win10
import pandas as pd
import numpy as np
import statsmodels.api as sm


# 自变量矩阵
def stepwise_linear_regression(element_parameters):
    X = element_parameters.iloc[:, 1:6]
    y = element_parameters.iloc[:, 0]
    final_model, included_features = stepwise_selection(X, y)
    # print(result)
    # print(b)
    print(final_model)
    print(included_features)


def stepwise_selection(X, y, initial_list=None, threshold=0.05, mark=True):
    # 初始化未被选中的因素
    if initial_list is None:
        initial_list = []
    b = None
    excluded = list(initial_list)
    # 初始化被选中的因素
    included = list(set(X.columns) - set(excluded))

    while mark:

        # 拟合选中的因素
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        aic = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit().aic
        print("AIC:", aic)
        # 得出被选中的因素里P值最大的
        p_values = model.pvalues

        # 排除常数项（截距项）的P值
        p_values = p_values.drop('const')

        # 找到最大的P值及其对应的特征
        max_p_value = p_values.max()
        print("最大的P值为:", max_p_value)
        if max_p_value < threshold:
            mark = False
            print("最终模型:")
            print(model.summary())
            # 获取回归系数
            b = model.params
        else:
            max_p_feature = p_values.idxmax()
            print("最大的P值对应的特征为:", max_p_feature)

            # 从被选中的因素中去除
            included.remove(max_p_feature)
            print("更新后的因素为:", included)

    return included, b



