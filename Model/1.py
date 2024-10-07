# @Time: 2024/10/7 17:23
# @Author: Shen Hao
# @File: 1.py
# @system: Win10
# 定义向前逐步回归函数
from statsmodels.formula.api import ols


def forward_select(data, target):
    variate = set(data.columns)  # 将字段名转换成字典类型
    variate.remove(target)  # 去掉因变量的字段名
    selected = []
    current_score, best_new_score = float('inf'), float('inf')  # 目前的分数和最好分数初始值都为无穷大（因为AIC越小越好）
    # 循环筛选变量
    while variate:
        aic_with_variate = []
        for candidate in variate:  # 逐个遍历自变量
            formula = "{}~{}".format(target, "+".join(selected + [candidate]))  # 将自变量名连接起来
            aic = ols(formula=formula, data=data).fit().aic  # 利用ols训练模型得出aic值
            aic_with_variate.append((aic, candidate))  # 将第每一次的aic值放进空列表
        aic_with_variate.sort(reverse=True)  # 降序排序aic值
        best_new_score, best_candidate = aic_with_variate.pop()  # 最好的aic值等于删除列表的最后一个值，以及最好的自变量等于列表最后一个自变量
        if current_score > best_new_score:  # 如果目前的aic值大于最好的aic值
            variate.remove(best_candidate)  # 移除加进来的变量名，即第二次循环时，不考虑此自变量了
            selected.append(best_candidate)  # 将此自变量作为加进模型中的自变量
            current_score = best_new_score  # 最新的分数等于最好的分数
            print("aic is {},continuing!".format(current_score))  # 输出最小的aic值
        else:
            print("for selection over!")
            break
    formula = "{}~{}".format(target, "+".join(selected))  # 最终的模型式子
    print("final formula is {}".format(formula))
    model = ols(formula=formula, data=data).fit()
    return (model)

