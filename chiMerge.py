import pandas as pd
import numpy as np
from scipy.stats import chi2

def calc_chiSquare(df, feature, target):
    """

    :param df: 样本集
    :param feature: 目标特征
    :param target: 目标Y值 (0或1) Y值为二分类变量
    :return: 卡方统计量dataframe
        feature: 特征名称
        act_target_cnt: 实际坏样本数
        expected_target_cnt：期望坏样本数
        chi_square：卡方统计量
    """

    # 计算样本期望频率
    target_cnt = df[target].sum()
    sample_cnt = len(df[target])
    expected_ratio = target_cnt * 1.0/sample_cnt
    # 对变量按属性值从大到小排序
    df = df[[feature, target]]
    col_value = sorted(list(set(df[feature])))
    # 计算每一个属性值对应的卡方统计量等信息

    chi_list = []; target_list = []; expected_target_list = []
    for value in col_value:
        df_target_cnt = df.loc[df[feature] == value, target].sum()
        df_cnt = len(df.loc[df[feature] == value, target])
        expected_target_cnt = df_cnt * expected_ratio
        chi_square = (df_target_cnt - expected_target_cnt)**2 / expected_target_cnt
        chi_list.append(chi_square)
        target_list.append(df_target_cnt)
        expected_target_list.append(expected_target_cnt)
    # 结果输出到dataframe, 对应字段为特征属性值, 卡方统计量, 实际坏样本量, 期望坏样本量
    chi_stats = pd.DataFrame({feature:col_value, 'chi_square':chi_list,
                               'act_target_cnt':target_list, 'expected_target_cnt':expected_target_list})
    return chi_stats[[feature, 'act_target_cnt', 'expected_target_cnt', 'chi_square']]

'''
分箱主体部分包括两种分箱方法的主体函数，其中merge_chiSquare()是对区间进行合并，
get_chiSquare_distribution()是根据自由度和置信度得到卡方阈值。
我在这里设置的是自由度为4，置信度为10%。两个自定义函数如下
'''
def get_chiSquare_distuibution(dfree=4, cf=0.1):
    '''
    根据自由度和置信度得到卡方分布和阈值
    params:
        dfree: 自由度, 最大分箱数-1, default 4
        cf: 显著性水平, default 10%
    return:
        卡方阈值
    '''
    percents = [0.95, 0.90, 0.5, 0.1, 0.05, 0.025, 0.01, 0.005]
    df = pd.DataFrame(np.array([chi2.isf(percents, df=i) for i in range(1, 30)]))
    df.columns = percents
    df.index = df.index+1
    # 显示小数点后面数字
    pd.set_option('precision', 3)
    return df.loc[dfree, cf]

def merge_chiSquare(chi_result, index, mergeIndex, a = 'expected_target_cnt',
                    b = 'act_target_cnt', c = 'chi_square'):
    '''
    params:
        chi_result: 待合并卡方数据集
        index: 合并后的序列号
        mergeIndex: 需合并的区间序号
        a, b, c: 指定合并字段
    return:
        分箱合并后的卡方dataframe
    '''
    chi_result.loc[mergeIndex, a] = chi_result.loc[mergeIndex, a] + chi_result.loc[index, a]
    chi_result.loc[mergeIndex, b] = chi_result.loc[mergeIndex, b] + chi_result.loc[index, b]
    chi_result.loc[mergeIndex, c] = (chi_result.loc[mergeIndex, b] - chi_result.loc[mergeIndex, a])**2 /chi_result.loc[mergeIndex, a]
    chi_result = chi_result.drop([index])
    chi_result = chi_result.reset_index(drop=True)
    return chi_result



def chiMerge_maxInterval(chi_stats, feature, maxInterval=5):
    '''
    卡方分箱合并--最大区间限制法
    params:
        chi_stats: 卡方统计量dataframe
        feature: 目标特征
        maxInterval：最大分箱数阈值
    return:
        卡方合并结果dataframe, 特征分割split_list
    '''
    group_cnt = len(chi_stats)
    split_list = [chi_stats[feature].min()]
    # 如果变量区间超过最大分箱限制，则根据合并原则进行合并
    while (group_cnt > maxInterval):
        min_index = chi_stats[chi_stats['chi_square'] == chi_stats['chi_square'].min()].index.tolist()[0]
        # 如果分箱区间在最前,则向下合并
        if min_index == 0:
            chi_stats = merge_chiSquare(chi_stats, min_index, min_index + 1)
        # 如果分箱区间在最后，则向上合并
        elif min_index == group_cnt - 1:
            chi_stats = merge_chiSquare(chi_stats, min_index - 1, min_index)
        # 如果分箱区间在中间，则判断与其相邻的最小卡方的区间，然后进行合并
        else:
            if chi_stats.loc[min_index - 1, 'chi_square'] > chi_stats.loc[min_index + 1, 'chi_square']:
                chi_stats = merge_chiSquare(chi_stats, min_index, min_index + 1)
            else:
                chi_stats = merge_chiSquare(chi_stats, min_index - 1, min_index)
        group_cnt = len(chi_stats)
    chiMerge_result = chi_stats
    split_list.extend(chiMerge_result[feature].tolist())
    return chiMerge_result, split_list

# var 表示需要分箱的变量，函数返回卡方统计结果，包括样本实例区间，卡方统计量，响应频率和期望响应频率。
def chiMerge_minChiSquare(chi_stats, feature, dfree=4, cf=0.1, maxInterval=5):
    '''
    卡方分箱合并--卡方阈值法
    params:
        chi_stats: 卡方统计量dataframe
        feature: 目标特征
        maxInterval: 最大分箱数阈值, default 5
        dfree: 自由度, 最大分箱数-1, default 4
        cf: 显著性水平, default 10%
    return:
        卡方合并结果dataframe, 特征分割split_list
    '''
    threshold = get_chiSquare_distuibution(dfree, cf)
    min_chiSquare = chi_stats['chi_square'].min()
    group_cnt = len(chi_stats)
    split_list = [chi_stats[feature].min()]
    # 如果变量区间的最小卡方值小于阈值，则继续合并直到最小值大于等于阈值
    while (min_chiSquare < threshold and group_cnt > maxInterval):
        min_index = chi_stats[chi_stats['chi_square'] == chi_stats['chi_square'].min()].index.tolist()[0]
        # 如果分箱区间在最前,则向下合并
        if min_index == 0:
            chi_stats = merge_chiSquare(chi_stats, min_index + 1, min_index)
        # 如果分箱区间在最后，则向上合并
        elif min_index == group_cnt - 1:
            chi_stats = merge_chiSquare(chi_stats, min_index - 1, min_index)
        # 如果分箱区间在中间，则判断与其相邻的最小卡方的区间，然后进行合并
        else:
            if chi_stats.loc[min_index - 1, 'chi_square'] > chi_stats.loc[min_index + 1, 'chi_square']:
                chi_stats = merge_chiSquare(chi_stats, min_index, min_index + 1)
            else:
                chi_stats = merge_chiSquare(chi_stats, min_index - 1, min_index)
        min_chiSquare = chi_stats['chi_square'].min()
        group_cnt = len(chi_stats)
    chiMerge_result = chi_stats
    split_list.extend(chiMerge_result[feature].tolist())
    return chiMerge_result, split_list