import pandas as pd
import numpy as np
from itertools import chain, combinations
import math


def indicator(a, b):
    return 1 if a == b else 0


def calc_distances(df, target):
    return np.sqrt(((df - target) ** 2).sum(axis=1))


def data_sort_with_distance_to_test(df, test, dis_metric):
    # print("type: ", type(df), type(test))

    # 计算距离并进行升序排序
    if dis_metric == "euclidean":
        distances = calc_distances(df, test)
        df_sorted = df.assign(distance=distances).sort_values(by='distance')
        # print(df_sorted.head(10))
        return df_sorted


def combinations_count(n, k):
    return int(math.factorial(n) / (math.factorial(k) * math.factorial(n - k)))
