import pandas as pd
import numpy as np

# 假设df是你的DataFrame
df = pd.DataFrame({
    'feature1': [1, 2, 3, 4],
    'feature2': [5, 6, 7, 8],
    'feature3': [9, 10, 11, 12]
})

# 定义一个目标数据点
target_point = df.loc[0]

# 计算每个数据点到目标点的欧氏距离
distances = np.sqrt(((df - target_point) ** 2).sum(axis=1))

print(distances.iloc[1])
