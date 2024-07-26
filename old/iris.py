from sklearn.datasets import load_iris
from Shapley import Shapley
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import data_sort_with_distance_to_test

# 加载鸢尾花数据集
iris = load_iris()

# 数据和标签
data = iris.data
labels = iris.target

# 查看数据
# print(type(data))  # 打印前五行数据
# print(type(labels))  # 打印前五个标签
# print(iris.feature_names)
# print(iris.target_names)
df = pd.DataFrame(iris.data, columns=iris.feature_names)
target = pd.Series(iris.target)
# print("target", iris.target)
# df.to_csv('iris.csv', index=False)

X, X_test, y, y_test = train_test_split(
    df, target, test_size=0.3, random_state=45)
# print(type(X_test.iloc[0]))
# data_sort_with_distance_to_test(X, X_test.iloc[0])
# print(y_test.iloc[2])
# print("end")

# shapley = Shapley(X, y, X_test, y_test, model_family='KNN', K=10)
# shapley.test_run()
# shapley.base_calc()
# shapley.MC_calc()

# shapley = Shapley(X, y, X_test, y_test, model_family='TKNN',
#                   threshold=1, dis_metric='euclidean')
# shapley.test_run()
# shapley.MC_calc()
# shapley.base_calc()

shapley = Shapley(X, y, X_test, y_test, model_family="DecisionTree")
shapley.base_calc()
