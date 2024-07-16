import itertools
import pandas as pd
import random
from model import DecisionTree, KNN, TKNN
from utils import data_sort_with_distance_to_test, indicator


class Shapley():
    def __init__(self, X, y, X_test, y_test, conv_threshold=0.05, window_size=5, model_family="KNN", task="classification", metric="f1", **model_args):
        self.X_test = X_test
        self.y_test = y_test
        self.df = X
        self.df['target'] = y
        self.df["df_index"] = range(len(self.df))
        self.df.set_index("df_index", inplace=True)

        self.SVs = {}
        self.MC_SVs = {}
        self.MC_SV_hist = {}

        self.conv_threshold = conv_threshold
        self.window_size = window_size

        self.model_family = model_family
        self.task = task
        self.metric = metric
        self.model_args = model_args

    def printSV(self):
        sum = 0.0
        for (i, sv) in list(self.SVs.items()):
            # print(sv)
            sum += sv
        print("SV_sum: ", sum)
        df = pd.DataFrame(list(self.SVs.items()), columns=['feature', 'SV'])
        df.to_csv("./res/SV_" + self.model_family + ".csv", index=False)

    def printMCSV(self):
        df = pd.DataFrame(list(self.MC_SVs.items()),
                          columns=['feature', 'MC_SV'])
        df.to_csv("./res/MC_SV_" + self.model_family + ".csv", index=False)

    def check_threshold_with_window(self):
        SV_num = len(self.MC_SV_hist[0])
        if SV_num <= self.window_size:
            return False
        for idx in range(len(self.df)):
            if SV_num != len(self.MC_SV_hist[idx]):
                raise RuntimeError("Error: SV_num != len(MC_SV_hist[idx])")
            new_window_SV = sum(
                self.MC_SV_hist[idx][SV_num - self.window_size:SV_num]) / self.window_size
            last_window_SV = sum(
                self.MC_SV_hist[idx][SV_num - self.window_size - 1:SV_num - 1]) / self.window_size
            if abs(new_window_SV) < 1e-18 and abs(last_window_SV) < 1e-18:
                continue
            last_window_SV = last_window_SV if abs(
                last_window_SV) > 1e-18 else 1e-18
            if abs(new_window_SV - last_window_SV) / last_window_SV > self.conv_threshold:
                print("Data point " + str(idx) +
                      " new_window_SV: " + str(new_window_SV))
                print("Data point " + str(idx) +
                      " last_window_SV: " + str(last_window_SV))
                print("Data point " + str(idx) + " not converged.")
                return False

            # print("Data point " + str(idx) + "converged.")

        return True

    def values(self, df):
        """calc the value of a data point set, which is the performance of the model trained on it."""
        if len(df) == 0:
            return 0
        df = df.reset_index(drop=True)
        X = df.drop(['target'], axis=1)
        y = df['target']

        # model = DecisionTree(X, self.X_test, y, self.y_test, self.model_args)
        if self.model_family == 'KNN':
            model = KNN(X, self.X_test, y, self.y_test, **self.model_args)
        elif self.model_family == 'DecisionTree':
            model = DecisionTree(
                X, self.X_test, y, self.y_test, **self.model_args)
        elif self.model_family == "TKNN":
            model = TKNN(X, self.X_test, y, self.y_test, **self.model_args)
        return model.score()

    def base_calc(self):
        # 计算Shapley值
        if self.model_family == "KNN":
            self.KNN_calc()
            return
        if self.model_family == "TKNN":
            self.TKNN_calc()
            return
        dp_num = len(self.df)
        for idx in range(dp_num):
            # 计算每个数据点的Shapley值
            print("Calc for data point: " + str(idx))
            SV = 0.0
            iter_time = 0
            for set_size in range(dp_num):
                print("set_size: " + str(set_size))
                for index_subset in itertools.combinations(range(dp_num), set_size):
                    index_sublist = list(index_subset)
                    print("List: " + str(index_sublist))
                    if idx in index_sublist:
                        continue
                    ori_value = self.values(self.df.loc[index_sublist])
                    index_sublist.append(idx)
                    new_value = self.values(self.df.loc[index_sublist])
                    SV += new_value - ori_value
                    iter_time += 1
            self.SVs[idx] = SV / iter_time
            print("Data point " + idx + " SV: " + str(self.SVs[idx]))
        self.printSV()

    def KNN_calc(self):
        N = len(self.df)
        K = self.model_args.get("K")
        dis_metric = self.model_args.get('dis_metric', 'euclidean')
        for idx in range(N):
            self.SVs[idx] = 0.0
        # Traverse each test point, and the final SV is the average value of SV based on each test point.
        for j in range(len(self.X_test)):
            # print("j:", j)
            SVs = [0.0 for _ in range(N + 1)]
            df_sorted = data_sort_with_distance_to_test(
                self.df, self.X_test.iloc[j], dis_metric)
            for i in range(N, 0, -1):
                if i == N:
                    SVs[i] = indicator(
                        df_sorted.loc[i - 1]['target'], self.y_test.iloc[j]) * 1.0 / N
                else:
                    SVs[i] = SVs[i + 1] + (indicator(df_sorted.loc[i - 1]['target'], self.y_test.iloc[j]) - indicator(
                        df_sorted.loc[i]['target'], self.y_test.iloc[j])) * min(K, i) * 1.0 / K / i

                self.SVs[i - 1] += SVs[i]
        for idx in range(N):
            self.SVs[idx] /= len(self.X_test)
        self.printSV()

    def TKNN_calc(self):
        pass

    def MC_calc(self):
        # 蒙特卡洛计算Shapley值
        dp_num = len(self.df)
        for idx in range(dp_num):
            self.MC_SVs[idx] = 0.0
            self.MC_SV_hist[idx] = []

        iter_time = 0
        while not self.check_threshold_with_window():
            iter_time += 1
            print("Monte Carlo Iteration: " + str(iter_time))
            permutation = random.sample(range(dp_num), dp_num)
            df = self.df.copy()
            df = df[0:0]
            ori_value = self.values(df)
            for idx in list(permutation):
                df.loc[idx] = self.df.loc[idx]
                new_value = self.values(df)
                self.MC_SVs[idx] += new_value - ori_value
                self.MC_SV_hist[idx].append(self.MC_SVs[idx] / iter_time)
                ori_value = new_value

        for idx in range(len(self.df)):
            self.MC_SVs[idx] /= iter_time

        self.printMCSV()

    def test_run(self):
        # df = self.df.reset_index(drop=True)
        # X = df.drop([self.goal], axis=1)
        # y = df[self.goal]
        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.3, random_state=46)
        # model = LogisticModel(X_train, X_test, y_train, y_test)
        # print(model.score())
        # print(self.values(self.df))
        df = self.df.reset_index(drop=True)
        X = df.drop(['target'], axis=1)
        y = df['target']
        model = KNN(X, self.X_test, y, self.y_test, **self.model_args)
        print(model.single_predict(self.X_test.iloc[[0]]))
        print(model.single_predict(self.X_test.iloc[[1]]))
        print(model.single_predict(self.X_test.iloc[[2]]))
        print(model.single_predict(self.X_test.iloc[[3]]))
        print(model.single_predict(self.X_test.iloc[[4]]))
        model.predict()
        print(model.score())
