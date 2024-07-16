from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, pairwise_distances
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode
import numpy as np
from utils import calc_distances, indicator, combinations_count


class Model():
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.prediction = self.model.predict(self.X_test)
        # print(self.prediction)
        pdf = pd.DataFrame(self.prediction, columns=['prediction'])
        pdf.to_csv('./res/prediction.csv', index=False)

    def score(self):
        self.train()
        self.predict()


class LogisticModel(Model):
    def __init__(self, X_train, X_test, y_train, y_test, **model_args):
        super().__init__(X_train, X_test, y_train, y_test)
        self.model = LogisticRegression(random_state=666)

    def train(self):
        # could do some data preprocessing here
        super().train()

    def predict(self):
        super().predict()

    def score(self):
        super().score()
        # print(self.y_test.shape, self.prediction.shape)
        df = pd.DataFrame(
            {'truth': self.y_test, 'prediction': self.prediction})
        df.to_csv('./res/truth_prediction.csv', index=False)
        return f1_score(self.y_test, self.prediction, average='weighted')


class DecisionTree(Model):
    def __init__(self, X_train, X_test, y_train, y_test, **model_args):
        super().__init__(X_train, X_test, y_train, y_test)
        self.model = DecisionTreeClassifier(random_state=42)

    def train(self):
        # could do some data preprocessing here
        super().train()

    def predict(self):
        super().predict()

    def score(self):
        super().score()
        # print(self.y_test.shape, self.prediction.shape)
        df = pd.DataFrame(
            {'truth': self.y_test, 'prediction': self.prediction})
        df.to_csv('./res/truth_prediction.csv', index=False)
        return f1_score(self.y_test, self.prediction, average='weighted')


class KNN(Model):
    def __init__(self, X_train, X_test, y_train, y_test, **model_args):
        super().__init__(X_train, X_test, y_train, y_test)
        n_neighbors = model_args.get('K', 3)
        n_neighbors = min(n_neighbors, len(X_train))
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self):
        # could do some data preprocessing here
        super().train()

    def predict(self):
        super().predict()

    def single_predict(self, df):
        return self.model.predict(df)

    def score(self):
        super().score()
        # print(self.y_test.shape, self.prediction.shape)
        # df = pd.DataFrame(
        #     {'truth': self.y_test, 'prediction': self.prediction})
        # df.to_csv('./res/truth_prediction.csv', index=False)
        return f1_score(self.y_test, self.prediction, average='weighted')


class TKNN(Model):
    def __init__(self, X_train, X_test, y_train, y_test, **model_args):
        super().__init__(X_train, X_test, y_train, y_test)
        self.threshold = model_args.get('threshold', 10)
        self.dis_metric = model_args.get('dis_metric', 'euclidean')

    def train(self):
        pass

    def predict(self):
        distances = pairwise_distances(
            self.X_test, self.X_train, metric=self.dis_metric)
        # Determine whether the distance between each test point and all points in the training set is less than the threshold.
        neighbors_within_threshold = distances <= self.threshold

        predictions = []
        for i in range(len(self.X_test)):
            neighbor_labels = self.y_train[neighbors_within_threshold[i]]
            if len(neighbor_labels) > 0:
                # If there are neighbors, predict the most common labels
                prediction = mode(neighbor_labels, keepdims=True).mode[0]
            else:
                # If no neighbor, returning None or the most common categories
                prediction = 0
            predictions.append(prediction)

        self.prediction = np.array(predictions)
        # print(self.prediction)
        # pdf = pd.DataFrame(self.prediction, columns=['prediction'])
        # pdf.to_csv('./res/prediction.csv', index=False)

    def score(self):
        super().score()
        df = pd.DataFrame(
            {'truth': self.y_test, 'prediction': self.prediction})
        df.to_csv('./res/truth_prediction_' + 'TKNN' + '.csv', index=False)
        return f1_score(self.y_test, self.prediction, average='weighted')


class TKNN_Shap():
    def __init__(self, X, y, X_val, y_val, t, num_of_classes):
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val
        self.t = t
        self.C = num_of_classes

        self.SVs = [0] * len(self.X)
        self.C_calc_init()

    def C_calc_init(self):
        self.cD = len(self.X)
        # 遍历每一个验证数据点，计算 Cz(val)(D)[i]，即 1 + D 中是验证点 i 的邻居的数量
        for val_i in range(len(self.X_val)):
            self.C_val_D[val_i] = 1
            self.C_val_D_plus[val_i] = 0
            self.is_neighbor[val_i] = [0] * self.cD
            self.is_same[val_i] = [0] * self.cD
            distances = calc_distances(self.X, self.X_val[val_i])
            for i, d in distances.iteritems():
                if self.y_val.iloc[val_i] == self.y.iloc[i]:
                    self.is_same[val_i][i] = 1
                if d < self.t:
                    self.is_neighbor[val_i][i] = 1
                    self.C_val_D[val_i] += 1
                    self.C_val_D_plus[val_i] += self.is_same[val_i][i]

    def C_val_i_for_i(self, val_i, i):
        return self.C_val_D[val_i] - self.is_neighbor[val_i][i]

    def C_val_i_for_i_plus(self, val_i, i):
        return self.C_val_D_plus[val_i] - self.is_neighbor[val_i][i] * self.is_same[val_i][i]

    def calc_SV_single_val(self, val_i):
        SVs = [0] * self.cD
        for i in range(self.cD):
            if self.is_neighbor[val_i][i] == 1:
                C_val_i_for_i = self.C_val_i_for_i(val_i, i)
                C_val_i_for_i_plus = self.C_val_i_for_i_plus(val_i, i)
                if C_val_i_for_i >= 2:
                    A1 = self.is_same[val_i][i] / C_val_i_for_i - \
                        C_val_i_for_i_plus / \
                        (C_val_i_for_i * (C_val_i_for_i - 1))
                    A2 = -1.0
                    for k in range(self.cD):
                        A2 += 1 / (k + 1) * (1 - combinations_count(self.cD - k,
                                                                    C_val_i_for_i) / combinations_count(self.cD + 1, C_val_i_for_i))
                    SVs[i] += A1 * A2
                SVs[i] += (self.is_same[val_i][i] -
                           1 / self.C) / C_val_i_for_i
        return SVs

    def calc_SVs(self):
        for val_i in range(len(self.X_val)):
            SVs = self.calc_SV_single_val(val_i)
            for i, SV in enumerate(SVs):
                self.SVs[val_i][i] += SV
        return self.SVs
