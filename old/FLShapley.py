import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
import numpy as np
import random
from math import log, floor


class Player:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.SV = 0

    def local_train(self, model):
        model.fit(self.X_train, self.y_train, epochs=5)
        return model.get_weights()


class FLShapley:
    def __init__(self, player_num, iter_time, participant_fraction_each_round):
        # 加载MNIST数据集
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()
        self.player_num = player_num
        num_train_per_player = self.X_train.shape[0] // player_num
        self.players = []
        for i in range(player_num):
            self.players.append(Player(self.X_train[i * num_train_per_player:(
                i + 1) * num_train_per_player], self.y_train[i * num_train_per_player:(i + 1) * num_train_per_player]))
        self.iter_time = iter_time
        self.participant_fraction_each_round = participant_fraction_each_round

    def create_model(self):
        model = Sequential([
            Flatten(input_shape=(28, 28)),  # 假设输入数据是28x28的图像
            Dense(128, activation='relu'),  # 一个有128个神经元的隐藏层
            Dense(128, activation='relu'),  # 一个有128个神经元的隐藏层
            Dense(10, activation='softmax')  # 输出层，对应10个类别
        ])
        # 编译模型
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def copy_model(self, model):
        copied_model = self.create_model()
        copied_model.set_weights(model.get_weights())
        return copied_model

    def global_train(self, model):
        player_weights = []
        for player in self.players:
            player_weights.append(player.local_train(
                self.copy_model(model)))
        new_weights = np.mean(np.array(player_weights), axis=0)
        model.set_weights(new_weights)
        return model

    def calc_T(self, m, epsilon=0.1, delta=0.1, r=1):
        return 2*r*r*log(2*m/delta)/(epsilon*epsilon)

    def value(self, model):
        loss, accuracy = model.evaluate(self.X_test, self.y_test)
        return accuracy

    def calc(self):
        model = self.create_model()
        for t in range(self.iter_time):
            local_models = []
            m = self.player_num * self.participant_fraction_each_round
            selected_players = random.sample(list(range(self.player_num)), m)
            for k in selected_players:
                local_models.append(
                    self.players[k].local_train(self.copy_model(model)))

            U_prev = self.value(model)
            s = [0] * m
            T = floor(self.calc_T(m))
            for t in range(T):
                for i in random.permutation(np.array(list(range(m)))):
                    new_model = model.set_weights(
                        np.mean(np.array(local_models[:i]), axis=0))
                    U = self.value(new_model)
                    s[i] = U - U_prev

            for i, k in enumerate(selected_players):
                self.players[k].SV += s[i] / T
        pass
