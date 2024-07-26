# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 15:36:16 2024

@author: admin
"""
from models.Nets import CNN  # , CNNCifar
from scipy.special import comb  # , perm
from models.aggregation_method import WeightedAvg
from arguments import args_parser
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import itertools
import numpy as np
import torch
import copy
import random
import math
import torch.nn.functional as F
all_gpus = range(torch.cuda.device_count())


class Shapley():
    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            'cuda:{}'.format(int(all_gpus[0]))
            if torch.cuda.is_available() else 'cpu'
        )

        # DA task settings
        self.task = args.task
        self.model_name = args.model_name
        self.Tst = torch.load(
            'data/%s%s/test.pt' % (args.dataset, args.data_allocation),
            map_location=self.device)
        self.X_test = []
        self.y_test = []
        for item in self.Tst:
            data, label = item[0], item[1]
            self.X_test.append(data.numpy().reshape(-1))
            self.y_test.append(label)
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)

        # player setting
        if self.task == 'DA':
            self.players = torch.load('data/%s%s/train0.pt' % (
                args.dataset, args.data_allocation),
                map_location=self.device)

        elif self.task == 'FL':
            self.players = [
                torch.load('data/%s%s/train%s.pt' % (
                    args.dataset, args.data_allocation, no),
                    map_location=self.device)
                for no in range(self.args.num_clients)]
        elif self.task == 'FA':
            self.players = None

        # utility setting
        self.utility_records = dict()
        self.utility_func = args.utility_func

        # SV settings
        self.SV = dict([(player_id, 0.0)
                       for player_id in range(len(self.players))])
        self.computation_method = args.computation_method

    def DA(self, player_idxs):
        utility = 0.0
        if self.model_name == 'KNN':
            # model initialize and training
            # (maybe expedited by some ML speedup functions)
            self.model = KNeighborsClassifier(
                n_neighbors=min(len(player_idxs), self.args.n_neighbors)
            )
            X_train = []
            y_train = []
            for idx in player_idxs:
                X_train.append(self.players[idx][0].numpy().reshape(-1))
                y_train.append(self.players[idx][1])
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            self.model.fit(X_train, y_train)

            # model testing (maybe expedited by some ML speedup functions)
            predictions = self.model.predict(self.X_test)
            # utility metric
            if self.utility_func == 'tst_accuracy':
                utility = accuracy_score(self.y_test, predictions)
            elif self.utility_func == 'tst_F1':
                utility = f1_score(self.y_test, predictions)
        else:
            # DA with other types of ML models are left for future experiments
            pass

        return utility

    def FL(self, player_idxs):
        utility = 0.0

        # model initialize and training
        # (maybe expedited by some ML speedup functions)
        self.model = CNN(args=self.args)
        loss_func = torch.nn.CrossEntropyLoss()

        for ridx in range(self.args.maxRound):
            localUpdates = dict()
            p_k = dict()
            for player_idx in player_idxs:
                p_k[player_idx] = self.players[player_idx].__len__
                ldr_train = DataLoader(self.players[player_idx],
                                       batch_size=self.args.local_bs,
                                       shuffle=True)

                local_model = copy.deepcopy(self.model).to(self.device)
                local_model.train()
                for (n, p) in local_model.named_parameters():
                    p.requires_grad = True
                optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=self.args.lr*(self.args.decay_rate**ridx),
                    momentum=self.args.momentum,
                    weight_decay=self.args.weight_decay)
                for iter_ in range(self.args.local_ep):
                    for batch_idx, batch in enumerate(ldr_train):
                        optimizer.zero_grad()
                        local_model.zero_grad()

                        data = batch[0].to(self.device)
                        labels = batch[1].to(self.device)
                        # forward
                        net_outputs = local_model(data)

                        # loss
                        loss = loss_func(net_outputs, labels)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            parameters=local_model.parameters(),
                            max_norm=self.args.max_norm,
                            norm_type=2)
                        optimizer.step()
                localUpdates[player_idx] = local_model.state_dict()

            # aggregation
            AggResults = WeightedAvg(localUpdates, p_k)
            self.model.load_state_dict(AggResults)

        # model testing (maybe expedited by some ML speedup functions)
        ldr_eval = DataLoader(self.Tst,
                              batch_size=self.args.test_bs,
                              shuffle=False)
        self.model = self.model.to(self.device)
        self.model.eval()
        data_size = 0
        batch_loss = []
        correct = 0
        for batch_idx, batch in enumerate(ldr_eval):
            data = batch[0].to(self.device)
            labels = batch[1].to(self.device)
            data_size += len(labels)
            # forward
            outputs = self.model(data)
            # metric
            if self.utility_func == 'tst_accuracy':
                y_pred = outputs.data.max(1, keepdim=True)[1]
                correct += y_pred.eq(labels.data.view_as(y_pred)
                                     ).long().cpu().sum()

            elif self.utility_func == 'tst_loss':
                loss = F.cross_entropy(outputs, labels, reduction='sum').item()
                batch_loss.append(loss)

        if self.utility_func == 'tst_accuracy':
            utility = float(correct.item() / data_size)*100
        elif self.utility_func == 'tst_loss':
            utility = sum(batch_loss) / data_size
        return utility

    def FA(self, player_idxs):
        utility = 0.0
        # not yet done
        # ...
        return utility

    def utilityComputation(self, players):
        if len(players) == 0:
            return 0.0

        utility_record_idx = str(sorted(players))
        if utility_record_idx not in self.utility_records:
            if self.task == 'DA':
                utility = self.DA(players)
            elif self.task == 'FL':
                utility = self.FL(players)
            elif self.task == 'FA':
                utility = self.FA(players)

            self.utility_records[utility_record_idx] = utility
        else:
            utility = self.utility_records[utility_record_idx]
        return utility

    def truncation(self, bef_addition, subset, player_idx):
        truncation_flag = False
        if np.abs(self.args.taskTotalUtility - bef_addition) < self.args.truncThreshold:
            utility_record_idx = str(sorted(list(subset)+[player_idx]))
            self.utility_records[utility_record_idx] = bef_addition
            truncation_flag = True
        return truncation_flag

    def Exact(self, truncation=False):
        for player_idx, player in enumerate(self.players):
            print('calculating SV for player %s...' % player_idx)
            self.SV[player_idx] = 0.0
            # tmp_playerSet = self.players[:player_idx] + self.players[player_idx+1:]
            tmp_playerSet = list(range(player_idx)) + \
                list(range(player_idx+1, len(self.players)))
            for subset_size in range(len(tmp_playerSet)):
                tmpSum = 0.0
                tmpCount = 0
                for subset_idx, subset in enumerate(
                        itertools.combinations(tmp_playerSet, subset_size)):
                    print('(Player %s Coalition Size %s/%s) sub-coalition No.%s/%s...' % (
                        player_idx, subset_size, len(tmp_playerSet),
                        subset_idx+1, comb(len(tmp_playerSet), subset_size)
                    ))
                    # utility before adding the targeted player
                    bef_addition = self.utilityComputation(subset)
                    # utility after adding the targeted player
                    aft_addition = self.utilityComputation(
                        list(subset)+[player_idx])

                    # gain in utility
                    tmpSum += aft_addition - bef_addition
                    tmpCount += 1
                self.SV[player_idx] += tmpSum / tmpCount
            self.SV[player_idx] /= len(self.players)

    def MC(self, truncation=False):
        # Monte Carlo sampling
        convergence = False
        iter_time = 0
        permutation = list(range(len(self.players)))
        while not convergence:
            iter_time += 1
            random.shuffle(permutation)
            convergence_diff = 0
            for order, player_idx in enumerate(permutation):
                print('Monte Carlo iteration %s for player %s at position %s...' % (
                    iter_time, player_idx, order))
                subset = permutation[:order]
                # utility before adding the targeted player
                bef_addition = self.utilityComputation(subset)

                if truncation:
                    if self.truncation(bef_addition, subset, player_idx):
                        aft_addition = bef_addition
                    else:
                        # utility after adding the targeted player
                        aft_addition = self.utilityComputation(
                            list(subset)+[player_idx])
                else:
                    # utility after adding the targeted player
                    aft_addition = self.utilityComputation(
                        list(subset)+[player_idx])

                # update SV
                old_SV = self.SV[player_idx]
                # start updating
                self.SV[player_idx] = (iter_time-1)/iter_time * self.SV[player_idx] + \
                    1/iter_time * (aft_addition - bef_addition)
                # compute difference
                convergence_diff += (self.SV[player_idx] - old_SV)**2

            if math.sqrt(convergence_diff) < self.args.convergence_threshold:
                convergence = True
            else:
                print(
                    'Monte Carlo iteration %s done ' % iter_time,
                    'with convergence_diff == %s...' % math.sqrt(
                        convergence_diff)
                )

    def SS(self, truncation=False):
        # stratified sampling (please refer to code in MC)
        pass

    def AS(self, truncation=False):
        # antithetic sampling (please refer to code in AS)
        pass

    def CalSV(self):
        if self.computation_method == 'exact':
            self.Exact()
        elif self.computation_method == 'MC':
            self.MC()
        elif self.computation_method == 'TMC':
            self.MC(truncation=True)
        else:
            print("No such computation method!!")


if __name__ == '__main__':
    args = args_parser()
    SVtask = Shapley(args)
    SVtask.CalSV()
    print('Experiment arguemtns: ', args)
    print("\n Final Resultant SVs: ", SVtask.SV)
