#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--manual_seed', type=int, default=42, 
                        help="random seed")
    parser.add_argument('--gpu', type=int, default=-1, 
                        help="GPU ID, -1 for CPU")
    parser.add_argument('--cuda', type=str, default=None, help="")
    
    # task parameters
    parser.add_argument('--task', type=str, default="DA", 
                        help="{DA, FL, FA}")
    # FL task parameter
    parser.add_argument('--maxRound', type=int, default=10, 
                        help="FL task  parameter")
    parser.add_argument('--local_ep', type=int, default=3, 
                        help="FL task  parameter")
    parser.add_argument('--local_bs', type=int, default=64, 
                        help="FL task  parameter")
    parser.add_argument('--num_clients', type=int, default=10, 
                        help="FL task  parameter")
    parser.add_argument('--test_bs', type=int, default=128, 
                        help="FL task  parameter")
    
    # dataset parameters
    parser.add_argument('--dataset', type=str, default="mnist", 
                        help="{MNIST, Iris}")
    parser.add_argument('--num_classes', type=int, default=10, 
                        help="number of classes")
    parser.add_argument('--num_trainDatasets', type=int, default=1, 
                        help="range(1,10)")
    parser.add_argument('--data_allocation', type=int, default=0, 
                        help="{0,1,2,3,4,5}")
    parser.add_argument('--data_size_group', type=int, default=1, 
                        help="data_size_group") #
    parser.add_argument('--group_size', type=str, default='10', 
                        help='group_size')
    parser.add_argument('--data_size_mean', type=float, default=100.0, 
                        help="data_size_mean")
    parser.add_argument('--multiplier', type=str, default='1', 
                        help='multiplier for data_size_mean of each group')
    
   
    # model parameters
    parser.add_argument('--model_name', type=str, default='KNN', 
                        help="{KNN, CNN}")
    # KNN model parameters
    parser.add_argument('--n_neighbors', type=int, default=5, 
                        help="KNN model parameter")
    # CNN model parameters
    parser.add_argument('--num_channels', type=int, default=1, 
                        help="CNN model parameter")
    parser.add_argument('--lr', type=float, default=0.1, 
                        help="CNN model parameter")
    parser.add_argument('--momentum', type=int, default=0, 
                        help="CNN model parameter")
    parser.add_argument('--decay_rate', type=int, default=1, 
                        help="CNN model parameter")
    parser.add_argument('--weight_decay', type=int, default=0, 
                        help="CNN model parameter")
    parser.add_argument('--max_norm', type=int, default=5, 
                        help="CNN model parameter")
    
    
    # SV parameters
    parser.add_argument('--utility_func', type=str, default="tst_accuracy", 
                        help="{tst_accuracy, tst_F1}")
    parser.add_argument('--computation_method', type=str, default="exact", 
                        help="{exact, MC, TMC}")
    parser.add_argument('--convergence_threshold', type=float, default=0.5, 
                        help="MC convergence_threshold")
    parser.add_argument('--truncThreshold', type=float, default=0.1, 
                        help="TMC threshold")
    
    args = parser.parse_args()
    return args
