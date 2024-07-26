# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 20:57:36 2021

@author: admin
"""
from arguments import args_parser
from utils.sampling import data_split
from utils.dataset_helper import ImageDataset 
from utils.tools import init_random_seed
import torch,os


if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) \
                               if torch.cuda.is_available() and args.gpu != -1\
                               else 'cpu')
    # parse args
    print(args)
    if args.manual_seed:
        init_random_seed(args.manual_seed)
    
    # load dataset and split workers
    dataset_train, dataset_test, validation_index, \
    dict_workers = data_split(args)
    trn_len = len(dataset_train)
    tst_len = len(dataset_test)
    sum_training_data = sum([len(data_indexs) for data_indexs in dict_workers.values()])
    print('sum_training_data: ', sum_training_data, '\n')   
    
    
    if os.path.exists('data/%s%s/'%(args.dataset, args.data_allocation_scheme))==False:
        os.makedirs('data/%s%s/'%(args.dataset, args.data_allocation_scheme))
    for idx in range(args.num_trainDatasets):
        filename = 'data/%s%s/train%s.pt'%(args.dataset, args.data_allocation_scheme, idx)
        trainset = ImageDataset(
                dataset_train, None, trn_len, dict_workers[idx],
                )
        #else:
        #    trainset = DatasetSplit(dataset, data_idxs)
            
        torch.save(trainset, filename)
        print('worker %s: data_size %s %s...'%(idx, len(dict_workers[idx]), 
                                            len(trainset)))
        
        
    test_dataset_path = 'data/%s%s/test.pt'%(args.dataset, args.data_allocation_scheme)
    # load validation data
    # if args.dataset == 'shakespeare':
    testset= ImageDataset(
            dataset_test, None, tst_len,
            set(range(len(dataset_test)))-set(validation_index),
            #dataset_train, None, dict_workers[idx],
            )        
    torch.save(testset, test_dataset_path)
    print('test data_size %s...'%len(testset))