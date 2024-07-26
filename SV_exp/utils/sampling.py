import numpy as np
from torchvision import datasets, transforms
import os, torch
from .text_helper import Corpus, centralized
from .image_helper import FEMNIST


def get_datasets(args):
    # load datasets
    if args.dataset == 'mnist':
        img_size = torch.Size([args.num_channels, 28, 28])
        
        trans_mnist = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST(
                root='./data/mnist/',
                train=True,
                download=True,
                transform=trans_mnist
            )

        dataset_test = datasets.MNIST(
                root='./data/mnist/',
                train=False,
                download=True,
                transform=trans_mnist
            )
    
    elif args.dataset == 'cifar':
        img_size = torch.Size([3, 32, 32])

        trans_cifar10_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
        trans_cifar10_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
    
        dataset_train = datasets.CIFAR10(
                root='./data/cifar',
                train=True,
                download=True,
                transform=trans_cifar10_train
            )
    
        dataset_test = datasets.CIFAR10(
                root='./data/cifar',
                train=False,
                download=True,
                transform=trans_cifar10_val
                )
    else:
        exit('Error: unrecognized dataset')
        
        
    return dataset_train, dataset_test, img_size


def data_split(args):
    # load datasets
    dataset_train, dataset_test, img_size = get_datasets(args)
    validation_index = []#np.random.choice(
            #len(dataset_test),int(len(dataset_test)*0.05), replace=False
            #)
    
    # sampling
    if args.data_allocation_scheme == 0 or args.num_trainDatasets==1:
        data_size_group = 1
        data_size_means = [len(dataset_train)/args.num_trainDatasets]
        group_size = [args.num_trainDatasets]
    else:
        args.group_size = [int(tmp) for tmp in args.group_size.split(",")]
        args.multiplier = [int(tmp) for tmp in args.multiplier.split(",")]
        
        data_size_group = args.data_size_group 
        group_size = args.group_size
        data_size_means = [args.data_size_mean*args.multiplier[gidx] \
                           for gidx in range(data_size_group)]
            
    data_quantity = []
    for i in range(data_size_group):
        tmp = np.random.normal(data_size_means[i], data_size_means[i]/4, 
                               group_size[i])
        tmp2 = []
        small_index = np.where(tmp<=data_size_means[i])[0]
        if len(small_index) >= group_size[i]/2:
            tmp2 += list(tmp[small_index][:int(group_size[i]/2)])
            tmp2 += list(2*data_size_means[i]-\
                         tmp[small_index][:group_size[i]-int(group_size[i]/2)])
        else:
            large_index = np.where(tmp>=data_size_means[i])[0]
            tmp2 += list(tmp[large_index][:int(group_size[i]/2)])
            tmp2 += list(2*data_size_means[i]-\
                         tmp[large_index][:group_size[i]-int(group_size[i]/2)])
        #tmp2 = tmp2[:group_size[i]]
        if len(tmp2)<group_size[i]:
            tmp2 += tmp2 + tmp2[int(group_size[i]/2):
                                int(group_size[i]/2)+(group_size[i]-len(tmp2))]
        data_quantity += tmp2
    data_quantity = np.array([(int(np.round(i)) if int(np.round(i)) >=2 else 2) \
                              for i in data_quantity])
    data_quantity = sorted(data_quantity)
    print(data_quantity)
    if len(group_size) <= 1:     
        data_idx = list(range(sum(data_quantity)))
        #print(data_idx)
        np.random.shuffle(data_idx)
        workers_idxs = [[] for _ in range(args.num_trainDatasets)]
        for idx in range(args.num_trainDatasets):
            print('sampling worker %s...'%idx)
            workers_idxs[idx] = np.random.choice(data_idx, 
                        data_quantity[idx], replace=False)
            data_idx = list(set(data_idx)-set(workers_idxs[idx]))
            np.random.shuffle(data_idx)
    else:
        try:
            idxs_labels = np.array(dataset_train.train_labels)  
        except:
            idxs_labels = np.array(dataset_train.targets)
           
        class_num = dict([(c,0) for c in range(args.num_classes)])
        worker_classes = dict()
        for idx in range(args.num_trainDatasets):
            worker_classes[idx] = range(args.num_classes)
            for tmp, c in enumerate(worker_classes[idx]):
                if tmp == len(worker_classes[idx])-1:
                    class_num[c] += data_quantity[idx] - \
                        int(data_quantity[idx]/len(worker_classes[idx]))*(len(worker_classes[idx])-1)
                else:
                    class_num[c] += int(data_quantity[idx]/len(worker_classes[idx]))
                
        class_indexes = dict()
        for c,num in class_num.items():
            original_index = list(np.where(idxs_labels==c)[0])
            appended_index = []
            count=0
            while len(appended_index)<num:
                appended_index += [tmp+count*len(idxs_labels) for tmp in original_index]
                count+=1
            np.random.shuffle(appended_index)
            class_indexes[c] = appended_index  
            
        
        workers_idxs = [[] for _ in range(args.num_trainDatasets)]
        for idx in range(args.num_trainDatasets):
            print('sampling worker %s...'%idx)
            workers_idxs[idx] = []
            for tmp, c in enumerate(worker_classes[idx]):
                if tmp == len(worker_classes[idx])-1:
                    sampled_idx = list(np.random.choice(
                        class_indexes[c], 
                        data_quantity[idx] - \
                        int(data_quantity[idx]/len(worker_classes[idx]))*(len(worker_classes[idx])-1), 
                        replace=False))
                else:
                    sampled_idx = list(np.random.choice(
                        class_indexes[c], 
                        int(data_quantity[idx]/len(worker_classes[idx])),
                        replace=False))
                workers_idxs[idx] += sampled_idx
                class_indexes[c] = list(set(class_indexes[c])-set(sampled_idx))
                np.random.shuffle(class_indexes[c])
            np.random.shuffle(workers_idxs[idx])
            print(data_quantity[idx], len(workers_idxs[idx]),
                  worker_classes[idx], set([idxs_labels[tmp%len(idxs_labels)] for tmp in workers_idxs[idx]]))
    
    dict_workers = {i: workers_idxs[i] for i in range(len(workers_idxs))}
    x=[]
    combine = [] 
    for i in dict_workers.values():
        x.append(len(i))
        combine.append(len(i))
    print('train data partition')
    print('sum:',np.sum(np.array(x)))
    print('mean:',np.mean(np.array(x)))
    print('std:',np.std(np.array(x)))
    print('max:',max(np.array(x)))
    print('min:',min(np.array(x)))
    return dataset_train, dataset_test, validation_index, dict_workers

