import os
import sys
import time 
import copy
import torch
import pickle 
import random
import argparse
import preprocess_data 
from torch.autograd import Variable
from torch import optim 
import torch.nn as nn
from model.model import Encoder, Decoder, EncDecAD
from pathlib import Path 
import random 
from sklearn.cluster import KMeans

def get_precision_recall(score, label, sz, beta=1.0, wsz=1): 
    
    # interval, max, ...
    maximum = score.max().item()
    th = torch.linspace(0, maximum, sz, dtype=torch.float64)
    
    precision = []
    recall = [] 

    for i in range(len(th)):
        anomaly = (score>th[i]).float() 
        idx = anomaly*2+label
        tn = (idx==0).sum().item() # tn
        fn = (idx==1).sum().item() # fn
        fp = (idx==2).sum().item() # fp
        tp = (idx==3).sum().item() # tp 
        
        p = tp/(tp+fp+1e-7)
        r = tp/(tp+fn+1e-7) 

        if p!=0 and r!=0:
            precision.append(p) 
            recall.append(r) 

    precision = torch.Tensor(precision)
    recall = torch.Tensor(recall) 

    beta = 0.1
    f01 = (1+beta**2)*torch.max((precision*recall).div(beta**2*precision+recall+1e-7))

    beta = 1
    f1 = (1+beta**2)*torch.max((precision*recall).div(beta**2*precision+recall+1e-7))

    recall_shift = recall.clone()
    recall_shift[1:] = recall[:-1]
    recall_space = recall_shift - recall 

    auc = (precision*recall_space).sum(0).item() 
        
    print('f01', f01, 'f1', f1, 'auc', auc)
    return precision, recall, f1

if __name__=='__main__': 
	
    parser = argparse.ArgumentParser(description='Argument Parser') 
    parser.add_argument('--data', type=str, default='ecg',
        help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
    
    parser.add_argument('--filename', type=str, default='chfdb_chf13_45590.pkl', 
        help='filename of the dataset')
   
    parser.add_argument('--bsz', type=int, default=8)
    parser.add_argument('--seqlen', type=int, default=32)
    parser.add_argument('--split_ratio', type=float, default=0.7) 

    args = parser.parse_args() 

    device = torch.device('cuda') 

    TimeseriesData = preprocess_data.PickleDataLoad(data_type=args.data, filename=args.filename, augment=False) 


    train_dataset = TimeseriesData.batchify(TimeseriesData.trainData, args.bsz) 
    test_dataset = TimeseriesData.batchify(TimeseriesData.testData, args.bsz) 
    gen_dataset = TimeseriesData.batchify(TimeseriesData.testData, 1) 
    gen_label = TimeseriesData.testLabel

    total_length = train_dataset.size(0)
    split_trainset = train_dataset[:int(total_length*args.split_ratio)]
    split_validset = train_dataset[int(total_length*args.split_ratio):]

    all_sample = [] 
    
    for i in range(0, split_trainset.size(0), args.seqlen//8): 

        if i+args.seqlen>split_trainset.size(0):
            break 

        all_sample.append(split_trainset[i:i+args.seqlen]) 

    all_sample = torch.cat(all_sample, 1).transpose(0,1).contiguous().view(-1,32)

    est = KMeans(n_clusters=10) 
    est.fit(all_sample)
    center = torch.tensor(est.cluster_centers_) 

    save_folder = Path('kmeans', args.data, args.filename)
    save_folder.mkdir(parents=True, exist_ok=True) 

    torch.save(center, str(save_folder.joinpath('kmeans.pt')))  
    
    # get valid error 
    
    valid_sample = [] 
    for i in range(0, split_validset.size(0), args.seqlen//8): 
        if i+args.seqlen>split_validset.size(0): 
            break 

        valid_sample.append(split_validset[i:i+args.seqlen]) 

    valid_sample = torch.cat(valid_sample, 1).transpose(0,1).contiguous().view(-1,32)
    valid_sample = torch.tensor(valid_sample, dtype=torch.float64) 

    valid_err = 0 
    errors = [] 
    for i in range(valid_sample.size(0)):
        min_dist = 1e10 
        min_idx = 0
        for j in range(center.size(0)): 
            dist = (center[j]-valid_sample[i]).pow(2).sum().item()
            if dist<min_dist:
                min_idx = j
                min_dist = dist 

        valid_err += (min_dist/32) 
        each_err = center[min_idx]-valid_sample[i] 
        each_err = each_err.view(32,-1) 
        errors.append(each_err) 

    valid_err /= valid_sample.size(0)
    errors = torch.cat(errors,0) 
    mean = errors.mean(0)
    cov = (errors.t()).mm(errors)/errors.size(0) - mean.unsqueeze(1).mm(mean.unsqueeze(1).t())

    print(str(save_folder), valid_err)

    # generation 
    gen_dataset = gen_dataset.view(-1, 1) 
    
    test_sample = [] 
    for i in range(0,gen_dataset.size(0), args.seqlen): 
        if i+args.seqlen>gen_dataset.size(0): 
            break 

        test_sample.append(gen_dataset[i:i+args.seqlen]) 
    
    test_sample = torch.stack(test_sample, 1).transpose(0,1).contiguous().view(-1,32)
    test_sample = torch.tensor(test_sample, dtype=torch.float64) 
    
    rec_sample = []     
    rec_errors = [] 

    for i in range(test_sample.size(0)):
        min_dist = 1e10
        min_idx = 0
        for j in range(center.size(0)):
            dist = (center[j]-test_sample[i]).pow(2).sum().item() 
            if dist<min_dist:
                min_idx = j
                min_dist = dist 

        rec_sample.append(center[min_idx]) 
        each_err = center[min_idx]-test_sample[i] 
        each_err = each_err.view(32,-1) 
        rec_errors.append(each_err) 

    rec_sample = torch.stack(rec_sample, 1).transpose(0,1).contiguous().view(-1,32)
    rec_errors = torch.cat(rec_errors, 0)

    xm = rec_errors-mean
    score = (xm).mm(cov.inverse())*xm 
    score = score.sum(1)

    gen_label = gen_label[:len(score)] 
    
    print(gen_label.size(), score.size())
    precision, recall, f1 = get_precision_recall(score, gen_label, 1000, beta=1.0) 


