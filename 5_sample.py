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

if __name__=='__main__': 
	
    parser = argparse.ArgumentParser(description='Argument Parser') 
    parser.add_argument('--data', type=str, default='ecg', 
        help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
    
#    parser.add_argument('--filename', type=str, default='qtdbsel102.pkl', 
 #       help='filename of the dataset')
 
    parser.add_argument('--filename', type=str, default='chfdb_chf13_45590.pkl', 
        help='filename of the dataset')
   
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--seqlen', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--clip', type=float, default=0.25)
    parser.add_argument('--nlayers', type=int, default=2) 
    parser.add_argument('--dropout', type=float, default=0.5) 
    parser.add_argument('--h_dropout', type=float, default=0.5) 
    parser.add_argument('--feedback', action='store_true') 
    parser.add_argument('--gated', action='store_true') 
    parser.add_argument('--split_ratio', type=float, default=0.7) 

    args = parser.parse_args() 
    
    device = torch.device('cuda') 

    TimeseriesData = preprocess_data.PickleDataLoad(data_type=args.data, filename=args.filename, augment=False) 

    ninp = TimeseriesData.trainData.size(-1) 

    train_dataset = TimeseriesData.batchify(TimeseriesData.trainData, args.bsz) 
    test_dataset = TimeseriesData.batchify(TimeseriesData.testData, args.bsz) 
    gen_dataset = TimeseriesData.batchify(TimeseriesData.testData, 1) 

# make block

    def get_batch(source, seqlen, i):
        seqlen = min(seqlen, len(source)-i) 
        input = source[i:i+seqlen] 
        target_idx = torch.range(input.size(0)-1, 0,-1, dtype=torch.int64) 
        target = input.index_select(0, target_idx) 
 
        return input.to(device), target.to(device)

    def make_block(dataset): 
        
        block = []
        for nbatch, i in enumerate(range(0, dataset.size(0), args.seqlen)): 
            input, _ = get_batch(dataset, args.seqlen, i) 
            if input.size()==torch.Size([args.seqlen, args.bsz, dataset.size(-1)]):
                block.append(input) 
        
        # seqlen batch dim 

        block = torch.cat(block, 1)
        return block
            
    block = make_block(train_dataset)
    
    # pca 
    def PCA(data, k=2): 
        # data: num_data by feature. 
        mean = data.mean(0) 
        data = data - mean
        
        # svd
        U,S,V = torch.svd(data.t()) 
        return data.mm(U[:,:k]) 

    block_pca = [] 
    block = block.transpose(0,2) # dim by all_sample by seqlen

    for channel in range(block.size(0)):
        
        b = block[channel].t() # all_sample by seqlen
        b_pca = PCA(b, 20)

        block_pca.append(b_pca.t()) # 10 by seqlen 

    block_pca = torch.stack(block_pca, 0).transpose(0,2) # seqlen by all_sample by dim

    print(block_pca.size())
    pickle.dump(block_pca, open('pca.pkl', 'wb')) 

        
        

    



