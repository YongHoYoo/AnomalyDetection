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
#        help='filename of the dataset')
 
    parser.add_argument('--filename', type=str, default='chfdb_chf13_45590.pkl', 
        help='filename of the dataset')
   
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--seqlen', type=list, default=[8,16,32,64,128]) 
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--clip', type=float, default=0.25)
    parser.add_argument('--nlayers', type=int, default=2) 
    parser.add_argument('--dropout', type=float, default=0.35) 
    parser.add_argument('--h_dropout', type=float, default=0.25) 
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

    encDecAD = EncDecAD(ninp, args.nhid, ninp, args.nlayers, dropout=args.dropout, h_dropout=args.h_dropout, feedback=args.feedback, gated=args.gated) 
    bestEncDecAD = EncDecAD(ninp, args.nhid, ninp, args.nlayers, dropout=args.dropout, h_dropout=args.h_dropout, feedback=args.feedback, gated=args.gated) 
 
    encDecAD.to(device) 
    bestEncDecAD.to(device) 

    # make save_folder 
    param_folder_name = 'nlayers:%d'%args.nlayers + '_nhid:%d'%args.nhid + ('_feedback:1' if args.feedback else '_feedback:0') + ('_gated:1' if args.gated else '_gated:0') 

    save_folder = Path('result', args.data, args.filename, param_folder_name) 
    save_folder.mkdir(parents=True, exist_ok=True)

    # if there is a file 'model_dictionary.pt', exit. 
    if save_folder.joinpath('model_dictionary.pt').is_file(): 
        print('There is already trained model in ') 
        print(str(save_folder)) 
        sys.exit() 
 

    criterion = torch.nn.MSELoss() 
    optimizer = optim.Adam(encDecAD.parameters(), lr=args.lr, weight_decay=1e-5)
    best_val_loss = None 
    early_stop = 0 

    def get_batch(source, seqlen, i):
        seqlen = min(seqlen, len(source)-i) 
        input = source[i:i+seqlen] 
        target_idx = torch.range(input.size(0)-1, 0,-1, dtype=torch.int64) 
        target = input.index_select(0, target_idx) 
 
        return input.to(device), target.to(device)

    def train(model, dataset): 
        model.train() 

        start_time = time.time() 

        total_train_loss = 0

        random.shuffle(args.seqlen) 
        for seqlen in args.seqlen: 
            hidden = None
            train_loss = 0 

            model.train()
            for nbatch, i in enumerate(range(0, dataset.size(0), seqlen)):
                input, target = get_batch(dataset, seqlen, i) 
                optimizer.zero_grad() 
                output, hidden = model(input.requires_grad_(), hidden) 
                loss = criterion(output, target) 
                loss.backward() 
                
                torch.nn.utils.clip_grad_norm_(encDecAD.parameters(), args.clip) 
                train_loss += loss.item()
                optimizer.step() 
    
                hidden = hidden[0].detach(), hidden[1].detach() 
        
            total_train_loss += train_loss/nbatch
            print('| epoch {:3d} | seqlen {:3d} | train loss {:5.2f}'.format(
                epoch, seqlen, train_loss/nbatch)) 

        total_train_loss /= len(args.seqlen) 
        args.seqlen.sort() 

        return total_train_loss, hidden

    def calculate_params(model, dataset): 
        model.eval() 

        # split dataset 
        total_length = dataset.size(0) 
        split_trainset = dataset[:int(total_length*args.split_ratio)] 
        split_validset = dataset[int(total_length*args.split_ratio):] 

        all_errors = [] 
        
        for seqlen in args.seqlen:
            hidden = None 
            
            errors = [] 

            for nbatch, i in enumerate(range(0, split_trainset.size(0), seqlen)):
                input, _ = get_batch(split_trainset, seqlen, i) 
                output, hidden = model(input, hidden) 
                hidden = hidden[0].detach(), hidden[1].detach() 
           
            for nbatch, i in enumerate(range(0, split_validset.size(0), seqlen)): 
                input, target = get_batch(split_validset, seqlen, i) 
                output, hidden = model(input, hidden) 

                error = output-target # seqlen by batch by dim
                errors.append(error) # to avoid out of gpu memory 
                hidden = hidden[0].detach(), hidden[1].detach() 
            
            all_errors.append(torch.cat(errors, 0).view(-1, split_validset.size(2))) 
        
        all_errors = torch.stack(all_errors, 0) # 5 by 51200 by 2

        means = [] 
        covs = [] 
        for channel in range(all_errors.size(-1)): 
            x = all_errors[:,:,channel] # 5 by 51200 
            mean = x.mean(1) 
            cov = x.mm(x.t())/x.size(-1)-mean.unsqueeze(1).mm(mean.unsqueeze(1).t()) 
            means.append(mean)
            covs.append(cov) 

        means = torch.stack(means, 1)
        covs = torch.stack(covs, 2) 

        return means, covs
            


    def evaluate(model, dataset, last_hidden=None): 
        model.eval() 
        start_time = time.time() 

        total_valid_loss = 0 

        for seqlen in args.seqlen: 

            valid_loss = 0 
            outputs = [] 
            hidden = (last_hidden[0].detach(), last_hidden[1].detach()) if last_hidden is not None else None

            for nbatch, i in enumerate(range(0, dataset.size(0), seqlen)):
                input, target = get_batch(dataset, seqlen, i) 
                output, hidden = model(input, hidden) 
                loss = criterion(output, target) 
                valid_loss += loss.item() 
            return valid_loss/len(args.seqlen) 

            total_valid_loss += valid_loss/nbatch

    try: 
        for epoch in range(args.epochs): 
    
            start_time = time.time() 

            # split dataset 
            total_length = train_dataset.size(0) 
            split_trainset = train_dataset[:int(total_length*0.7)] 
            split_validset = train_dataset[int(total_length*0.7):] 

            train_loss, last_hidden = train(encDecAD, split_trainset) 
            valid_loss = evaluate(encDecAD, split_validset, last_hidden) 

            if best_val_loss is None or best_val_loss>valid_loss: 
                best_val_loss = valid_loss 
                bestEncDecAD.load_state_dict(copy.deepcopy(encDecAD.state_dict()))
                early_stop = 0 
            else: 
                early_stop += 1 
                if early_stop==10: 
                    print('Validation loss is not updated more.')
                    print('The iteration is terminated at iteration %d.'%epoch) 
                    break 

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.4f} | valid loss {:5.4f} | '.format(epoch, (time.time() - start_time), train_loss, valid_loss))
            print('-' * 89)
    
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # get errors' mean & std, save model 
    means, covs = calculate_params(bestEncDecAD, train_dataset) 
 
    model_dictionary = {'state_dict': bestEncDecAD.state_dict(), 
        'mean': means, 
        'covariance': covs, 
        'args': args,
        }    

    torch.save(model_dictionary, str(save_folder.joinpath('model_dictionary.pt')))
    print('The model is saved in ' + str(save_folder))

