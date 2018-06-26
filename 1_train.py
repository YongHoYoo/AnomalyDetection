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
    
    parser.add_argument('--filename', type=str, default='chfdb_chf13_45590.pkl', 
        help='filename of the dataset')
 
 #   parser.add_argument('--filename', type=str, default='power_data.pkl',
  #      help='filename of the dataset')
   
    parser.add_argument('--bsz', type=int, default=8)  
    parser.add_argument('--seqlen', type=int, default=84)
    parser.add_argument('--epochs', type=int, default=180)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--clip', type=float, default=0.25) 
    parser.add_argument('--nlayers', type=int, default=2) 
    parser.add_argument('--dropout', type=float, default=0.3) 
    parser.add_argument('--h_dropout', type=float, default=0.3)
    parser.add_argument('--feedback', action='store_true') 
    parser.add_argument('--gated', action='store_true') 
    parser.add_argument('--hidden_tied', action='store_true') 
    parser.add_argument('--split_ratio', type=float, default=0.7) 

    args = parser.parse_args() 

    if args.feedback:
        args.h_dropout+=0.05

    device = torch.device('cuda') 

    TimeseriesData = preprocess_data.PickleDataLoad(data_type=args.data, filename=args.filename, augment=False) 

    ninp = TimeseriesData.trainData.size(-1) 

    train_dataset = TimeseriesData.batchify(TimeseriesData.trainData, args.bsz) 
    test_dataset = TimeseriesData.batchify(TimeseriesData.testData, args.bsz) 
    gen_dataset = TimeseriesData.batchify(TimeseriesData.testData, 1) 

    valid_errs = [] 

    encDecAD = EncDecAD(ninp, args.nhid, ninp, args.nlayers, dropout=args.dropout, h_dropout=args.h_dropout, feedback=args.feedback, gated=args.gated, hidden_tied=args.hidden_tied) 
    bestEncDecAD = EncDecAD(ninp, args.nhid, ninp, args.nlayers, dropout=args.dropout, h_dropout=args.h_dropout, feedback=args.feedback, gated=args.gated, hidden_tied=args.hidden_tied) 
 
    encDecAD.to(device) 
    bestEncDecAD.to(device) 

    # make save_folder 
    param_folder_name = 'nlayers:%d'%args.nlayers + '_nhid:%d'%args.nhid + ('_F:1' if args.feedback else '_F:0') + ('_G:1' if args.gated else '_G:0') + ('_H:1' if args.hidden_tied else '_H:0') 

    save_folder = Path('result', args.data, args.filename, param_folder_name) 
    save_folder.mkdir(parents=True, exist_ok=True)

# if there is a file 'model_dictionary.pt', exit. 
#    if save_folder.joinpath('model_dictionary.pt').is_file(): 
#        print('There is already trained model in ') 
#        print(str(save_folder)) 
#        sys.exit() 
 

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


        all_train_loss = 0 
        all_seqlen = [21,42,84,168,336]
#        all_seqlen = [16,32,64,128,256]

        for seqlen  in all_seqlen: 

            train_loss = 0 
            model.train()
            hidden = None

            for nbatch, i in enumerate(range(0, dataset.size(0), seqlen)):
                input, target = get_batch(dataset, seqlen, i) 
                noise = 1e-3*torch.randn(input.size()).to(device)
                input_n = input + noise
                optimizer.zero_grad() 
                output, hidden, enc_hiddens, dec_hiddens = model(input_n.requires_grad_(), hidden) 
                loss = criterion(output, target) 

                if enc_hiddens is not None: 
                    loss_hidden = criterion(dec_hiddens[0][:], enc_hiddens[0][:].data) 
                    loss += loss_hidden
            
                loss.backward() 
            
                torch.nn.utils.clip_grad_norm_(encDecAD.parameters(), args.clip) 
                train_loss += loss.item()
                optimizer.step() 
    
                hidden = hidden[0].detach(), hidden[1].detach() 
 
            train_loss/=(nbatch+1) 
            all_train_loss+=train_loss

        all_train_loss/=len(all_seqlen) 
        return all_train_loss, hidden

    def calculate_params(model, dataset): 
        model.eval() 

        # split dataset 
        total_length = dataset.size(0) 
        split_trainset = dataset[:int(total_length*args.split_ratio)] 
        split_validset = dataset[int(total_length*args.split_ratio):] 

        all_seqlen= [84]
        all_errors = [] 

        for seqlen in all_seqlen: 

            hidden = None 
            errors = [] 

            for nbatch, i in enumerate(range(0, split_trainset.size(0), seqlen)):
                input, _ = get_batch(split_trainset, seqlen, i) 
                output, hidden, _, _ = model(input, hidden) 
                hidden = hidden[0].detach(), hidden[1].detach() 

            for nbatch, i in enumerate(range(0, split_validset.size(0), seqlen)): 
                input, target = get_batch(split_validset, seqlen, i) 
                output, hidden, _, _ = model(input, hidden) 
    
                error = output-target # seqlen by batch by dim
                errors.append(error) 
                hidden = hidden[0].detach(), hidden[1].detach() 

            errors = torch.cat(errors, 0).view(-1, errors[0].size(-1)) # x by 2 
            all_errors.append(errors) 

        all_errors = torch.cat(all_errors, 1) # x by (2*5) 

        mean = all_errors.mean(0) 
        cov = (all_errors.t()).mm(all_errors)/all_errors.size(0) - mean.unsqueeze(1).mm(mean.unsqueeze(1).t()) 

        return mean, cov 


    def evaluate(model, dataset, last_hidden=None): 
        model.eval() 
        start_time = time.time() 
        
        valid_loss = 0 
 
#        hidden = None  
        hidden = (last_hidden[0].detach(), last_hidden[1].detach()) if last_hidden is not None else None
  
        for nbatch, i in enumerate(range(0, dataset.size(0), args.seqlen)):
            input, target = get_batch(dataset, args.seqlen, i) 
            output, hidden, _, _ = model(input, hidden) 
            loss = criterion(output, target) 
            valid_loss += loss.item() 
  
        valid_loss/=(nbatch+1)
        return valid_loss

    try: 
      for epoch in range(args.epochs): 
               
            start_time = time.time() 
  
            # split dataset 
            total_length = train_dataset.size(0) 
            split_trainset = train_dataset[:int(total_length*args.split_ratio)] 
            split_validset = train_dataset[int(total_length*args.split_ratio):] 
  
            train_loss, last_hidden = train(encDecAD, split_trainset) 
            valid_loss = evaluate(encDecAD, split_validset, last_hidden) 

            if best_val_loss is None or best_val_loss>valid_loss: 
                best_val_loss = valid_loss 
                bestEncDecAD.load_state_dict(copy.deepcopy(encDecAD.state_dict()))
                early_stop = 0 
            else: 
                early_stop += 1 
             #   if early_stop==50:  
              #      print('Validation loss %f is not updated more.'%best_val_loss)
               #     print('The iteration is terminated at iteration %d.'%epoch) 
                #    break 

            valid_errs.append(best_val_loss) 
  
            print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.4f} | valid loss {:5.4f} | '.format(epoch, (time.time() - start_time), train_loss, best_val_loss))
            print('-' * 89)
 
 
    
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
  
    print('best_val', best_val_loss) 

    # get errors' mean & std, save model 
    mean, cov = calculate_params(bestEncDecAD, train_dataset) 
  
    model_dictionary = {'state_dict': bestEncDecAD.state_dict(), 
         'mean': mean,
         'covariance': cov,
         'best_loss': best_val_loss, 
         'args': args,
         }    
             
    torch.save(model_dictionary, str(save_folder.joinpath('model_dictionary.pt')))

    add = args.data + '_' + ('1' if args.feedback else '0') + ('1' if args.gated else '0') + ('1' if args.hidden_tied else '0') 

    print('The model is saved in ' + str(save_folder))
    
    pickle.dump(valid_errs, open('valid_err_' + add +'.pt','wb')) 
