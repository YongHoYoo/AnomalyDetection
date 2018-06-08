import sys
import time
import torch
import pickle 
import argparse
import preprocess_data 
from model.model import Encoder, Decoder, EncDecAD
from pathlib import Path
import torch.nn as nn

def get_precision_recall(score, label, sz, beta=1.0, wsz=1): 
    
    # interval, max, ...
    maximum = score.max().item()
    th = torch.linspace(0, maximum, sz) 
    
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

    f1 = (1+beta**2)*torch.max((precision*recall).div(beta**2*precision+recall+1e-7))
        
    return precision, recall, f1


if __name__ == '__main__': 

    parser = argparse.ArgumentParser(description='Argument Parser') 
    parser.add_argument('--data', type=str, default='ecg', 
        help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
    
    parser.add_argument('--filename', type=str, default='chfdb_chf13_45590.pkl', 
        help='filename of the dataset')

  #  parser.add_argument('--filename', type=str, default='xmitdb_x108_0.pkl', 
   #     help='filename of the dataset')


    parser.add_argument('--seqlen', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100) 
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--clip', type=float, default=0.25)
    parser.add_argument('--nlayers', type=int, default=2) 
    parser.add_argument('--dropout', type=float, default=0.25) 
    parser.add_argument('--h_dropout', type=float, default=0.25) 
    parser.add_argument('--feedback', action='store_true') 
    parser.add_argument('--gated', action='store_true') 
    parser.add_argument('--hidden_tied', action='store_true') 
    parser.add_argument('--verbose', action='store_true') 

    args = parser.parse_args() 

    device = torch.device('cuda') 

    # check whether if there is a trained file in saved folder 
    param_folder_name = 'nlayers:%d'%args.nlayers + '_nhid:%d'%args.nhid + ('_F:1' if args.feedback else '_F:0') + ('_G:1' if args.gated else '_G:0') + ('_H:1' if args.hidden_tied else '_H:0') 
    save_folder = Path('result', args.data, args.filename, param_folder_name) 

    if save_folder.joinpath('model_dictionary.pt').is_file() is not True: 
        print('There is no trained model in ')
        print(str(save_folder)) 
        sys.exit() 

    if save_folder.joinpath('recall.pkl').is_file() is True: 
        print('The precision, and recall were already calculated!') 
        print(str(save_folder)) 
        sys.exit() 

    def get_batch(source, seqlen, i):
        seqlen = min(seqlen, len(source)-i) 
        input = source[i:i+seqlen] 
        target_idx = torch.LongTensor(range(input.size(0)-1, -1, -1))
        target = input.index_select(0, target_idx) 
    
        return input.to(device), target.to(device) 


    def evaluate(model, dataset): 
        model.eval() 
        total_loss = 0 
        start_time = time.time() 

        hidden = None

        for nbatch, i in enumerate(range(0, dataset.size(0), args.seqlen)):
            input, target = get_batch(dataset, args.seqlen, i) 
            output, hidden, _, _ = model(input, hidden) 
            loss = criterion(output, target) 
            total_loss += loss.item() 

        return total_loss/(nbatch+1) 


    def get_anomaly_score(model, dataset, mean, cov): 

        assert(dataset.size(1)==1) 

        model.eval() 

        all_seqlen = [4,8,16,32,64]
        all_errors = [] 
        all_outputs = [] 

        for seqlen in all_seqlen: 
        
            hidden = None
            outputs = [] 
            errors = [] 

            for nbatch, i in enumerate(range(0, dataset.size(0), seqlen)):
                input, target = get_batch(dataset, seqlen, i) 
                output, hidden, _, _ = model(input, hidden)  # input 8 1 2
                
                output_idx = torch.arange(output.size(0)-1, -1, -1).to(device).long() 
                reverse_output = output.index_select(0, output_idx) 
                outputs.append(reverse_output) 
    
                error = output-target 
                reverse_error = error.index_select(0, output_idx) 
                errors.append(reverse_error) 

            outputs = torch.cat(outputs, 0).view(-1,dataset.size(-1))
            errors = torch.cat(errors, 0).view(-1, dataset.size(-1))

            print(outputs)
#            print(outputs.size(), errors.size())
#            assert(False)

            all_errors.append(errors)
            all_outputs.append(outputs) 

        all_errors = torch.cat(all_errors, 1) 
        all_outputs = torch.cat(all_outputs, 1) 

        xm = (all_errors - mean)
        cov_eps = cov + 1e-5*torch.eye(cov.size(0)).to('cuda')
        score = (xm).mm(cov_eps.inverse())*xm
        score = score.sum(1) 

        return all_outputs, score
 
    # save   
    checkpoint = torch.load(str(save_folder.joinpath('model_dictionary.pt')))
    mean = checkpoint['mean'] 
    covariance = checkpoint['covariance'] 
    best_val_loss = checkpoint['best_loss'] 

    TimeseriesData = preprocess_data.PickleDataLoad(data_type=args.data, filename=args.filename, augment=False)
    
    gen_dataset = TimeseriesData.batchify(TimeseriesData.testData, 1) 
    gen_label = TimeseriesData.testLabel
    
    ninp = TimeseriesData.testData.size(-1)    

    encDecAD = EncDecAD(ninp, args.nhid, ninp, args.nlayers, dropout=args.dropout, h_dropout=args.h_dropout, feedback=args.feedback, gated=args.gated) 
    encDecAD.to(device)
    
    encDecAD.load_state_dict(checkpoint['state_dict']) 

    out_dataset, gen_score = get_anomaly_score(encDecAD, gen_dataset, mean, covariance) 

    print(out_dataset)

    pickle.dump(gen_dataset, open(str(save_folder.joinpath('gen_dataset.pkl')), 'wb')) 
    pickle.dump(out_dataset, open(str(save_folder.joinpath('out_dataset.pkl')), 'wb'))

    pickle.dump(gen_score, open(str(save_folder.joinpath('scores.pkl')), 'wb')) 
    pickle.dump(gen_label, open(str(save_folder.joinpath('labels.pkl')), 'wb')) 

    # Get precision, recall
    precision, recall, f1 = get_precision_recall(gen_score.cpu(), gen_label.cpu(), 1000, beta=1.0) 
    
    pickle.dump(precision, open(str(save_folder.joinpath('precision.pkl')), 'wb'))
    pickle.dump(recall, open(str(save_folder.joinpath('recall.pkl')), 'wb')) 
    
    print(str(save_folder), best_val_loss, f1)
