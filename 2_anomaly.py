import sys
import time
import torch
import pickle 
import argparse
import preprocess_data 
from torch.autograd import Variable
from model.model import Encoder, Decoder, EncDecAD
from pathlib import Path
import torch.nn as nn

def get_precision_recall(score, label, sz, beta=1.0, wsz=1): 
    
    # interval, max, ...
    maximum = score.max()
    th = torch.linspace(0, maximum, sz) 
    
    precision = torch.zeros(len(th))
    recall = torch.zeros(len(th)) 

    
    for i in range(len(th)):
        anomaly = (score>th[i]).float() 
        idx = anomaly*2+label
        tn = (idx==0).sum() # tn
        fn = (idx==1).sum() # fn
        fp = (idx==2).sum() # fp
        tp = (idx==3).sum() # tp 
        
        precision[i] = tp/(tp+fp+1e-7)
        recall[i] = tp/(tp+fn+1e-7) 

    f1 = (1+beta**2)*torch.max((precision*recall).div(beta**2*precision+recall+1e-7))
        
    return precision, recall, f1


if __name__ == '__main__': 

    parser = argparse.ArgumentParser(description='Argument Parser') 
    parser.add_argument('--data', type=str, default='ecg', 
        help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
    
    parser.add_argument('--filename', type=str, default='chfdb_chf13_45590.pkl', 
        help='filename of the dataset')

    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--seqlen', type=list, default=[4,8,16,32,64]) 
    parser.add_argument('--epochs', type=int, default=100) 
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--ninp', type=int, default=2) 
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--clip', type=float, default=0.25)
    parser.add_argument('--nlayers', type=int, default=2) 
    parser.add_argument('--dropout', type=float, default=0.35) 
    parser.add_argument('--h_dropout', type=float, default=0.35) 
    parser.add_argument('--log_interval', type=int, default=10) 
    parser.add_argument('--feedback', action='store_true') 
    parser.add_argument('--gated', action='store_true') 
    parser.add_argument('--verbose', action='store_true') 

    args = parser.parse_args() 

    # check whether if there is a trained file in saved folder 
    param_folder_name = 'nlayers:%d'%args.nlayers + '_nhid:%d'%args.nhid + ('_feedback:1' if args.feedback else '_feedback:0') + ('_gated:1' if args.gated else '_gated:0') 
    save_folder = Path('result', args.data, args.filename, param_folder_name) 

    if save_folder.joinpath('model_dictionary.pt').is_file() is not True: 
        print('There is no trained model in ')
        print(str(save_folder)) 
        sys.exit() 

    if save_folder.joinpath('recall.pkl').is_file() is True: 
        print('The precision, and recall were already calculated!') 

    def get_batch(source, seqlen, i):
        seqlen = min(seqlen, len(source)-i) 
        input = source[i:i+seqlen] 
        target_idx = torch.LongTensor(range(input.size(0)-1, -1, -1))
        target = input.index_select(0, target_idx) 
    
        return Variable(input.cuda()), Variable(target.cuda()) 

    def evaluate(model, dataset, reconstruct=False): 
        model.eval() 
        total_loss = 0 
        start_time = time.time() 

        if reconstruct is True:
            outputs = [] 

        for seqlen in args.seqlen: 

            hidden = None
            for nbatch, i in enumerate(range(0, dataset.size(0), seqlen)):
                input, target = get_batch(dataset, seqlen, i) 
                output, hidden = model(input, hidden) 
                loss = criterion(output, target) 
                total_loss += loss.data[0]
    
                if reconstruct is True: 
                   # output reverse 
                        output_idx = Variable(torch.LongTensor(range(output.size(0)-1, -1, -1)).cuda())
                        reverse_output = output.index_select(0, output_idx) 
                        outputs.append(reverse_output) 


        if reconstruct is True: 
            outputs = torch.cat(outputs, 0) 
            return total_loss/nbatch, outputs
        else:
            return total_loss/nbatch

    def get_anomaly_score(model, dataset, means, covs): 

        assert(dataset.size(1)==1) 

        all_errors = [] 
        all_outputs = []     

        for seqlen in args.seqlen: 
            hidden = None
            errors = [] 
            outputs = [] 

            for nbatch, i in enumerate(range(0, dataset.size(0), seqlen)):
                input, target = get_batch(dataset, seqlen, i) 
                output, hidden = model(input, hidden)  # input 8 1 2
                
                output_idx = Variable(torch.LongTensor(range(output.size(0)-1, -1, -1)).cuda()) 
                reverse_output = output.index_select(0, output_idx) 
                outputs.append(reverse_output) 

                error = output-target 
                errors.append(error) 
                hidden = (Variable(hidden[0].data), Variable(hidden[1].data)) 

            outputs = torch.cat(outputs, 0) 
            all_outputs.append(outputs) 
            all_errors.append(torch.cat(errors, 0).squeeze()) # n 1 2 -> n 2

        all_errors = torch.stack(all_errors, 0)  # 5 n 2

        scores = [] 
        for channel in range(all_errors.size(-1)):
            x = all_errors[:,:,channel].t() # n by 5
    
            xm = x-means[:,channel]
            score = (xm.mm(covs[:,:,channel].inverse()))*xm
            scores.append(score.sum(1)) # n 

        scores = torch.stack(scores, 1) # n by 2
        return all_outputs, scores         
 
    # save   
    checkpoint = torch.load(str(save_folder.joinpath('model_dictionary.pt')))
    mean = checkpoint['mean'] 
    covariance = checkpoint['covariance'] 

    TimeseriesData = preprocess_data.PickleDataLoad(data_type=args.data, filename=args.filename)   
    
    gen_dataset = TimeseriesData.batchify(TimeseriesData.testData, 1) 
    gen_label = TimeseriesData.testLabel
    
    encDecAD = EncDecAD(args.ninp, args.nhid, args.ninp, args.nlayers, dropout=args.dropout, h_dropout=args.h_dropout, feedback=args.feedback, gated=args.gated) 
    encDecAD.cuda() 
    
    encDecAD.load_state_dict(checkpoint['state_dict']) 

    criterion = torch.nn.MSELoss() 


    out_dataset, gen_score = get_anomaly_score(encDecAD, gen_dataset, mean, covariance) 

    pickle.dump(gen_dataset, open(str(save_folder.joinpath('gen_dataset.pkl')), 'wb')) 
    pickle.dump(out_dataset, open(str(save_folder.joinpath('out_dataset.pkl')), 'wb'))

    pickle.dump(gen_score, open(str(save_folder.joinpath('scores.pkl')), 'wb')) 
    pickle.dump(gen_label, open(str(save_folder.joinpath('labels.pkl')), 'wb')) 

    # Get precision, recall
    precision, recall, f1 = get_precision_recall(gen_score[:,0].data.cpu(), gen_label.cpu(), 1000, beta=1.0) 
    
    pickle.dump(precision, open(str(save_folder.joinpath('precision.pkl')), 'wb'))
    pickle.dump(recall, open(str(save_folder.joinpath('recall.pkl')), 'wb')) 
    
    print(str(save_folder), f1)
