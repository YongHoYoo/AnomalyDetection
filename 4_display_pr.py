import os
import sys
import torch 
import pickle
import argparse
from pathlib import Path

import plotly 
import plotly.graph_objs as go
from plotly import tools 

if __name__ == '__main__': 

    parser = argparse.ArgumentParser(description='Argument Parser') 
    parser.add_argument('--data', type=str, default='ecg', 
        help='type of the dataset (ecg, gesture, power_demand, space_shuttle, respiration, nyc_taxi')
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--nlayers', type=int, default=2) 

    args = parser.parse_args() 

    root_path = Path('result', args.data) 

    for rp in root_path.iterdir():
        param_names = [] 
        param_names.append('nlayers:%d'%args.nlayers + '_nhid:%d'%args.nhid + '_F:0_G:0_H:0') 
        param_names.append('nlayers:%d'%args.nlayers + '_nhid:%d'%args.nhid + '_F:0_G:1_H:0') 
        param_names.append('nlayers:%d'%args.nlayers + '_nhid:%d'%args.nhid + '_F:1_G:1_H:0')
        param_names.append('nlayers:%d'%args.nlayers + '_nhid:%d'%args.nhid + '_F:0_G:0_H:1') 
        param_names.append('nlayers:%d'%args.nlayers + '_nhid:%d'%args.nhid + '_F:1_G:1_H:1') 

        all_saved = True

        for param_name in param_names: 
            
            subroot_path = rp.joinpath(param_name) 
 
            if subroot_path.joinpath('precision.pkl').is_file() is False: # if there is no precision.pkl file
                all_saved = False
        
        if all_saved is False: 
            continue 

        trace_prs = [] 
        f_0_1 = []
        f_1_0 = []        
        valid = [] 
        auc = [] 

        for param_name in param_names: 
            
            subroot_path = rp.joinpath(param_name) 
            precision = pickle.load(open(str(subroot_path.joinpath('precision.pkl')), 'rb'))
            recall = pickle.load(open(str(subroot_path.joinpath('recall.pkl')), 'rb')) 

            recall_shift = recall.clone()
            recall_shift[1:] = recall[:-1] 
            recall_space = recall_shift - recall 

            auc.append((precision*recall_space).sum().item())
            
            checkpoint = torch.load(str(subroot_path.joinpath('model_dictionary.pt')))
            best_val_loss = checkpoint['best_loss'] 

            def f_score(beta=1.0):
                return (1+beta**2)*torch.max((precision*recall).div(beta**2*precision+recall+1e-7)) 
        
            f_0_1.append(f_score(0.1))
            f_1_0.append(f_score(1.0)) 
            valid.append(best_val_loss) 
        
            # precision-recall 
            trace_pr = go.Scatter( 
                x = recall, 
                y = precision, 
                mode = 'lines', 
                line = dict(shape='spline'), 
                name = str(subroot_path).split('/')[-1],
                )

            trace_prs.append(trace_pr) 

        plotly.offline.plot({
                'data': trace_prs, 
                'layout': go.Layout(title=str(rp), xaxis=dict(title='Recall'), yaxis=dict(title='Precision')),
# , plot_bgcolor='rgb(239,239,239)', xaxis = go.XAxis(gridcolor='rgb(255,255,255)'), yaxis = go.YAxis(gridcolor='rgb(255,255,255)')), 
                }, filename=str(rp.joinpath('pr.html')))

        # table 
        f_score = [param_names, valid, auc, f_0_1, f_1_0]
    
        trace_table = go.Table(
            name=str(rp),
            header=dict(values=['filename', 'valid', 'auc', 'beta 0.1', 'beta 1.0'], 
                line = dict(color='#7D7F80'), 
                fill = dict(color='#a1c3d1'), 
                align = ['left']*5), 
            cells=dict(values=f_score, 
                    line = dict(color='#7D7F80'), 
                    align = ['left']*5)) 
    
        plotly.offline.plot({
            'data': [trace_table], 
            'layout': go.Layout(title=str(rp))}, filename=str(rp.joinpath('_table.html')))
       
        


