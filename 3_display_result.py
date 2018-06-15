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
#    parser.add_argument('--filename', type=str, default='ann_gun_CentroidA.pkl', 
#    parser.add_argument('--filename', type=str, default=.pkl', 
#        help='filename of the dataset')

    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--seqlen', type=list, default=[4,8,16,32,64]) 
    parser.add_argument('--ninp', type=int, default=2) 
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--nlayers', type=int, default=2) 
    parser.add_argument('--feedback', action='store_true') 

    args = parser.parse_args() 

    root_path = Path('result', args.data)

    for rp in root_path.iterdir(): 
        
#        param_name = 'nlayers:%d'%args.nlayers + '_nhid:%d'%args.nhid + '_F:1_G:1_H:1' 
        param_name = 'nlayers:%d'%args.nlayers + '_nhid:%d'%args.nhid + '_F:0_G:0_H:0' 

        subroot_path = rp.joinpath(param_name) 

        if subroot_path.joinpath('precision.pkl').is_file() is False:
            continue 


        gen = pickle.load(open(str(subroot_path.joinpath('gen_dataset.pkl')), 'rb')) 
        outs = pickle.load(open(str(subroot_path.joinpath('out_dataset.pkl')), 'rb'))

        labels = pickle.load(open(str(subroot_path.joinpath('labels.pkl')), 'rb')) 
        scores = pickle.load(open(str(subroot_path.joinpath('scores.pkl')), 'rb'))
        
        precision = pickle.load(open(str(subroot_path.joinpath('precision.pkl')), 'rb')) 
        recall = pickle.load(open(str(subroot_path.joinpath('recall.pkl')), 'rb')) 
        
        seqlen = gen.size(0)
        
        # original sequence 
        gen = gen.view(seqlen, -1) 
        fig = tools.make_subplots(rows=gen.size(1)+1, cols=1) 

        for channel in range(gen.size(1)): 

            normal = list(gen[:seqlen, channel])
            abnormal = list(gen[:seqlen,channel]) 
            
            for i in range(seqlen):
                
                if labels[i]==1 and labels[i-1]==0 and i>0: # normal 
                    abnormal[i-1] = normal[i-1]
                elif labels[i]==0 and labels[i-1]==1 and i>0: # abnormal
                    pass
                elif labels[i]==1:
                    normal[i]=None 
                else:
                    abnormal[i]=None 
            
            trace_normal= go.Scatter(
                x = list(range(1, 1+seqlen)),
                y = normal, 
                mode = 'lines+markers', 
                marker=dict(size=1,),
                line=dict(color='rgba(119,119,119,1)'),
               ) 
            
            trace_abnormal= go.Scatter(
                x = list(range(1, 1+seqlen)),
                y = abnormal, 
                mode = 'lines+markers', 
                marker=dict(size=1,),
                line=dict(color='rgb(205, 12, 24)'), 
               ) 
            
            fig.append_trace(trace_normal, channel+1, 1) 
            fig.append_trace(trace_abnormal, channel+1, 1) 


            for k in range(1): 
                
                trace_out = go.Scatter(
                    x = torch.Tensor(range(1, 1+seqlen)), 
                    y = outs[:seqlen, channel+k*2].data.cpu(), 
                    mode = 'lines+markers',
                    line = dict(color='rgb(22,96,167)'), 
                    marker=dict(size=2,),
                    ) 
                fig.append_trace(trace_out, channel+1, 1) 
            
        trace_score = go.Scatter(   
            x = torch.Tensor(range(1, 1+seqlen)), 
            y = scores[:seqlen].data.cpu(), 
            mode = 'lines+markers', 
            marker=dict(size=1,), 
            line = dict(color='rgb(205,12,24)'), 
            ) 
        
        fig.append_trace(trace_score, gen.size(1)+1, 1) 
        
        fig['layout'].update(title=str(subroot_path), xaxis = go.XAxis(gridcolor='rgb(255,255,255)'))# xaxis = go.XAxis(), yaxis=go.YAxis()) # , plot_bgcolor='rgb(239,239,239)') 
        plotly.offline.plot(fig, filename=str(subroot_path.joinpath('result.html'))) 
        
