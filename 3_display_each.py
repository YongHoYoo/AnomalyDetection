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
    
    parser.add_argument('--filename', type=str, default='chfdb_chf13_45590.pkl', 
        help='filename of the dataset')

    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--seqlen', type=list, default=[4,8,16,32,64]) 
    parser.add_argument('--ninp', type=int, default=2) 
    parser.add_argument('--nhid', type=int, default=64)
    parser.add_argument('--nlayers', type=int, default=2) 
    parser.add_argument('--feedback', action='store_true') 

    args = parser.parse_args() 

    # check whether if there is a trained file in saved folder 
    param_folder_name = 'nlayers:%d'%args.nlayers + '_nhid:%d'%args.nhid + ('_feedback:1' if args.feedback else '_feedback:0') + ('_gated:0') 
    save_folder = Path('result', args.data, args.filename, param_folder_name) 

    if save_folder.joinpath('model_dictionary.pt').is_file() is not True: 
        print('There is no trained model in ')
        print(str(save_folder)) 
        sys.exit() 

    

    gen = pickle.load(open(str(save_folder.joinpath('gen_dataset.pkl')), 'rb')) 
    outs = pickle.load(open(str(save_folder.joinpath('out_dataset.pkl')), 'rb'))
    
    labels = pickle.load(open(str(save_folder.joinpath('labels.pkl')), 'rb')) 
    scores = pickle.load(open(str(save_folder.joinpath('scores.pkl')), 'rb'))
    
    precision = pickle.load(open(str(save_folder.joinpath('precision.pkl')), 'rb')) 
    recall = pickle.load(open(str(save_folder.joinpath('recall.pkl')), 'rb')) 
    
    seqlen = 1500
    
    # original sequence 
    gen = gen.squeeze() 
    fig = tools.make_subplots(rows=2*gen.size(1), cols=1) 

    for channel in range(gen.size(1)): 
    
        normal = list(gen[:seqlen, channel])
        abnormal = list(gen[:seqlen,channel]) 
        
        for i in range(seqlen):
            if labels[i]==1:
                normal[i]=None 
            else:
                abnormal[i]=None 
        
        trace_normal= go.Scatter(
            x = list(range(1, 1+seqlen)),
            y = normal, 
            mode = 'lines+markers', 
            marker=dict(size=1,),
           ) 
        
        trace_abnormal= go.Scatter(
            x = list(range(1, 1+seqlen)),
            y = abnormal, 
            mode = 'lines+markers', 
            marker=dict(size=1,),
           ) 
        
        print(channel)
        fig.append_trace(trace_normal, channel*2+1, 1) 
        fig.append_trace(trace_abnormal, channel*2+1, 1) 
        
        for out in outs:
            out = out.squeeze() 
        
            trace_out = go.Scatter(
                x = torch.Tensor(range(1, 1+seqlen)), 
                y = out[:seqlen, channel].data.cpu(), 
                mode = 'lines+markers',
                line = dict(dash='dot'), 
                marker=dict(size=2,),
                ) 
            fig.append_trace(trace_out, channel*2+1, 1) 
        
        trace_score = go.Scatter(   
            x = torch.Tensor(range(1, 1+seqlen)), 
            y = scores[:seqlen, channel].data.cpu(), 
            mode = 'lines+markers', 
            marker=dict(size=1,), 
            ) 

        fig.append_trace(trace_score, channel*2+2, 1) 
    
    
    fig['layout'].update(title='Result', plot_bgcolor='rgb(239,239,239)') 
    plotly.offline.plot(fig, filename=str(save_folder.joinpath('one.html'))) 
    
    # precision-recall 
    trace_pr = go.Scatter( 
        x = recall, 
        y = precision, 
        mode = 'line_markers', 
        name = 'prcurve', 
        )
    
    plotly.offline.plot({
        'data': [trace_pr], 
        'layout': go.Layout(title='prcurve', plot_bgcolor='rgb(239,239,239)', xaxis = go.XAxis(gridcolor='rgb(255,255,255)'), yaxis = go.YAxis(gridcolor='rgb(255,255,255)')), 
    
        }, filename=str(save_folder.joinpath('prcurve.html'))) 
