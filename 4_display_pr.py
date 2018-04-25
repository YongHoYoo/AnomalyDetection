import sys
import torch 
import pickle
import argparse
from pathlib import Path

import plotly 
import plotly.graph_objs as go
from plotly import tools 

if __name__ == '__main__': 

    data = 'ecg'
    filename = 'chfdb_chf13_45590.pkl' 
    nlayers = 2
    nhid=64
    feedback = 1
    gated = 1

    param_names = [] 
    trace_prs = [] 

    f_0_1 = [] # beta 0.1
    f_1_0 = [] # beta 1.0 

    param_names.append('nlayers:%d'%nlayers + '_nhid:%d'%nhid + '_feedback:0' + '_gated:0')
    param_names.append('nlayers:%d'%nlayers + '_nhid:%d'%nhid + '_feedback:1' + '_gated:0')
    param_names.append('nlayers:%d'%nlayers + '_nhid:%d'%nhid + '_feedback:0' + '_gated:1')
    param_names.append('nlayers:%d'%nlayers + '_nhid:%d'%nhid + '_feedback:1' + '_gated:1') 

    for param_name in param_names: 
        save_folder = Path('result', data, filename, param_name) 
        
        precision = pickle.load(open(str(save_folder.joinpath('precision.pkl')), 'rb')) 
        recall = pickle.load(open(str(save_folder.joinpath('recall.pkl')), 'rb')) 

        beta = 0.1

        def f_score(beta=1.0):
            return (1+beta**2)*torch.max((precision*recall).div(beta**2*precision+recall+1e-7)) 
        
        f_0_1.append(f_score(0.1))
        f_1_0.append(f_score(1.0)) 
        
        # precision-recall 
        trace_pr = go.Scatter( 
            x = recall, 
            y = precision, 
            mode = 'lines+markers', 
            name = param_name, 
            )

        trace_prs.append(trace_pr) 

    plotly.offline.plot({
            'data': trace_prs, 
            'layout': go.Layout(title='prcurve', plot_bgcolor='rgb(239,239,239)', xaxis = go.XAxis(gridcolor='rgb(255,255,255)'), yaxis = go.YAxis(gridcolor='rgb(255,255,255)')), 
        
            }, filename='prcurve.html')

    # table 
    f_score = [param_names, f_0_1, f_1_0]

    trace_table = go.Table(
        header=dict(values=['filename', 'beta 0.1', 'beta 1.0'], 
            line = dict(color='#7D7F80'), 
            fill = dict(color='#a1c3d1'), 
            align = ['left']*5), 
        cells=dict(values=f_score, 
                line = dict(color='#7D7F80'), 
                align = ['left']*5)) 

    plotly.offline.plot({
        'data': [trace_table], 
        'layout': go.Layout(title='table')}, filename='table.html') 

           

