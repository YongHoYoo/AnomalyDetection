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

    block_pca = pickle.load(open('pca.pkl','rb')) # seqlen by all_sample by dim 
    print(block_pca.size())
    #16 by 10 2
    block_pca = block_pca.transpose(0,2) # dim all_sample seqlen
    # 2 by 10 by 16
        
    for b, block in enumerate(block_pca): 

        # num_sample by seqlen 

        fig = tools.make_subplots(rows=5, cols=1) 
        
        for i in range(block.size(0)): 
            
            trace_block = go.Scatter(
                x = torch.arange(block.size(1)), 
                y = block[i,:], 
                mode = 'lines+markers', 
                marker=dict(size=1,), 
            ) 

            fig.append_trace(trace_block, i+1, 1) 

        fig['layout'].update(title='pca', plot_bgcolor='rgb(239,239,239)') 
        plotly.offline.plot(fig, filename='pca_'+str(b)+'.html') 
        
        
