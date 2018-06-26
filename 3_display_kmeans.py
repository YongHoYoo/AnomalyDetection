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
    parser.add_argument('--filename', type=str, default='chfdb_chf13_45590.pkl', help='filename of the dataset') 

    args = parser.parse_args() 

    root_path = Path('kmeans', args.data, args.filename)
    center = torch.load(str(root_path.joinpath('kmeans.pt')))   
    centers = center.view(-1,64,2) 

    fig = tools.make_subplots(rows=4, cols=5) 

    for i in range(centers.size(0)):

        for j in range(2): 

            if j==0:
                c = 'rgb(205,12,24)' 
            else:
                c = 'rgb(22, 96, 167)' 

            trace_normal = go.Scatter( 
                x = torch.range(0, 63), 
                y = centers[i][:,j], 
                mode = 'lines', 
                line = dict(color=c), 
            )
    
            fig['layout']['xaxis%d'%(i+1+j*10)].update(showgrid=False, zeroline=False, showline=True, mirror='ticks', showticklabels=False)
            fig['layout']['yaxis%d'%(i+1+j*10)].update(showgrid=False, zeroline=False, showline=True, mirror='ticks', showticklabels=False)
    
            print(i//5+1+(j)*2, i%5+1) 
            fig.append_trace(trace_normal, i//5+1+(j)*2, i%5+1) 

    plotly.offline.plot(fig) 
