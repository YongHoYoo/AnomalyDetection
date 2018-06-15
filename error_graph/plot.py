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

    filenames = ['valid_err_000.pt', 'valid_err_010.pt', 'valid_err_110.pt'] 

    trace_all = []
    for filename in filenames: 
        valid_err = pickle.load(open(filename, 'rb')) 
        trace_valid = go.Scatter(
            x = list(range(0, len(valid_err))), 
            y = valid_err, 
            mode = 'lines+markers', 
            line = dict(shape='spline'), 
            marker = dict(size=2), 
        ) 
        trace_all.append(trace_valid) 

    plotly.offline.plot({
        'data': trace_all, 
        'layout': go.Layout(title='abc')}, filename='valid_error.html') 

        
