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

  #  data = 'ecg' 
  #  data = 'power_demand'
  #  data = 'gesture'
    data = 'space_shuttle'
    option = ['000.pt', '010.pt', '110.pt', '001.pt', '111.pt'] 
    filenames = ['valid_err_' + data + '_'+ option[i] for i in range(5)] 
        
    trace_all = []
    for filename in filenames: 
        valid_err = pickle.load(open(filename, 'rb')) 
        trace_valid = go.Scatter(
            x = list(range(0, len(valid_err),10)), 
            y = valid_err[0::10], 
            mode = 'lines+markers', 
            marker = dict(size=7), 
            line = dict(shape='spline'), 
        ) 
        trace_all.append(trace_valid) 

    plotly.offline.plot({
        'data': trace_all, 
        'layout': go.Layout(title='abc')}, filename='valid_error_'+data+'.html') #)#,xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))}, filename='valid_error.html') 

        
