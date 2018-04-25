#!/bin/bash

data="ecg"
#filename="chfdb_chf13_45590.pkl" 
#filename="chfdb_chf01_275.pkl"
#xfilename="chfdbchf15.pkl"
#filename="qtdbsel102.pkl" 
#filename="mitdb__100_180.pkl"
filename="stdb_308_0.pkl"
#filename="ltstdb_20321_240.pkl"
#filename="xmitdb_x108_0.pkl"
#filename="ltstdb_20221_43.pkl" 


python3 2_anomaly.py --data $data --filename $filename
python3 2_anomaly.py --data $data --filename $filename --feedback
python3 2_anomaly.py --data $data --filename $filename --gated
python3 2_anomaly.py --data $data --filename $filename --feedback --gated


