#!/bin/bash

data="ecg"
#filename="chfdb_chf13_45590.pkl"  lab
#filename="chfdb_chf01_275.pkl"    note
#filename="chfdbchf15.pkl"         #note
#filename="qtdbsel102.pkl"       
#filename="mitdb__100_180.pkl"    lab
filename="stdb_308_0.pkl"   #     note
#filename="ltstdb_20321_240.pkl"
#filename="xmitdb_x108_0.pkl"
#filename="ltstdb_20221_43.pkl" 

CUDA_VISIBLE_DEVICES=0 python3 1_train.py --data $data --filename $filename&
CUDA_VISIBLE_DEVICES=0 python3 1_train.py --data $data --filename $filename --feedback&
CUDA_VISIBLE_DEVICES=0 python3 1_train.py --data $data --filename $filename --gated&
CUDA_VISIBLE_DEVICES=0 python3 1_train.py --data $data --filename $filename --feedback --gated&
