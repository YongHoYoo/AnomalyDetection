#!/bin/bash

data="ecg"

declare -a filenames=("chfdb_chf13_45590.pkl" "chfdb_chf01_275.pkl" "chfdbchf15.pkl" "qtdbsel102.pkl" "mitdb__100_180.pkl" "stdb_308_0.pkl" "ltstdb_20321_240.pkl" "xmitdb_x108_0.pkl" "ltstdb_20221_43.pkl" ) 
#declare -a filenames=("mitdb__100_180.pkl" "stdb_308_0.pkl" "ltstdb_20321_240.pkl" "ltstdb_20221_43.pkl" ) 

#declare -a filenames=("ltstdb_20321_240.pkl" "chfdbchf15.pkl" )

for filename in "${filenames[@]}"
do
    CUDA_VISIBLE_DEVICES=0 python3 2_anomaly.py --data $data --filename $filename
    CUDA_VISIBLE_DEVICES=0 python3 2_anomaly.py --data $data --filename $filename --gated
    CUDA_VISIBLE_DEVICES=0 python3 2_anomaly.py --data $data --filename $filename --gated --feedback
    CUDA_VISIBLE_DEVICES=0 python3 2_anomaly.py --data $data --filename $filename --hidden_tied   
    CUDA_VISIBLE_DEVICES=0 python3 2_anomaly.py --data $data --filename $filename --gated --hidden_tied --feedback
 done   
