#!/bin/bash

data="ecg"
declare -a filenames=("chfdbchf15.pkl" "qtdbsel102.pkl" "stdb_308_0.pkl" ) 

for filename in "${filenames[@]}"
do

    CUDA_VISIBLE_DEVICES=0 python3 2_anomaly.py --data $data --filename $filename
    CUDA_VISIBLE_DEVICES=0 python3 2_anomaly.py --data $data --filename $filename --feedback
    CUDA_VISIBLE_DEVICES=0 python3 2_anomaly.py --data $data --filename $filename --gated
    CUDA_VISIBLE_DEVICES=0 python3 2_anomaly.py --data $data --filename $filename --hidden_tied   
    CUDA_VISIBLE_DEVICES=0 python3 2_anomaly.py --data $data --filename $filename --gated --hidden_tied --feedback&
 done   
