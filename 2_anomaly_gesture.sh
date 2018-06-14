#!/bin/bash

data="gesture"

declare -a filenames=("ann_gun_CentroidA.pkl" ) 

for filename in "${filenames[@]}"
do
    CUDA_VISIBLE_DEVICES=0 python3 2_anomaly.py --data $data --filename $filename
#    CUDA_VISIBLE_DEVICES=0 python3 2_anomaly.py --data $data --filename $filename --gated
#    CUDA_VISIBLE_DEVICES=0 python3 2_anomaly.py --data $data --filename $filename --gated --feedback
#    CUDA_VISIBLE_DEVICES=0 python3 2_anomaly.py --data $data --filename $filename --hidden_tied   
#    CUDA_VISIBLE_DEVICES=0 python3 2_anomaly.py --data $data --filename $filename --gated --hidden_tied --feedback
 done   
