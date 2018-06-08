#!/bin/bash

maxjob=6 

data="ecg"

#declare -a filenames=("chfdb_chf13_45590.pkl" "chfdb_chf01_275.pkl" "chfdbchf15.pkl" "qtdbsel102.pkl" "mitdb__100_180.pkl" "stdb_308_0.pkl" "ltstdb_20321_240.pkl" "xmitdb_x108_0.pkl" "ltstdb_20221_43.pkl" ) 
declare -a filenames=("chfdbchf15.pkl" "qtdbsel102.pkl" "stdb_308_0.pkl" )  


for filename in "${filenames[@]}"
do
    while [[ $(jobs -p | wc -l) -ge $maxjob ]]
    do
        wait
    done

    CUDA_VISIBLE_DEVICES=0 python3 1_train.py --data $data --filename $filename&
    CUDA_VISIBLE_DEVICES=0 python3 1_train.py --data $data --filename $filename --feedback&
    CUDA_VISIBLE_DEVICES=0 python3 1_train.py --data $data --filename $filename --gated&
    CUDA_VISIBLE_DEVICES=0 python3 1_train.py --data $data --filename $filename --hidden_tied&
    CUDA_VISIBLE_DEVICES=0 python3 1_train.py --data $data --filename $filename --feedback --gated --hidden_tied&

done

