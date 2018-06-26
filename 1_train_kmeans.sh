#!/bin/bash

maxjob=6 

#data="ecg"

#declare -a filenames=("chfdb_chf13_45590.pkl" "chfdb_chf01_275.pkl" "chfdbchf15.pkl" "qtdbsel102.pkl" "mitdb__100_180.pkl" "stdb_308_0.pkl" "ltstdb_20321_240.pkl" "xmitdb_x108_0.pkl" "ltstdb_20221_43.pkl" ) 

#data="gesture" 
#
#declare -a filenames=("ann_gun_CentroidA.pkl" ) 

data="power_demand"

declare -a filenames=("power_data.pkl" ) 

#data="space_shuttle"
#
#declare -a filenames=("TEK14.pkl" "TEK16.pkl" "TEK17.pkl" )


for filename in "${filenames[@]}"
do
    while [[ $(jobs -p | wc -l) -ge $maxjob ]]
    do
        wait
    done

    CUDA_VISIBLE_DEVICES=0 python3 1_kmeans.py --data $data --filename $filename

done

