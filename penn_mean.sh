#!/bin/bash

maxjob=4
declare -a atype=("mean")
declare -a nfed=(2 3) 
declare -a nrec=(1 2 3 4 5 10 20 50 100) 

for fed in ${nfed[@]} 
do
	
	for rec in ${nrec[@]} 
	do 
		for a in "${atype[@]}"
		do
			while [[ $(jobs -p | wc -l) -ge $maxjob ]]
			do
				wait
			done
	
			echo "nfed" $fed , "nrec" $rec, "atype" $a
			CUDA_VISIBLE_DEVICES=1 ipython3 main.py -- --data penn --model DenseLSTM --nhid 1500 --dropout 0.65 --epochs 80 --tied --feedback --nlayers $fed --nrec $rec --atype "${a}"& # feedback
			CUDA_VISIBLE_DEVICES=1 ipython3 main.py -- --data penn --model DenseLSTM --nhid 1500 --dropout 0.65 --epochs 80 --tied  --nlayers $fed --nrec $rec --atype "${a}"&           # no-feedback
	
		done
	done

done
