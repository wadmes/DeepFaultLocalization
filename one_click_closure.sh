  #! /bin/bash
start=$1
end=$2
gpu_count=$3
name=$4
iter=$5
result=$6
for i in $(seq $start $((end-1)))
do
	echo "$name $i $gpu_count"
	# python3 main.py ./DeepFLwFB/Deep/ ./result/$result $name $i attention FB softmax $iter 1 $gpu_count;\
	python3 main.py . ./result/$result $name $i attention DeepFL softmax $iter 1 $gpu_count;	
done
