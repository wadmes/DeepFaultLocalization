#!/bin/bash
subs=(Chart Lang Time Math Mockito Closure)
vers=(26 65 27 106 38 133)
for idx in {0..6}
do
	sub=${subs[$idx]}
	for model in {mlp,mlp2,birnn,dfl1,dfl2}
	do
		ver_for_sub=${vers[$idx]}
		
		for v in `seq 1 $ver_for_sub`
		do
			echo $sub
			echo $v
			python main.py /home/xia/DeepFL/Data/ /home/xia/DeepFL/result $sub $v $model DeepFL softmax 2 1 1;
		done
	done
done
