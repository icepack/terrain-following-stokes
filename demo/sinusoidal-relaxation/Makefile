
all: relaxation.h5 relaxation-splitting.h5 relaxation-radau.h5

relaxation.h5 relaxation.log &: relaxation.py
	python relaxation.py --output relaxation --num-steps 40
	python relaxation.py --output relaxation-n1 --num-steps 1

relaxation-splitting.h5 relaxation-splitting.log &: relaxation.py
	python relaxation.py --output relaxation-splitting --lag-h --num-steps 320
	python relaxation.py --output relaxation-splitting-n300 --lag-h --num-steps 300

relaxation-radau.h5 relaxation-radau.log &: relaxation.py
	python relaxation.py --output relaxation-radau --num-steps 20 --method radau
