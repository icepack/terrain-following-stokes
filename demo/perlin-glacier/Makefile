
all: perlin.h5 perlin-degree3.h5 perlin-nz10.h5 perlin-radau.h5

perlin.h5 perlin.log &: perlin.py
	python perlin.py --output perlin.h5 > perlin.log

perlin-degree3.h5 perlin-degree3.log &: perlin.py
	python perlin.py --output perlin-degree3.h5 --degree 3 > perlin-degree3.log

perlin-nz10.h5 perlin-nz10.log &: perlin.py
	python perlin.py --output perlin-nz10.h5 --num-zcells 10 > perlin-nz10.log

perlin-radau.h5 perlin-radau.log &: perlin.py
	python perlin.py --output perlin-radau.h5 --method radau --timestep 2.0 > perlin-radau.log
