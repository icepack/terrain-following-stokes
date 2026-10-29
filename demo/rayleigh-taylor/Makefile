all: rayleigh-taylor-density.png

rayleigh-taylor-density.png: make_plots.py rayleigh-taylor.h5
	python make_plots.py --input rayleigh-taylor.h5 --output rayleigh-taylor-density.png

rayleigh-taylor.mp4: make_movie.py rayleigh-taylor.h5
	python make_movie.py --input rayleigh-taylor.h5 --output rayleigh-taylor.mp4

rayleigh-taylor.h5 rayleigh-taylor.log &: rayleigh_taylor.py
	python rayleigh_taylor.py --output rayleigh-taylor.h5 > rayleigh-taylor.log
