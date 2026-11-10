all: rayleigh-taylor-density.png

rayleigh-taylor-density.png: make_plots.py rayleigh-taylor.h5
	python make_plots.py --input rayleigh-taylor.h5 --output rayleigh-taylor-density.png

rayleigh-taylor-thickness.png: make_thickness_plot.py rayleigh-taylor.h5
	python make_thickness_plot.py --input rayleigh-taylor.h5 --output rayleigh-taylor-thickness.png

rayleigh-taylor.mp4: make_movie.py rayleigh-taylor.h5
	python make_movie.py --input rayleigh-taylor.h5 --output rayleigh-taylor.mp4

rayleigh-taylor.h5 rayleigh-taylor.log &: rayleigh_taylor.py
	python rayleigh_taylor.py --output rayleigh-taylor.h5 > rayleigh-taylor.log


# Work-precision runs: Radau-IIA vs backward Euler over the first 2 Ma.
# File names give the method and timestep, e.g. work-prec-radau-50.h5 is Radau-IIA with dt = 50 ka.
# Each run writes its wall time (s) and peak memory (KiB) to work-prec-<method>-<dt>.time.
# Use `make work-precision` for clean timings; `make -j3` is faster but timings are approximate.
FINAL_TIME := 2000
ATOL := 1e-8
RADAU_DTS := 200 100 50 25
RANA_DTS := 200 100 50 25
EULER_DTS := 200 100 50 25 12.5 6.25
REFERENCE_DT := 12.5

WORK_PREC_RUNS := $(sort $(RADAU_DTS:%=work-prec-radau-%.h5) work-prec-radau-$(REFERENCE_DT).h5) $(EULER_DTS:%=work-prec-euler-%.h5) \
    $(RANA_DTS:%=work-prec-rana-%.h5)

.PHONY: work-precision
work-precision: $(WORK_PREC_RUNS)

work-precision.png convergence.png &: make_work_precision.py $(WORK_PREC_RUNS)
	python make_work_precision.py --reference work-prec-radau-$(REFERENCE_DT).h5 \
	    --inputs $(WORK_PREC_RUNS) --output work-precision.png \
	    --convergence-output convergence.png

work-prec-radau-%.h5 work-prec-radau-%.time: rayleigh_taylor.py solver_parameters.py | work-prec-warmup.stamp
	/usr/bin/time -f "%e %M" -o work-prec-radau-$*.time \
	    python rayleigh_taylor.py --method radau --timestep $* --final-time $(FINAL_TIME) \
	    --atol $(ATOL) --output work-prec-radau-$*.h5 > work-prec-radau-$*.log

work-prec-euler-%.h5 work-prec-euler-%.time: rayleigh_taylor.py solver_parameters.py | work-prec-warmup.stamp
	/usr/bin/time -f "%e %M" -o work-prec-euler-$*.time \
	    python rayleigh_taylor.py --method euler --timestep $* --final-time $(FINAL_TIME) \
	    --atol $(ATOL) --output work-prec-euler-$*.h5 > work-prec-euler-$*.log

# Radau-IIA solved by Newton-Krylov, preconditioned with Rana LD split by stage.
work-prec-rana-%.h5 work-prec-rana-%.time: rayleigh_taylor.py solver_parameters.py | work-prec-warmup.stamp
	/usr/bin/time -f "%e %M" -o work-prec-rana-$*.time \
	    python rayleigh_taylor.py --method radau --solver rana --timestep $* --final-time $(FINAL_TIME) \
	    --atol $(ATOL) --output work-prec-rana-$*.h5 > work-prec-rana-$*.log

# One short run of each method so form compilation isn't counted in the timed runs.
work-prec-warmup.stamp: rayleigh_taylor.py solver_parameters.py
	for method in radau euler; do \
	    python rayleigh_taylor.py --method $$method --timestep 50 --final-time 50 \
	        --atol $(ATOL) --output work-prec-warmup.h5 > /dev/null || exit 1; \
	done
	python rayleigh_taylor.py --method radau --solver rana --timestep 50 --final-time 50 \
	    --atol $(ATOL) --output work-prec-warmup.h5 > /dev/null
	rm -f work-prec-warmup.h5
	touch $@
