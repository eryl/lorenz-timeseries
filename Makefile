N=10000
T=10
RANDOM_SEED=1729
all: split
data/lorenz.h5:
	python bin/generate_lorenz_data.py $(N) -t $(T) --output $@ --random-seed $(RANDOM_SEED)
data/random_hp_lorenz.h5:
	python bin/generate_lorenz_data.py $(N) -t $(T) --hp-noise-level 0.1 --hp-n-per-perturbance 100 --output $@ --random-seed $(RANDOM_SEED)
split: data/lorenz.h5 data/random_hp_lorenz.h5
	python bin/split_lorenz_dataset.py data/lorenz.h5 --split train 0.9 --split validation 0.05 --split test 0.05 --random-seed $(RANDOM_SEED)
	python bin/split_lorenz_dataset.py data/random_hp_lorenz.h5 --split train 0.9 --split validation 0.05 --split test 0.05 --random-seed $(RANDOM_SEED)
clean:
	rm data/lorenz* data/random_hp_lorenz*
