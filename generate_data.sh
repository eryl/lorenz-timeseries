N=10000
T=10
RANDOM_SEED=1729

echo "Generating fixed Lorenz system data with $N trajectories and $T timesteps"
python bin/generate_lorenz_data.py $N -t $T --output data/lorenz.h5 --random-seed $RANDOM_SEED
echo "Generating random Lorenz system data with $N trajectories and $T timesteps"
python bin/generate_lorenz_data.py $N -t $T --hp-noise-level 0.1 --hp-n-per-perturbance 100 --output data/random_hp_lorenz.h5 --random-seed $RANDOM_SEED
echo "Splitting fixed Lorenz system data into training, validation and test set"
python bin/split_lorenz_dataset.py data/lorenz.h5 --split train 0.9 --split validation 0.05 --split test 0.05 --random-seed $RANDOM_SEED
echo "Splitting random Lorenz system data into training, validation and test set"
python bin/split_lorenz_dataset.py data/random_hp_lorenz.h5 --split train 0.9 --split validation 0.05 --split test 0.05 --random-seed $RANDOM_SEED

