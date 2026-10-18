#!/bin/bash
# runs every experiment in order, each step writes its own results file
cd "$(dirname "$0")"
set -e
python3 tune_random_forest.py
python3 hyperparameter_sweeps.py
python3 ensemble_size_experiment.py
python3 diversity_experiment.py
python3 depth_experiment.py
python3 final_comparison.py
python3 extra_experiments.py leaf repeat
echo "all done"
