# Random forest for network intrusion classification

Code for the CS441/741 Assignment 4 report (option 1, ensemble learning)
All models, pre-processing steps and metrics are implemented with NumPy, in `classifiers.py` and `ml_utils.py`. Optuna is used only for the hyperparameter search

## Data

The scripts read the modified UNSW-NB15 dataset from `../assignment2/networkTraffic.csv`. Change the path in `pipelines.py` if the file is somewhere else.

## Running

```bash
./run_experiments.sh                  # every experiment, writes JSON and CSV files to results/
python3 make_figures.py optuna_plots sweep_plots size_plot diversity_plot depth_plot final_results extra_results
```

The figures are written to `figures/`, and the report source is in `report/`.
