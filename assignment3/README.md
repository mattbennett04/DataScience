# Elman, Jordan and multi-recurrent networks for time series forecasting

CS441 assignment 3, Compares three simple recurrent networks for one-step-ahead forecasting
on five M4 series (`data/`), and tests retraining after a regime change. The report is in `report/`.

- `preprocess.py` - transforms, scaling, folds and input windows
- `models.py` - the three networks
- `train.py` - Optuna tuning and training per fold, results in `results/`
- `analyse.py` - tables, statistical tests and drift measures
- `retrain.py` - retraining after a detected change, results in `results_retrain*/`
- `report/make_figures.py` - figures for the report

To run everything, from this folder:

```bash
python train.py
python analyse.py
python analyse.py --results results_M26487_raw
python retrain.py
python retrain.py --fixed --datasets M6864
python report/make_figures.py
cd report && pdflatex 26913348RW441assignment3.tex && pdflatex 26913348RW441assignment3.tex
```

Finished runs are skipped, so delete a result file to rerun it.


AI Declaration:
Chatgpt was used to understand concepts, including how the different Nueral Networks, especially
statistical testing and to help word certain parts of the report. Claude was used to print the results
in a readable and neat format, and also for formatting the latex report.
