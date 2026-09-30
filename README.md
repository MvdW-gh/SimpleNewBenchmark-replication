# Reproducibility package: "A Simple New Benchmark for Forecasting Bond Risk Premia"

Tobias Hoogteijling, Martin Martens and Michel van der Wel — *International Journal of Forecasting*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23065218.svg)](https://doi.org/10.5281/zenodo.23065218)

Archived on Zenodo: https://doi.org/10.5281/zenodo.23065218 (version v1.0).

This package contains all data and code to reproduce the tables and figures of the paper and its online appendix.
The quickest check takes a few minutes: open `replicate.ipynb`, which rebuilds every table and figure from the
stored results and shows the number printed in the paper next to each reproduced number.

---

## 1. Assembly date and authorship

- **Assembled:** 30 September 2026
- **Authors of the package:** Tobias Hoogteijling (Robeco Quantitative Investing), Martin Martens (Robeco
  Quantitative Investing), Michel van der Wel (Erasmus University Rotterdam)
- **Contact:** Michel van der Wel, vanderwel@ese.eur.nl

The views expressed in the paper and this package are those of the authors and do not necessarily reflect the
position of Robeco.

## 2. Repository structure

```
.
├── README.md             this file
├── LICENSE               licences (MIT for code, CC BY 4.0 for other own material)
├── environment.yml       conda environment (exact package versions)
├── replicate.ipynb       MAIN FILE: all tables and figures in paper order, next to the printed numbers (step 1)
├── run_linear.py         runs all linear-model analyses            (step 2, ~10 minutes)
├── run_nn.py             runs all neural-network analyses          (step 3, ~13 hours)
├── run.py                runs any set of methods; used by the two scripts above
├── compare.py            short text comparison of all results with the paper
├── data/                 input data (Section 4)
├── src/
│   ├── snb.py            all estimation code (models, forecasts, tests, economic value)
│   ├── tables.py         builds the tables and figures from the stored results
│   ├── sharpe_test.py    significance of Sharpe-ratio gains (Ledoit-Wolf bootstrap)  (step 4, ~3 minutes)
│   ├── latex_tables.py   writes the LaTeX tables of the paper (Tables 1, 3-7, B.1, H.1) from the stored results
│   └── paper_values.py   the numbers as printed in the paper, used for the comparison
└── output/               stored results (intermediary data, Section 4.2)
    ├── <run name>/       one folder per run of run_linear.py (results.pkl, results.xlsx, figures/, run_info.json)
    ├── nn_n20/<model>/   one folder per neural-network specification of run_nn.py
    └── tables_tex/       LaTeX tables written by src/latex_tables.py
```

## 3. Computing environment

- **Language:** Python 3.10.
- **Packages:** listed with versions in `environment.yml`; main ones: numpy 1.26, pandas 1.5.3, scipy 1.11,
  statsmodels 0.14, scikit-learn 1.3, matplotlib 3.8, TensorFlow 2.17.0 with Keras 3.9.0, joblib, jupyterlab.
- **Set-up** (with [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda):

  ```
  conda env create -f environment.yml
  conda activate simplenewbenchmark
  ```

- **Note on versions:** the neural networks were originally estimated on Databricks Runtime 16.4 LTS ML, which uses
  TensorFlow 2.17 / Keras 3.9; the environment uses the same versions. Keras 2 (TensorFlow ≤ 2.15) gives
  materially different networks, because `keras.regularizers.l1_l2` then adds an extra L2 penalty by default.
  On Windows, the environment uses OpenBLAS, because scipy 1.11 fails with recent versions of Intel MKL.
- **Operating system:** developed and run on Windows 11; the code is platform independent.
- **Licences:** see `LICENSE`. Code: MIT. README, notebook text and stored results: CC BY 4.0. Input data: terms of
  their original sources (Section 4).

## 4. Data

### 4.1 Input data (`data/`)

All input data are publicly available and included in the package.

| File | Content | Source | Format |
|---|---|---|---|
| `LW_monthly.csv` | Zero-coupon Treasury yields of Liu and Wu (2021), monthly, June 1961 – December 2018, maturities 1–360 months, in percent | https://sites.google.com/view/jingcynthiawu/ (as cited in the paper) | CSV; first column `Year` = YYYYMM, then one column per maturity in months |
| `data_mccracken.csv` | FRED-MD macro data (McCracken and Ng, 2016), 128 series, March 1959 – December 2018, transformed to stationarity | https://www.stlouisfed.org/research/economists/mccracken/fred-databases, vintage 2019-01 (January 2019) | CSV without header; one column per series in the FRED-MD order (numbering of Table H.1), one row per month |
| `fredmd_series.csv` | Documentation of `data_mccracken.csv`: column number, FRED-MD mnemonic, description and block for each of the 128 series (the variable list of Table H.1) | FRED-MD vintage 2019-01; descriptions follow McCracken and Ng (2016) | CSV |
| `CPI data for cieslak and povala (FRED download).csv` | Consumer Price Index; only `CPILFESL` (core CPI, index 1982–84 = 100) is used, for the Cieslak–Povala trend inflation | FRED, https://fred.stlouisfed.org/series/CPILFESL | CSV with columns `DATE`, `CPIAUCSL`, `CPILFESL`, `INDPRO` |

- **Liu–Wu yields:** the version used in the paper; the authors later updated their data set, which gives
  numerically slightly different but qualitatively similar results. The CSV is a conversion of the monthly Excel
  file published by Liu and Wu; only the maturities of 12, 24, ..., 120 months are used.
- **FRED-MD:** `data_mccracken.csv` is intermediary data, created from the January 2019 FRED-MD vintage with the
  MATLAB code published by McCracken (`fredfactors.m` in `fred-databases_code.zip` on the FRED-MD page): every series
  is transformed with its transformation code, the first two months are dropped, outliers
  (|x − median| > 10 × interquartile range) are set to missing, and missing values are filled with the EM algorithm
  (`factors_em`, 8 factors); the sample is then restricted to March 1959 – December 2018. The columns follow the order
  of the series in the vintage file. We verified this against the archived January 2019 vintage (historical
  vintages 2015–2025 on the FRED-MD page): 127 of the 128 columns agree with its 127 series (differences only in the
  observations treated as outliers or missing). The remaining column (no. 75, S&P Industrials) was part of the
  vintage as originally published but is not included in the later archive.
- **Sample:** all analyses use August 1971 – December 2018 (the first ten-year yield is available in August 1971).
- **Usage:** the data are freely available for research; their terms of use are those of the respective providers.

### 4.2 Intermediary data (`output/`)

The estimation results are stored in `output/`, so that all tables and figures can be checked without
re-running the estimation (in particular the neural networks, which take about 13 hours).

| Files | Generated by | Used by |
|---|---|---|
| `output/<run>/results.pkl`, `results.xlsx`, `figures/`, `run_info.json` for the 16 linear runs (`baseline_main`, `baseline_diebold_li`, `baseline_macro`, `baseline_macro_ex_yields`, `hanson_12m`, `hanson_6m`, `nonoverlapping`, `rolling`, `holding_1/3/6/24`, `lookback_1/3/6/24`) | `run_linear.py` | `replicate.ipynb`, `compare.py`, `src/latex_tables.py` |
| `output/nn_n20/<model>/…` for the 13 neural-network specifications | `run_nn.py` | same |
| `output/sharpe_tests.csv` (p-values of the Sharpe-ratio tests, Tables 3, 5, 7) | `src/sharpe_test.py` | `replicate.ipynb`, `src/latex_tables.py` |
| `output/tables_tex/*.tex` | `src/latex_tables.py` | the paper |

`results.pkl` contains all numerical outputs of a run (a Python dictionary of pandas objects); `results.xlsx`
contains the same tables for inspection without Python; `run_info.json` records the settings and runtime.

## 5. Which code produces which outputs

The figures and tables are displayed, in the order of the paper, by `replicate.ipynb`. They are computed by:

| Paper | Run (script) | Shown in `replicate.ipynb` by |
|---|---|---|
| Figure 1 | `baseline_main` (`run_linear.py`) | `tables.figure_1()` |
| Figures 2–5 | `baseline_main` (`run_linear.py`) | `tables.show_figure(...)` |
| Table 1 | `baseline_main` | `tables.full_sample("Table 1")` |
| Table 2 | `baseline_main`, `baseline_diebold_li` | `tables.oos_r2("Table 2")` |
| Table 3 | `baseline_main`, `baseline_diebold_li` | `tables.economic_value("Table 3")` |
| Table 4 | `nn_n20` (`run_nn.py`) | `tables.oos_r2("Table 4")` |
| Table 5 | `nn_n20` (`run_nn.py`) | `tables.economic_value("Table 5")` |
| Table 6 | `baseline_macro`, `nn_n20` | `tables.oos_r2("Table 6")` |
| Table 7 | `baseline_macro`, `nn_n20` | `tables.economic_value("Table 7")` |
| Numbers in the text (Sections 2.1, 4.2) | `baseline_main`, `baseline_macro` | `tables.text_numbers()` |
| Footnote 14 (macro data excluding interest-rate series) | `baseline_main`, `baseline_macro`, `baseline_macro_ex_yields` | `tables.ex_yields()` |
| Table A.1 | `hanson_12m`, `hanson_6m` | `tables.hanson()` |
| Table B.1 | `baseline_main` | `tables.forward_change_loadings()` |
| Table B.2 | `baseline_main` | `tables.full_sample("Table B.2")` |
| Table B.3 | `baseline_main`, `baseline_macro` | rows of Tables 2 and 6 |
| Table D.1 | `nonoverlapping` | `tables.oos_r2("Table D.1")` |
| Table E.1 | `rolling` | `tables.oos_r2("Table E.1")` |
| Table F.1 | `holding_1`, `holding_3`, `holding_6`, `holding_24` (12 months: `baseline_*`) | `tables.oos_r2("Table F.1")` |
| Table F.2 | `lookback_1`, `lookback_3`, `lookback_6`, `lookback_24` | `tables.oos_r2("Table F.2")` |
| Table H.1 | `baseline_macro` | `tables.macro_loadings_named()` |

The settings of every run are defined in `run_linear.py` (dictionary `RUNS`) and `run_nn.py` (list `NN_METHODS`).
All estimation code is in `src/snb.py`, function `pc_regression`.

### How to reproduce

1. **Check the tables and figures from the stored results** (minutes): open `replicate.ipynb` in Jupyter
   (`jupyter lab replicate.ipynb`) and run all cells. Every table shows the printed value (`paper`) above the
   reproduced value (`reproduced`) with a check mark; the first cell gives an overview per table.
2. **Re-run the linear models** (about 10 minutes): `python run_linear.py`, then re-run the notebook.
3. **Re-run the neural networks** (about 13 hours): `python run_nn.py --tag n20`, then re-run the notebook.
4. **Re-run the Sharpe-ratio tests** (about 3 minutes, after steps 2 and/or 3): `python src/sharpe_test.py`. The
   significance stars on the Sharpe ratios in Tables 3, 5 and 7 are based on these p-values (Ledoit and Wolf, 2008).
   A single specification can be run with e.g. `python run_nn.py --tag check --methods "NN-3-1-Yield changes"`
   (about one hour) and shown by setting `tables.NN_TAG = "check"` in the notebook.

A quick text comparison of all results with the paper is given by `python compare.py`.

## 6. Hardware and expected runtime

Results were produced on a laptop with an Intel Core i7-13700H (14 cores, 20 threads), 64 GB RAM, Windows 11.

| Step | Runtime |
|---|---|
| `replicate.ipynb` from stored results | < 1 minute |
| `run_linear.py` (16 runs, 6–7 in parallel) | about 10 minutes |
| `run_nn.py` (13 specifications, 18 parallel processes) | about 13.5 hours (55–75 minutes per specification) |
| `src/sharpe_test.py` | about 3 minutes |

Memory use is modest (a few GB). Runtimes scale roughly inversely with the number of CPU cores for `run_nn.py`.

## 7. Special setup

- No GPU is needed; everything runs on CPU.
- The neural networks are estimated in parallel over forecast dates with joblib; the number of processes is set with
  `--n-jobs` (default: all cores). Each process uses one thread.
- Random seeds are fixed (networks: seed 100 for every forecast date; bootstrap: numpy seed 0). Re-running on the
  same machine and software reproduces the stored results; on other hardware small numerical differences in the
  neural-network results are possible.
- On Windows, `run_nn.py` prevents the computer from going to sleep while it runs.
- On Windows, place the package in a short folder path (e.g. `C:\snb`): Windows limits file paths to 260 characters,
  and some figure file names are long. If a path is too long, the affected figures are skipped with a warning; the
  results are not affected.

## Notes on the code

The estimation code was originally written for a local Jupyter environment (linear models) and Databricks (neural
networks). `src/snb.py` combines both in one local version. Relative to the code used for the earliest drafts of
the paper, a bug was corrected in the neural networks on principal components (Tables 4 and 5, rows *Y* and
*ΔY*): the principal-component transformation was only applied during hyperparameter tuning, not when forecasting.
The published tables are based on the corrected code.

## Acknowledgements

The estimation code was originally written by Tobias Hoogteijling and partly run in a company computing environment
(Databricks). The stand-alone version in this package, this README and the reproduction checks were prepared with the
assistance of Claude Code (Anthropic), under the supervision of the authors, who verified all results.

## References

- Ledoit, O. and Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe ratio. *Journal of Empirical
  Finance*, 15(5), 850–859.
- Liu, Y. and Wu, J. C. (2021). Reconstructing the yield curve. *Journal of Financial Economics*, 142(3), 1395–1425.
- McCracken, M. W. and Ng, S. (2016). FRED-MD: A monthly database for macroeconomic research. *Journal of Business &
  Economic Statistics*, 34(4), 574–589.
