"""
Significance of Sharpe-ratio gains relative to the expectations-hypothesis benchmark (Tables 3, 5 and 7).

Ledoit and Wolf (2008) studentized circular block bootstrap for the difference between two Sharpe ratios,
implemented by Martin Martens. The Sharpe ratios are computed from monthly observations of overlapping 12-month
trading returns, so the test uses blocks of 12 months and HAC standard errors with 11 lags. As for the certainty
equivalent returns, a p-value is only computed when the Sharpe ratio of the model exceeds that of the benchmark.

    python src/sharpe_test.py        # writes output/sharpe_tests.csv (a few minutes)

Reference: Ledoit, O. and Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe ratio.
Journal of Empirical Finance, 15(5), 850-859.
"""

import numpy as np
import pandas as pd

import tables

BLOCK_SIZE = 12
N_BOOT = 3000
SEED = 42


def _sharpe_ratio(x):
    return float(np.mean(x) / np.std(x, ddof=1))


def _sharpe_diff(data):
    return _sharpe_ratio(data[:, 0]) - _sharpe_ratio(data[:, 1])


def _influence_sharpe(x):
    mu, sigma = float(np.mean(x)), float(np.std(x, ddof=1))
    centered = x - mu
    return centered / sigma - 0.5 * (mu / sigma**3) * (centered**2 - sigma**2)


def _hac_long_run_variance(series, lags):
    x = np.asarray(series, dtype=float) - np.mean(series)
    t = len(x)
    lrv = float(np.dot(x, x) / t)
    for k in range(1, lags + 1):
        lrv += 2.0 * (1.0 - k / (lags + 1)) * float(np.dot(x[k:], x[:-k]) / t)
    return max(lrv, 1e-12)


def _se_sharpe_diff_hac(data, lags):
    psi = _influence_sharpe(data[:, 0]) - _influence_sharpe(data[:, 1])
    return float(np.sqrt(_hac_long_run_variance(psi, lags=lags) / len(data)))


def _circular_block_indices(rng, n, block_size):
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=n_blocks)
    return (starts[:, None] + np.arange(block_size)[None, :]).reshape(-1)[:n] % n


def sharpe_diff_lw_bootstrap(r1, r2, block_size=BLOCK_SIZE, n_boot=N_BOOT, random_state=SEED):
    """Two-sided p-value for H0: SR(r1) = SR(r2), studentized circular block bootstrap with the null imposed
    by shifting the mean of r1 (Ledoit and Wolf, 2008)."""
    r1, r2 = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)
    rng = np.random.default_rng(random_state)
    data = np.column_stack((r1, r2))
    t = len(data)
    lags = min(max(block_size - 1, 1), max(t - 2, 1))

    delta_hat = _sharpe_diff(data)
    t_obs = delta_hat / _se_sharpe_diff_hac(data, lags=lags)

    shift = (_sharpe_ratio(r2) - _sharpe_ratio(r1)) * float(np.std(r1, ddof=1))
    data_null = np.column_stack((r1 + shift, r2))
    t_null = np.empty(n_boot)
    for b in range(n_boot):
        sample = data_null[_circular_block_indices(rng, t, block_size)]
        t_null[b] = _sharpe_diff(sample) / _se_sharpe_diff_hac(sample, lags=lags)
    return {"sharpe_diff": delta_hat, "t_statistic": t_obs, "p_value": float(np.mean(np.abs(t_null) >= abs(t_obs)))}


# ----------------------------------------------------------------------------------------------------------
# All Sharpe ratios in Tables 3, 5 and 7

TESTS = {
    "Table 3": [("baseline_main", m) for m in ["Yields", "Yield changes", "Forwards", "Forward changes",
                                                "Cochrane Piazessi"]] + [("baseline_diebold_li", "NN-3-1-Diebold Li direct")],
    "Table 5": [("nn", m) for m in ["NN-3-1-Yields", "NN-3-1-Yield changes", "NN-3-1-Forwards",
                                     "NN-3-1-Forward changes", "NN-3-1-Yields PC", "NN-3-1-Yield changes PC",
                                     "NN-3-1-Cochrane Piazessi", "NN-3-1-Diebold Li"]],
    "Table 7": [("baseline_macro", m) for m in ["Yields + macro (revised)", "Forwards + macro (revised)",
                                                 "Yield changes + macro (revised)", "Forward changes + macro (revised)",
                                                 "Cieslak Povala (param 0.987)"]]
               + [("nn", m) for m in ["NN-32-1-Yields + macro (revised)", "NN-32-1-Forwards + macro (revised)",
                                       "NN-32-1-Yield changes + macro (revised)",
                                       "NN-32-1-Forward changes + macro (revised)",
                                       "NN-3-1-Cieslak Povala (param 0.987)"]],
}


def run_all():
    bench = tables.load("baseline_main")["Trading returns: benchmark"].astype(float)
    rows = []
    for table, entries in TESTS.items():
        for run, method in entries:
            model = tables.load(run, method)[f"Trading returns: {method}"].astype(float)
            for m in tables.MATURITIES:
                idx = model[m].dropna().index.intersection(bench[m].dropna().index)
                r1, r2 = model[m].loc[idx].to_numpy(), bench[m].loc[idx].to_numpy()
                sr1, sr2 = _sharpe_ratio(r1), _sharpe_ratio(r2)
                p = sharpe_diff_lw_bootstrap(r1, r2)["p_value"] if sr1 > sr2 else np.nan
                rows.append([table, run, method, m, sr1, sr2, p])
    out = pd.DataFrame(rows, columns=["table", "run", "method", "maturity", "sharpe", "sharpe_benchmark", "p_value"])
    out.to_csv(tables.OUTPUT / "sharpe_tests.csv", index=False)
    return out


if __name__ == "__main__":
    res = run_all()
    sig = res[res.p_value < 0.1]
    print(f"{len(res)} Sharpe ratios, {res.p_value.notna().sum()} above the benchmark, {len(sig)} significant at 10%:")
    print(sig.to_string(index=False))
