"""
Build the paper's tables and figures from the stored results in output/ and show them next to the printed values.

Used by replicate.ipynb. Linear results come from run_linear.py, neural-network results from run_nn.py
(output/nn_<NN_TAG>/).
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paper_values as pv

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
NN_TAG = "n20"

MATURITIES = [24, 36, 48, 60, 84, 120]

LABELS = {
    "Benchmark": "Benchmark (EH)",
    "Yields": "Y", "Yields without PC1": "Y (without PC1)", "Yield changes": "ΔY",
    "Yields + yield changes": "Y and ΔY", "Forwards": "F", "Forward changes": "ΔF",
    "Cochrane Piazessi": "Cochrane and Piazzesi", "NN-3-1-Diebold Li direct": "Diebold and Li",
    "Yields + macro (revised)": "Y and Z", "Forwards + macro (revised)": "F and Z",
    "Yield changes + macro (revised)": "ΔY and Z", "Forward changes + macro (revised)": "ΔF and Z",
    "Cieslak Povala (param 0.987)": "Cieslak and Povala",
    "NN-3-1-Yields": "NN: y", "NN-3-1-Yield changes": "NN: Δy", "NN-3-1-Forwards": "NN: f",
    "NN-3-1-Forward changes": "NN: Δf", "NN-3-1-Yields PC": "NN: Y", "NN-3-1-Yield changes PC": "NN: ΔY",
    "NN-3-1-Cochrane Piazessi": "NN: Cochrane and Piazzesi", "NN-3-1-Diebold Li": "NN: Diebold and Li",
    "NN-32-1-Yields + macro (revised)": "NN: y and Z", "NN-32-1-Forwards + macro (revised)": "NN: f and Z",
    "NN-32-1-Yield changes + macro (revised)": "NN: Δy and Z",
    "NN-32-1-Forward changes + macro (revised)": "NN: Δf and Z",
    "NN-3-1-Cieslak Povala (param 0.987)": "NN: Cieslak and Povala",
}
for _m in ["Yields", "Yield changes", "Forwards", "Forward changes"]:
    LABELS[f"{_m} (nonoverlapping)"] = LABELS[_m]


# ----------------------------------------------------------------------------------------------------------
# Loading results


def _folder(run, method):
    if run == "nn":
        return f"nn_{NN_TAG}/" + "".join(c if c.isalnum() or c in "-_" else "_" for c in method)
    return run


_cache = {}


def load(run, method=None):
    """Stored output dictionary of a run (None if the run has not been done yet)."""
    folder = _folder(run, method)
    if folder not in _cache:
        path = OUTPUT / folder / "results.pkl"
        _cache[folder] = pickle.load(open(path, "rb")) if path.exists() else None
    return _cache[folder]


def _maturity_columns(n_values):
    return [f"n={int(m / 12)}" for m in MATURITIES][-n_values:]


def _stars(p):
    if p is None or pd.isna(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def _check(ours, paper, tol):
    if ours is None:
        return "not run yet"
    diffs = [abs(a - b) for a, b in zip(ours, paper) if a is not None and b is not None]
    return "✓" if max(diffs) <= tol else f"✗ (max diff {max(diffs):.3f})"


# ----------------------------------------------------------------------------------------------------------
# Out-of-sample R2 and economic value


def oos_r2(table):
    """Out-of-sample R2 table (Tables 2, 4, 6, D.1, E.1, F.1, F.2): paper row, reproduced row with stars and
    Clark-West p-values, and a check."""
    rows = []
    for (run, method), paper in pv.R2[table].items():
        outp = load(run, method)
        label = LABELS.get(method, method)
        if run.startswith(("holding", "lookback")):
            label = f"{run.replace('_', ' ')}: {label}"
        ours, cells = None, ["" for _ in paper]
        if outp is not None:
            r2 = outp["R2"].loc[method]
            pvals = outp["R2_pval"].loc[method]
            cols = [m for m in MATURITIES if m in r2.index][-len(paper):]
            ours = [float(r2[m]) for m in cols]
            cells = [f"{float(r2[m]):.3f}{_stars(pvals[m])}" + (f" ({float(pvals[m]):.3f})" if pd.notna(pvals[m]) else "")
                     for m in cols]
        rows.append([label, "paper"] + [f"{x:.3f}" for x in paper] + [""])
        rows.append([label, "reproduced"] + cells + [_check(ours, paper, 0.0015)])
    return _frame(rows, len(next(iter(pv.R2[table].values()))))


def sharpe_pvalues():
    """p-values of the Sharpe-ratio tests (output/sharpe_tests.csv, written by src/sharpe_test.py), keyed by
    (method, maturity); empty if the tests have not been run."""
    path = OUTPUT / "sharpe_tests.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {(r.method, int(r.maturity)): r.p_value for r in df.itertuples()}


def economic_value(table):
    """Certainty equivalent returns (%) with p-values and Sharpe ratios (Tables 3, 5, 7)."""
    rows = []
    for metric, source, fmt, tol in [("CER (%)", pv.CER, "{:.2f}", 0.015), ("Sharpe ratio", pv.SHARPE, "{:.3f}", 0.0015)]:
        for (run, method), paper in source[table].items():
            outp = load(run, method)
            key = "CER" if metric.startswith("CER") else "Sharpe Ratio"
            ours, cells = None, ["" for _ in paper]
            if outp is not None:
                util = outp["Utilities"]
                stat = util.loc[(key, method, "stat")].astype(float) * (100 if key == "CER" else 1)
                ours = [float(stat[m]) for m in MATURITIES]
                pvals = util.loc[(key, method, "p-val")] if (key, method, "p-val") in util.index else None
                if key == "Sharpe Ratio":
                    sr_p = sharpe_pvalues()
                    pvals = pd.Series({m: sr_p.get((method, m), np.nan) for m in MATURITIES})
                cells = []
                for m in MATURITIES:
                    p = None if pvals is None else pvals[m]
                    cells.append(fmt.format(stat[m]) + (f"{_stars(p)} ({float(p):.3f})" if p is not None and pd.notna(p) else ""))
            label = f"{metric}: {LABELS.get(method, method)}"
            rows.append([label, "paper"] + [fmt.format(x) for x in paper] + [""])
            rows.append([label, "reproduced"] + cells + [_check(ours, paper, tol)])
    return _frame(rows, 6)


def _frame(rows, n_values):
    df = pd.DataFrame(rows, columns=["row", "source"] + _maturity_columns(n_values) + ["check"])
    return df.set_index(["row", "source"])


# ----------------------------------------------------------------------------------------------------------
# Full-sample regressions (Table 1, Table B.2)


def full_sample(table):
    """Coefficients with Newey-West t-statistics and R2 of the full-sample predictive regressions."""
    run = "baseline_main"
    outp = load(run)
    rows = []
    for method, per_n in pv.FULL_SAMPLE[table].items():
        cof = outp[f"{method}: full sample regression"].astype(float)
        for n, (coefs, tstats, r2) in per_n.items():
            m = n * 12
            ours_c = [cof.loc[(m, "coef"), c] for c in ["const", 1, 2, 3]]
            ours_t = [cof.loc[(m, "t-stat"), c] for c in ["const", 1, 2, 3]]
            ours_p = [cof.loc[(m, "p-value"), c] for c in ["const", 1, 2, 3]]
            ours_r2 = cof.loc[(m, "coef"), "R2"]
            label = f"{LABELS[method]}, n={n}"
            rows.append([label, "paper"] + [f"{c} ({t:.2f})" for c, t in zip(coefs, tstats)] + [f"{r2:.3f}", ""])
            rows.append([label, "reproduced"]
                        + [f"{c:.2f}{_stars(p)} ({t:.2f})" for c, t, p in zip(ours_c, ours_t, ours_p)]
                        + [f"{ours_r2:.3f}",
                           _check(ours_c + ours_t + [ours_r2], coefs + tstats + [r2], 0.0051)])
    df = pd.DataFrame(rows, columns=["row", "source", "const", "PC1", "PC2", "PC3", "R2", "check"])
    return df.set_index(["row", "source"])


# ----------------------------------------------------------------------------------------------------------
# Appendix tables


def hanson():
    """Table A.1: regressions of ten-year excess returns on level, slope and their changes, pre- and post-2000."""
    rows = []
    for (subsample, horizon), paper in pv.HANSON.items():
        run = "hanson_12m" if horizon == 12 else "hanson_6m"
        outp = load(run)
        res = outp[f"Hanson Lucca Wright: {subsample} regression (10Y)"]
        for var, paper_vals in paper.items():
            if var == "adj-R2":
                ours = [float(res.loc["adj-R2", (s, "coef")]) for s in range(1, 6)]
                cells = [f"{x:.2f}" for x in ours]
            else:
                ours = [None if pd.isna(res.loc[var, (s, "coef")]) else float(res.loc[var, (s, "coef")]) for s in range(1, 6)]
                cells = ["" if c is None else f"{c:.2f}{_stars(res.loc[var, (s, 'p-value')])} ({float(res.loc[var, (s, 't-stat')]):.2f})"
                         for c, s in zip(ours, range(1, 6))]
            label = f"{subsample}, {horizon} months: {var}"
            rows.append([label, "paper"] + ["" if x is None else f"{x:.2f}" for x in paper_vals] + [""])
            rows.append([label, "reproduced"] + cells + [_check(ours, paper_vals, 0.0051)])
    df = pd.DataFrame(rows, columns=["row", "source"] + [f"({s})" for s in range(1, 6)] + ["check"])
    return df.set_index(["row", "source"])


def forward_change_loadings():
    """Table B.1: loadings of the principal components of forward-rate changes and their eigenvalues.

    Principal components are only defined up to sign, and the code does not sort the eigenvalues; the
    comparison with the paper is therefore made on absolute loadings, with components in descending order
    of eigenvalue."""
    outp = load("baseline_main")
    X = outp["forward changes"].astype(float).dropna().to_numpy()
    eigenvalues = np.real(np.linalg.eig(np.cov(X - X.mean(axis=0), rowvar=False))[0])
    order = np.argsort(eigenvalues)[::-1]
    loadings = np.real(outp["Forward changes: PC loadings"])[:, order]
    index = [f"n={m}" for m in range(12, 132, 12)] + ["λ"]
    columns = [f"PC {i}" for i in range(1, 11)]
    ours = pd.DataFrame(np.vstack([loadings, eigenvalues[order]]), index=index, columns=columns)
    paper = pd.DataFrame(pv.FORWARD_CHANGE_LOADINGS + [pv.FORWARD_CHANGE_EIGENVALUES], index=index, columns=columns)
    max_diff_abs_loadings = (ours.iloc[:10].abs() - paper.iloc[:10].abs()).abs().max().max()
    eigenvalue_ratio = (paper.loc["λ"] / ours.loc["λ"]).iloc[:4]
    summary = (f"max difference in absolute loadings: {max_diff_abs_loadings:.3f}; "
               f"paper / reproduced eigenvalues (PC1-4): {', '.join(f'{r:.3f}' for r in eigenvalue_ratio)}")
    return ours.round(2), paper, summary


def macro_loadings():
    """Table H.1: loadings of the 128 FRED-MD series on the first eight macro principal components
    (sign convention of the code: first loading of every component positive)."""
    outp = load("baseline_macro")
    loadings = pd.DataFrame(np.real(outp["Yields + macro (revised): macro PC loadings"]),
                            index=range(1, 129), columns=[f"PC{i}" for i in range(1, 9)])
    first_row = pd.DataFrame([pv.MACRO_LOADINGS_ROW1, loadings.iloc[0].round(3).tolist()],
                             index=["paper", "reproduced"], columns=loadings.columns)
    return loadings.round(3), first_row


def macro_loadings_named():
    """Table H.1 with the FRED-MD mnemonic and description of every series (data/fredmd_series.csv)."""
    series = pd.read_csv(ROOT / "data" / "fredmd_series.csv", index_col="column", encoding="utf-8")
    loadings, _ = macro_loadings()
    return series[["mnemonic", "description", "block"]].join(loadings)


def ex_yields():
    """Footnote 14: out-of-sample R2 when the 17 interest-rate series are removed from the macro data."""
    main, macro, ex = load("baseline_main"), load("baseline_macro"), load("baseline_macro_ex_yields")
    rows = []
    for m, label in [("Yield changes", "ΔY"), ("Forward changes", "ΔF"), ("Yields", "Y"), ("Forwards", "F")]:
        for name, outp, method in [(f"{label}", main, m), (f"{label} and Z", macro, f"{m} + macro (revised)"),
                                   (f"{label} and Z (ex yields)", ex, f"{m} + macro (ex yields, revised)")]:
            rows.append([name] + [round(float(outp["R2"].loc[method, x]), 3) for x in MATURITIES])
    return pd.DataFrame(rows, columns=["factors"] + _maturity_columns(6)).set_index("factors")


# ----------------------------------------------------------------------------------------------------------
# Numbers quoted in the text


def text_numbers():
    """Numbers quoted in the text of Section 2 and 4."""
    main, macro = load("baseline_main"), load("baseline_macro")
    X = main["yield changes"].astype(float).dropna().to_numpy()
    ev = np.sort(np.linalg.eigvalsh(np.cov(X - X.mean(axis=0), rowvar=False)))[::-1]
    corr = main["Yields and change in yields PC correlations"].astype(float)
    ours = {"variance explained by first three PCs of yield changes (%)": 100 * ev[:3].sum() / ev.sum()}
    for k in (1, 2, 3):
        ours[f"correlation PC{k} of changes vs change in PC{k} (%)"] = \
            100 * corr.loc[f"Yield change PC {k}", f"Change in PC of yields{k}"]
    cp = macro["Cieslak Povala (param 0.987) prediction"].astype(float)
    dy = main["Yield changes prediction"].astype(float)
    cors = [100 * cp[m].corr(dy[m]) for m in MATURITIES]
    ours["correlation Cieslak-Povala vs yield-change forecasts, min (%)"] = min(cors)
    ours["correlation Cieslak-Povala vs yield-change forecasts, max (%)"] = max(cors)
    df = pd.DataFrame({"paper": pd.Series(pv.TEXT), "reproduced": pd.Series(ours).round(2)})
    df["check"] = ["✓" if abs(a - b) <= (0.05 if a >= 90 else 0.5) else "✗" for a, b in zip(df["paper"], df["reproduced"])]
    return df


# ----------------------------------------------------------------------------------------------------------
# Figures


def figure_1():
    """Figure 1: yields (left) and 12-month changes in yields (right), August 1971 - December 2018."""
    outp = load("baseline_main")
    yields, changes = outp["yields"].astype(float), outp["yield changes"].astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for m in [12, 36, 60, 84, 120]:
        axes[0].plot(yields.index, yields[m], label=f"{m // 12}Y yield")
        axes[1].plot(changes.index, changes[m], label=f"Change in {m // 12}Y yield")
    for ax in axes:
        ax.set_ylabel("%")
        ax.legend(fontsize="x-small")
    fig.tight_layout()
    return fig


def show_figure(run, name):
    """Display a figure stored by run.py (output/<run>/figures/<name>.png)."""
    from IPython.display import Image, display
    display(Image(filename=str(OUTPUT / run / "figures" / f"{name}.png"), width=700))


# ----------------------------------------------------------------------------------------------------------
# Overview


def summary():
    """Number of rows that match the paper, per table."""
    rows = []
    for table in pv.R2:
        c = oos_r2(table).xs("reproduced", level=1)["check"]
        rows.append([table, "out-of-sample R2", c])
    for table in pv.CER:
        c = economic_value(table).xs("reproduced", level=1)["check"]
        rows.append([table, "CER and Sharpe ratio", c])
    for table in pv.FULL_SAMPLE:
        c = full_sample(table).xs("reproduced", level=1)["check"]
        rows.append([table, "full-sample regressions", c])
    c = hanson().xs("reproduced", level=1)["check"]
    rows.append(["Table A.1", "coefficients", c])
    c = text_numbers()["check"]
    rows.append(["Text", "quoted numbers", c])
    out = pd.DataFrame([[t, what, (c == "✓").sum(), c.str.startswith("✗").sum(), (c == "not run yet").sum()]
                        for t, what, c in rows],
                       columns=["table", "statistic", "match", "differ", "not run yet"])
    return out.set_index("table")
