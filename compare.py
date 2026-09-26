"""
Compare reproduced results (output/<run>/results.pkl) with the numbers printed in the paper (src/paper_values.py).

    python compare.py                 # all tables for which results exist
    python compare.py --tol 0.0015    # tolerance for "match" (default: half a unit in the last printed digit)
"""

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import paper_values as pv  # noqa: E402

MATURITIES = [24, 36, 48, 60, 84, 120]


def load(run_name, cache={}):
    if run_name not in cache:
        path = ROOT / "output" / run_name / "results.pkl"
        cache[run_name] = pickle.load(open(path, "rb")) if path.exists() else None
    return cache[run_name]


def run_folder(run_name, method, nn_tag):
    """Neural networks are stored per method under output/nn_<tag>/ (see run_nn.py)."""
    if run_name == "nn":
        return f"nn_{nn_tag}/" + "".join(c if c.isalnum() or c in "-_" else "_" for c in method)
    return run_name


def reproduced(kind, outp, method):
    if kind == "R2":
        row = outp["R2"].loc[method]
    else:
        metric = "CER" if kind == "CER" else "Sharpe Ratio"
        row = outp["Utilities"].loc[(metric, method, "stat")]
        if kind == "CER":
            row = row * 100
    return [float(row[m]) for m in MATURITIES if m in row.index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--nn-tag", default="n20", help="which neural-network run to compare (output/nn_<tag>/)")
    args = parser.parse_args()

    rows = []
    for kind, tables, default_tol in [("R2", pv.R2, 0.0015), ("CER", pv.CER, 0.015), ("SR", pv.SHARPE, 0.0015)]:
        tol = args.tol if args.tol is not None else default_tol
        for table, entries in tables.items():
            for (run_name, method), paper in entries.items():
                run_name = run_folder(run_name, method, args.nn_tag)
                outp = load(run_name)
                if outp is None:
                    continue
                try:
                    ours = reproduced(kind, outp, method)[-len(paper):]
                except KeyError:
                    rows.append([table, kind, run_name, method, "missing", None, ""])
                    continue
                diff = max(abs(a - b) for a, b in zip(ours, paper))
                rows.append([table, kind, run_name, method, "match" if diff <= tol else "DIFF", round(diff, 4),
                             " ".join(f"{x:.3f}" for x in ours)])

    df = pd.DataFrame(rows, columns=["table", "stat", "run", "method", "result", "max abs diff", "reproduced"])
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 60, "display.width", 250):
        print(df.to_string(index=False))
    print(f"\n{(df['result'] == 'match').sum()} of {len(df)} rows match the paper.")


if __name__ == "__main__":
    main()
