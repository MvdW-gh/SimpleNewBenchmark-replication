"""
Run pc_regression for a set of methods and store the results.

Defaults are the paper's baseline settings. Results are written to output/<name>/: an Excel file with all tables, the figures as PNG,
a pickle with all non-figure outputs, and run_info.json with the settings and runtime.

Examples:
    python run.py --name linear_baseline --methods "Yields" "Yield changes"
    python run.py --name nn_test --methods "NN-3-1-Yield changes" --end-date 1992-12 --num-nns 2 --num-best 1
"""

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

# One thread per process: parallelism comes from running several processes (joblib), and letting each
# process also use all cores for TensorFlow / numpy only oversubscribes the machine.
for _var in ["TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(_var, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import snb  # noqa: E402

DEFAULTS = dict(holding_period=12, lookback=12, window="expanding", T_in=221, begin_date="1971-08",
                end_date="2018-12", no_macro=False, num_simulations=5000, num_nns=20, num_best=10,
                n_jobs=-1, seed=0)


def stored_results(outp):
    """Outputs kept in results.pkl: everything except figures and the simulated yield curves of the bootstrap
    (several hundred MB, not used for any table in the paper)."""
    return {k: v for k, v in outp.items()
            if not isinstance(v, plt.Figure) and not (k.startswith("simulated") or k.endswith("simulated t-vals"))}


def run(name, methods, **settings):
    """Run pc_regression for the given methods and store all results in output/<name>/."""
    s = {**DEFAULTS, **settings}
    out_dir = ROOT / "output" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    yields, macro, cpi = snb.load_data()

    start = time.time()
    outp = snb.pc_regression(
        yields=yields, macro=None if s["no_macro"] else macro, cpi=cpi, recession=None,
        window=s["window"], T_in=s["T_in"], holding_period=s["holding_period"], lookback=s["lookback"],
        cp_params=[0.987], methods_provided=list(methods),
        begin_date=s["begin_date"], end_date=s["end_date"],
        num_simulations=s["num_simulations"],
        ann_num_nns=s["num_nns"], ann_num_best=s["num_best"], n_jobs=s["n_jobs"],
        seed=s["seed"],
    )
    runtime = time.time() - start

    snb.write_to_excel(outp, out_dir / "results.xlsx")
    snb.save_figures(outp, out_dir / "figures")
    with open(out_dir / "results.pkl", "wb") as f:
        pickle.dump(stored_results(outp), f)
    with open(out_dir / "run_info.json", "w") as f:
        json.dump({"name": name, "methods": list(methods), **s, "runtime_seconds": round(runtime, 1),
                   "finished": datetime.now().isoformat(timespec="seconds")}, f, indent=2)
    plt.close("all")

    print(f"\n{name}: finished in {runtime / 60:.1f} minutes. Results in {out_dir}")
    return outp


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, help="name of the output folder under output/")
    parser.add_argument("--methods", nargs="+", required=True, help="method labels as used in snb.pc_regression")
    parser.add_argument("--holding-period", type=int, default=DEFAULTS["holding_period"])
    parser.add_argument("--lookback", type=int, default=DEFAULTS["lookback"])
    parser.add_argument("--window", choices=["expanding", "rolling"], default=DEFAULTS["window"])
    parser.add_argument("--T-in", type=int, default=DEFAULTS["T_in"])
    parser.add_argument("--begin-date", default=DEFAULTS["begin_date"])
    parser.add_argument("--end-date", default=DEFAULTS["end_date"])
    parser.add_argument("--no-macro", action="store_true", help="pass macro=None (as some original runs did)")
    parser.add_argument("--num-simulations", type=int, default=DEFAULTS["num_simulations"])
    parser.add_argument("--num-nns", type=int, default=DEFAULTS["num_nns"], help="networks fitted per forecast")
    parser.add_argument("--num-best", type=int, default=DEFAULTS["num_best"], help="networks averaged per forecast")
    parser.add_argument("--n-jobs", type=int, default=DEFAULTS["n_jobs"], help="parallel processes for neural networks")
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    args = vars(parser.parse_args())

    outp = run(args.pop("name"), args.pop("methods"), **args)
    print("\nR2 (out of sample):")
    print(outp["R2"].astype(float).round(3))


if __name__ == "__main__":
    main()
