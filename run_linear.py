"""
Run all linear-model analyses of the paper (everything except the neural networks).

Each entry in RUNS is one call of run.run(); results go to output/<name>/. The runs are independent and are
executed in parallel (default: 6 at a time).

    python run_linear.py              # all runs
    python run_linear.py --only baseline_main rolling
    python run_linear.py --parallel 4
"""

import argparse
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import run

YIELD_METHODS = ["Yields", "Yield changes", "Forwards", "Forward changes"]
MACRO_METHODS = [f"{m} + macro (revised)" for m in YIELD_METHODS]

RUNS = {
    # Tables 1-3, Figures 2-5, Tables B.1-B.2
    "baseline_main": dict(methods=["Yields", "Yields without PC1", "Yield changes", "Yields + yield changes",
                                   "Forwards", "Forward changes", "Cochrane Piazessi"]),
    # Diebold-Li row of Tables 2-3 (a linear VAR forecast, implemented inside the neural-network routine)
    "baseline_diebold_li": dict(methods=["NN-3-1-Diebold Li direct"]),
    # Tables 6-7 (regression panel), B.3, H.1
    "baseline_macro": dict(methods=MACRO_METHODS + ["Cieslak Povala (param 0.987)"]),
    # Footnote 14: macro data excluding the interest-rate series ("ex yields")
    "baseline_macro_ex_yields": dict(methods=[f"{m} + macro (ex yields, revised)" for m in YIELD_METHODS]),
    # Table A.1
    "hanson_12m": dict(methods=["Hanson Lucca Wright"]),
    "hanson_6m": dict(methods=["Hanson Lucca Wright"], holding_period=6, lookback=6),
    # Table D.1
    "nonoverlapping": dict(methods=[f"{m} (nonoverlapping)" for m in YIELD_METHODS]),
    # Table E.1
    "rolling": dict(methods=YIELD_METHODS + MACRO_METHODS, window="rolling"),
}
# Table F.1: holding periods (lookback 12); 12 is the baseline
for hp in [1, 3, 6, 24]:
    RUNS[f"holding_{hp}"] = dict(methods=YIELD_METHODS + MACRO_METHODS, holding_period=hp)
# Table F.2: lookback periods (holding period 12)
for lb in [1, 3, 6, 24]:
    RUNS[f"lookback_{lb}"] = dict(methods=["Yield changes", "Forward changes",
                                           "Yield changes + macro (revised)", "Forward changes + macro (revised)"],
                                  lookback=lb)


def _run(name):
    settings = dict(RUNS[name])
    methods = settings.pop("methods")
    start = time.time()
    try:
        run.run(name, methods, **settings)
    except Exception:
        return name, time.time() - start, traceback.format_exc()
    return name, time.time() - start, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=list(RUNS), help="run only these")
    parser.add_argument("--parallel", type=int, default=6, help="number of runs executed at the same time")
    args = parser.parse_args()

    names = args.only or list(RUNS)
    start = time.time()
    failed = []
    with ProcessPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(_run, name): name for name in names}
        for fut in as_completed(futures):
            name, seconds, error = fut.result()
            if error:
                failed.append(name)
                print(f"[FAILED] {name} ({seconds / 60:.1f} min)", error, sep="\n", flush=True)
            else:
                print(f"[done] {name} ({seconds / 60:.1f} min)", flush=True)
    print(f"{len(names) - len(failed)} of {len(names)} runs finished in {(time.time() - start) / 60:.1f} minutes."
          + (f" Failed: {', '.join(failed)}" if failed else ""))


if __name__ == "__main__":
    main()
