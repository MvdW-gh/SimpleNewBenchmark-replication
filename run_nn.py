"""
Run the neural-network analyses of the paper (Tables 4-7).

Each network specification is run separately and stored in output/nn_<tag>/<method>/, so that finished
specifications are kept if a long run is interrupted; already finished specifications are skipped.
The networks are fitted in parallel over forecast dates (n_jobs processes).

    python run_nn.py --tag n20                                   # all specifications, 20 networks (as in the paper)
    python run_nn.py --tag n100 --num-nns 100                    # 100 networks, as in Bianchi et al. (2021)
    python run_nn.py --tag test --methods "NN-3-1-Yield changes"
"""

import argparse
import sys
import time

import run

NN_METHODS = [
    # Table 4 / 5
    "NN-3-1-Yields", "NN-3-1-Yield changes", "NN-3-1-Forwards", "NN-3-1-Forward changes",
    "NN-3-1-Yields PC", "NN-3-1-Yield changes PC", "NN-3-1-Cochrane Piazessi", "NN-3-1-Diebold Li",
    # Table 6 / 7
    "NN-32-1-Yields + macro (revised)", "NN-32-1-Forwards + macro (revised)",
    "NN-32-1-Yield changes + macro (revised)", "NN-32-1-Forward changes + macro (revised)",
    "NN-3-1-Cieslak Povala (param 0.987)",
]


def keep_awake(on):
    """On Windows, ask the system not to go to sleep while the run is in progress (screen may still turn off)."""
    if sys.platform == "win32":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def folder_name(method):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in method)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="output goes to output/nn_<tag>/")
    parser.add_argument("--methods", nargs="+", default=NN_METHODS, choices=NN_METHODS)
    parser.add_argument("--num-nns", type=int, default=20, help="networks fitted per forecast (paper: 20)")
    parser.add_argument("--num-best", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--end-date", default="2018-12", help="shorter sample for testing")
    args = parser.parse_args()

    keep_awake(True)
    start = time.time()
    try:
        for i, method in enumerate(args.methods, 1):
            name = f"nn_{args.tag}/{folder_name(method)}"
            if (run.ROOT / "output" / name / "run_info.json").exists():
                print(f"[{i}/{len(args.methods)}] skipping {method} (already done)", flush=True)
                continue
            print(f"[{i}/{len(args.methods)}] starting {method} at {time.strftime('%H:%M')}", flush=True)
            run.run(name, [method], num_nns=args.num_nns, num_best=args.num_best, n_jobs=args.n_jobs,
                    end_date=args.end_date)
    finally:
        keep_awake(False)
    print(f"All done in {(time.time() - start) / 3600:.1f} hours.", flush=True)


if __name__ == "__main__":
    main()
