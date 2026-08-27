"""CLI: metadata baselines on either dataset with any feature-group selection.

    .venv/bin/python 03_Models/run_baselines.py --source retrospective --data 02_Data/processed --groups meta
    .venv/bin/python 03_Models/run_baselines.py --source retrospective --data 02_Data/processed --groups meta,sched,vis,aud
    .venv/bin/python 03_Models/run_baselines.py --source prospective --data 02_Data/tracking --horizon-days 7 --groups meta,sched
    ... --synthetic 1200      # generate a fake retrospective tree and run on it (pipeline check)

Results go to 04_Experiments/runs/<name>/results.json (gitignored). --test
evaluates the held-out test split: do that ONCE, at the end.
"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ytdiag.adapters import load_prospective, load_retrospective  # noqa: E402
from ytdiag.baselines import format_results, run_baselines  # noqa: E402
from ytdiag.features import available_groups  # noqa: E402
from ytdiag.synthetic import generate  # noqa: E402

RUNS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "04_Experiments", "runs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["retrospective", "prospective"], default="retrospective")
    ap.add_argument("--data", help="processed/ dir (retrospective) or tracking/ dir (prospective)")
    ap.add_argument("--synthetic", type=int, metavar="N", help="generate N synthetic retrospective videos instead of --data")
    ap.add_argument("--groups", default="meta", help="comma-separated feature groups, e.g. meta,sched,vis,aud")
    ap.add_argument("--horizon-days", type=int, default=7)
    ap.add_argument("--name", help="run name under 04_Experiments/runs/")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test", action="store_true", help="also evaluate the held-out test split (once!)")
    a = ap.parse_args()

    if a.synthetic:
        data = generate(tempfile.mkdtemp(prefix="ytdiag_synth_"), n=a.synthetic, seed=a.seed)
        df = load_retrospective(data)
    elif a.source == "retrospective":
        df = load_retrospective(a.data)
    else:
        df = load_prospective(a.data, horizon_days=a.horizon_days)
    print(f"{len(df)} rows, {int(df.label.notna().sum())} labeled; groups available: {available_groups(df)}")
    groups = tuple(g.strip() for g in a.groups.split(","))
    name = a.name or f"{'synthetic' if a.synthetic else a.source}_{'-'.join(groups)}"
    results = run_baselines(df, groups, out_dir=os.path.join(RUNS_DIR, name), seed=a.seed, evaluate_test=a.test)
    print(format_results(results))
    print(f"-> {os.path.join(RUNS_DIR, name, 'results.json')}")


if __name__ == "__main__":
    main()
