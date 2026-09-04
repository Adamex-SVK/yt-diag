#!/usr/bin/env python3
"""Audit whether seeded retrospective 'test' folds remained globally unseen."""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ytdiag.adapters import load_retrospective  # noqa: E402
from ytdiag.split import split_indices  # noqa: E402


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.path.join(ROOT, "02_Data", "processed"))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--output", default=os.path.join(
        ROOT, "05_Reports", "final_report", "results", "split_exposure.json",
    ))
    args = parser.parse_args()
    df = load_retrospective(args.data)
    df = df[df.label.notna()].reset_index(drop=True)
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    splits = {seed: split_indices(df, seed=seed) for seed in seeds}
    rows = []
    globally_unseen = set(range(len(df)))
    for split in splits.values():
        globally_unseen -= set(split["train"]) | set(split["val"])
    for seed, split in splits.items():
        other_train = set().union(*(
            set(other["train"]) for other_seed, other in splits.items() if other_seed != seed
        ))
        other_val = set().union(*(
            set(other["val"]) for other_seed, other in splits.items() if other_seed != seed
        ))
        nominal_test = set(split["test"])
        rows.append({
            "seed": seed,
            "nominal_test_rows": len(nominal_test),
            "seen_in_other_training": len(nominal_test & other_train),
            "seen_in_other_validation": len(nominal_test & other_val),
            "seen_in_other_development": len(nominal_test & (other_train | other_val)),
            "development_exposure_fraction": len(nominal_test & (other_train | other_val)) / len(nominal_test),
        })
    payload = {
        "dataset": "retrospective",
        "n_labelled": len(df),
        "seeds": seeds,
        "per_seed": rows,
        "rows_never_used_for_training_or_validation": len(globally_unseen),
        "conclusion": "no globally unseen retrospective test set exists",
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
