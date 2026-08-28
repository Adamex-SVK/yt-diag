"""End-to-end pipeline checks on synthetic data (no network, no real data).
Run:  .venv/bin/python -m pytest 03_Models/tests -q   (or plain python)."""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from ytdiag.adapters import load_prospective, load_retrospective  # noqa: E402
from ytdiag.baselines import run_baselines  # noqa: E402
from ytdiag.features import LABEL_ONLY, available_groups, select_columns  # noqa: E402
from ytdiag.split import split_indices  # noqa: E402
from ytdiag.synthetic import generate  # noqa: E402


def _synthetic_df(n=1200):
    return load_retrospective(generate(tempfile.mkdtemp(prefix="ytdiag_test_"), n=n, seed=1))


def test_retrospective_adapter_and_registry():
    df = _synthetic_df()
    assert len(df) == 1200 and df.label.notna().all()
    assert abs(df.label.mean() - 0.25) < 0.02  # top quartile per category
    assert available_groups(df) == ["meta", "sched", "vis", "aud", "asset"]
    assert len(select_columns(df, ("meta",))) == 8
    assert sum(c.startswith("aud__egemaps__") for c in select_columns(df, ("aud",))) == 88
    assert not any(c in LABEL_ONLY for c in select_columns(df, ("meta", "sched", "vis", "aud")))
    assert "asset__thumbnail_path" not in select_columns(df, ("asset",))  # assets are never tabular inputs
    try:
        select_columns(df, ("bogus",)); raise AssertionError("unknown group accepted")
    except ValueError:
        pass


def test_split_is_channel_grouped_and_sized():
    df = _synthetic_df()
    idx = split_indices(df, seed=3)
    n = len(df)
    assert 0.5 < len(idx["train"]) / n < 0.7 and 0.12 < len(idx["val"]) / n < 0.28
    ch = df.channel_id.to_numpy()
    assert not (set(ch[idx["train"]]) & set(ch[idx["val"]]) | set(ch[idx["train"]]) & set(ch[idx["test"]]))
    assert sorted(np.concatenate(list(idx.values()))) == list(range(n))  # partition


def test_baselines_recover_planted_signal():
    df = _synthetic_df()
    res = run_baselines(df, ("meta", "sched", "vis"), seed=0)
    val = {k: v["val"]["auc_roc"] for k, v in res["models"].items()}
    assert abs(val["dummy_prior"] - 0.5) < 1e-9
    assert val["logistic_regression"] > 0.62, val  # planted signal is recoverable
    assert "test" not in res["models"]["logistic_regression"]  # test untouched by default
    res_meta = run_baselines(df, ("meta",), seed=0)
    assert res_meta["n_features"] == 8


def test_prospective_adapter_on_live_tracking_dir_if_present():
    tracking = os.path.join(os.path.dirname(os.path.dirname(HERE)), "02_Data", "tracking")
    if not os.path.exists(os.path.join(tracking, "cohort.csv")):
        return  # not on the tracking machine
    df = load_prospective(tracking, horizon_days=7)
    assert len(df) > 0 and "sched" in available_groups(df) and "meta" in available_groups(df)
    assert df.asset__thumbnail_path.notna().mean() > 0.9
    assert (df.track__n_snapshots >= 1).all()
    # no video has reached day 7 yet at the time of writing -> outcomes NaN, labels NaN; must not crash
    assert df.outcome_views.isna().all() or df.label.notna().sum() > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"{name} OK")
    print("ALL PIPELINE TESTS PASSED")
