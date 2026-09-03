"""Structural checks for the reproducible descriptive EDA outputs.

These checks use the checked-in retrospective dataset but do not modify it and
do not render figures. They pin the split boundary, table shapes and matrix
properties without freezing incidental correlation values.

    .venv/bin/python 02_Data/tests/test_eda_features.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

DATA_MODULE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DATA_MODULE)
import eda_features as ef  # noqa: E402


def test_dashboard_artifacts() -> None:
    data = ef.load()
    assert "label" not in data.columns

    train = ef.training_partition(data, seed=0)
    assert 0 < len(train) < len(data)
    assert train["channel_id"].nunique() < data["channel_id"].nunique()

    correlation, pair_counts = ef.correlation_tables(train)
    expected = list(ef.DASHBOARD_FEATURES)
    assert correlation.index.tolist() == expected
    assert correlation.columns.tolist() == expected
    assert pair_counts.index.tolist() == expected
    assert pair_counts.columns.tolist() == expected
    assert np.allclose(correlation, correlation.T, equal_nan=True)
    assert (np.diag(pair_counts) > 0).all()

    audio = ef.audio_correlation_table(data)
    assert audio.shape == (88, 88)
    assert np.allclose(audio, audio.T, equal_nan=True)

    coverage = ef.modality_coverage_table(data)
    coverage_columns = [column for column in coverage if column.endswith("_pct")]
    assert len(coverage) == 10  # two overall rows + four categories x two formats
    assert coverage[coverage_columns].notna().all().all()
    assert coverage[coverage_columns].ge(0).all().all()
    assert coverage[coverage_columns].le(100).all().all()

    pivot = ef.category_format_pivot(data)
    assert len(pivot) == 4
    assert set(pivot["category"]) == set(data["meta__category"].dropna().unique())
    assert {"n__regular", "n__shorts"}.issubset(pivot.columns)

    associations = ef.outcome_association_table(train)
    assert "log1p views" not in set(associations["feature"])
    assert {
        "rho_overall", "rho_regular", "rho_shorts",
        "format_sign_reversal", "pooled_vs_format_reversal",
    }.issubset(associations.columns)


if __name__ == "__main__":
    test_dashboard_artifacts()
    print("EDA feature tests passed")
