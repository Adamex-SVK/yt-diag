"""Metadata baselines (evaluation_and_planning.md): dummy floor, logistic
regression, gradient boosting (XGBoost; sklearn HistGradientBoosting if
xgboost is unavailable) on any selection of feature groups.

Protocol: channel-grouped 60/20/20 split; models fit on train, threshold
picked on val (max F1), reported on val; the TEST split is evaluated only
when `evaluate_test=True` -- touch it once, at the end (MILESTONES.md).
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import select_columns
from .split import split_indices

CATEGORICAL = ("meta__category",)
MIN_LABELED_ROWS = 50  # below this a 60/20/20 grouped split is meaningless


def _preprocessor(cols, scale):
    cat = [c for c in cols if c in CATEGORICAL]
    num = [c for c in cols if c not in CATEGORICAL]
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])


def _boosting():
    try:
        from xgboost import XGBClassifier
        return "xgboost", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                                        random_state=0, n_jobs=4)
    except Exception:  # missing wheel / libomp
        from sklearn.ensemble import HistGradientBoostingClassifier
        return "hist_gradient_boosting", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05,
                                                                        max_iter=300, random_state=0)


def _metrics(y, p, threshold):
    return {"auc_roc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "f1": float(f1_score(y, (p >= threshold).astype(int))), "n": int(len(y)),
            "positive_rate": float(np.mean(y))}


def _best_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 91)
    return float(max(grid, key=lambda t: f1_score(y, (p >= t).astype(int))))


def run_baselines(df, groups, out_dir=None, seed=0, evaluate_test=False):
    total = len(df)
    df = df[df.label.notna()].reset_index(drop=True)
    if len(df) < MIN_LABELED_ROWS:
        raise ValueError(
            f"only {len(df)} of {total} rows have a label (need >= {MIN_LABELED_ROWS}). "
            "Retrospective data: run compute_labels_v2.py first. Prospective data: outcomes are "
            "interpolated at the horizon, so videos younger than --horizon-days have no label yet "
            "(the cohort started 2026-08-26 -> first 7-day labels on 2026-09-02).")
    cols = select_columns(df, groups)
    if not cols:
        raise ValueError(f"no input columns for groups {groups}")
    idx = split_indices(df, seed=seed)
    X, y = df[cols], df.label.astype(int).to_numpy()
    boost_name, boost = _boosting()
    models = {
        "dummy_prior": Pipeline([("prep", _preprocessor(cols, scale=False)),
                                 ("clf", DummyClassifier(strategy="prior"))]),
        "logistic_regression": Pipeline([("prep", _preprocessor(cols, scale=True)),
                                         ("clf", LogisticRegression(max_iter=2000, C=1.0))]),
        boost_name: Pipeline([("prep", _preprocessor(cols, scale=False)), ("clf", boost)]),
    }
    results = {"groups": list(groups), "n_features": len(cols), "seed": seed,
               "split_sizes": {k: int(len(v)) for k, v in idx.items()}, "models": {}}
    for name, model in models.items():
        model.fit(X.iloc[idx["train"]], y[idx["train"]])
        p_val = model.predict_proba(X.iloc[idx["val"]])[:, 1]
        thr = _best_threshold(y[idx["val"]], p_val)
        res = {"val": _metrics(y[idx["val"]], p_val, thr), "threshold": thr}
        cats = df.meta__category.iloc[idx["val"]].to_numpy()
        res["val_auc_by_category"] = {
            c: (float(roc_auc_score(y[idx["val"]][cats == c], p_val[cats == c]))
                if len(set(y[idx["val"]][cats == c])) == 2 else None)
            for c in sorted(set(cats))}
        if evaluate_test:
            p_test = model.predict_proba(X.iloc[idx["test"]])[:, 1]
            res["test"] = _metrics(y[idx["test"]], p_test, thr)
        results["models"][name] = res
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    return results


def format_results(results):
    lines = [f"groups={results['groups']}  features={results['n_features']}  "
             f"split={results['split_sizes']}"]
    for name, r in results["models"].items():
        v = r["val"]
        line = f"  {name:24s} val AUC {v['auc_roc']:.3f}  PR-AUC {v['pr_auc']:.3f}  F1@{r['threshold']:.2f} {v['f1']:.3f}"
        if "test" in r:
            line += f"  |  TEST AUC {r['test']['auc_roc']:.3f}"
        lines.append(line)
    return "\n".join(lines)
