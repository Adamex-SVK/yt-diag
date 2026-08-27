"""Channel-grouped, label-stratified train/val/test split (60/20/20).

No channel may appear in more than one split (evaluation_and_planning.md;
FEATURES.md: `channel_id` drives the split). Implemented with
StratifiedGroupKFold(5): folds 0-2 train, 3 val, 4 test -- deterministic
for a given seed, stratified on the label as far as grouping allows.
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def split_indices(df, label_col="label", group_col="channel_id", seed=0):
    y = df[label_col].to_numpy()
    groups = df[group_col].to_numpy()
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = [test_idx for _, test_idx in skf.split(np.zeros(len(df)), y, groups)]
    train = np.concatenate(folds[:3])
    val, test = folds[3], folds[4]
    assert not (set(groups[train]) & set(groups[val])) and not (set(groups[train]) & set(groups[test])) \
        and not (set(groups[val]) & set(groups[test])), "channel leaked across splits"
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}
