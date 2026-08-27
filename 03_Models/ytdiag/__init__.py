"""YT-Diag modelling package (03_Models).

One pipeline, many feature configurations. Both datasets -- Adam's
retrospective per-video directory tree (02_Data/processed/) and Emmanuel's
prospective tracker CSVs (02_Data/tracking/) -- are loaded by adapters
(adapters.py) onto ONE canonical table whose columns are prefixed by feature
GROUP (features.py). Every model, baseline, and ablation is then just a
choice of groups, never a second implementation.
"""
