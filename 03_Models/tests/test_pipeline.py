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
    assert len(select_columns(df, ("meta",))) == 13  # incl. language, both caption flags, channel age/count/first-upload
    assert df.meta__channel_age_days.gt(0).all() and set(df.meta__is_first_upload.unique()) <= {0, 1}
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
    assert res_meta["n_features"] == 13


def _synthetic_tracking_dir():
    """A tiny tracking/ dir with the 2026-08-27 schema and REAL values, to pin
    the prospective adapter's parsing and policies (the live-dir test only
    smoke-checks shape)."""
    import csv
    import json
    d = tempfile.mkdtemp(prefix="ytdiag_tracking_")
    os.makedirs(os.path.join(d, "texts"))
    cohort_fields = ["video_id", "category", "channel_id", "published_at_utc", "discovered_at_utc", "discovery_source",
                     "sampling_arm", "window_start_utc", "window_end_utc", "search_rank", "duration_sec",
                     "definition", "caption_available", "default_language", "default_audio_language"]
    pub = "2026-08-20T10:00:00Z"
    rows = []
    N = 24  # >= 8 main-arm videos per category, so the min-label-group rule leaves labels in place
    for i in range(N):
        arm = "non_english" if i == N - 1 else "date_window"
        rows.append(dict(video_id=f"v{i:02d}", category="comedy" if i % 2 else "howto", channel_id=f"ch{i}",
                         published_at_utc=pub, discovered_at_utc=pub, discovery_source="search:23:comedy:medium",
                         sampling_arm=arm, window_start_utc="", window_end_utc="", search_rank=i + 1, duration_sec=600,
                         definition="hd" if i % 3 else "sd", caption_available=("" if i == 10 else ("true" if i % 2 else "false")),
                         default_language="", default_audio_language="hi" if i == N - 1 else ("en-US" if i % 2 else "en")))
    with open(os.path.join(d, "cohort.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cohort_fields); w.writeheader(); w.writerows(rows)
    # two observations per video: day 1 and day 8 (brackets the 7-day horizon); title edited on day 8
    snap_fields = ["video_id", "observed_at_utc", "age_hours", "view_count", "like_count", "comment_count", "status",
                   "youtube_category_id", "title", "description_length", "tag_count"]
    snaps = []
    for i in range(N):
        # v09's first snapshot predates the description_length/tag_count (and title) columns -> fallback to first text
        snaps.append(dict(video_id=f"v{i:02d}", observed_at_utc="2026-08-21T10:00:00Z", age_hours=24.0, view_count=100 * (i + 1),
                          like_count=1, comment_count=0, status="ok", youtube_category_id="23",
                          title="" if i == 9 else "first title",
                          description_length="" if i == 9 else 50, tag_count="" if i == 9 else 4))
        snaps.append(dict(video_id=f"v{i:02d}", observed_at_utc="2026-08-28T10:00:00Z", age_hours=192.0, view_count=1000 * (i + 1),
                          like_count=1, comment_count=0, status="ok", youtube_category_id="23", title="EDITED much longer title",
                          description_length=999, tag_count=9))
    with open(os.path.join(d, "video_snapshots.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=snap_fields); w.writeheader(); w.writerows(snaps)
    chan_fields = ["channel_id", "observed_at_utc", "subscriber_count", "hidden_subscriber_count", "channel_video_count",
                   "country", "channel_created_at"]
    with open(os.path.join(d, "channel_snapshots.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=chan_fields); w.writeheader()
        for i in range(N):
            # ch0 has an EARLIER row (from a previous video) with a different count: the video's
            # channel features must come from the row at/after the video's first observation
            if i == 0:
                w.writerow(dict(channel_id="ch0", observed_at_utc="2026-08-10T10:00:00Z", subscriber_count=10,
                                hidden_subscriber_count="false", channel_video_count=0, country="US", channel_created_at=""))
            w.writerow(dict(channel_id=f"ch{i}", observed_at_utc="2026-08-21T10:00:01Z", subscriber_count=5000,
                            hidden_subscriber_count="false", channel_video_count=1 if i == 0 else 30, country="US",
                            channel_created_at="2026-08-01T00:00:00Z"))
    with open(os.path.join(d, "thumbnail_snapshots.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "observed_at_utc", "sha256", "quality", "file", "changed"]); w.writeheader()
        for i in range(N):
            w.writerow(dict(video_id=f"v{i:02d}", observed_at_utc="2026-08-21T10:00:00Z", sha256="a", quality="maxres", file=f"v{i:02d}_v0.jpg", changed="true"))
            w.writerow(dict(video_id=f"v{i:02d}", observed_at_utc="2026-08-28T10:00:00Z", sha256="b", quality="maxres", file=f"v{i:02d}_v1.jpg", changed="true"))
    with open(os.path.join(d, "text_snapshots.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "observed_at_utc", "sha256", "file", "changed"]); w.writeheader()
        for i in range(N):
            for n, (obs, desc) in enumerate((("2026-08-21T10:00:00Z", "original description"),
                                             ("2026-08-28T10:00:00Z", "edited description"))):
                fname = f"v{i:02d}_v{n}.json"
                json.dump({"title": "t", "description": desc, "tags": []}, open(os.path.join(d, "texts", fname), "w"))
                w.writerow(dict(video_id=f"v{i:02d}", observed_at_utc=obs, sha256=f"h{n}", file=fname, changed="true"))
    return d


def test_prospective_adapter_parsing_and_policies():
    df = load_prospective(_synthetic_tracking_dir(), horizon_days=7)
    assert len(df) == 24
    # flags survive pandas' bool parsing; definition/language mapped and normalized
    assert set(df.meta__caption_available.dropna().unique()) == {0, 1} and df.meta__caption_available.isna().sum() == 1  # '' -> NaN, strings parsed
    assert set(df.meta__definition_hd.unique()) == {0, 1}
    assert set(df.meta__language.unique()) == {"en", "hi"}  # 'en-US' -> 'en'
    # upload-time policy: FIRST observation, never the edited one
    v9 = df[df.video_id == "v09"].iloc[0]  # old-schema first snapshot -> falls back to the FIRST text version
    assert v9.meta__title_length == len("t") and v9.meta__description_length == len("original description") and v9.meta__tag_count == 0
    others = df[df.video_id != "v09"]
    assert (others.meta__title_length == len("first title")).all()
    assert (others.meta__description_length == 50).all() and (others.meta__tag_count == 4).all()
    assert (df.asset__description == "original description").all()
    assert (df.track__text_changes == 1).all()
    assert df.asset__thumbnail_path.str.endswith("_v0.jpg").all() and (df.track__thumbnail_changes == 1).all()
    # outcome interpolated at 168h between the 24h and 192h observations
    v0 = df[df.video_id == "v00"].iloc[0]
    assert abs(v0.outcome_views - (100 + (1000 - 100) * (168 - 24) / (192 - 24))) < 1e-6
    # label only within the main arm; comparison arm gets none
    assert df[df.sampling_arm == "non_english"].label.isna().all()
    main = df[df.sampling_arm == "date_window"]
    assert main.label.notna().all() and 0 < main.label.sum() < len(main)
    # channel maturity
    assert (df.meta__channel_age_days > 0).all()
    assert df[df.video_id == "v00"].meta__is_first_upload.iloc[0] == 1 and df[df.video_id == "v01"].meta__is_first_upload.iloc[0] == 0
    assert df[df.video_id == "v00"].meta__channel_follower_count.iloc[0] == 5000  # row at/after first observation, not the earlier one


def test_prospective_adapter_on_live_tracking_dir_if_present():
    tracking = os.path.join(os.path.dirname(os.path.dirname(HERE)), "02_Data", "tracking")
    if not os.path.exists(os.path.join(tracking, "cohort.csv")):
        return  # not on the tracking machine
    df = load_prospective(tracking, horizon_days=7)
    assert len(df) > 0 and "sched" in available_groups(df) and "meta" in available_groups(df)
    assert df.asset__thumbnail_path.notna().mean() > 0.9
    assert (df.track__n_snapshots >= 1).all()
    # labels exist only for main-arm videos past the horizon (none at the time of writing); must not crash
    main_reached = df.outcome_views.notna() & (df.sampling_arm == "date_window")
    assert df[~main_reached].label.isna().all()
    assert main_reached.sum() < 8 or df.label.notna().sum() > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"{name} OK")
    print("ALL PIPELINE TESTS PASSED")
