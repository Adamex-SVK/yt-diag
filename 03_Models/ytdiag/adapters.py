"""Dataset adapters: each maps one raw source onto the canonical table
(features.py naming). Missing groups are simply absent/NaN -- the model
config decides what to use, the adapter never pretends.

  load_retrospective(processed_dir)  Adam's per-video directory tree
      metadata.json, visual_features.json, audio_features.json, label.json
      (compute_labels_v2), metadata_extra.json (backfill_published_at.py),
      transcript.txt, thumbnail.jpg, frames/
  load_prospective(tracking_dir, horizon_days)  Emmanuel's tracker CSVs
      cohort.csv + video_snapshots.csv + channel_snapshots.csv +
      thumbnail_snapshots.csv; outcome = views interpolated at the horizon
      from the bracketing observations; label = within-category top
      quartile of log(1+outcome) among videos that have reached the horizon
      (a stand-in for the v2 stratified label -- see FEATURES.md 6)
"""
import csv
import datetime
import json
import math
import os

import numpy as np
import pandas as pd

from .features import VIS_FRAMES, VIS_THUMB, AUD_PAUSES


def _schedule_features(published_at_utc):
    """sched__* from a full UTC timestamp (FEATURES.md 6). NaN if unknown."""
    if not published_at_utc:
        return {"sched__hour_sin": np.nan, "sched__hour_cos": np.nan,
                "sched__weekday": np.nan, "sched__is_weekend": np.nan}
    dt = datetime.datetime.strptime(published_at_utc[:19], "%Y-%m-%dT%H:%M:%S")
    h = dt.hour + dt.minute / 60.0
    return {"sched__hour_sin": math.sin(2 * math.pi * h / 24), "sched__hour_cos": math.cos(2 * math.pi * h / 24),
            "sched__weekday": dt.weekday(), "sched__is_weekend": int(dt.weekday() >= 5)}


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_retrospective(processed_dir):
    rows = []
    for category in sorted(os.listdir(processed_dir)):
        cat_dir = os.path.join(processed_dir, category)
        if not os.path.isdir(cat_dir):
            continue
        for vid in sorted(os.listdir(cat_dir)):
            d = os.path.join(cat_dir, vid)
            if not os.path.exists(os.path.join(d, ".done")):
                continue
            meta = _read_json(os.path.join(d, "metadata.json"))
            if not meta:
                continue
            label = _read_json(os.path.join(d, "label.json")) or {}
            extra = _read_json(os.path.join(d, "metadata_extra.json")) or {}
            vis = _read_json(os.path.join(d, "visual_features.json")) or {}
            aud = _read_json(os.path.join(d, "audio_features.json")) or {}
            tinfo = _read_json(os.path.join(d, "transcript_info.json")) or {}
            row = {
                "dataset": "retrospective", "video_id": vid, "channel_id": meta.get("channel_id"),
                "label": {"viral": 1, "typical": 0}.get(label.get("label"), np.nan),
                "view_count": meta.get("view_count"),
                "meta__duration_sec": meta.get("duration"),
                "meta__definition_hd": {"hd": 1, "sd": 0}.get(meta.get("definition"), np.nan),
                "meta__title_length": meta.get("title_length"),
                "meta__description_length": meta.get("description_length"),
                "meta__tag_count": meta.get("tag_count"),
                "meta__channel_follower_count": meta.get("channel_follower_count"),
                "meta__caption_available": (int(extra["caption_available"]) if "caption_available" in extra
                                            else (int(tinfo["had_auto_captions"]) if "had_auto_captions" in tinfo else np.nan)),
                "meta__category": category,
                **_schedule_features(extra.get("published_at_utc")),
                "asset__thumbnail_path": next((os.path.join(d, f) for f in ("thumbnail.jpg", "thumbnail.webp", "thumbnail.png")
                                               if os.path.exists(os.path.join(d, f))), None),
                "asset__frames_dir": os.path.join(d, "frames") if os.path.isdir(os.path.join(d, "frames")) else None,
                "asset__transcript_path": os.path.join(d, "transcript.txt") if os.path.exists(os.path.join(d, "transcript.txt")) else None,
                "asset__title": meta.get("title"), "asset__description": meta.get("description"),
            }
            thumb = vis.get("thumbnail") or {}
            for c in VIS_THUMB:
                v = thumb.get(c)
                row[f"vis__thumb_{c}"] = (float(v) if isinstance(v, bool) else v) if v is not None else np.nan
            for c in VIS_FRAMES:
                row[f"vis__{c}"] = vis.get(c, np.nan)
            for name, v in (aud.get("egemaps") or {}).items():
                row[f"aud__egemaps__{name}"] = v
            for c in AUD_PAUSES:
                row[f"aud__pause_{c}"] = (aud.get("pauses") or {}).get(c, np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def _interp_at(ages, values, horizon_h):
    """Linear interpolation of `values` at age=horizon_h from the bracketing
    observations; NaN if the horizon is not yet bracketed (never extrapolate)."""
    pairs = sorted((a, v) for a, v in zip(ages, values) if v == v)
    if not pairs or pairs[-1][0] < horizon_h:
        return np.nan
    prev = None
    for a, v in pairs:
        if a >= horizon_h:
            if prev is None or a == horizon_h:
                return v
            (a0, v0) = prev
            return v0 + (v - v0) * (horizon_h - a0) / (a - a0)
        prev = (a, v)
    return np.nan


def load_prospective(tracking_dir, horizon_days=7):
    cohort = pd.read_csv(os.path.join(tracking_dir, "cohort.csv"))
    snaps = pd.read_csv(os.path.join(tracking_dir, "video_snapshots.csv"))
    chans = pd.read_csv(os.path.join(tracking_dir, "channel_snapshots.csv"))
    thumbs_p = os.path.join(tracking_dir, "thumbnail_snapshots.csv")
    thumbs = pd.read_csv(thumbs_p) if os.path.exists(thumbs_p) else pd.DataFrame(columns=["video_id", "changed", "file"])
    horizon_h = horizon_days * 24

    first_chan = chans.sort_values("observed_at_utc").groupby("channel_id").first()
    rows = []
    for r in cohort.itertuples(index=False):
        s = snaps[snaps.video_id == r.video_id]
        ok = s[s.status == "ok"]
        outcome = _interp_at(ok.age_hours.to_numpy(float), ok.view_count.to_numpy(float), horizon_h) if len(ok) else np.nan
        ch = first_chan.loc[r.channel_id] if r.channel_id in first_chan.index else None
        t = thumbs[thumbs.video_id == r.video_id]
        first_file = t[t.changed == True]["file"].iloc[0] if (t.changed == True).any() else None  # noqa: E712
        rows.append({
            "dataset": "prospective", "video_id": r.video_id, "channel_id": r.channel_id,
            "sampling_arm": r.sampling_arm, "outcome_views": outcome, "label": np.nan,
            "meta__duration_sec": r.duration_sec if r.duration_sec == r.duration_sec else np.nan,
            "meta__definition_hd": np.nan, "meta__title_length": len(str(ok.title.iloc[0])) if len(ok) else np.nan,
            "meta__description_length": np.nan, "meta__tag_count": np.nan,
            "meta__channel_follower_count": float(ch.subscriber_count) if ch is not None and ch.subscriber_count == ch.subscriber_count else np.nan,
            "meta__caption_available": np.nan, "meta__category": r.category,
            **_schedule_features(r.published_at_utc),
            "asset__thumbnail_path": os.path.join(tracking_dir, "thumbnails", first_file) if first_file else None,
            "asset__frames_dir": None, "asset__transcript_path": None,
            "asset__title": ok.title.iloc[0] if len(ok) else None, "asset__description": None,
            "track__thumbnail_changes": int((t.changed == True).sum()) - 1 if len(t) else np.nan,  # noqa: E712
            "track__title_changes": int(ok.title.nunique() - 1) if len(ok) else np.nan,
            "track__n_snapshots": int(len(ok)),
        })
    df = pd.DataFrame(rows)
    # within-category top quartile of log(1+outcome) among videos that reached the horizon
    reached = df.outcome_views.notna()
    for cat, idx in df[reached].groupby("meta__category").groups.items():
        sub = df.loc[idx, "outcome_views"]
        k = max(1, len(sub) // 4)
        top = sub.sort_values(ascending=False).index[:k]
        df.loc[idx, "label"] = 0
        df.loc[top, "label"] = 1
    return df
