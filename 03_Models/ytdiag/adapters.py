"""Dataset adapters: each maps one raw source onto the canonical table
(features.py naming). Missing groups are simply absent/NaN -- the model
config decides what to use, the adapter never pretends.

  load_retrospective(processed_dir)  Adam's per-video directory tree
      metadata.json, visual_features.json, audio_features.json, label.json
      (compute_labels_v2), metadata_extra.json (backfill_published_at.py),
      transcript.txt, thumbnail.jpg, frames/
  load_prospective(tracking_dir, horizon_days)  Emmanuel's tracker CSVs
      cohort.csv + video_snapshots.csv + channel_snapshots.csv +
      thumbnail_snapshots.csv + text_snapshots.csv (+ texts/ JSON);
      outcome = views interpolated at the horizon from the bracketing
      observations; label = within-category top quartile of outcome among
      MAIN-ARM (sampling_arm == "date_window") videos that reached the
      horizon (comparison arms short_form / non_english get no label -- a
      stand-in for the v2 stratified label, see FEATURES.md 6).
      Upload-time policy: every feature that a creator can edit after
      publishing (title, description, tags, thumbnail) is taken from the
      FIRST observation, never the latest -- otherwise post-publish edits
      made in response to performance would leak into upload-time inputs.
      Edit counts are exposed separately as track__* columns.
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
    if not isinstance(published_at_utc, str) or not published_at_utc:  # None / float NaN from pandas
        return {"sched__hour_sin": np.nan, "sched__hour_cos": np.nan,
                "sched__weekday": np.nan, "sched__is_weekend": np.nan}
    dt = datetime.datetime.strptime(published_at_utc[:19], "%Y-%m-%dT%H:%M:%S")
    h = dt.hour + dt.minute / 60.0
    return {"sched__hour_sin": math.sin(2 * math.pi * h / 24), "sched__hour_cos": math.cos(2 * math.pi * h / 24),
            "sched__weekday": dt.weekday(), "sched__is_weekend": int(dt.weekday() >= 5)}


def _channel_features(published_at_utc, channel_created_at, channel_video_count):
    """meta__channel_age_days / meta__channel_video_count / meta__is_first_upload.
    Prospective: video count at FIRST observation (near-publish). Retrospective:
    CURRENT count from the backfill (post-outcome, coarse) -- disclose."""
    age = np.nan
    if isinstance(published_at_utc, str) and published_at_utc and isinstance(channel_created_at, str) and channel_created_at:
        try:
            pub = datetime.datetime.strptime(published_at_utc[:19], "%Y-%m-%dT%H:%M:%S")
            cre = datetime.datetime.strptime(channel_created_at[:19], "%Y-%m-%dT%H:%M:%S")
            age = (pub - cre).total_seconds() / 86400.0
        except ValueError:
            pass
    try:
        count = float(channel_video_count) if channel_video_count not in (None, "") else np.nan
    except (TypeError, ValueError):
        count = np.nan
    return {"meta__channel_age_days": age, "meta__channel_video_count": count,
            "meta__is_first_upload": (int(count <= 1) if count == count else np.nan)}


def _flag(v):
    """'true'/'false' strings (CSV written by the tracker) OR bools (pandas
    parses a pure true/false column as bool) -> 1/0; anything else NaN."""
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, str) and v.lower() in ("true", "false"):
        return int(v.lower() == "true")
    return np.nan


def _language(*candidates):
    """First non-empty declared language, reduced to its primary subtag
    ('en-US' -> 'en') so the one-hot doesn't fragment by region."""
    for v in candidates:
        if isinstance(v, str) and v.strip():
            base = v.strip().split("-")[0].lower()
            if base in ("zxx", "und"):  # 'no linguistic content' / 'undetermined' are not languages
                continue
            return base
    return np.nan


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
                # two different things (FEATURES.md 1): uploader-provided captions
                # (contentDetails.caption, via the backfill) vs YouTube auto-captions
                # (transcript_info.json, collection time). Never conflate them.
                "meta__caption_available": _flag(extra["caption_available"]) if "caption_available" in extra else np.nan,
                "meta__auto_captions": int(tinfo["had_auto_captions"]) if "had_auto_captions" in tinfo else np.nan,
                "meta__category": category,
                "meta__language": _language(extra.get("default_audio_language"), extra.get("default_language")),
                "meta__is_short": _flag(extra.get("is_short")),
                **_schedule_features(extra.get("published_at_utc")),
                **_channel_features(extra.get("published_at_utc"), extra.get("channel_created_at"),
                                    extra.get("channel_video_count")),
                "asset__thumbnail_path": next((os.path.join(d, f) for f in ("thumbnail.jpg", "thumbnail.webp", "thumbnail.png")
                                               if os.path.exists(os.path.join(d, f))), None),
                "asset__frames_dir": os.path.join(d, "frames") if os.path.isdir(os.path.join(d, "frames")) else None,
                # cleaned transcript first (clean_retrospective.py --fix-transcripts):
                # raw auto-caption text repeats every phrase ~3x (rolling SRT
                # windows), which distorts any text feature computed from it
                "asset__transcript_path": next(
                    (os.path.join(d, f) for f in ("transcript_clean.txt", "transcript.txt")
                     if os.path.exists(os.path.join(d, f))), None),
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


MIN_LABEL_GROUP = 8  # a within-category quartile over fewer main-arm videos is noise, not a label


def load_prospective(tracking_dir, horizon_days=7):
    cohort = pd.read_csv(os.path.join(tracking_dir, "cohort.csv"))
    snaps = pd.read_csv(os.path.join(tracking_dir, "video_snapshots.csv"))
    chans = pd.read_csv(os.path.join(tracking_dir, "channel_snapshots.csv"))
    thumbs_p = os.path.join(tracking_dir, "thumbnail_snapshots.csv")
    thumbs = pd.read_csv(thumbs_p) if os.path.exists(thumbs_p) else pd.DataFrame(columns=["video_id", "file", "changed"])
    texts_p = os.path.join(tracking_dir, "text_snapshots.csv")
    texts = pd.read_csv(texts_p) if os.path.exists(texts_p) else pd.DataFrame(columns=["video_id", "file", "changed"])
    horizon_h = horizon_days * 24

    # schema tolerance: files written before 2026-08-27 lack these columns
    for c in ("definition", "caption_available", "default_language", "default_audio_language", "is_short"):
        if c not in cohort.columns:
            cohort[c] = np.nan
    for c in ("description_length", "tag_count"):
        if c not in snaps.columns:
            snaps[c] = np.nan
    if "channel_created_at" not in chans.columns:
        chans["channel_created_at"] = np.nan

    # group once -- per-video boolean filtering of the full frames is O(cohort x rows)
    snaps = snaps.sort_values("observed_at_utc")
    chans = chans.sort_values("observed_at_utc")
    snaps_by = {k: g for k, g in snaps.groupby("video_id")}
    chans_by = {k: g for k, g in chans.groupby("channel_id")}
    thumbs_by = {k: g for k, g in thumbs.groupby("video_id")}
    texts_by = {k: g for k, g in texts.groupby("video_id")}
    empty = snaps.iloc[0:0]

    def _str(v):
        return v if isinstance(v, str) and v else None

    def _num(v):
        try:
            return float(v) if v == v and v not in ("", None) else np.nan
        except (TypeError, ValueError):
            return np.nan

    def _changed(frame):
        return frame.changed.astype(str).str.lower() == "true"

    def _text_versions(video_id):
        t = texts_by.get(video_id)
        if t is None:
            return []
        return list(t[_changed(t)]["file"])

    def _read_text(fname):
        with open(os.path.join(tracking_dir, "texts", fname), encoding="utf-8") as f:
            return json.load(f)

    def _channel_at(channel_id, first_obs):
        """Channel snapshot taken at/after the video's FIRST observation (channel
        rows are written in the same tick, just after the video rows); falls
        back to the latest row. Using the channel's own first row would describe
        the channel at some EARLIER video's time."""
        g = chans_by.get(channel_id)
        if g is None:
            return None
        if first_obs is not None:
            later = g[g.observed_at_utc >= first_obs]
            if len(later):
                return later.iloc[0]
        return g.iloc[-1]

    rows = []
    for r in cohort.itertuples(index=False):
        s = snaps_by.get(r.video_id, empty)
        ok = s[s.status == "ok"]
        outcome = _interp_at(ok.age_hours.to_numpy(float), ok.view_count.to_numpy(float), horizon_h) if len(ok) else np.nan
        first = ok.iloc[0] if len(ok) else None  # upload-time policy: FIRST observation
        first_obs = first.observed_at_utc if first is not None else None
        ch = _channel_at(r.channel_id, first_obs)
        created = None
        g = chans_by.get(r.channel_id)
        if g is not None:
            known = g.channel_created_at.dropna()
            known = known[known.astype(str).str.len() > 0]
            created = known.iloc[-1] if len(known) else None
        t = thumbs_by.get(r.video_id)
        first_file = t[_changed(t)]["file"].iloc[0] if t is not None and _changed(t).any() else None
        versions = _text_versions(r.video_id)
        first_text = _read_text(versions[0]) if versions else None
        # first-observation values; rows snapshotted before 2026-08-27 have empty
        # description_length/tag_count (and the day-1 rows no title) -- fall
        # back to the first stored text version, which is the same policy
        first_title = first.title if first is not None and isinstance(first.title, str) else (first_text.get("title") if first_text else None)
        desc_len = _num(first.description_length) if first is not None else np.nan
        if desc_len != desc_len and first_text is not None:
            desc_len = float(len(first_text.get("description") or ""))
        tag_cnt = _num(first.tag_count) if first is not None else np.nan
        if tag_cnt != tag_cnt and first_text is not None:
            tag_cnt = float(len(first_text.get("tags") or []))
        rows.append({
            "dataset": "prospective", "video_id": r.video_id, "channel_id": r.channel_id,
            "sampling_arm": r.sampling_arm, "outcome_views": outcome, "label": np.nan,
            "meta__duration_sec": _num(r.duration_sec),
            "meta__definition_hd": {"hd": 1, "sd": 0}.get(_str(r.definition), np.nan),
            "meta__title_length": len(first_title) if first_title is not None else np.nan,
            "meta__description_length": desc_len,
            "meta__tag_count": tag_cnt,
            "meta__channel_follower_count": _num(ch.subscriber_count) if ch is not None else np.nan,
            "meta__caption_available": _flag(r.caption_available),
            "meta__auto_captions": np.nan,  # not observable without downloading (retrospective only)
            "meta__category": r.category,
            "meta__language": _language(r.default_audio_language, r.default_language),
            "meta__is_short": _flag(r.is_short),
            **_schedule_features(r.published_at_utc),
            **_channel_features(r.published_at_utc, created, ch.channel_video_count if ch is not None else np.nan),
            "asset__thumbnail_path": os.path.join(tracking_dir, "thumbnails", first_file) if first_file else None,
            "asset__frames_dir": None, "asset__transcript_path": None,
            "asset__title": first_title,
            "asset__description": first_text.get("description") if first_text else None,
            "track__thumbnail_changes": int(_changed(t).sum()) - 1 if t is not None and len(t) else np.nan,
            "track__text_changes": len(versions) - 1 if versions else np.nan,
            "track__title_changes": int(ok.title.dropna().nunique() - 1) if len(ok) else np.nan,
            "track__n_snapshots": int(len(ok)),
        })
    df = pd.DataFrame(rows)
    # within-category top quartile of outcome among MAIN-ARM videos that reached
    # the horizon; comparison arms (short_form, non_english) are never pooled in
    # -- their systematic differences would otherwise become label proxies; and
    # tiny groups get no label rather than a forced 'viral'
    reached = df.outcome_views.notna() & (df.sampling_arm == "date_window")
    for cat, idx in df[reached].groupby("meta__category").groups.items():
        if len(idx) < MIN_LABEL_GROUP:
            continue
        sub = df.loc[idx, "outcome_views"]
        k = max(1, len(sub) // 4)
        top = sub.sort_values(ascending=False).index[:k]
        df.loc[idx, "label"] = 0
        df.loc[top, "label"] = 1
    return df
