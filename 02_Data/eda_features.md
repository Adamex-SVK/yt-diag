# Descriptive feature EDA — what every modality actually contains

_Emmanuel, 2026-09-01. Companion to `eda.md`, which decided the label. This one profiles the
features themselves: what they measure, how Shorts and regular video differ, and which are broken.
Six modalities analysed, every serious finding independently re-derived before being written down;
corrections that survived that pass are noted inline._

## The one-sentence version

Six of the features we ship measure something other than their name, and the cause is almost always
the same: **video format leaks into everything**. The fixes are in `derived_features.py`.

---

## 1. What was broken, and what fixed it

| Feature | What it actually measured | Fix | Evidence |
|---|---|---|---|
| `vis__thumb_brightness` | whether the video is a Short | crop to the content region | format AUC **0.925 → 0.559**; sign reversal eliminated |
| `aud__egemaps__VoicedSegmentsPerSec` | duration (a saturated counter) | drop it | ρ = **−0.975** with duration inside regular video |
| `meta__title_length` | how many hashtags the title has | strip hashtags first | Shorts 71 chars raw → **31** stripped; ordering inverts |
| `pause_*` (4 fields) | nothing, on 65% of Shorts | restrict to regular video | 65.1% of Shorts have `pause_count` exactly 0 |
| `frames_std_cct` | partly the frame sampling interval | expose interval as a covariate | ρ ≈ +0.21…+0.24 with interval |
| — (missing) | single-scene video was unmeasurable | new `is_single_scene` flag | 10.8% of the collection |

### The thumbnail fix is the important one

**91.4% of Shorts thumbnails are pillarboxed** — a vertical frame dropped into a 16:9 box with
blurred, darkened bars either side. Those bars are most of the image, so `vis__thumb_brightness`
measures padding:

|  | raw brightness | content-crop brightness |
|---|---|---|
| regular | 144.3 | 145.1 |
| Shorts | **80.3** | **151.8** |
| AUC predicting format | **0.925** | **0.559** |

A 64-point gap collapses to 7. And it removes a genuine sign reversal:

| | pooled | regular | Shorts |
|---|---|---|---|
| raw | **−0.036** | +0.124 | +0.112 |
| content-crop | **+0.122** | +0.103 | +0.105 |

Raw, the pooled number says *darker thumbnails perform better* — false in both subgroups, and
actively harmful as advice. Cropped, pooled agrees with both strata.

This matters beyond one column: an attribution layer surfacing raw brightness would tell a creator
to *"brighten your thumbnail"* when the model has actually learned *"this is a Short"*.

### `VoicedSegmentsPerSec` is a capped counter

It saturates at exactly 1,000 voiced segments. For the median regular video,
`VoicedSegmentsPerSec × duration = 1000.008`; 75.1% of regular videos sit in [998, 1002] and
capping begins at 278 s. Inside regular video it correlates **−0.975** with duration. It is a
duration proxy dressed as prosody, and duration is already a label confound. The four
segment-length features come off the same buffer, so for capped videos they describe only the
first ~5 minutes.

---

## 2. Shorts and regular video are two different populations

Almost every pooled statistic in this dataset is a mixture of two distributions. Some highlights,
all with the format split that makes them meaningful:

**Duration** is a disguised format flag: ρ(duration, `is_short`) = **−0.849**. A pooled
per-category duration median describes almost no real video, and the category ordering is
*different* in each format — product_reviews is second-longest among Shorts but last among regular.

**Audio** differs by mastering, not by delivery. Shorts are **4.6 dB louder** with about a third
less dynamic range (loudness `stddevNorm` 0.555 vs 0.669). Pitch *level* differs but pitch *range*
does not (p = 0.056) — so the honest phrasing is "Shorts are louder and flatter", not "Shorts
presenters are more energetic".

**Speaking rate** (new feature): 163 wpm overall, but **151 wpm regular vs 190 Shorts**, and 224 for
vlogs Shorts where 55% exceed 220 wpm. Regular video sits squarely in the normal human 120–160
band. The Shorts excess is not a measurement error — it is jump-cut editing removing the silence
between sentences, so on a Short this measures **edited speech density**, not articulation rate.

**Pauses do not exist on Shorts.** 65.1% have `pause_count` exactly 0, versus 3.3% of regular
video, and duration does not explain it: duration-matched regular videos under 180 s zero out only
16.9% of the time. Two causes, both real — jump cuts remove the silence, and the VAD scores music
as speech (274 videos report zero silence anywhere while having no usable transcript). The pause
fields are only measurements on regular video.

**Language is confounded with format**: Shorts are 50.4% English against 78.1% for regular.

**Text volume differs 15.6×.** A text encoder sees a median 1,919 words for a regular video and far
fewer for a Short; 41.7% of Shorts are mostly hashtags, and a quarter of Shorts transcripts are
unusable. Transcript availability itself is an artefact: the 50-word floor is absolute, so within
Shorts it correlates **+0.58** with duration — it measures length, not quality.

---

## 3. Things that turned out not to be measurable

Recorded because they cost real time and someone will otherwise try again.

**Cutting pace cannot be recovered from 20 frames.** The plan was to count shot changes by
thresholding consecutive-frame differences. Calibration killed it: the mean consecutive difference
has median 43.9 on a 0–255 scale, so a "large" difference is the *normal* case — at a threshold of
25, 83% of videos exceed it on average. The frames are seconds to minutes apart, so nearly every
consecutive pair already straddles several cuts. Real cut detection needs ~1 fps, which means
re-downloading every video.

What survives is the **zero-scene-change flag**: 200 videos (10.8%) whose 20 frames contain no
large jump at all — locked-camera stand-up, one-angle tutorials, held photo cards. Concentrated in
howto Shorts (29.7%). Spot-checked by eye and every sampled case was genuinely single-take.

**Time-normalising frame differences does not work either.** Dividing by the elapsed gap gives a
quantity correlating **−0.829** with duration — very nearly `1/duration`. Frame difference
*saturates*: two unrelated frames differ by roughly a fixed ceiling however far apart they are, so
dividing by a growing gap just reconstructs duration. `visual_diversity` itself is mildly
duration-confounded (+0.20 pooled, +0.39 within Shorts) and needs conditioning before use.

---

## 4. The honest headline: none of it predicts the label

Across all six modalities, after conditioning on the label cells, **nothing reaches a useful
effect size**. Speech and pause features land at AUC 0.477–0.523 pooled and stay there within
cells. No visual feature separates the label within cells. Text is inert once conditioned
(nothing exceeds |AUC − 0.5| = 0.043). The strongest surviving text signal is `#shorts` in the
title — which is a format detector, not content.

Two ways to read that, and the report should give both:

1. **The label design is working.** These features were *supposed* to lose their apparent power,
   because most of it was format, channel size and age. The engineered blocks scoring at chance is
   the same result as the visual-only baseline sitting at 0.504.
2. **Engineered features may simply be too coarse.** Fifteen numbers cannot capture what makes a
   thumbnail compelling. This is the case *for* the deep model rather than against it — and it also
   means the deep model has a real bar to clear rather than an easy win.

The metadata baseline sits around **AUC 0.56–0.61** and that is the number to beat.

---

## 5. What this changes

**Implemented** in `derived_features.py` (all 1,860 videos, `processed/derived_features.csv`):
content-crop thumbnail statistics, hashtag-stripped title length, speaking rate, single-scene flag,
frame sampling interval.

**Recommended for the feature registry**, not yet applied:

- drop `aud__egemaps__VoicedSegmentsPerSec` and the four segment-length features, or mask them
  above 278 s
- replace `vis__thumb_brightness` with `thumb_crop_brightness`
- replace `meta__title_length` with `title_chars_nohash`, and expose `title_has_shorts_tag`
  separately as the format tell it is
- restrict the pause block to regular video, or add an explicit `pause_measurable` mask
- pass `frame_interval_sec` to the temporal branch as a covariate

**For the report**: never print a pooled per-category duration; report Shorts and regular
separately everywhere; and disclose that a 12,002-second video and a 4-second Short are both
represented by 20 frames.

## Reproducing

```bash
.venv/bin/python 02_Data/derived_features.py     # the fixes -> processed/derived_features.csv
.venv/bin/python 02_Data/motion_features.py      # frame-change statistics
```
