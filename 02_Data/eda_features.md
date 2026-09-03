# Descriptive feature EDA

_Emmanuel, 2026-09-01. Reproducible with `eda_features.py`; generated tables,
figures and machine-readable decisions live in `02_Data/eda_features/`._

This EDA first asks what the measured modalities contain before asking whether
they predict the target. The core tables drop the adapter's label column and
remain label-blind. The dashboard's correlation and view-association panels are
the explicit exception: they use only the pre-specified seed-0,
channel-grouped outer training partition and are descriptive, not a feature
selection procedure. Full-dataset outcome scans would leak validation/test
information into the design and remain prohibited.

## What we added

The checked-in tables report both formats separately and then category × format.
They use medians and interquartile ranges for skewed quantities; CCT also gets a
mean because an average was explicitly requested.

- Dataset composition: sample size, duration, views, subscribers, collection
  age, transcript usability and audio coverage.
- Visual profile: thumbnail/frame CCT and coverage, raw and heuristic-cropped
  brightness, saturation, contrast, face presence, face count/area/centrality
  conditional on a detected face, and frame face ratio.
- Audio and speech profile: VAD pause ratio, pauses per minute, mean detected
  pause, equivalent sound level, loudness level/variability, and cleaned
  transcript words per video minute.
- Sampling audit: the mean frame-gap proxy remains available because 20 frames
  represent videos ranging from seconds to hours.

## Correlation dashboard and pivot table

The familiar EDA layer is now reproducible rather than assembled by hand:

- `category_format_pivot.csv` places category on rows and regular/Shorts
  measurements on columns: counts, median views/subscribers/duration,
  brightness, CCT, face prevalence, pause ratio, speech density and modality
  coverage.
- `correlation_heatmap_train_seed0.png` is a 19-variable Spearman matrix over
  1,113 training videos from 798 channels. Spearman is used because counts and
  durations are extremely skewed; categorical names are not converted into
  meaningless integer codes.
- `audio_correlation_clustered.png` separates all 88 eGeMAPS variables from the
  interpretable matrix. It reveals 23 feature pairs with absolute Spearman
  correlation above 0.95, supporting dimensionality reduction or strong
  regularisation for the audio block.
- `modality_coverage_heatmap.png` makes structural missingness visible by
  category and format. The largest gap is usable transcript coverage: 92% for
  regular videos versus 60% for Shorts, while stored thumbnails and frames are
  complete.
- `shortcut_scatterplots_train_seed0.png` facets subscriber/view and age/view
  relationships by category, while
  `outcome_associations_train_seed0.png` contrasts pooled, regular-only and
  Shorts-only rank correlations.

The correlation matrix quantifies several redundancies that a model can exploit:
duration identifies format (ρ = −0.848), pause ratio and pauses/minute are almost
the same signal (ρ = 0.98), thumbnail and frame CCT move together (ρ = 0.53),
cropped-thumbnail and frame brightness move together (ρ = 0.63), and detected
face presence is strongly tied to detected face area (ρ = 0.79). Subscriber
count is by far the strongest pooled association with log views (ρ = 0.705).
Duration is a useful Simpson warning: its pooled association with views is
slightly negative, while it is positive within both regular videos and Shorts.
These are associations in one exploratory training partition, not causal advice
and not justification for selecting features after looking at validation data.

The format split is essential. Median duration is 843 s for regular videos and
37 s for Shorts. Raw thumbnail brightness is 144 versus 80 and predicts format
at direction-free AUC 0.925; after the heuristic content crop it is 145 versus
152 and the format AUC falls to 0.559. This supports using crop brightness as a
candidate, but the crop flag itself is not a verified pillarbox detector: the
heuristic also crops 45.2% of regular thumbnails.

CCT v3 is now complete and well covered. All 1,860 visual files use
`nearest_planckian_locus_cie1960uv_1pct_lut`; thumbnail CCT is present for
1,854 videos and 36,652/37,200 frames (98.53%) yield valid CCT. Median thumbnail
CCT is 5,312 K for regular and 5,453 K for Shorts; median frame CCT is 5,762 K
and 5,603 K respectively. The manifest now distinguishes all 1,860 files
conforming to the frozen policy from 1,852 rows with the complete model-ready
thumbnail/frame aggregate triplet; 172 videos have at least one missing CCT
measurement (usually only part of their 20-frame sequence). These are
descriptive associations, not advice to make a video warmer or cooler.

The audio table must be read as whole-track acoustics, not “audio quality.”
Shorts are about 4.7 dB higher in median equivalent level and have lower
loudness variability. Their median detected pause ratio and pauses/minute are
both zero; 64.7% have zero detected pauses versus 3.1% of regular videos. This
can reflect editing, continuous speech, music being classified as voice, or a
combination. The pause fields are therefore VAD-silence diagnostics, especially
weak for Shorts—not literal ground-truth human pauses.

Cleaned transcript density is 151 words/minute for regular videos and 190 for
Shorts. It measures words retained in the final edited video, not a speaker's
physical articulation rate. Coverage also differs sharply (92.4% regular,
60.2% Shorts), partly because transcript usability uses an absolute 50-word
floor.

## Features that needed correction or qualification

| Existing/proposed field | Decision | Reason |
|---|---|---|
| raw thumbnail brightness | keep for audit; prefer crop candidate | strong format proxy caused by thumbnail composition/padding |
| heuristic crop flag | diagnostic only | crops many ordinary thumbnails; not geometry ground truth |
| `VoicedSegmentsPerSec` and segment-length fields | exclude or mask in a later registry change | the openSMILE segment buffer saturates near 1,000 segments on long videos; `VoicedSegmentsPerSec` has within-regular ρ ≈ −0.975 with duration |
| raw title length | pair with hashtag-stripped length | hashtags reverse the Shorts/regular ordering |
| transcript words/minute | keep as edited-speech-density interpretation | cleaned text and editing density, not articulation |
| pause fields | keep as descriptive diagnostics with coverage/masks | VAD confounds music and speech; zero is especially common for Shorts |
| mean frame interval | keep as sampling diagnostic/covariate | fixed 20-frame budget gives very different temporal coverage |

## What we deliberately did not add

- **Music amount:** caption tags and VAD cannot quantify music reliably.
- **Cut rate/editing pace:** 20 sparse frames miss almost all cuts; dividing
  frame differences by time reconstructs roughly `1/duration` rather than a
  stable editing rate.
- **Single-scene labels:** no threshold crossing among 20 sparse frames cannot
  prove a video used one scene. The old `is_single_scene` proposal was removed.
- **One audio-quality score:** eGeMAPS values describe a mixed track containing
  speech, music and ambience; they do not grade microphone or mastering quality.
- **Full-dataset label correlations:** the earlier “none predicts the label”
  scan is exploratory only and is not used for feature selection or a reported
  confirmatory result.

`motion_features.py` remains as a negative-result audit so the shot-rate idea is
not repeatedly rediscovered. Its outputs are not registered as model inputs.

## Reproduce

```bash
.venv/bin/python 02_Data/derived_features.py
.venv/bin/python 02_Data/motion_features.py
.venv/bin/python 02_Data/eda_features.py
```
