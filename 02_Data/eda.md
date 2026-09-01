# EDA — retrospective dataset (1,860 videos)

_Emmanuel, 2026-09-01. Run on the dataset as cleaned on 2026-08-31 (see CHANGELOG). Every number
here is reproduced by `.venv/bin/python 02_Data/eda_retrospective.py`, which writes
`02_Data/eda/eda_stats.json` and four figures; findings were independently re-derived before being
written down, and the corrections that survived that pass are noted inline._

## What this changes

Three decisions, all with evidence attached, plus one warning that affects every modality.

| Decision | Current default | Recommended | Why |
|---|---|---|---|
| **Age floor** (`--min-age-days`) | 30 | **0 — remove it** | Costs 46% of the data and buys nothing measurable once bands do their job |
| **Shorts** | not stratified | **`is_short` as a third stratifier** | Unstratified it is a ~2× prior on the label *and* 99% readable off the frames |
| **Band counts** | 4 age × 4 size | **3 age × 3 size** (× 2 format) | 4×4×2 shatters cells (median 13, 102 cells < 20); 2×2×2 lets the confounds back in |

Net: **1,854 labelled videos instead of 998** (comedy 463, howto 373, product_reviews 480, vlogs 538), the format confound neutralised, cells that stay
big enough to rank within, and per-category test sets roughly 2.5× larger.

**The warning:** the format bit (Shorts vs regular) is legible from the pixels — `frames_portrait`
recovers `is_short` for 99.0% of videos, and 96% of Shorts thumbnails are a blurred pillarbox. Any
vision result must be read against a format-only baseline, or it is measuring aspect ratio.

---

## 1. The target, and the shortcut ceiling

`view_count` is power-law: pooled skew 24.4, Gini 0.914, and the top 10% of videos hold 87.6% of
all 13.8 bn views. `log1p` makes it tractable (per-category log-skew −0.60 to +0.53).

The four categories are **four different populations on that scale**. Median views: comedy
1,254,042, howto 883,149, vlogs 314,043, **product_reviews 9,953** — 126× below comedy, and
P(a comedy video out-views a product_reviews video) = 0.887. A single pooled threshold would label
almost no product_reviews video viral. Within-category comparison is mandatory, not stylistic.

**The shortcut ceiling.** A model given only four things it never has to look at the content
for — subscriber count, video age at collection, duration, `is_short` — predicts `log(views)` at
**out-of-fold R² = 0.584** (channel-grouped 5-fold; subscribers alone reach 0.43, and adding
category dummies pushes it to ~0.70).

Two consequences worth stating plainly in the report:

- **Never quote an R² on log(views) as evidence of content signal.** A no-content baseline already
  gets ~0.60 of it.
- This is the strongest possible argument *for* the v2 within-cell binary label. On the actual
  stratified label the same four confounds fall to **AUC ≈ 0.53–0.58** — the stratification is
  doing real work. Quoting the 0.58 R² figure as "the bar for the deep model" would be wrong by
  construction: the bar is ~0.55 AUC on the label, not 0.58 R² on views.

## 2. The age floor: remove it

`compute_labels_v2.py` excludes videos younger than 30 days at collection. That inherits a v1
justification — views-per-day decays with age — which **v2 does not share**, because v2 ranks
within age bands rather than dividing by age. So the floor has to re-earn its place.

**It cannot.** Cost and benefit, measured:

- **Cost:** 861 videos, 46% of the collection. It is not a random 46%: the dropped videos are 64%
  Shorts vs 43% retained, median duration 66s vs 231s. Comedy falls 464 → 193 and vlogs 541 → 142,
  which leaves those categories with 40 and 31 positives respectively.
- **Benefit:** essentially nil. With `is_short` stratified, the AUC of predicting the label from
  all four confounds is 0.576 at min-age 0, 0.572 at min-age 7, 0.571 at min-age 14. The floor
  also makes cells *worse*: 22 cells under 20 videos at min-age 0 versus 37 at min-age 7 and 55 at
  min-age 14.

**Is a 1-day-old view count meaningful at all?** Yes, and this is measurable rather than
arguable — the prospective panel observes the same videos twice daily, so it can answer directly.
Over 1,257 main-arm videos with a full day-1→day-5 curve:

| early observation | rank correlation with day 5 | top-quartile agreement |
|---|---|---|
| day 1 | 0.946 | 89.3% |
| day 2 | 0.984 | 93.2% |
| day 3 | 0.994 | 97.4% |
| day 4 | 0.999 | 99.1% |

A video's *rank among its peers* is ~90% settled within 24 hours. What is **not** settled is the
raw count — views roughly double between day 1 and day 5 — which is exactly why the label must
stay a within-band ranking and never a raw-count threshold.

Caveat to disclose: this measures stability out to day 5, not day 30; the prospective cohort is
not old enough yet. The curve is strongly asymptotic, but the extrapolation is an assumption.
Re-check it after 2026-09-25, when the first cohort videos reach the 30-day horizon.

## 3. Shorts: two populations sharing a folder

975 Shorts and 880 regular videos. They are not two flavours of one thing:

- **Duration separates them almost perfectly.** Every Short in this dataset is ≤ 180.0s; a
  `duration ≤ 180s` rule recovers `is_short` with recall 1.000 and precision 0.942. (Do **not**
  port that ceiling to the tracker — the prospective cohort contains a verified 250s Short.)
- **Views:** Shorts take 2.5× the median views pooled — but this reverses inside howto, so the
  pooled ratio is a Simpson artifact. Report it per category or not at all.
- **Text:** 388 Shorts have no usable transcript versus 67 regular videos.

**The labelling consequence.** Sharing a quartile cell, Shorts are labelled viral at 29.5–35.0%
versus 14.0–17.7% for regular videos — under the current default that is 29 of 226 viral labels
handed to Shorts on format alone. Adding `is_short` as a stratifier closes the gap to ~0.5
percentage points and drops AUC(label | `is_short`) from **0.596 to 0.447** — i.e. below chance.

Three designs were measured; **B is recommended**:

| design | labelled n | format AUC | note |
|---|---|---|---|
| A: `is_short` unstratified | 1,859 | 0.596 | format is a free ~2× prior on the label |
| **B: `is_short` as a stratifier** | **1,854** | **0.447** | keeps everything, confound removed |
| C: regular videos only | 572 (at min-age 30) | 0.500 | clean, but discards 69% and leaves vlogs at n=66 |

Design C deserves a mention to Adam rather than a silent rejection: it is the option that makes
the retrospective set **commensurable with the prospective cohort**, whose main arm is
English-only, ≥4 min and non-Shorts by construction. If the two datasets are ever to be pooled or
compared directly, C is the honest choice. At n=572 with vlogs at 66 it is not viable as the
primary cohort here, but it is the natural robustness check.

**Band counts, given the extra dimension.** Stratifying on format halves every cell, so the age
and size band counts must come down with it:

| config | n | median cell | cells < 20 | AUC (4 confounds) | Shorts over-rep |
|---|---|---|---|---|---|
| **3 age × 3 size × 2 format** | **1,854** | **24** | **22** | **0.576** | **0.99×** |
| 3×3×2, min-age 7 | 1,448 | 19 | 37 | 0.572 | 1.00× |
| 4×4×2 | 1,854 | 13 | 102 | 0.529 | 1.01× |
| 2×2×2 | 1,854 | 52 | 0 | 0.660 | 1.00× |

4×4×2 controls the confounds best but shatters the grid — 102 of its cells fall below the
labeler's own `SMALL_CELL_WARN`, and cells of ≤ 4 videos force a ~36% viral rate through the
`max(1, floor(n/4))` rule. 2×2×2 has beautiful cells and lets the confounds straight back in
(0.660). **3×3×2 is the knee.**

## 4. Format leaks into every modality — the thing to control

This is the finding most likely to produce a wrong conclusion if ignored. `is_short` is not a
label the model has to infer; it is written on the pixels:

- `frames_portrait` matches `is_short` for **99.0%** of videos.
- **96.3%** of Shorts thumbnails are a blurred, darkened pillarbox (939/975) versus 13/880 regular
  — so `vis__thumb_brightness`, the single strongest visual separator in the engineered set, is a
  **format detector, not a content feature**.
- Titles carry `#shorts` hashtags, which turn the text encoder into a format classifier too.
- At least six columns encode `is_short` near-perfectly, so it cannot be "removed" from the
  feature set by deleting a column.

With format stratified out of the *label*, this leakage stops being able to inflate a score — the
model can still see the format, but the format no longer predicts the target. That is the right
fix. What must still be reported: **a format-only baseline alongside every vision result**, and an
explicit note that the engineered visual features are contaminated by pillarboxing.

## 5. Channels, the split, and how little we can prove

1,319 channels for 1,860 videos: **81.7% of channels contribute exactly one video**, and 58% of
the corpus comes from those singletons; the largest channel has 19.

- The **channel-grouped split is load-bearing**: a non-grouped split leaks AUC 0.86 of pure
  channel memorisation. Keep `StratifiedGroupKFold` and say so in the report.
- Channel identity predicts the label at AUC 0.79 under a naive within-category quartile label but
  only **0.59 under the real v2 label** — more evidence the stratification works.
- **Test-set power is the binding constraint on what we can claim.** Under the current default
  (998 labelled), a 20% test slice holds ~6.9 comedy positives and ~5.7 vlogs positives. One video
  is worth 17 points of recall. A per-category AUC would carry a 95% CI of roughly [0.46, 0.94] —
  i.e. no information. Reaching ±0.10 needs n ≈ 171 per category; ±0.05 needs 684.

  **Therefore: report pooled test metrics with confidence intervals; treat per-category numbers as
  descriptive only.** The recommended config (1,854 labelled) raises the pooled test set to ~371
  videos and per-category to 63–127, which makes the pooled comparison meaningful and per-category
  suggestive. Detecting a deep-model AUC of 0.70 against a 0.60 metadata baseline on the same test
  set needs n ≈ 220 — pooled, that is now within reach; per category it is not.

## 6. The feature matrix — fix before baselines

125 input columns (meta 14, sched 4, vis 15, aud 92). Concrete defects, all confirmed:

- **`meta__definition_hd` is 100% NaN** in the retrospective set (yt-dlp never populated it) though
  99.8% present in the tracking cohort. Drop it here; keep it there.
- **CCT is broken, not merely noisy.** `vis__frames_std_cct` reaches 5.0 × 10⁹ and 27 videos have
  *negative* Kelvin — the McCamy approximation diverges near a green-deficient chromaticity. 83
  videos are flagged `vis_cct_valid = 0`. After standardisation 99% of rows span 0.006 sd, so the
  column carries nothing. NaN the out-of-range values (per column: a band of 1,000–20,000 K for
  the two *mean* columns; a std has no lower bound, so use a magnitude cap there instead) and
  propagate the flag as a feature.
- **Two string columns** (`meta__category`, `meta__language`) will crash sklearn unencoded, and
  `meta__is_first_upload` has exactly 2 positives — drop or fold it in.
- **Listwise deletion is impossible**: 0 of 1,860 rows are complete. Missingness is structural and
  blocky (all 92 `aud__` columns absent together for 35 videos; `face_centrality` null exactly when
  no face). Use explicit missing-indicators plus imputation, and prefer models that handle NaN.
- eGeMAPS is redundant — 12 feature pairs correlate above 0.95. Consider a PCA block, or accept it
  and use a tree model.
- **Events per variable ≈ 2** at 125 columns and ~430 positives. The tabular baseline is
  under-powered as specified; feature-group selection is not optional.

## 7. Text and language

- **A native-English-only text model is not viable per category**: within the labelled cohort,
  native-English *and* usable transcript leaves comedy 77 and vlogs 64 videos. Use
  `native_en + translated_en` (1,411 videos) as the primary text cohort and keep `native_en`-only
  as an ablation, disclosing that ~33% of the text is machine-translated.
- The **transcript is auto-translated to English but the title and description are not**, so the
  "text modality" mixes languages *within a single video*. That must be stated in methodology.
- A dataset-level English-only filter would cost 36.7% of videos and halve comedy and vlogs.
  Not recommended; keep `meta__language` as a covariate.
- **Script detection cannot identify language here** (translated captions are Latin-script);
  `lang_declared` is the only usable signal.

## 8. Other confounds worth one line each

- **Duration vs views reverses under pooling** (Simpson): pooled ρ = −0.042, but +0.281 within
  regular videos and +0.085 within Shorts. Never report the pooled figure. After controlling for
  subscribers and age, only product_reviews long-form survives multiple-comparison correction.
- **87 sub-HD 4:3 thumbnails** are an age-and-quality confound concentrated in product_reviews
  (50/482): median age 1,566 days vs 46, and they under-perform on views even *within* age
  quartiles. Keep `thumb_subhd` as a stratifier and a disclosed limitation, not a feature.
- Roughly **5–6% of videos are visually static** across their 20 frames (locked-camera stand-up,
  slideshows). The temporal branch has nothing to learn there; if the attribution layer ever says
  "your pacing hurt you", gate it on a frame-difference threshold — and note the naive threshold of
  15 does *not* separate them (the locked-camera specials score 20–25).
- **The viral set is configuration-sensitive**: across 16 configurations only 83 of 438
  ever-viral videos are viral in all of them (Jaccard 0.44–0.79). The label is a defensible
  convention, not a ground truth — say so, and run the chosen config as a sensitivity analysis.

## 9. For Adam

1. **Sign-off needed** on the three label changes in the table at the top — they modify
   `compute_labels_v2.py`'s defaults, which were pending your sign-off anyway.
2. **Design C** (regular videos only) is the alternative worth a conversation: it is the only
   option that makes this dataset directly comparable to the prospective cohort.
3. **product_reviews is a different animal** — median 9,953 views against comedy's 1.25 M, Gini
   0.965, half its videos under 10k views. That is the Entertainment-category keyword workaround
   showing up in the data. It is also the category with the most sub-HD thumbnails and the most
   low-subscriber channels. Worth deciding whether it stays a headline category or becomes a
   documented caveat — the `tech_reviews` backup arm in the tracker exists for exactly this.

## Reproducing

```bash
.venv/bin/python 02_Data/eda_retrospective.py               # stats + 4 figures
.venv/bin/python 02_Data/eda_retrospective.py --no-figures  # stats only
```

Outputs land in `02_Data/eda/`: `eda_stats.json`, `views_and_age.png`, `rank_stability.png`,
`label_configs.png`, `shorts_viral_share.png`.
