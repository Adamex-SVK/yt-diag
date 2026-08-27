# FEATURES.md — YT-Diag Feature Dictionary

_The single source of truth for every feature in the project: what it is, where it comes from, and what it may be used for. Update this file whenever a feature is added, removed, or its role changes. Where this file and a research doc disagree, this file wins (research docs are proposals; this documents what is actually collected)._

**Roles** (the column that matters most — mixing these up caused the v1 label bugs):

- **model input** — may be fed to baselines and/or the deep model
- **label-only** — used to construct the viral/typical label; must NEVER be a model input
- **stratifier** — used to define label-v2 ranking cells; may also be a model input, with a caveat: banding neutralizes the *band IDs* (every cell has the same viral rate), but the exact underlying values can retain residual within-band signal — coarse stratification reduces direct label shortcuts, it does not license these as leakage-free inputs (see the §5 limitations note)
- **identifier / bookkeeping** — never a feature at all (but `channel_id` drives channel-grouped train/val/test splits)

---

## 1. Structured metadata — _Adam, 2026-08-18/20 (`collect_and_extract.py` → `metadata.json`)_

| Feature | Role | Definition | Source field |
|---|---|---|---|
| `duration` | model input | Video length in seconds | yt-dlp `duration` |
| `upload_date` | model input (derived) | Upload date, `YYYYMMDD` — **date only, no time** (see §6 backfill) | yt-dlp `upload_date` |
| `definition` | model input | HD vs SD | yt-dlp `definition` |
| category | model input / stratifier | One of comedy, howto, vlogs, product_reviews (the collection category; also the outermost label-v2 stratum) | Collection folder + yt-dlp `categories` |
| `title_length` | model input | Character count of the title | Derived from `title` at collection |
| `description_length` | model input | Character count of the description | Derived from `description` |
| `tag_count` | model input | Number of creator tags | Derived from `tags` |
| `channel_follower_count` | model input / stratifier | Channel subscriber count **at collection time** (post-outcome — see label v2 notes, §5) | yt-dlp `channel_follower_count` |
| `license` | bookkeeping | CC vs standard license (kept for provenance; CC constraint dropped 2026-08-14) | yt-dlp `license` |
| `view_count` | **label-only** | Views at collection time | yt-dlp `view_count` |
| `like_count` | **label-only** | Likes at collection time | yt-dlp `like_count` |
| `comment_count` | **label-only** | Comments at collection time | yt-dlp `comment_count` |
| caption availability | model input | Whether YouTube auto-captions existed (feature #12 of the original 12) — **not in `metadata.json`**; read `had_auto_captions` from `transcript_info.json` | `transcript_info.json` |
| `id`, `channel_id`, `collected_at` | identifier / bookkeeping | Video id; channel id (**drives channel-grouped splits** — no channel may span train/val/test); collection timestamp (used with `upload_date` to compute video age) | yt-dlp |

> ⚠️ The "12 metadata features" table in `data_retrieval.md` §5.4 includes view/like/comment counts as features #1–3. They are **label-only** (see `02_Data/README.md`). The actual model-facing metadata list is the 9 rows marked "model input" above, plus the §6 schedule features once backfilled.

## 2. Visual engineered features — _Adam, spec 2026-08-14 (`additional_features.md` §1), implemented 2026-08-18 (`visual_features.json`)_

Computed on the thumbnail and on every sampled frame (before frame deletion), then aggregated across frames. All are **model inputs** (baseline columns; optionally a 4th projected modality in the fusion head).

**Per image** (stored fully for the thumbnail under `thumbnail.*`):

| Feature | Definition |
|---|---|
| `cct` | Correlated color temperature in Kelvin (McCamy's approximation from mean RGB) — "warm vs cool" look |
| `brightness` | Mean HSV value channel (0–255) |
| `saturation` | Mean HSV saturation channel (0–255) — color vividness |
| `contrast` | Std of the HSV value channel |
| `has_face` / `face_count` | Any face detected / how many (OpenCV Haar cascade; mediapipe fallback path unused, see README pitfalls) |
| `max_face_area_ratio` | Area of the largest face box ÷ image area |
| `face_centrality` | Distance of the largest face's center from image center (0 = dead center); `null` if no face |

**Aggregates across the ~20 sampled frames**: `frames_mean_cct`, `frames_std_cct`, `frames_mean_brightness`, `frames_mean_saturation`, `frames_mean_contrast`, `frames_has_face_ratio` (fraction of frames with a face), `frames_mean_max_face_area_ratio`.

## 3. Audio / prosodic features — _Adam, spec 2026-08-14 (`additional_features.md` §2), implemented 2026-08-18 (`audio_features.json`)_

Extracted from the audio track (16 kHz mono WAV) before deletion. All are **model inputs**.

| Feature group | Definition |
|---|---|
| `egemaps.*` | The 88 openSMILE **eGeMAPSv02 functionals** — the standard minimal acoustic set for voice/affect research: pitch (F0) statistics, loudness, jitter/shimmer (voice stability), spectral balance/slope, formants, voicing rate. Individual names follow openSMILE's eGeMAPS naming; treat as an 88-column block rather than documenting each here. |
| `pauses.pause_count` | Number of silences > 300 ms (webrtcvad, aggressiveness 2) |
| `pauses.total_pause_sec` | Total paused seconds |
| `pauses.pause_ratio` | Paused time ÷ total audio time — "dead air" fraction |
| `pauses.mean_pause_sec` | Mean pause length |

## 4. Raw modalities (encoder inputs, not tabular features) — _Adam, 2026-08-18/20_

| Modality | Files | Encoder (locked Tier-1 architecture) |
|---|---|---|
| Thumbnail | `thumbnail.jpg` (1280×720) | DINOv2 ViT-S/14, frozen |
| Frames | `frames/` — ~20 frames, 60% densely sampled in the first 60 s | Same DINOv2 + attention pooling |
| Transcript | `transcript.txt` (+ `transcript_info.json`: `source` = auto-captions or Whisper, `had_auto_captions`) | ModernBERT-base, frozen |
| Title + description | `metadata.json` `title`, `description` | Same ModernBERT-base |

## 5. Label fields — _Emmanuel, 2026-08-26 (`compute_labels_v2.py` → `label.json`, `labels_summary.csv`)_

All **label machinery — never model inputs**. Produced by the v2 stratified label (top quartile of `log(1+views)` within category × age-band × channel-size-band; supersedes v1's views-per-day ÷ subs, which had an age bias and a post-outcome subscriber denominator — see CHANGELOG 2026-08-26).

| Field | Definition |
|---|---|
| `label` | `viral` / `typical` — **the prediction target**. Exactly the top ⌊n/4⌋ videos (min 1) of each cell of size n |
| `label_version` | 2 |
| `days_since_upload` | Video age in days at collection (from `upload_date` → `collected_at`) |
| `log_views` | `log(1 + view_count)` — the ranking score (monotonic, so ranking-equivalent to raw views; kept as a readable EDA quantity) |
| `age_band` / `size_band` | Equal-count quantile band indices (default 4×4 per category) by age / subscriber count, **assigned by value** — identical (API-rounded) values always share a band |
| `cell_size` / `cell_percentile` | Number of videos in the video's stratification cell / rank percentile within it (diagnostic) |
| `warnings` | Per-video flags, e.g. exact view-count tie at the cell's viral cutoff (broken deterministically by `video_id`) |
| `labels_excluded.csv` | Videos excluded from the cohort with reasons (< 30 days old, missing dates/views, hidden subs) — excluded, never imputed |

> Known limitations (disclose in the report): the coarse subscriber stratifier *reduces, does not eliminate,* the post-outcome-subs problem; models still see exact age/subs values, so stratification removes direct band shortcuts without guaranteeing performance comes from content; labels are a retrospective cohort-relative ranking computed on the full cohort before any split (transductive). Planned sensitivity checks: halved-subs band-crossing rate; train-only band edges.

## 6. Planned features (not yet collected) — _Emmanuel, 2026-08-26_

| Feature | Role | Definition / plan | Status |
|---|---|---|---|
| `published_at` | model input (derived) | Full UTC publish timestamp — `videos.list snippet.publishedAt`; `metadata.json` only has the date. Backfill for all collected videos: ~160 batched calls for 8,000 videos | `backfill_published_at.py` built 2026-08-26 — run on the collected data |
| `publish_hour_utc` (sin/cos) | model input | UTC hour as sine/cosine cyclical encoding (hour 23 and 0 are neighbors); local-time interpretation only where timezone is known | after backfill |
| `publish_weekday` | model input | Day of week (Mon–Sun) from `published_at` — robust to timezone error | after backfill |
| `is_weekend` | model input | Sat/Sun flag | after backfill |
| `publish_time_of_day` | model input | Coarse local-time bucket (morning/afternoon/evening/night), converted via channel country's primary timezone; **missing when country unknown** — never guessed; record the timezone source | after backfill + country |
| `channel_country` | model input / analysis | Channel's self-declared country — `channels.list snippet.country` (often missing; free alongside the subscriber call; creator country ≠ audience location). Used for timezone conversion and a US-only *sensitivity analysis*, **not** for filtering the dataset | needs fetch |
| Prospective tracking cohort | future label / validation | **Fixed capped cohort (12,000 main-category budget = 3,000 × 4, plus the backup category's own 3,000 on top → 15,000 main-arm ceiling; raised from the initial 2,000–4,000 plan on 2026-08-27: quota is use-it-or-lose-it, storage/RAM aren't binding, and a larger clean panel lets the extension *select* its extraction subset; still capped, not an unbounded sweep)**: new videos discovered at age < 1 day (`search.list publishedAfter/Before`, `order=date`), views/likes/comments + channel stats snapshotted daily for 30 days (`videos.list`/`channels.list`, batched 50 per call). Store per snapshot: `published_at_utc`, `observed_at_utc`, `age_hours`, raw counts, discovery source. Gives fixed-horizon outcomes (`views_at_7d` preliminary, `views_at_30d` primary — keep continuous, don't reduce to quartiles at collection time) and `subscriber_count_at_first_observation` (+ observation age + `hiddenSubscriberCount`) — **not** "day-0 subs": even hours-old observations may include video-driven gains, and public counts are rounded to 3 significant figures. Supplement search discovery with a small stratified panel of known channels (note: upload checks are ~1 call *per channel* per poll — stagger by upload frequency, don't poll thousands). Any day-1-views-derived feature belongs to a separate early-warning model, never the upload-time model. Also captured per snapshot: per-video `youtube_category_id`, current `title`, and the current **thumbnail** (hashed; new image stored only on change) — thumbnail/title *change events* are themselves candidate features, and the first capture preserves the **first-observed (near-publish)** thumbnail the retrospective dataset can never see; discovery can lag publication by up to ~24h, so pre-discovery changes are invisible — disclose in methodology **Main arm requires duration ≥ 4min** (discovery `videoDuration=medium/long` + verified `contentDetails` duration at admission — 94% of unfiltered order=date results were Shorts ≤180s). The `short_form` arm is **only the retagged day-1 (2026-08-26) unfiltered cohort**, kept tracked as a Shorts-vs-regular comparison; ongoing discovery *rejects* new sub-4min candidates rather than adding to it | `track_new_videos.py` built 2026-08-26; **cohort live since 2026-08-26** (day-1 main arm >1,000 across the four categories + `tech_reviews` backup category (id 28) accumulating as a product_reviews replacement candidate + `short_form` comparison arm) — **scheduling active**: two launchd agents on Emmanuel's MacBook (09:05 full tick = first run of the Pacific quota day, 21:05 `--no-discover`; missed runs fire on wake); migration to an always-on server is preferred if one becomes available |

> Interpretation note for all schedule features: feature importance supports "publish time was *associated* with the prediction," not "posting at that hour *caused* underperformance" — causal claims would need a stronger design. Word the diagnostic output accordingly.

---

_Created 2026-08-26 (Emmanuel). Sections 1–4 document features Adam built into the collection pipeline (spec: `additional_features.md`, 2026-08-14; implementation: `collect_and_extract.py`, 2026-08-18/20). Sections 5–6 are the 2026-08-26 label revision and newly planned features._
