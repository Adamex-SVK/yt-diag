# 02_Data — README

## Purpose
Dataset documentation, YouTube API data collection, preprocessing pipelines, and exploratory data analysis (EDA).

## Data source

**YouTube Data API v3** — 4 categories, 2,000 videos each, 8,000 total (locked ticket #3, 2026-08-20):
- Comedy/skits (categoryId=23)
- Tutorials & how-to (categoryId=26)
- Vlogs (categoryId=22 + `q=vlog` workaround)
- Product reviews (categoryId=24, Entertainment, + `q=product review` workaround — locked ticket #4, 2026-08-20)

## Per-video features

| Signal | Format | Encoder |
|--------|--------|---------|
| 12 metadata features | Structured (duration, publish hour/day, tag count, title/desc length, category, channel subs/age) | Raw → baseline models |
| Thumbnail | 1280×720 JPG | Pretrained vision backbone (DINOv2 ViT-S/14, frozen) |
| 20 frames (16–24 range) | **Not uniform** — 60% of frames densely sampled in the first 60s, remaining 40% spread across the rest (locked in the Architecture Digest: engagement signal concentrates early) | Same DINOv2 backbone + attention-pool temporal aggregator |
| Transcript | YouTube auto-captions (~70-80% coverage) + Whisper fallback | Pretrained text transformer (ModernBERT-base, frozen) |
| Title + description | Text | Same ModernBERT-base |
| Visual engineered features | Color temperature (CCT), brightness/saturation/contrast, face presence/count/area/centrality — thumbnail + per-frame, mean/std across frames | Structured columns → metadata baseline; optional 4th projected modality into the fusion head |
| Audio/prosodic engineered features | eGeMAPS (88-feature openSMILE set: pitch, loudness, spectral, voicing) + explicit pause count/ratio/length via VAD | Same as above |

Architecture (frozen encoders, late-fusion MLP, Integrated Gradients attribution) locked in the "YT-Diag Architecture Digest" artifact, 2026-08-14. Engineered-feature scope from `01_Research/2026-08-14_scaling_and_features/additional_features.md`.

**Label**: views/likes/comments used ONLY for label construction (never model input).

## Contents

| File | Purpose | Status |
|------|---------|--------|
| `cc_availability_scan.py` | CC-license availability scan (YouTube Data API v3 `search.list`) across candidate categories | Done — run 2026-08-14 |
| `cc_availability_scan_results.json` | Raw scan output | Done |
| `cc_availability_scan_findings.md` | Findings write-up: real CC-video counts per category, `totalResults` unreliability, `videoCategoryId` filter quirk for categoryId 22/24 | Done — feeds wayfinder tickets #3 and #4 |
| `collect_and_extract.py` | End-to-end per-video pipeline: discover IDs (`search.list`, no license filter, `relevanceLanguage=en`) → yt-dlp metadata (+3 derived columns) + thumbnail + auto-subs → download video+audio → ffmpeg dense-first-60s frame extraction (20 frames) → OpenCV visual features → ffmpeg audio-track extraction → openSMILE eGeMAPS + VAD pause stats → transcript (auto-captions if they pass a quality heuristic, else Whisper `small.en` fallback) → **delete the video and audio files**. Only derived features are kept. Idempotent/resumable via a `.done` marker per video. | **Done — feature-complete, verified end-to-end, category mapping (#4) and dataset size (#3) both locked 2026-08-20.** Ready for a real collection run. |
| `compute_labels.py` | Computes the CLAUDE.md label (top-quartile views-per-day-since-upload, normalized by subscriber count, within category) from `metadata.json` for every `.done` video in a category. Writes `label.json` per video + `labels_summary.csv` per category. Separate from `collect_and_extract.py` on purpose — a label is a within-category ranking, so it can only be computed once a category's whole collected cohort is on disk, unlike every other feature here. | Done — built + tested 2026-08-20 on a real 2-video batch (correctly ranked the higher-normalized-rate video as "viral") |
| `data_collection_plan.md` | API query strategy, rate limits, category selection | TBD |
| `preprocessing.md` | Cleaning, tokenization, frame extraction pipeline | TBD |
| `eda.md` | Distributions, category stats, missing data report | TBD |

## Status
- **Environment**: Python venv + `yt-dlp` + `ffmpeg` set up and smoke-tested (metadata, license field, thumbnail, auto-subs all confirmed working against real videos). YouTube Data API v3 key provisioned in gitignored `.env`.
- **CC-availability scan**: done — see `cc_availability_scan_findings.md`. Real CC-video counts are much lower via bare category+license filtering (~95–110/category) than `totalResults` estimates suggested (off by 30–1,000,000x); categoryId 22 (vlogs) and 24 (Entertainment) need a keyword-assisted workaround due to an API filter quirk.
- **Data collection**: pipeline script complete and verified end-to-end (`collect_and_extract.py`) — all 12 metadata-baseline columns, transcript (auto-captions + Whisper fallback), visual + audio engineered features, and the dense-first-60s frame sampling all confirmed working on a real 2-video batch. Category mapping (ticket #4) and dataset size (ticket #3) both closed 2026-08-20 — 8,000 total, 2,000/category, product_reviews = Entertainment+keyword. Not yet run for real collection at that scale.
- **Label computation**: `compute_labels.py` built and verified on the same test batch.
- **Preprocessing pipeline**: frame/thumbnail/metadata/visual/audio/transcript extraction all done as part of the collection script.
- **EDA**: not started

## Notes
- Actual data files (CSV, frames, transcripts) go in the GitHub repo, not here.
- CC-license constraint on frame/video download was explicitly dropped by team decision (see `cc_availability_scan_findings.md` addendum) — accepted as outside YouTube TOS for the download step specifically.
- YouTube API quota: `collect_and_extract.py`'s discovery step uses `search.list` (100/day free bucket); metadata/thumbnail/captions/video come from `yt-dlp` and don't touch official-API quota at all.
- `collect_and_extract.py` requires `ffmpeg` + `ffprobe` on PATH in addition to `yt-dlp`, plus `pip install -r requirements.txt` for the feature-extraction deps (numpy, opencv-python-headless, opensmile, webrtcvad) — set this up on any new machine (e.g. the school computer) before running it. `02_Data/processed/`, `collection_manifest.csv`, and `collection_log.txt` are the script's outputs/logs (gitignored — not yet added, see `.gitignore` `02_Data/` rules).
- **opencv packaging pitfall (found 2026-08-18, fixed):** `opencv-python-headless>=5` silently ships without `CascadeClassifier`, breaking the face-detection fallback. Pinned to `<5` in `requirements.txt`. Also: never have more than one of `opencv-python`/`opencv-python-headless`/`opencv-contrib-python` installed at once — they share the same `cv2` package directory and uninstalling one corrupts the others (`cv2.__version__`/`cv2.data` disappear). If this happens, `pip uninstall` all opencv variants and reinstall clean.
- **mediapipe pitfall (found 2026-08-18, not fixed, not needed):** `mediapipe==1.0.1` (current PyPI release) removed the `solutions.face_detection` API `collect_and_extract.py` was written against, and pulls in a conflicting `opencv-contrib-python` build as a dependency. Left out of `requirements.txt` entirely — `_face_detector()` in the script catches this and falls back to the OpenCV Haar cascade automatically (the sanctioned fallback per `additional_features.md` 1.3), so no functionality is lost, just some accuracy on small/angled faces.
- `download_video()` must request `bestvideo+bestaudio` (not `bestvideo` alone) — video-only streams have no audio track to extract eGeMAPS/prosody features from.
- **yt-dlp version pitfall (found 2026-08-20, fixed):** `yt-dlp<2026.08.19` fails every video download against YouTube's current anti-bot JS challenge (`HTTP Error 403`). Fixed by pinning `yt-dlp==2026.8.19` in `requirements.txt`, installing `deno` on PATH (`brew install deno` — now a hard requirement, checked by `check_dependencies()`), and passing `--remote-components ejs:github` on every yt-dlp call (`YT_DLP_EXTRA_ARGS` in the script). This is a moving target on YouTube's side — if downloads start failing again on the school computer, `pip install --upgrade yt-dlp` first before debugging anything else.
- **Whisper is English-only (`small.en`)**: discovery now sets `relevanceLanguage=en` on `search.list` to keep non-English videos out (found 2026-08-20 — a Korean-language test video had no English auto-captions and Whisper produced a garbage transcript). `relevanceLanguage` is a soft ranking hint, not a hard filter, so a small fraction of non-English videos can still slip through; not solved further since it matches the mitigation `data_retrieval.md` §2.3 already called for, and stronger filtering would need a post-hoc language-detection pass, which is out of scope until it's shown to be a real problem at collection scale.
- All three scripts (frame sampling, engineered features, label computation) were verified together on a real 2-video test batch (`jNQXAC9IVRw`, `9bZkp7q19f0`) end-to-end on 2026-08-20 before being called done — not just unit-tested in isolation.
- **Category-mapping quirk spread further (found 2026-08-22):** the bare-`videoCategoryId`-returns-0 quirk (previously only categoryId 22/24) now also hits comedy(23) and howto(26) — confirmed directly against the live API. All four categories now carry a keyword (`comedy`, `tutorial`, `vlog`, `product review`).
- **Windows encoding pitfall (found 2026-08-22, fixed):** every `open(..., "w")` in the script now passes `encoding="utf-8"` explicitly. Without it, Python on Windows defaults to the system codepage (cp1252) for text writes, which crashes (`'charmap' codec can't encode character...`) the instant a video's title/description/transcript contains a non-Latin-1 character (hit this on a real school-computer run — Arabic text in one video, an ₹ symbol in another). macOS/Linux default to UTF-8, which is why this didn't surface in local testing.
- **Chrome cookie pitfalls on Windows (found 2026-08-22):** `--cookies-from-browser chrome` fails two different ways — "Could not copy Chrome cookie database" if Chrome is still running (even in the background/system tray), and "Failed to decrypt with DPAPI" (yt-dlp issue #10927) against current Chrome's App-Bound Encryption, which has no clean fix for external tools. Added a `--cookies PATH` flag as the reliable alternative: export a Netscape-format `cookies.txt` via a browser extension (e.g. "Get cookies.txt LOCALLY") instead — that goes through Chrome's own extension API, not external DB decryption, so it isn't affected by either issue.
- **Single-threaded was far too slow for 8,000 videos (found 2026-08-22):** a real school-computer run averaged ~50s/video, which extrapolates to ~5 days for the full run. Added `--workers` (default 3, `ThreadPoolExecutor`) to process videos concurrently, with submissions staggered by `--pacing/--workers` so `workers` requests don't all fire in one burst (the exact pattern that triggered the bot-check above). Whisper inference is still serialized behind a lock (`_whisper_lock`) — one shared model instance, since loading a `small.en` copy per worker would cost too much RAM — so concurrency mainly overlaps downloads/frame-extraction/visual-features across videos, not Whisper transcription itself. Verified correct (no output corruption, genuine overlap, not just interleaved logging) on a real 3-video concurrent run before shipping.
