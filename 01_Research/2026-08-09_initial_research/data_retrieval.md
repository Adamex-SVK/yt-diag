# Data Retrieval Research — YT-Diag

_Research date: 2026-08-09. Sources: official YouTube Data API docs, yt-dlp README, OpenAI Whisper README, decord README, Wikipedia._

---

## 1. YouTube Data API v3 — Practical Collection Strategy

### 1.1 Quota Model (as of 2026)

The YouTube Data API v3 uses a **quota-unit** system. Every API request costs a certain number of units, deducted from a daily allocation.

**Default free quota:**
- **10,000 units/day** for general endpoints (all list/read operations)
- **100 `search.list` calls/day** (separate quota bucket, 1 unit each)
- **100 `videos.insert` calls/day** (separate bucket, 1 unit each)

Daily quotas reset at **midnight Pacific Time (PT)**.

Source: [YouTube Data API Overview — Quota Usage](https://developers.google.com/youtube/v3/getting-started)

### 1.2 Per-Request Quota Costs

From the official [Quota Calculator](https://developers.google.com/youtube/v3/determine_quota_cost):

| Method | Quota Cost |
|--------|-----------|
| `search.list` | 1 (own bucket: 100/day) |
| `videos.list` | 1 |
| `channels.list` | 1 |
| `captions.list` | **50** |
| `captions.download` | **50** (estimated — listed under captions resource costs) |
| `videoCategories.list` | 1 |
| `playlistItems.list` | 1 |

### 1.3 What One Video Costs in Quota

To collect full data for **one video** via the official API:

| Step | Method | Quota Cost |
|------|--------|-----------|
| Discover videos | `search.list` | 1 (separate bucket) |
| Get metadata + stats | `videos.list` (50 videos per call, `part=snippet,contentDetails,statistics,topicDetails`) | 1 |
| Get channel info (subscriber count) | `channels.list` (50 channels per call) | 1 |
| List available captions | `captions.list` | **50** |
| Download caption track | `captions.download` | **~50** (estimated) |

**Total per video: ~102 quota units** (mainly driven by caption operations).

### 1.4 Timeline for 2,000 Videos

- At 10,000 units/day: you can process roughly **~98 videos/day** through the official API
- 2,000 videos ÷ 98/day ≈ **~21 days** of consecutive daily collection
- The `search.list` bucket (100/day) further constrains discovery to ~100 new video IDs/day

**This is a significant bottleneck.** The caption endpoints (`captions.list` at 50 units, `captions.download` at ~50 units) consume the vast majority of quota. Without captions, metadata-only collection costs ~2 units/video, allowing ~5,000 videos/day.

### 1.5 Workaround Strategies

**A. Use yt-dlp instead of the official API (RECOMMENDED)**

`yt-dlp` bypasses the official API quota entirely. It scrapes YouTube's internal Innertube API (the same one the web client uses), which has **no documented quota**. This is the approach used by most research projects and is the practical standard.

```bash
# Extract metadata only (no download)
yt-dlp --dump-json --skip-download "VIDEO_URL"

# Download auto-generated subtitles
yt-dlp --write-auto-subs --sub-langs en --skip-download "VIDEO_URL"

# Download thumbnail
yt-dlp --write-thumbnail --skip-download "VIDEO_URL"
```

yt-dlp can extract: title, description, tags, categories, view_count, like_count, comment_count, duration, upload_date, channel_id, channel_url, channel_follower_count, thumbnails, subtitles, and more — all without touching the official API quota.

Source: [yt-dlp README — Output Template fields](https://github.com/yt-dlp/yt-dlp#output-template)

**B. Multiple API keys**

Creating multiple Google Cloud projects gives multiple 10,000-unit/day quotas. This is technically against ToS for quota circumvention but is commonly practiced in research. For 2,000 videos: 2–3 projects would suffice.

**C. Staggered collection**

Spread collection over weeks. For a university project with a one-month timeline, this is realistic — metadata-only collection can be done in 1 day; the bottleneck is captions and frame downloads.

**D. Request quota extension**

Google offers a [quota extension request form](https://support.google.com/youtube/contact/yt_api_form). Approval is not guaranteed but is documented.

### 1.6 Python Libraries

| Library | Purpose | Notes |
|---------|---------|-------|
| `google-api-python-client` | Official YouTube Data API v3 wrapper | Requires API key, subject to quota |
| `yt-dlp` (Python package) | Metadata extraction, subtitle download, video download | **No API key needed**, bypasses quota. Actively maintained (2026). |
| `youtube-dl` | Predecessor of yt-dlp | **Inactive**. Do not use. |

**Recommendation**: Use `yt-dlp` as a Python library for all metadata and subtitle collection. Use the official API only as a fallback for structured search/discovery.

---

## 2. Filtering and Sampling Strategy

### 2.1 CC License Filtering via the API

The `search.list` endpoint has a **`videoLicense`** parameter:

- `videoLicense=creativeCommon` — returns only CC-licensed videos
- `videoLicense=youtube` — standard YouTube license
- `videoLicense=any` — both (default)

Additionally, `search.list` supports:
- `videoCategoryId` — filter by YouTube category ID (e.g., 22 = People & Blogs/vlogs, 27 = Education, 24 = Entertainment, 26 = Howto & Style)
- `videoDuration` — `short` (<4 min), `medium` (4–20 min), `long` (>20 min)
- `type=video` — return only videos
- `relevanceLanguage=en` — prefer English-language results
- `order=date` / `order=viewCount` / `order=rating` / `order=relevance`

Source: [YouTube Data API — Search.list parameters](https://developers.google.com/youtube/v3/docs/search/list)

**Using yt-dlp for CC filtering**: yt-dlp exposes a `license` field in its output template and can filter via `--match-filters "license ~= 'Creative Commons'"`. However, yt-dlp's search capabilities (`ytsearch:`) do not directly support CC filtering. The most practical approach is:

1. Use the official API `search.list` with `videoLicense=creativeCommon` to discover CC video IDs (100/day, cheap at 1 unit each)
2. Feed those IDs to yt-dlp for full metadata + subtitle extraction

### 2.2 How Many YouTube Videos Are CC-Licensed?

According to the **State of the Commons 2017 report** (cited by Wikipedia): approximately **49 million CC-licensed works on YouTube** as of 2017. This is a substantial corpus — enough to find 2,000 videos without scarcity issues, even with category filtering.

Source: [Wikipedia — Creative Commons](https://en.wikipedia.org/wiki/Creative_Commons) (citing State of the Commons 2017)

**⚠️ Caveat**: The absolute number is large, but the percentage of CC-licensed videos within specific categories (comedy/skits, vlogs, product reviews) may be much smaller. Many CC videos are educational, music, or documentary content. **The proposal's four categories may need adjustment if CC availability within them is too sparse.** This should be empirically tested early.

### 2.3 Sampling Within Categories

**Recommended stratified sampling approach:**

1. For each of the 4 categories, collect **all available CC-licensed video IDs** via `search.list` (can use multiple API keys for this discovery phase; 100 calls/day/key × 50 results/call = 5,000 IDs/day/key)
2. From the pool, randomly sample 500 videos per category
3. Ensure diversity: stratify by upload date (recent vs. older), view count deciles, and channel size

**Bias risks to mitigate:**

| Bias | Mitigation |
|------|-----------|
| Only sampling highly-viewed videos | Sample uniformly across view-count deciles |
| Only sampling recent uploads | Use `publishedAfter`/`publishedBefore` to span multiple years |
| Only English-language content | Add `relevanceLanguage` filter explicitly; decide whether non-English content is in or out |
| "Viral" label imbalance | Since "viral" = top quartile within category, labeling is relative and self-balancing. But ensure enough low-view videos are sampled for the non-viral class. |

### 2.4 Category ID Mapping

YouTube's `videoCategoryId` values (from `videoCategories.list`):

| Category | ID | Notes |
|----------|----|-------|
| Comedy | 23 | aka "Comedy" |
| Howto & Style | 26 | Closest to "tutorials & how-to" |
| People & Blogs | 22 | Closest to "vlogs" |
| Science & Technology | 28 | Could proxy "product reviews" but not perfect. Consider Entertainment (24) as an alternative |

**⚠️ Issue**: "Product reviews" is not a native YouTube category. Most review content is tagged under Science & Technology (28), Entertainment (24), or Howto & Style (26). The team should decide whether to:
- Use a keyword-based search (`ytsearch:"product review"`) for this category instead
- Replace "product reviews" with a native category (e.g., Gaming/20, Music/10)

---

## 3. Frame Extraction

### 3.1 Tools for Frame Extraction

| Tool | Pros | Cons |
|------|------|------|
| **decord** | Fast, hardware-accelerated (GPU via NVDEC), built for DL training, random-access efficient, `get_batch()` for multiple frame indices | Requires building from source for GPU, less actively maintained (last commit ~2022), Python bindings can be finicky |
| **OpenCV (cv2)** | Universal, well-documented, `VideoCapture.set(cv2.CAP_PROP_POS_FRAMES)` for seeking | Slower than decord for random access, decodes sequentially |
| **ffmpeg** (subprocess) | Most reliable, handles any format, can seek to exact timestamps | Subprocess overhead, need to parse output |
| **torchvision.io** | Native PyTorch integration, GPU decoding on CUDA | Limited format support, requires building from source with FFmpeg |

**Recommendation**: Use **decord** if you can install it (CPU version is on PyPI: `pip install decord`). Fall back to **OpenCV** if decord installation is problematic. Both are mature enough for a one-month project.

```python
# decord: efficient batch frame extraction
from decord import VideoReader
import numpy as np

vr = VideoReader('video.mp4')
total_frames = len(vr)
# Sample 20 frames uniformly
indices = np.linspace(0, total_frames - 1, 20, dtype=int)
frames = vr.get_batch(indices)  # shape: (20, H, W, 3)
```

Source: [decord GitHub README](https://github.com/dmlc/decord)

### 3.2 Download vs. Streaming Trade-offs

**Full download approach (recommended for 2,000 videos):**
- Download video once via yt-dlp → extract frames from local file
- Pro: frame extraction is fast and repeatable; no re-download needed for re-extraction
- Con: storage cost (see §3.4)

**Streaming approach:**
- Extract frames during download (pipe yt-dlp output to ffmpeg)
- Pro: no permanent video storage needed
- Con: if frames are the wrong resolution or count, must re-download

### 3.3 Legal Path for CC-Licensed Videos

For CC-licensed videos: **yt-dlp** is the standard tool. CC BY license explicitly permits downloading, copying, and redistribution with attribution. yt-dlp can:

```bash
# Download video (for frame extraction)
yt-dlp -f "bestvideo[height<=720]" -o "%(id)s.%(ext)s" "VIDEO_URL"

# Extract audio only (for Whisper)
yt-dlp -f "bestaudio" -x --audio-format wav "VIDEO_URL"

# Download thumbnail
yt-dlp --write-thumbnail --skip-download "VIDEO_URL"
```

**⚠️ TOS note**: The proposal mentions "full video frame extraction only for CC-licensed videos (YouTube TOS)." This is correct — downloading non-CC videos via yt-dlp violates YouTube's Terms of Service. The project should strictly enforce `license=creativeCommon` filtering before any download.

### 3.4 Storage Estimates

Assuming 720p resolution and 20 frames per video:

- **Per frame** (720p JPEG, medium quality): ~50–150 KB
- **Per video** (20 frames): ~1–3 MB
- **2,000 videos**: ~2–6 GB for frames alone

For full video downloads (temporary, for frame extraction):
- **Per video** (720p, ~10 min average): ~50–150 MB
- **Not all videos need to be stored simultaneously** — process in batches of 100, delete after frame extraction
- **Peak storage**: ~15 GB (100 videos × 150 MB)

**Thumbnails**: ~100–300 KB each → ~200–600 MB for 2,000

**Total dataset size** (frames + thumbnails + metadata + transcripts): **~5–10 GB** — well within any modern laptop/cloud VM.

---

## 4. Transcript Handling

### 4.1 YouTube Auto-Captions Reliability

YouTube auto-captions (ASR) are generated using Google's speech recognition and are available for the **vast majority** of English-language videos. Key facts:

- **Availability**: Auto-captions are generated for almost all videos with clear English speech. The `captions.list` API response includes a `trackKind` field: value `ASR` = auto-generated, `standard` = manually uploaded.
- **Quality**: Generally good for clear, well-enunciated English speech. Degrades with: background noise, overlapping speakers, heavy accents, technical vocabulary, music, non-English words.
- **Language coverage**: English auto-captions are near-universal. Other languages vary widely. The proposal's 4 categories (comedy, tutorials, vlogs, reviews) are likely to have English captions.

Source: [YouTube Data API — Captions resource](https://developers.google.com/youtube/v3/docs/captions)

### 4.2 Whisper as Fallback

**OpenAI Whisper** can transcribe audio locally when auto-captions are missing or poor quality.

| Model | Parameters | VRAM | Relative Speed | Use Case |
|-------|-----------|------|---------------|----------|
| `tiny` / `tiny.en` | 39M | ~1 GB | ~10x | Fastest, lowest accuracy |
| `base` / `base.en` | 74M | ~1 GB | ~7x | Good balance for clean audio |
| `small` / `small.en` | 244M | ~2 GB | ~4x | **Recommended for this project** |
| `medium` / `medium.en` | 769M | ~5 GB | ~2x | High accuracy, slower |
| `large` | 1550M | ~10 GB | 1x | Best accuracy, slowest |
| `turbo` | 809M | ~6 GB | ~8x | Optimized large-v3, fast + accurate |

**⚠️ Important**: `turbo` is **not trained for translation** — only transcription in the original language. For English-only transcription, `turbo` or `small.en` are the best practical choices.

**Latency estimate** (on a laptop GPU, e.g., RTX 3060):
- `small.en`: ~30–60 seconds per 10-minute video (real-time factor ~0.05–0.1x)
- `base.en`: ~15–30 seconds per 10-minute video
- `turbo`: ~10–20 seconds per 10-minute video
- 2,000 videos × 10 min average would take ~8–33 hours on a single GPU for `base.en`/`small.en`

On **CPU-only**: multiply by 3–5x. Still feasible overnight or over a weekend.

Source: [OpenAI Whisper README](https://github.com/openai/whisper#available-models-and-languages)

### 4.3 Decision Heuristic: Auto-Captions vs. Whisper

**When to trust auto-captions:**
- `trackKind = "standard"` (manually uploaded) → always use
- `trackKind = "ASR"` AND language is English AND video has clear solo speech → use
- `trackKind = "ASR"` but `isDraft = true` → captions still processing, use Whisper

**When to fall back to Whisper:**
- No captions available (`captions.list` returns empty)
- Auto-captions exist but are in a non-English language (unless the project wants multilingual transcripts)
- Multiple speakers, heavy background noise, music over speech, or technical jargon (can be detected heuristically: unusually short average caption segment length, or captions with many `[Music]` / `[Applause]` tags)

**Practical hybrid approach:**
1. Use yt-dlp to download auto-captions: `yt-dlp --write-auto-subs --sub-langs en --skip-download --convert-subs srt "URL"`
2. Check caption quality with a simple heuristic (e.g., word count > 50, no more than 20% `[Music]`/`[Applause]` tags)
3. If captions fail the heuristic, run Whisper `small.en` on the audio track

This avoids running Whisper on all 2,000 videos — likely only 10–30% will need it.

---

## 5. Metadata Features

### 5.1 Which Features Are Available From a Single API Call?

A single `videos.list` call with `part=snippet,contentDetails,statistics,topicDetails,status` returns:

| # | Feature | Part | Field | Directly Available? |
|---|---------|------|-------|--------------------|
| 1 | Title | snippet | `title` | ✅ Yes |
| 2 | Description | snippet | `description` | ✅ Yes |
| 3 | Published date | snippet | `publishedAt` | ✅ Yes |
| 4 | Category ID | snippet | `categoryId` | ✅ Yes |
| 5 | Tags | snippet | `tags[]` | ✅ Yes |
| 6 | Thumbnail URLs | snippet | `thumbnails` | ✅ Yes (URLs only — actual images fetched separately) |
| 7 | Duration | contentDetails | `duration` | ✅ Yes (ISO 8601 format) |
| 8 | Definition (HD/SD) | contentDetails | `definition` | ✅ Yes |
| 9 | Caption availability | contentDetails | `caption` | ✅ Yes (boolean) |
| 10 | View count | statistics | `viewCount` | ✅ Yes |
| 11 | Like count | statistics | `likeCount` | ✅ Yes |
| 12 | Comment count | statistics | `commentCount` | ✅ Yes |

### 5.2 Features Requiring Additional Calls

| Feature | Additional Call Needed | Quota Cost |
|---------|----------------------|------------|
| Channel subscriber count | `channels.list` with `part=statistics` | 1 unit (50 channels/call) |
| Channel total video count | Same as above | Included |
| Channel country | `channels.list` with `part=snippet` | 1 unit |
| Video category name (human-readable) | `videoCategories.list` | 1 unit (one-time, cached) |
| Caption tracks | `captions.list` | 50 units per video |
| Actual caption text | `captions.download` | ~50 units per video |

### 5.3 Undocumented or Rate-Limited Fields

- **`dislikeCount`**: The `statistics.dislikeCount` field still exists in the API response but always returns `0` since YouTube removed public dislike counts in December 2021.
- **`favoriteCount`**: Deprecated, always returns `0`.
- **Exact subscriber count**: Returned as an integer but truncated for display on youtube.com (e.g., "1.2M"). The API returns the exact count.
- **`topicDetails`**: Returns Freebase topic IDs — useful for content categorization but Freebase is deprecated. The `topicCategories[]` field still provides useful broad categories.

### 5.4 The 12 Metadata Features — Proposed Mapping

Based on the proposal's "12 structured metadata features," here is a likely mapping to API fields:

| # | Likely Feature | API Field | Status |
|---|---------------|-----------|--------|
| 1 | View count | `statistics.viewCount` | ✅ Direct |
| 2 | Like count | `statistics.likeCount` | ✅ Direct |
| 3 | Comment count | `statistics.commentCount` | ✅ Direct |
| 4 | Video duration | `contentDetails.duration` | ✅ Direct |
| 5 | Upload date/time | `snippet.publishedAt` | ✅ Direct |
| 6 | Title length | Derived from `snippet.title` | ✅ Derived |
| 7 | Description length | Derived from `snippet.description` | ✅ Derived |
| 8 | Tag count | `len(snippet.tags[])` | ✅ Derived |
| 9 | HD/SD | `contentDetails.definition` | ✅ Direct |
| 10 | Category | `snippet.categoryId` | ✅ Direct |
| 11 | Subscriber count | `channels.list` → `statistics.subscriberCount` | ⚠️ Extra call |
| 12 | Caption availability | `contentDetails.caption` | ✅ Direct |

**Note**: Subscriber count (feature #11) requires one extra `channels.list` call per batch of 50 channels — negligible quota cost (~1 unit per 50 videos).

---

## Practical Recommendations

### Priority 1: Choose yt-dlp as Primary Collection Tool (immediate)

The official API quota model makes collecting 2,000 videos with captions impractical in a one-month timeline. **Use yt-dlp's Python API** for all metadata + subtitle extraction. It exposes `view_count`, `like_count`, `comment_count`, `duration`, `upload_date`, `channel_follower_count`, `license`, `tags`, `categories`, `description`, `title`, `thumbnails`, `subtitles`, and more — all without quota.

### Priority 2: Test CC Availability in Target Categories (this week)

Before committing to the four categories, run a quick scan:
1. Use the official API `search.list` (100 calls/day is free) with `videoLicense=creativeCommon` + `videoCategoryId=23` (Comedy), `26` (Howto), `22` (People & Blogs), and either `28` (Science) or keyword-based `q=product review`
2. Check how many CC results exist per category. If any category has <500 CC videos, adjust scope.

### Priority 3: Decide "Product Reviews" Category Mapping

"Product reviews" is not a native YouTube category. Options:
- **Keyword search**: `q=product review` + CC filter — but search quality varies
- **Switch to Science & Technology (28)** or **Entertainment (24)** — natively filterable
- **Drop to 3 categories** with 667 videos each — simpler, still valid for a DL course project

### Priority 4: Build the Collection Pipeline (next week)

Suggested pipeline architecture:
```
1. Discovery: Official API search.list → collect CC video IDs per category
2. Metadata: yt-dlp (Python) → extract all metadata + subtitle availability
3. Subtitles: yt-dlp --write-auto-subs OR yt-dlp --write-subs → SRT files
4. Quality check: Heuristic → Whisper small.en fallback for poor/no captions
5. Thumbnails: yt-dlp --write-thumbnail OR direct HTTP download from thumbnail URL
6. Video download: yt-dlp -f "bestvideo[height<=720]" → local file
7. Frame extraction: decord VideoReader.get_batch() → 20 uniformly sampled frames
8. Cleanup: Delete video files after frame extraction
```

### Priority 5: Set Up Whisper (as needed, not upfront)

Install `openai-whisper` + `ffmpeg`. Use `small.en` model. Have it ready but only invoke it when auto-captions are missing or flagged as poor quality. Running Whisper on all 2,000 videos is overkill for a course project and will consume significant compute time.

### ⚠️ Key Unknowns & Risks

1. **CC video availability in entertainment categories**: Comedy and vlog categories may have very few CC-licensed entries. Verify empirically before committing.
2. **yt-dlp reliability**: YouTube actively changes its internal APIs. yt-dlp is actively maintained (last release: July 2026) but can break without warning. Have the official API as fallback for metadata at minimum.
3. **Frame extraction for CC-only constraint**: If CC videos are predominantly short, low-production-value content, the visual modality may add less signal than anticipated. Validate with a small sample.
4. **Category ID mismatch**: YouTube's category system doesn't cleanly map to the proposal's four categories. This may introduce noise in the labeling — be transparent about this in the final report.
5. **Language consistency**: If auto-captions exist in multiple languages for the same video, decide on a consistent policy (e.g., always use English, or use the video's declared primary language).

---

_Research conducted by AI agent. Sources cited inline. Last updated: 2026-08-09._
