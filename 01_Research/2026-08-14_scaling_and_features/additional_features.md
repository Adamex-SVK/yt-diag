# Additional Engineered Features — Visual & Prosodic

_Research date: 2026-08-14. Prompted by Adam's idea to look beyond raw pixel/text embeddings: color temperature and face presence/size in frames, and voice modulation (pauses, volume, pitch, tempo) in the audio track. This note scopes those into concrete, extractable features, picks tools, and places them in the existing pipeline from `../2026-08-09_initial_research/`._

---

## 0. Why this is worth doing (and where it fits)

Everything in `model_architectures.md` treats vision and audio as opaque embeddings (DINOv2 features, no audio modality at all — the original proposal only used text, thumbnail, and frames). Two gaps that this note addresses:

1. **No audio modality exists yet.** Rajaram & Manchanda (2020) — the closest comparable system — explicitly model "how it is said" through audio and found it contributes less than text but more than nothing. YT-Diag currently has no equivalent signal at all. Voice modulation features are a cheap way to add it without needing a full audio deep-learning pipeline.
2. **Embeddings aren't interpretable.** DINOv2 features going into a 256-dim projection give the attribution layer (Integrated Gradients) something to attribute *importance* to, but not something a human reads as "the thumbnail has a close-up face." Explicit, named features (face-area %, color temperature, pause ratio) are directly usable in the metadata-only XGBoost/LR baselines (`evaluation_and_planning.md` §2) — where they add real predictive and interpretable signal — and serve as a sanity check on the deep model's attribution output: if Integrated Gradients flags "thumbnail" as important, does that track with an explicit face-size feature also being important in the tree-model's feature importance?

These are **structured features**, not a fourth deep-learning modality. They plug into the existing 12-metadata-feature baseline (`evaluation_and_planning.md` §2.1) as additional columns, and optionally into the deep model as a fourth projected input alongside text/thumbnail/temporal.

Compute cost is trivial relative to §3–5 of `compute_scaling.md` — this is CPU-bound, embarrassingly parallel work well suited to the 96 cores, not something that competes with GPU training time.

---

## 1. Visual features (per sampled frame + thumbnail)

### 1.1 Color temperature

**What it measures:** whether a frame/thumbnail reads as "warm" (orange/red-shifted, common in vlog/lifestyle thumbnails) or "cool" (blue-shifted, common in tech/gaming). Thumbnail color grading is a known lever in the creator community for grabbing attention — this operationalizes it as a number instead of leaving it inside an opaque embedding.

**How to compute:** convert average frame RGB to correlated color temperature (CCT) via McCamy's approximation from CIE chromaticity coordinates — a standard, well-documented formula, implementable in a few lines with `colour-science` (Python) or `opencv` + manual xy-chromaticity conversion. No pretrained model needed.

**Features per video:**
- Thumbnail CCT (single value)
- Mean/std of CCT across sampled frames (captures whether color grading is consistent or shifts, e.g., a dim intro vs. a bright main segment)

### 1.2 Brightness, saturation, contrast

Cheap complements to color temperature, same computation path (HSV histogram stats via OpenCV/PIL, no model needed):
- Mean brightness (V channel), mean saturation (S channel)
- Contrast (std of luminance)
- These are established correlates of thumbnail CTR in the creator-analytics literature (bright, high-saturation thumbnails tend to outperform — worth testing as a hypothesis here, not asserting as fact).

### 1.3 Face presence and face-area ratio

**What it measures:** whether a face is present, how many, and what fraction of the frame the largest face occupies — "big face close-up" is one of the most cited thumbnail conventions among YouTube creators.

**Tooling options (ranked by fit for this project):**

| Tool | Type | Notes |
|---|---|---|
| **MediaPipe Face Detection** | Lightweight CNN (BlazeFace) | Fast on CPU (~10 ms/frame), no GPU needed, good for running across 96 cores in parallel, permissive license, actively maintained by Google |
| **RetinaFace** | Deeper detector | More accurate on small/angled faces, heavier, better run on the A40 rather than CPU if used at scale |
| OpenCV Haar cascade | Classical | Fast but noticeably less accurate; fallback only if the above are unavailable |

**Recommendation:** MediaPipe Face Detection for the full dataset (CPU-parallel, trivial cost at 5,000–8,000 videos × ~20 frames), since accuracy needs here are "is there a face and how big," not precise landmark localization.

**Features per video:**
- `has_face` (thumbnail, boolean)
- `face_count` (thumbnail and per-frame)
- `max_face_area_ratio` = largest detected face bounding-box area ÷ frame area (thumbnail and mean across sampled frames)
- `face_centrality` = distance of the largest face's bounding-box center from the frame center, normalized — captures whether the face is centered (common convention) or off to a side

**Optional stretch (only if time allows, per the same P0/P1/P2 discipline as `MILESTONES.md`):** facial expression/emotion via a lightweight pretrained SER-style classifier (e.g., a small FER model) — "smiling/surprised face" is a stronger thumbnail-psychology hypothesis than presence alone, but adds a second model dependency and labeling ambiguity. Treat as P2.

### 1.4 Motion / editing pace (cheap bonus, same frame set)

Since frames are already sampled and cached (`model_architectures.md` §7.2 recommends pre-computing frame embeddings — this reuses the same frames):
- Frame-to-frame pixel difference / histogram difference across the sampled frames, as a rough proxy for cut frequency / edit pace. Fast-paced editing is a plausible virality correlate independent of what's actually shown.

---

## 2. Audio / prosodic features

None of this requires transcript accuracy — it operates on the raw audio waveform, so it's independent of the Whisper/auto-caption transcript pipeline in `data_retrieval.md` §4. Extract once per video from the audio track already being pulled for transcription (no extra download).

### 2.1 Pipeline

```
1. Audio already extracted via yt-dlp -f "bestaudio" -x --audio-format wav (data_retrieval.md §3.3)
2. Voice activity detection (VAD) → speech vs. silence segments
3. Pitch (F0) tracking on speech segments
4. Loudness/energy tracking
5. Aggregate to per-video summary statistics
```

### 2.2 Tooling

| Task | Tool | Notes |
|---|---|---|
| Voice activity detection (pauses) | `webrtcvad` (fast, robust) or `librosa.effects.split` (energy-based) | webrtcvad is purpose-built for this, very fast, CPU-only |
| Pitch (F0) extraction | `praat-parselmouth` (Python bindings for Praat) or `librosa.pyin` | Praat/parselmouth is the standard in speech-prosody research; pYIN is a solid pure-Python fallback |
| Loudness / energy | `librosa` (RMS energy) | Straightforward |
| Combined feature suites (alternative to hand-rolling the above) | `openSMILE` (via `opensmile-python`) or `pyAudioAnalysis` | Off-the-shelf prosodic/paralinguistic feature sets (e.g., openSMILE's eGeMAPS set is specifically designed for exactly this: pitch, loudness, pauses, spectral features, validated in affective-computing literature) — worth using directly instead of hand-computing each feature separately, given the time budget |

**Recommendation:** use `opensmile-python` with the eGeMAPS (Extended Geneva Minimalistic Acoustic Parameter Set) feature set as the primary path — it's a published, standardized 88-feature set covering pitch, loudness, spectral balance, and temporal (pause/voicing) statistics in one call, purpose-built for exactly "voice modulation" analysis, and saves reimplementing pitch/pause extraction by hand. Use `webrtcvad` + `librosa` only for the specific hand-picked features below if eGeMAPS's output isn't granular enough or for a quick first pass.

### 2.3 Features per video

**Pauses:**
- Pause count (silence segments above a duration threshold, e.g., >300 ms)
- Total pause duration, and pause ratio = silence time ÷ total duration
- Mean/std of pause length — regular short pauses (rehearsed pacing) vs. occasional long pauses (hesitation, or dramatic effect) are different signals

**Volume / loudness:**
- Mean RMS energy, std of RMS energy (dynamic range — flat delivery vs. expressive)
- Loudness range (max − min over a sliding window)

**Pitch (F0):**
- Mean F0, std of F0 (pitch variability — monotone vs. expressive delivery)
- F0 range (max − min)
- Voiced-frame ratio (fraction of speech that has detectable pitch, vs. unvoiced/noisy)

**Tempo / speaking rate:**
- Speaking rate proxy: voiced-segment count per minute, or (if transcript word timestamps are available from Whisper/auto-captions) actual words-per-minute — the latter is more accurate and effectively free since transcripts are already being collected
- Speaking-rate variability across the video (fast intro vs. slower explanation, or the reverse) — compute WPM in rolling windows and take the std

**Derived "modulation" summary (directly answers Adam's framing):**
- A single "expressiveness" composite (e.g., z-scored combination of pitch std, loudness std, and speaking-rate variability) is tempting but resist collapsing to one number pre-analysis — keep the components separate for the baseline model and ablations; a composite can be built later during EDA if the components turn out correlated.

---

## 3. Where these plug into the existing plan

| Destination | How |
|---|---|
| Metadata-only baseline (`evaluation_and_planning.md` §2) | Append as additional columns to the existing 12 structured features → immediately testable in LR/XGBoost/Random Forest without touching the deep model at all. **Cheapest way to find out if these features matter before investing in deep-model integration.** |
| Deep model (`model_architectures.md` §4) | Optional 4th modality: project the feature vector (visual stats + audio stats, ~20–30 dims total) through a small Linear+LN layer like the other three modalities, concatenate into the fusion step |
| Ablations (`evaluation_and_planning.md` §4) | A natural 5th ablation: full model vs. –engineered-features, isolating whether hand-crafted interpretable features add anything beyond what DINOv2/ModernBERT embeddings already implicitly capture |
| Attribution layer (`model_architectures.md` §5) | Cross-check: do Integrated Gradients attributions on the thumbnail modality correlate with the explicit face-area/color-temperature features being highly weighted in the metadata baseline? Agreement adds credibility to both. |

## 4. Effort and priority

Given the same P0/P1/P2 discipline as `MILESTONES.md`, this is **not** on the critical path (data collection → baselines → deep model still comes first) but is cheap enough to run in parallel once metadata + frames + audio are collected:

- **P1 (do if the metadata baseline shows headroom):** color temperature, brightness/saturation, face presence/area — all CPU-only, minutes of runtime even at 8,000 videos across 96 cores.
- **P1:** eGeMAPS audio features via openSMILE — one call per video, batchable.
- **P2:** motion/edit-pace proxy, facial expression classification, WPM-from-transcript speaking rate (needs transcript timestamps, slightly more plumbing).

None of these require GPU time — they're a good task to run on CPU *while* the A40 is busy training the deep model, not a sequential add-on.

---

## References

- Rajaram, P., & Manchanda, P. (2020). Unboxing Engagement in YouTube Influencer Videos: An Attention-Based Approach. _arXiv:2012.12311_. (Motivates modeling "how it is said" via audio.)
- Eyben, F., et al. (2016). The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing. _IEEE Transactions on Affective Computing_. (eGeMAPS feature set, implemented in openSMILE.)
- McCamy, C. S. (1992). Correlated color temperature as an explicit function of chromaticity coordinates. _Color Research & Application_. (CCT approximation formula.)
- Bazarevsky, V., et al. (2019). BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs. _arXiv:1907.05047_. (MediaPipe's face detector.)

---

_Companion note: `compute_scaling.md` (hardware upgrade path). Source docs: `../2026-08-09_initial_research/{model_architectures,data_retrieval,evaluation_and_planning}.md`._
