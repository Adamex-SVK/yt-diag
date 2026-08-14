# Compute Scaling — What Changes With an A40 + 96 Cores

_Research date: 2026-08-14. Context: the original architecture research (`../2026-08-09_initial_research/model_architectures.md`) was designed around a laptop/Colab-class GPU (6–8 GB VRAM) and a 2,000-video dataset. We now have an NVIDIA A40 (48 GB VRAM), 96 CPU cores, 1.5 TB RAM, 7 TB SSD, plus the option of renting additional cloud GPU time. This note re-evaluates every "we can't afford this" constraint from the Aug 9–10 research against the new hardware._

---

## 1. What the original constraints actually were

Re-reading `model_architectures.md` and `evaluation_and_planning.md`, every conservative call traced back to one of two limits, not to a belief that bigger was worse:

| Original limit | Where it showed up | Still a limit on this hardware? |
|---|---|---|
| ≤8 GB VRAM | Frozen backbones only, ViT-S not ViT-B/L, no TimeSformer, batch size capped | **No** — A40 has 48 GB, ~6–16x headroom |
| 2,000-video collection time budget | Sample size, overfitting risk of fine-tuning | **Partially** — collection speed is now a parallelization problem the 96 cores solve; CC-license *availability* per category is still an external constraint, not a compute one |
| 1-month wall-clock, 2 students, 10–20 hrs/week each | Ablation count, hyperparameter search depth, temporal model choice | **Yes, still binding** — more compute doesn't buy more calendar time. This is the real remaining bottleneck. |

The upshot: hardware removes the *architecture* excuses (frozen backbones, small ViT, no cross-attention, no TimeSformer) and the *data-volume* excuse (assuming CC videos exist). It does not remove the *time* excuse — every upgrade below still has to fit inside four weeks, so they're ordered by ROI, not just by what's technically possible.

## 2. Sample size: the highest-ROI upgrade, and it's a pipeline problem now

The 2,000-video cap came from `data_retrieval.md`'s estimate that collection (yt-dlp downloads, frame extraction via decord, Whisper fallback for ~10–30% of videos) was the Week 1 bottleneck — not GPU capacity. That estimate assumed sequential or lightly-parallel processing on a laptop.

**With 96 cores:**
- yt-dlp metadata/subtitle/thumbnail collection parallelizes near-linearly (it's I/O + light CPU bound) — a `multiprocessing.Pool(64)` or `asyncio` fan-out turns a multi-day sequential run into hours.
- decord frame extraction is CPU-bound per video and embarrassingly parallel across videos.
- Whisper `small.en` fallback (needed for ~10–30% of videos per the original estimate) was quoted at 8–33 hours single-GPU for 2,000 videos; on CPU it was estimated at 3–5x that. On the A40, batched Whisper inference (or `faster-whisper` with CTranslate2) cuts this to a few hours even at 5,000–8,000 videos, and it isn't competing with model training for the same GPU if scheduled first.

**The real remaining constraint is CC-license density per category** (Wikipedia/State of the Commons cites ~49M CC videos on YouTube overall, but density inside "comedy," "vlogs," etc. is unverified — `data_retrieval.md` §2.2 flagged this as untested). Compute cannot manufacture more CC-licensed comedy videos.

**Action, not just capacity:** rerun the CC-availability scan (`MILESTONES.md` immediate task #4) now, per category, and size the target dataset to what's actually available — realistically **5,000–8,000 videos** if density holds, versus the original 2,000. This directly benefits every downstream choice in §3, since more data is what makes fine-tuning safe.

## 3. Fine-tune instead of freeze

`model_architectures.md` §4.3 was explicit about *why* it recommended frozen backbones + a ~3M-parameter trainable fusion head: fine-tuning encoders means 200M+ trainable parameters, and "likely won't beat frozen encoders without more data." That was a data-size argument, not a compute-capability argument — and both premises just changed (more VRAM, likely more data from §2).

Recommended change from Tier 1 → a Tier 2/3 hybrid, skippable in one step rather than incrementally:

| Component | Tier 1 (original default) | Upgraded |
|---|---|---|
| Text encoder | ModernBERT-base, frozen | ModernBERT-base **or large**, LoRA fine-tuned (adds ~1–2% params per `model_architectures.md` §4.3, not full fine-tune — still avoid full FT even with headroom, since it's the least sample-efficient option per that section) |
| Vision backbone | DINOv2 ViT-S/14, frozen | DINOv2 ViT-B/14 (or L/14 if the dataset lands at the high end), fine-tuned last few blocks |
| Temporal | Same DINOv2 + attention pooling | TimeSformer (Bertasius et al. 2021, pretrained on Kinetics-400), fine-tune last layers — this was Tier 3 ("only if weeks 1–2 go perfectly") specifically because of GPU memory and training-time risk on the original hardware. That risk is largely gone on an A40. |
| Fusion | Late fusion, concat + MLP | Hybrid: late fusion **plus** one cross-attention layer (text ↔ vision) before the classification head — the Tier 2 upgrade path already specified in `model_architectures.md` §7.1 |

GPU memory reality check, scaled from the original estimates in §6.1 of `model_architectures.md` (~3–5 GB frozen, ~6–8 GB fine-tuned on a 6–8 GB budget): moving to fine-tuned ModernBERT-base/large + fine-tuned DINOv2 ViT-B + TimeSformer-B (~121M params) sits comfortably in the 15–25 GB range even with generous batch sizes and mixed precision — well inside the A40's 48 GB, leaving room to avoid gradient accumulation and to keep iteration fast.

## 4. Frame density

16–24 frames was sized for GPU memory and encoding time on constrained hardware, not because more frames stop helping (`model_architectures.md` §3.3 notes 96 frames is only marginal over 8–16 for many tasks, but that finding is from TimeSformer's own ablations at whatever compute *they* had — not a hard ceiling for this project). Given the "first 30 seconds carry disproportionate signal" finding (Rajaram & Manchanda 2020, cited in §3.3), a denser hybrid schedule is now cheap to run:

- **Upgraded sampling:** 16 frames in the first 60 seconds + 32 uniformly across the remainder (vs. the original 8+8–16 split), pre-computed and cached per the existing recommendation (`model_architectures.md` §7.2 table) so it's a one-time cost, not a per-epoch one.

## 5. Evaluation rigor

`evaluation_and_planning.md` sized validation around what was affordable: 5-fold CV, four priority ablations, one hyperparameter pass. With 96 cores / 1.5 TB RAM, the *evaluation* side (not just training) stops being cost-constrained:

- Run the two "nice to have if time permits" ablations from §4.4 (temporal-only removal, thumbnail-only removal) in addition to the four required ones — they were deprioritized for time, not because they're uninformative.
- Add the two supplementary baselines flagged in §2.3 as "medium priority": Random Forest and an MLP-on-metadata-only model, both essentially free on this hardware and useful for isolating "deep architecture helps" from "multimodal signal helps."
- Run a real hyperparameter sweep (Optuna or W&B Sweeps, 50–100+ trials in parallel across CPU/GPU-light configs) instead of a single manual pass — this was implicitly out of scope before given the 1-month/2-person time budget interacting with slow iteration; faster training on the A40 makes more trials fit in the same calendar time.
- Swap single-baseline Integrated Gradients for **Expected Gradients** (multiple baselines drawn from the training distribution) — `model_architectures.md` §7.1 listed this as a Tier 3 item specifically because IG with many baselines multiplies compute cost; that cost is no longer the constraint.

## 6. When cloud GPU is actually worth it

The A40 alone covers everything in §3–5. Reach for rented cloud GPU only for:
- **Multi-GPU data-parallel training** to cut wall-clock during the hyperparameter sweep (§5) — the sweep is the part most likely to be time-constrained, not memory-constrained.
- **Larger backbone experiments** (ModernBERT-large + DINOv2 ViT-L + TimeSformer-L simultaneously, large batch) if the 48 GB starts to pinch — unlikely at the parameter counts in §3, but worth knowing the ceiling exists.

Don't rent cloud GPU for baseline training, frozen-backbone inference, or feature extraction (§7 in the companion note) — the A40 and 96 CPU cores are already overkill for those.

## 7. Revised realistic targets

`evaluation_and_planning.md` §5.3 rated "Excellent" (AUC-ROC > 0.78) as **low confidence**, explicitly because of the 2,000-video ceiling and an external noise floor that no architecture change removes (algorithmic promotion, platform luck, timing — listed in §1.3). That noise floor is unchanged by better hardware. What changes is everything *upstream* of it:

| Target | Original confidence (2,000 videos, frozen Tier 1) | Revised confidence (5–8k videos, fine-tuned Tier 2/3) |
|---|---|---|
| Minimum viable (>0.65) | High | High |
| Good (0.70–0.78) | Medium | High |
| Excellent (>0.78) | Low | **Medium** — plausible with more data + fine-tuning, still capped by the same external-noise ceiling the original research identified |

**The framing from `evaluation_and_planning.md` §5.3 still holds and should stay in the report regardless of outcome: the gap between the full model and the metadata-only baseline is the real result, not the absolute AUC number.** More compute makes a bigger gap more achievable; it doesn't change what the honest headline claim is.

---

_Companion note: `additional_features.md` (candidate visual and audio features enabled by the same compute headroom). Source docs: `../2026-08-09_initial_research/{model_architectures,data_retrieval,evaluation_and_planning}.md`._
