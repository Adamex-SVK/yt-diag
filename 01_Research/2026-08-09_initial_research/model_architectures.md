# Model Architectures — YT-Diag

_Research date: 2026-08-10. Sources: arXiv, HuggingFace blog, Distill.pub, Captum docs, project proposal references._

---

## Table of Contents

1. [Text Encoder](#1-text-encoder)
2. [Vision Backbone (Thumbnail)](#2-vision-backbone-thumbnail)
3. [Temporal Transformer (Sampled Frames)](#3-temporal-transformer-sampled-frames)
4. [Fusion Strategy](#4-fusion-strategy)
5. [Attribution & Explainability](#5-attribution--explainability)
6. [Practical Feasibility & Resource Budget](#6-practical-feasibility--resource-budget)
7. [Practical Recommendations](#7-practical-recommendations)

---

## 1. Text Encoder

### 1.1 Candidates

| Model | Params | Context Window | Key Characteristics | Source |
|-------|--------|---------------|---------------------|--------|
| **BERT-base** | 110M | 512 tokens | Original workhorse; strong bidirectional representations but slow and context-limited | Devlin et al. 2018 |
| **RoBERTa-base** | 125M | 512 tokens | BERT with better training recipe (dynamic masking, no NSP, more data); ~2–5% better on GLUE | Liu et al. 2019, [arXiv:1907.11692](https://arxiv.org/abs/1907.11692) |
| **DistilBERT** | 66M | 512 tokens | Distilled from BERT; 40% smaller, 60% faster, retains 97% of BERT's performance | Sanh et al. 2019, [arXiv:1910.01108](https://arxiv.org/abs/1910.01108) |
| **ModernBERT-base** | 149M | **8,192 tokens** | Pareto-improvement over all previous encoders; RoPE, GeGLU, alternating global/local attention; 2x faster than DeBERTaV3, 4x on mixed-length inputs; trained on 2T tokens including code | Warner et al. 2024, [arXiv:2412.13663](https://arxiv.org/abs/2412.13663); [HF blog](https://huggingface.co/blog/modernbert) |
| **ModernBERT-large** | 395M | 8,192 tokens | Larger variant; ~2.6x params of base for marginal classification gains | Warner et al. 2024 |

### 1.2 Analysis

**For ~2,000 videos, the text modality is the most important signal.** Rajaram & Manchanda (2020) found that _"what is said" through words (text) is more important than "how it is said" through imagery (video images) or acoustics (audio)_ for predicting video engagement. This matches intuition for YouTube: title, description, and transcript carry dense semantic information about topic, style, and audience intent.

**Should title, description, and transcript share one encoder?** Yes — share one encoder. Three separate encoders add parameters and complexity without clear benefit for a 2,000-sample dataset. Instead, concatenate the three text fields with separator tokens (`[SEP]`) and pass through a single encoder. This is the standard approach in multimodal video understanding (used in VideoBERT, UniVL, and others). The model learns to attend across fields via the shared attention mechanism.

**How to handle long transcripts?** YouTube transcripts can run 2,000–10,000+ words (3,000–15,000 tokens). This exceeds BERT/RoBERTa/DistilBERT's 512-token limit. Options:

1. **ModernBERT's 8,192-token context (RECOMMENDED):** ModernBERT natively handles up to 8,192 tokens — enough for most transcripts. Its alternating global/local attention makes long-context inference ~3x faster than other long-context encoders. This eliminates chunking entirely for ~90%+ of transcripts. For videos with very long transcripts, truncate to 8,192 tokens (roughly ~6,000 words), keeping the beginning and end (intro/conclusion are usually the most informative).

2. **Chunk + mean-pool (fallback for BERT/RoBERTa):** Split transcript into overlapping 512-token chunks, encode each independently, mean-pool the `[CLS]` embeddings. This loses cross-chunk context but works adequately.

3. **Hierarchical: segment-level encoding → document-level pooling.** More complex; not worth it for 2,000 samples.

### 1.3 Recommendation

**ModernBERT-base** is the clear winner for this project:
- **8,192-token context** solves the transcript-length problem without chunking hacks
- **Better downstream accuracy** than DeBERTaV3 on classification while using <1/5th the memory
- **2–4x faster inference** than comparable encoders (critical for iteration speed in a 1-month project)
- **149M params** is manageable — roughly the same ballpark as BERT-base (110M)
- **Drop-in replacement** for any BERT pipeline via HuggingFace Transformers

If GPU memory is extremely tight, **DistilBERT** (66M params, 97% of BERT performance) is the pragmatic fallback — but you'll need to chunk transcripts.

---

## 2. Vision Backbone (Thumbnail)

### 2.1 Candidates

| Model | Params | Embedding Dim | Key Characteristics | Source |
|-------|--------|---------------|---------------------|--------|
| **ResNet-50** | 25.6M | 2048 | ConvNet workhorse; ImageNet pretrained; fast, well-understood | He et al. 2016 |
| **ViT-B/16** | 86M | 768 | Patch-based transformer; strong on ImageNet; needs more data to shine | Dosovitskiy et al. 2020 |
| **CLIP ViT-B/32** | 88M | 512 | Trained on 400M (image, text) pairs via contrastive learning; text-image aligned embedding space | Radford et al. 2021, [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) |
| **DINOv2 ViT-S/14** | 21M | 384 | Self-supervised; SOTA frozen features across diverse tasks; distilled from 1B-param teacher | Oquab et al. 2023, [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) |
| **DINOv2 ViT-B/14** | 86M | 768 | Larger DINOv2 variant; stronger features but heavier | Oquab et al. 2023 |

### 2.2 Is CLIP's Text+Vision Bridge Useful Here?

**Yes, but not in the most obvious way.** CLIP's aligned text-vision embedding space is interesting because YT-Diag already has text and vision modalities. In theory, CLIP could produce embeddings for both modalities in a shared space, simplifying fusion. However:

- **The text modality is better served by a dedicated text encoder** (ModernBERT). CLIP's text encoder is a GPT-2-style autoregressive model — weaker for classification than bidirectional encoders.
- **CLIP's vision encoder alone is still a strong thumbnail encoder** because it was trained to produce semantically rich features (what the thumbnail "means," not just what's in it).
- DINOv2 generally produces **better frozen features for downstream tasks** than CLIP, because its self-supervised training explicitly optimizes for feature quality rather than cross-modal alignment.

**Practical take:** Use CLIP ViT if you want to experiment with shared embedding spaces; use DINOv2 if you want the strongest frozen visual features.

### 2.3 Fine-tune or Freeze?

For **~2,000 videos**, the safest approach is to **freeze the vision backbone** and only train a lightweight projection head. Fine-tuning a vision transformer on 2,000 samples risks overfitting unless aggressive regularization is applied. This is the standard practice for small datasets in multimodal settings.

If you do want to fine-tune: use a low learning rate (1e-5 or lower), strong weight decay, and LoRA adapters rather than full fine-tuning.

### 2.4 Recommendation

**DINOv2 ViT-S/14 (21M params)** — the pragmatic choice:
- **Smallest** of the strong options (21M vs 86M for ViT-B variants)
- **Best frozen features** across diverse benchmarks (surpasses OpenCLIP)
- **384-dim embeddings** are compact, reducing fusion-head parameter count
- If thumbnail content turns out to matter more than expected, swap to DINOv2 ViT-B/14 (86M, 768-dim)

**Fallback:** ResNet-50 if simplicity is paramount (25M params, well-understood, every framework supports it).

---

## 3. Temporal Transformer (Sampled Frames)

### 3.1 How YouTube-8M Works (Abu-El-Haija et al. 2016)

The YouTube-8M baseline approach is conceptually simple:

1. **Sample frames** at 1 frame-per-second across the video
2. **Extract frame-level features** using a pretrained ImageNet CNN (Inception), taking the hidden representation before the classification layer
3. **Aggregate** frame features via pooling (mean-pool, max-pool, or attention-pool) into a single video-level embedding
4. **Classify** the pooled embedding with a shallow MLP

This is **not** what YT-Diag needs — YouTube-8M's frame-level CNN approach was designed for a dataset of 8M videos where end-to-end training was infeasible. With 2,000 videos and modern hardware, we can train something more expressive.

What YouTube-8M contributes to YT-Diag is the **sampling strategy**: uniform frame sampling at ~1 fps is a validated, simple baseline that avoids the complexity of keyframe detection.

### 3.2 Alternatives

| Model | Approach | Frames | Key Characteristics | Source |
|-------|----------|--------|---------------------|--------|
| **Frame-level CNN + Pooling** (YouTube-8M style) | Per-frame CNN → temporal pooling | 1 fps (~hundreds) | Simplest; loses temporal order; good baseline | Abu-El-Haija et al. 2016 |
| **TimeSformer** | Divided space-time attention: temporal attention + spatial attention applied separately in each block | 8–96 frames | SOTA on Kinetics-400/600; faster to train than 3D CNNs; handles >1-min clips | Bertasius et al. 2021, [arXiv:2102.05095](https://arxiv.org/abs/2102.05095) |
| **VideoMAE** | Masked autoencoder: masks 90–95% of video patches, reconstructs them; data-efficient pretraining | 16 frames | Achieves 87.4% on Kinetics-400 with vanilla ViT + no extra data; works with only 3k–4k videos | Tong et al. 2022, [arXiv:2203.12602](https://arxiv.org/abs/2203.12602) |
| **Per-frame DINOv2 + Lightweight Temporal Aggregator** | Encode frames with frozen DINOv2, aggregate with small Transformer | 16–24 frames | Practical hybrid; reuses vision backbone; small temporal model | — |

### 3.3 How Many Frames? Sampling Strategies?

For YouTube videos (typically 5–20 minutes):

- **16–24 frames** is the sweet spot. TimeSformer and VideoMAE both use this range and achieve strong results.
- **Uniform sampling** across the full video duration is standard and works well. Example: for a 10-minute video with 16 frames, sample at 0:00, 0:37, 1:15, ..., 9:22.
- **More frames ≠ always better.** TimeSformer found that 96 frames is only marginally better than 8–16 for many tasks, and the GPU cost scales more than linearly due to attention.
- **The first 30 seconds matter disproportionately.** Rajaram & Manchanda found that auditory and visual stimuli in the first 30 seconds of a video are associated with viewer engagement sentiment. Consider sampling more densely at the beginning (e.g., 8 frames in the first 60 seconds, 8 frames uniformly across the rest).

**Sampling strategy recommendation:** Hybrid — 8 frames from the first 60 seconds (denser sampling where it matters most), 8–16 frames uniformly across the remaining duration.

### 3.4 Recommendation

**Two-tier approach:**

**Tier 1 (pragmatic default):** Per-frame frozen DINOv2 ViT-S + lightweight temporal aggregator.

- Encode each of 16–24 frames with the same frozen DINOv2 ViT-S used for thumbnails (no separate backbone)
- Aggregate frame embeddings with a small 1–2 layer Transformer encoder (4 heads, 256 dim) or even a simple attention-pooling layer
- Total temporal module: ~1–2M additional parameters
- Why: reuses the vision backbone, keeps the temporal module tiny, and the frozen DINOv2 features are high-quality

**Tier 2 (if time and GPU permit):** TimeSformer with divided space-time attention, 16 frames.

- Stronger temporal modeling (actual spatiotemporal attention rather than post-hoc pooling)
- But: ~121M params for TimeSformer-B, needs more GPU memory, and 2,000 videos may be borderline for training from scratch
- Prefer using a TimeSformer pretrained on Kinetics-400 and fine-tuning only the last few layers

**Fallback (fastest path):** YouTube-8M-style mean pooling of frame-level features. No temporal modeling at all — just average the frame embeddings. This is a surprisingly strong baseline and establishes a lower bound quickly.

---

## 4. Fusion Strategy

### 4.1 The Fusion Spectrum

The taxonomy follows Liang et al. (2022), _"Foundations and Trends in Multimodal Machine Learning"_ [arXiv:2209.03430](https://arxiv.org/abs/2209.03430):

| Strategy | Description | Pros | Cons |
|----------|-------------|------|------|
| **Early fusion** | Concatenate raw inputs (pixels + tokens) before any encoding | Maximal interaction | Impractical for heterogeneous modalities; different dimensionalities and scales |
| **Mid fusion** | Encode each modality independently → fuse at intermediate layers via cross-attention | Rich cross-modal interaction | Complex architecture; harder to debug; more parameters |
| **Late fusion** | Encode each modality independently → concatenate final embeddings → classification head | Simple; modular; each encoder can be developed independently; easy to ablate | No cross-modal interaction during encoding |
| **Hybrid** | Late fusion + a small cross-attention layer before the classification head | Best of both for modest cost | Slightly more complex than pure late fusion |

### 4.2 Handling Different Embedding Dimensions

A typical setup will produce embeddings of different sizes:

| Modality | Encoder | Embedding Dim |
|----------|---------|---------------|
| Text (title + desc + transcript) | ModernBERT-base | 768 |
| Thumbnail | DINOv2 ViT-S/14 | 384 |
| Temporal (16–24 frames) | DINOv2 ViT-S + Pooling | 384 |

**Strategy:** Project each modality to a common dimension (e.g., 256 or 512) with a learned linear layer + LayerNorm, then concatenate or apply cross-attention.

```
text_emb    (768) → Linear(768, 512) → LN → text_proj    (512)
thumb_emb   (384) → Linear(384, 512) → LN → thumb_proj   (512)
temporal_emb(384) → Linear(384, 512) → LN → temporal_proj(512)

→ Concatenate → [512*3 = 1536] → MLP → binary logit
```

### 4.3 Joint vs. Staged Training

**Joint end-to-end training** is the proposal's ambition, but for 2,000 videos with frozen backbones it means training only the fusion head and projection layers — which is fine.

**Staged training** (pretrain each encoder separately, then fuse) is unnecessary here since all encoders are pretrained. The only question is whether to fine-tune the encoders or keep them frozen:

- **Frozen encoders + trainable fusion head (RECOMMENDED):** ~2–5M trainable parameters. Fast training, low overfitting risk. Can train on a single GPU in hours.
- **Fine-tuned encoders + fusion head:** 200M+ trainable parameters. Higher ceiling but high overfitting risk with 2,000 samples. Requires careful regularization and likely won't beat frozen encoders without more data.

If you do want to fine-tune: use LoRA adapters (low-rank matrices injected into attention layers) rather than full fine-tuning. This adds only ~1–2% more parameters while allowing the model to adapt.

### 4.4 Recommendation

**Late fusion with projection layers → concatenation → 2-layer MLP classifier.**

This is:
- **Simple to implement** (each modality is a standalone module)
- **Easy to ablate** (remove one modality and measure the drop)
- **Appropriate for the dataset size** (cross-attention fusion would overfit on 2,000 samples)
- **Interpretable** (the projection weights give a first-order signal of modality importance)

**Hybrid upgrade path (if time permits):** Add a single cross-attention layer where text attends to vision and vision attends to text before the concatenation step. This is the smallest cross-modal interaction you can add.

---

## 5. Attribution & Explainability

### 5.1 How Rajaram & Manchanda (2020) Implement Attention Attribution

Rajaram & Manchanda (2020), _"Unboxing Engagement in YouTube Influencer Videos"_ [arXiv:2012.12311](https://arxiv.org/abs/2012.12311), use an interpretable deep learning framework with:

1. **Model attention to video elements:** Their architecture includes attention mechanisms over text, audio, and visual modalities. The attention weights directly indicate which parts of each modality the model relied on.

2. **Ex-post interpretation via pruning spurious associations:** They introduce a novel approach that prunes spurious associations from attention weights. This is critical because raw attention weights can be noisy and capture correlations rather than causal importance.

3. **Modality-level findings:** They found that text ("what is said") dominates over imagery and audio for predicting engagement, and that the first 30 seconds of a video carry disproportionate weight.

4. **Validation:** They validate through multiple methods, including connecting findings to marketing theory.

**What YT-Diag can adopt directly:** The idea of using model attention weights as a first-order explanation, combined with a pruning/post-processing step. However, attention-as-explanation is controversial (Jain & Wallace 2019 showed attention weights don't always correlate with feature importance). So we should supplement with gradient-based methods.

### 5.2 Alternatives

| Method | How It Works | Pros | Cons | Source |
|--------|-------------|------|------|--------|
| **Attention weights** | Use learned attention scores as importance | Built into the model; no extra computation | Controversial validity; may not reflect true importance | Rajaram & Manchanda 2020 |
| **Integrated Gradients** | Integrates gradients along path from baseline to input; satisfies completeness axiom | Axiomatically grounded; model-agnostic; works for any differentiable model; implemented in Captum | Requires choosing a baseline; computationally heavier than raw gradients (needs ~50–200 forward passes) | Sundararajan et al. 2017, [arXiv:1703.01365](https://arxiv.org/abs/1703.01365) |
| **SHAP (SHapley Additive exPlanations)** | Game-theoretic; computes Shapley values for each feature | Theoretically well-founded; unified framework (Lundberg & Lee 2017) | Extremely expensive for high-dim inputs (exponential in features); practical approximations exist | Lundberg & Lee 2017 |
| **LIME** | Local surrogate model (linear) around a prediction | Model-agnostic; simple to understand | Unstable; sensitive to perturbation strategy; explanations can differ between runs | Ribeiro et al. 2016 |
| **Gradient × Input** | Element-wise product of gradient and input | Simplest gradient-based method; zero extra infrastructure | Noisy; prone to saturation issues | Simonyan et al. 2013 |
| **Ablation-based** | Remove features one at a time, measure prediction change | Direct causal interpretation; no gradients needed | Very expensive (one forward pass per feature); feature correlation confounds results | — |

### 5.3 Evaluating Explanation Quality

This is an open problem. As the Distill.pub article by Sturmfels et al. (2020), _"Visualizing the Impact of Feature Attribution Baselines"_, makes clear: there is **no ground truth** for explanations, and every evaluation method has flaws.

Practical evaluation approaches for YT-Diag:

1. **Ablation sanity check:** Rank features by importance score → ablate top-k → measure prediction drop. A good explanation should cause a steeper drop than random. (Caveat: distribution shift from ablated inputs.)

2. **Completeness axiom check:** For Integrated Gradients, verify that $\sum_i \phi_i = f(x) - f(x')$. If it doesn't sum, the approximation didn't converge.

3. **Human qualitative check:** For a handful of example videos, do the explanations match intuition? E.g., does the model flag a clickbait title as helping virality, or a low-quality thumbnail as hurting it?

4. **Modality ablation:** Remove entire modalities and measure the classification drop. This validates which modalities matter, providing a coarse check on attribution methods.

### 5.4 Recommendation

**Two-method approach:**

1. **Integrated Gradients (primary)** — via PyTorch Captum library:
   - Apply to the fused model to get per-modality attribution scores
   - Then drill into each modality: which tokens in the transcript? Which regions in the thumbnail?
   - Use a zero-embedding baseline for text and a black image or blurred image for vision
   - 50–100 integration steps is typically enough for convergence

2. **Attention weight analysis (supplementary)** — if the fusion head uses cross-attention:
   - Extract attention weights as a fast, qualitative signal
   - Flag when attention and Integrated Gradients disagree (this is itself informative)

**For the final output:** Convert raw attribution scores into a short, ranked, human-readable list. E.g.:
> "This video is predicted to underperform because: (1) Weak thumbnail — lacks faces and text overlays (35% contribution), (2) Title too generic — doesn't signal the specific value proposition (28% contribution), (3) Slow first 30 seconds — hook doesn't arrive until 0:45 (18% contribution)..."

This matches the proposal's description of "a short, ranked, human-readable explanation."

---

## 6. Practical Feasibility & Resource Budget

### 6.1 GPU Memory Estimate

For the recommended architecture (frozen ModernBERT-base + frozen DINOv2 ViT-S + small temporal aggregator + fusion head):

| Component | Approx. Memory (FP32) |
|-----------|----------------------|
| ModernBERT-base (inference) | ~600 MB |
| DINOv2 ViT-S (inference) | ~85 MB |
| Temporal aggregator (trainable) | ~10 MB |
| Fusion head (trainable) | ~5 MB |
| Activations (batch=8, mixed precision) | ~2–4 GB |
| Optimizer states (Adam, trainable params only) | ~40 MB |
| **Total (frozen backbones)** | **~3–5 GB** |

**This fits comfortably on any GPU with ≥6 GB VRAM** — including an RTX 3060, T4, or free-tier Colab GPU.

If fine-tuning encoders:

| Component | Approx. Memory (FP32) |
|-----------|----------------------|
| ModernBERT-base (trainable) | ~2.4 GB (params + grads + optimizer) |
| DINOv2 ViT-S (trainable) | ~340 MB |
| Everything else | ~3 GB |
| **Total (fine-tuned)** | **~6–8 GB** |

Still fits on most GPUs, but use gradient accumulation if batch size needs to drop below 4.

### 6.2 Training Time Estimate

With frozen backbones on a single RTX 4090 or A100:

- **Forward pass per sample:** ~50–100ms (dominated by ModernBERT on transcript)
- **2,000 samples × 20 epochs:** ~40,000 forward+backward passes
- **Total training:** ~2–4 hours

With fine-tuned backbones: ~8–15 hours.

### 6.3 What's Realistic in 1 Month?

Given 2 students, both in other courses, with ~10–20 hours/week each for the project:

| Week | Milestone |
|------|-----------|
| **Week 1** | Data collection complete (using yt-dlp); EDA; label computation (viral/non-viral within category) |
| **Week 2** | Baseline models (logistic regression, XGBoost on metadata only); set up frozen-backbone multimodal pipeline |
| **Week 3** | Train multimodal model; hyperparameter sweep; ablation studies (remove one modality at a time) |
| **Week 4** | Attribution layer (Integrated Gradients); qualitative analysis; write report; prepare presentation |

**This timeline is tight but feasible if:**
- Data collection starts immediately (the biggest bottleneck — see data retrieval research)
- You don't get stuck in architecture exploration (start with the simplest thing that works)
- Baselines are run in parallel with multimodal model development

---

## 7. Practical Recommendations

### 7.1 Recommended Architecture (Ranked by Feasibility)

#### 🥇 Tier 1: The Pragmatic Stack (START HERE)

```
Text:       ModernBERT-base (frozen) → Linear(768→256)
Thumbnail:  DINOv2 ViT-S/14 (frozen) → Linear(384→256)
Temporal:   16 frames → DINOv2 ViT-S/14 (frozen, shared with thumbnail)
            → Attention Pooling → Linear(384→256)
Fusion:     Concat([256, 256, 256]) → 2-layer MLP → binary logit
Attribution: Integrated Gradients via Captum
```

**Why this first:**
- ~600M total params but only ~3M trainable (fusion head only)
- Fits on any GPU with 6GB+ VRAM
- 2–4 hours to train
- Easy to ablate (drop one modality, measure impact)
- ModernBERT's 8K context handles transcripts without chunking
- DINOv2 reused for both thumbnail and frames (one less model to manage)

#### 🥈 Tier 2: The Moderate Upgrade (IF TIME PERMITS)

Add to Tier 1:
- **LoRA fine-tuning** on ModernBERT (adds ~1–2% params, enables domain adaptation to YouTube language)
- **Hybrid fusion:** one cross-attention layer where text attends to visual features before concatenation
- **Denser temporal sampling:** 8 frames in first 60 seconds + 16 uniformly across rest

#### 🥉 Tier 3: The Full Ambition (ONLY IF WEEKS 1–2 GO PERFECTLY)

Add to Tier 2:
- **TimeSformer** in place of the simple temporal aggregator (pretrained on Kinetics-400)
- **Multi-head cross-attention fusion** with separate cross-attention for each modality pair
- **Expected Gradients** (multiple baselines from training distribution) for more robust attributions

### 7.2 Where Ambitions Need Scaling Down

| Proposal Ambition | Reality Check | Recommendation |
|-------------------|---------------|----------------|
| "Trained end-to-end" | 2,000 samples is not enough to train 3 large models from scratch | Freeze backbones, train only fusion head |
| "Cross-attention fusion" | Will overfit on 2,000 samples unless aggressively regularized | Start with late fusion (concatenation); add cross-attention only as a final experiment |
| "Ablation-based attribution layer" | Full feature ablation is computationally prohibitive | Use Integrated Gradients instead; it's faster and more principled |
| "4 categories × 500 videos each" | Within-category virality means only ~125 positive examples per category | Consider binary classification (all categories pooled) first, then per-category as a secondary analysis |
| "16–24 frames sampled uniformly" | 24 frames × DINOv2 encoding × 2,000 videos = 48,000 frame encodings — manageable but adds pre-computation step | Pre-compute and cache frame embeddings; don't encode on-the-fly during training |

### 7.3 Simpler Fallbacks If Nothing Works

If the multimodal approach doesn't converge or overfits severely:

1. **Metadata-only baseline:** Logistic regression or XGBoost on 12 structured features (duration, upload time, title length, tag count, etc.). This is already a deliverable — the project proposal lists it as a baseline. If this baseline performs reasonably well, you have a valid result even without the deep model.

2. **Text-only model:** Fine-tune ModernBERT on title + description + transcript alone. Given Rajaram & Manchanda's finding that text dominates, this might get you 80% of the way with 20% of the complexity.

3. **Two-modality model:** Text + thumbnail, drop temporal. The temporal modality is the most uncertain in terms of added value for virality prediction — a static thumbnail may already capture the visual signal.

4. **Scrap the deep learning entirely:** If data collection fails or training doesn't converge, the Wu et al. (2018) baseline approach (structured features + relative engagement metric) is a legitimate, publishable result for a course project. The project proposal explicitly lists this as an adaptation target.

---

## References

1. Abu-El-Haija, S., Kothari, N., Lee, J., Natsev, P., Toderici, G., Varadarajan, B., & Vijayanarasimhan, S. (2016). YouTube-8M: A Large-Scale Video Classification Benchmark. _arXiv:1609.08675_.

2. Bertasius, G., Wang, H., & Torresani, L. (2021). Is Space-Time Attention All You Need for Video Understanding? (TimeSformer). _ICML 2021_. arXiv:2102.05095.

3. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. _arXiv:1810.04805_.

4. Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT). _ICLR 2021_. arXiv:2010.11929.

5. Liang, P. P., Zadeh, A., & Morency, L. P. (2022). Foundations and Trends in Multimodal Machine Learning: Principles, Challenges, and Open Questions. _arXiv:2209.03430_.

6. Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. _arXiv:1907.11692_.

7. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP). _NeurIPS 2017_.

8. Oquab, M., et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. _arXiv:2304.07193_.

9. Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP). _arXiv:2103.00020_.

10. Rajaram, P., & Manchanda, P. (2020). Unboxing Engagement in YouTube Influencer Videos: An Attention-Based Approach. _arXiv:2012.12311_.

11. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. _NeurIPS 2019 EMC2 Workshop_. arXiv:1910.01108.

12. Sturmfels, P., Lundberg, S., & Lee, S. I. (2020). Visualizing the Impact of Feature Attribution Baselines. _Distill_. DOI: 10.23915/distill.00022.

13. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks (Integrated Gradients). _ICML 2017_. arXiv:1703.01365.

14. Tong, Z., Song, Y., Wang, J., & Wang, L. (2022). VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training. _NeurIPS 2022_. arXiv:2203.12602.

15. Warner, B., et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference (ModernBERT). _arXiv:2412.13663_. [HuggingFace blog](https://huggingface.co/blog/modernbert).

16. Wu, S., Rizoiu, M. A., & Xie, L. (2018). Beyond Views: Measuring and Predicting Engagement in Online Videos. _ICWSM 2018_.

---

_This file covers the MODEL ARCHITECTURE dimension of the YT-Diag project. Companion files: `data_retrieval.md` (data collection strategy)._
