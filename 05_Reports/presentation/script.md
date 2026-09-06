# YT-Diag — Presentation Script (v4, matches the 9-slide deck)

Matches the current merged deck (`ytdiag_deck.html`, 9 slides after the
2026-09-06 merge of foundation-model+five-inputs and fusion-math+attribution,
plus folding "why this architecture" into the architecture slide's own
caption). Rule for this version: **never say a number or fact that's already
legible on the slide.** The slide shows the bar chart, the formula, the stat
card; the voice explains *why it's there* or *what it means*, not what it
says. If a line could be replaced by silently pointing at the screen, cut
the line.

No em dashes; periods, colons, or "because"/"so" instead.

Slides 1-4 are Adam's own rewrite (2026-09-06), kept close to his original
wording with only light grammar cleanup. Slides 5-9 are the earlier
conversational draft, re-mapped onto the merged slide structure.

---

## 1. Cover

Video content is the most convenient format for a creator to convey a
meaning, help the audience understand, and create a desired feeling. To
spread a message to the masses, we need to understand what makes a video
land. We use YouTube's promotion algorithm to find just that

## 2. The question

The real question is: if we strip out channel size, age, category and
format, how much does the actual content add to the success rate on
YouTube?

## 3. The dataset

We built our own dataset using 1,860 videos across 4 categories, and sourced
the features: for visual, it is the thumbnail and 20 frames, for text it is the title of the video, 
the transcript and additionally metadata and audio.

## 4. Structured comparator

The baseline for us to beat was metadata and schedule, which properly tuned,
reach 0.619 AUC. That alone sets a surprisingly high bar for
anything fancier to clear.


## 5. Deep learning architecture

> This is where deep learning actually earns its place: frozen vision and
> language models turn thumbnails, frames and text into embeddings, and one
> small trainable head fuses them together.

## 6. The model, end to end

> Every one of these encoders, whether it's reading pixels or words, is
> really the same block underneath: attention, then a feed-forward layer,
> stacked over and over. But vision, text and audio still get fed in
> completely differently, on purpose, because none of them actually behave
> the same way.

## 7. The fusion head

> Only that very last layer actually learns anything new here. Everything
> upstream is borrowed knowledge, frozen, never touched during training.
> And because that layer is linear, we can show exactly which input
> mattered most for any single prediction.

## 8. Results

> And here's the honest part: the fancy model doesn't actually beat the
> boring one. It's close, it's informative, but at this size, content isn't
> reliably winning yet. Audio looked like it might change that after the
> fact, but the moment we actually tested it properly, that promise mostly
> evaporated.

## 9. Conclusion

> The pipeline works. The proof that content matters just isn't there yet,
> mostly because our error bars are still wide on this little data. We're
> already collecting o bigger dataset of 12,000 videos, tracked over 30 days, which will finally settle the results.

---

## Timing

Roughly 260-280 words across 9 slides once slide 1 is completed. At a
natural pace with real pauses (~110-120 wpm), that's **~2:15-2:30**, with
room to spare under 3:00. If it still runs long after rehearsing:

- Slide 4's note above is the first candidate: dropping the restated 0.619
  number costs nothing since it's already on screen.
- Slide 3 (the dataset) is the next most compressible beat: "we built our
  own dataset, video by video, six signals each" says the same thing in
  fewer words without listing every signal by name.
- Don't cut slides 2, 5, 7, or 8: the research question, why deep learning
  is used at all, the exact-attribution point, and the honest result are
  the actual argument.
