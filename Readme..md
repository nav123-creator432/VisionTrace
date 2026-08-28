# VisionTrace

**Image Captioning with Benchmarking, Interpretability, and Robustness Analysis**

A ViT + DistilGPT2 encoder-decoder image captioning model built in PyTorch, benchmarked against a zero-shot vision-language model, analyzed via cross-attention visualization, and stress-tested under systematic image corruption.

This project goes beyond "train a captioning model and report BLEU." The core question it asks is: **when the model gets it wrong, *why* does it get it wrong?**

---

## Table of Contents

- [Architecture](#architecture)
- [Dataset](#dataset)
- [Training Setup](#training-setup)
- [Results](#results)
  - [Captioning Metrics](#captioning-metrics)
  - [Benchmark vs. SmolVLM](#benchmark-vs-smolvlm)
- [Decoding: A Degeneration Bug](#decoding-a-degeneration-bug)
- [Interpretability: Cross-Attention Analysis](#interpretability-cross-attention-analysis)
- [Robustness Analysis](#robustness-analysis)
- [Model Authorship Classifier](#model-authorship-classifier)
- [Known Limitations](#known-limitations)
- [Repository Structure](#repository-structure)
- [Setup](#setup)

---

## Architecture

```
Image (224x224)
      |
      v
+---------------------------+
|  ViT-Small (frozen)       |   WinKawaks/vit-small-patch16-224
|  22M params               |
+---------------------------+
      |
      | 196 patch embeddings (14x14 grid, CLS token dropped)
      | shape: (B, 196, 384)
      v
+---------------------------+
|  Linear Projection        |   384 -> 768
|  (trainable)              |
+---------------------------+
      |
      | encoder_hidden_states: (B, 196, 768)
      v
+---------------------------+
|  DistilGPT2 (trainable)   |   add_cross_attention=True
|  82M params               |   6 layers, cross-attn newly initialized
+---------------------------+
      |
      v
   Caption tokens
```

**Key design decisions:**

| Decision | Choice | Rationale |
|---|---|---|
| Encoder output | All 196 patch embeddings, not just CLS | CLS collapses the image to one vector, leaving cross-attention nothing spatial to attend over. Full patches preserve location information and enable attention visualization. |
| Encoder training | Frozen | ~7K training images is too little to fine-tune 22M vision params without overfitting. |
| Decoder | DistilGPT2 with `add_cross_attention=True` | Cross-attention layers are randomly initialized (absent from the pretrained checkpoint) and learn image grounding during training. |
| Projection layer | `nn.Linear(384, 768)` | ViT hidden size and GPT2 embedding size differ; a learned mapping is more expressive than reshaping. |

---

## Dataset

**Flickr8k** — 8,091 images, 5 human-written captions per image.

| Split | Images | Caption rows |
|---|---|---|
| Train | 6,189 | 30,946 (all 5 captions/image) |
| Validation | 688 | 688 (1 caption/image) |
| Test | 1,214 | 1,214 (1 caption/image) |

Splits are made **by unique image**, not by caption row — otherwise the same image's five captions would scatter across train and test, leaking information.

Train keeps all five captions per image as free augmentation. Validation and test use one caption per image to avoid redundantly scoring near-duplicate references.

---

## Training Setup

| Component | Value |
|---|---|
| Optimizer | AdamW, `lr=3e-5`, `weight_decay=0.01` |
| LR schedule | Linear warmup (10% of steps) then linear decay |
| Loss | Cross-entropy, `ignore_index=pad_token_id`, `label_smoothing=0.1` |
| Batch size | 32 |
| Teacher forcing | Input `[:-1]`, labels `[1:]` |
| Early stopping | Patience 3 on validation loss |
| Hardware | Single T4 GPU (Google Colab), ~9 min/epoch |

**Training curve:**

```
Epoch    Train Loss    Val Loss
  1        5.2614       4.2406
  2        4.1584       3.9671
  3        3.8901       3.8464
  4        3.7307       3.7883
  5        3.6138       3.7556
  6        3.5189       3.7428
  7        3.4374       3.7395
  8        3.3668       3.7361   <- best checkpoint saved
  9        3.3045       3.7395   (no improvement 1/3)
 10        3.2525       3.7494   (no improvement 2/3)
 11        3.2089       3.7602   (no improvement 3/3) -> early stop
```

Validation loss bottoms out at epoch 8 while training loss keeps falling — the classic overfitting signature. Checkpointing on validation improvement means the saved model is epoch 8, not epoch 11.

> Note: absolute loss values are not comparable across configurations here. Label smoothing puts a floor under the loss (a perfect model can no longer reach 0), so a run with smoothing enabled will report higher loss than one without, even if the model is better.

---

## Results

### Captioning Metrics

Evaluated on the 1,214-image held-out test set with beam search (`num_beams=4`).

| Metric | Score |
|---|---|
| BLEU | 0.0362 |
| ROUGE-1 | 0.3054 |
| ROUGE-2 | 0.0914 |
| ROUGE-L | 0.2343 |
| METEOR | 0.3047 |

### Benchmark vs. SmolVLM

`SmolVLM-256M-Instruct`, zero-shot (no fine-tuning), same test images, same scoring pipeline.

| Metric | This model (fine-tuned) | SmolVLM (zero-shot) |
|---|---|---|
| BLEU | **0.0362** | 0.0250 |
| ROUGE-1 | **0.3054** | 0.2408 |
| ROUGE-2 | **0.0914** | 0.0559 |
| ROUGE-L | **0.2343** | 0.2071 |
| METEOR | **0.3047** | 0.2479 |

**The fine-tuned model wins on every metric — but this needs an honest caveat.**

BLEU, ROUGE, and METEOR are n-gram *overlap* metrics. They reward matching the reference's phrasing, not being factually correct. Consider the same image captioned by both models:

> **Ground truth:** *A couple and an infant, being held by the male, sitting next to a pond with a near by stroller.*
>
> **This model:** *man and woman are sitting on a bench in front of a tree. one of them is holding a camera. the other is taking a picture.*
>
> **SmolVLM:** *In this image we can see a woman sitting on the ground and holding a baby. There is a baby in the baby carrier. There is a tree trunk. There is a water body.*

SmolVLM is **more factually grounded** here — it correctly identifies the woman, the baby, the tree, and the water. This model hallucinates a camera that does not exist. Yet this model scores higher, because it learned Flickr8k's terse phrasing style during fine-tuning and SmolVLM did not.

**Winning on automatic metrics is not the same as producing better captions.** This is a well-documented limitation of reference-overlap metrics in captioning and NLG generally.

---

## Decoding: A Degeneration Bug

Initial beam search output collapsed into repetition:

```
Greedy:      man and woman sitting on a bench . " 's face . " " " " " " " " " "
Beam search: man and woman sitting on a bench with their backs to the camera . . . . . . . . .
```

This is **neural text degeneration** (Holtzman et al., *The Curious Case of Neural Text Degeneration*). Greedy and beam search are deterministic — they always take the highest-scoring token. Once the model runs out of confident content, low-risk tokens (periods, quotes) score highest, and the decoder loops.

**Fix — decoding constraints, no retraining:**

```python
model.decoder.generate(
    ...,
    no_repeat_ngram_size=2,    # hard block on repeated bigrams
    repetition_penalty=1.5,    # soft down-weighting of used tokens
    length_penalty=0.6,        # favor shorter, caption-like outputs
)
```

**Impact:**

| Metric | Before | After | Change |
|---|---|---|---|
| BLEU | 0.0236 | 0.0362 | **+53%** |
| ROUGE-1 | 0.2436 | 0.3054 | +25% |
| ROUGE-2 | 0.0752 | 0.0914 | +22% |
| ROUGE-L | 0.1851 | 0.2343 | +27% |
| METEOR | 0.2836 | 0.3047 | +7% |

Generation also became ~8x faster (fewer decode steps per image).

**An important nuance:** `no_repeat_ngram_size` blocks verbatim repeats but does not force the model to say anything *meaningful*. After the fix, some outputs stopped looping and instead rambled — e.g. listing colors (`blue and yellow paddle . white . yellow . pink . black ...`). Decode-time constraints treat the symptom. The root cause is an undertrained model on limited data.

---

## Interpretability: Cross-Attention Analysis

Cross-attention weights were extracted from the decoder's final layer, averaged across attention heads, reshaped from 196 values back into the original 14×14 patch grid, and overlaid on the source image — one heatmap per generated word.

> **Implementation note:** this requires `attn_implementation="eager"` when loading the decoder. The default SDPA attention is a fused kernel that never materializes attention weights, so `output_attentions=True` silently returns an empty tuple.

<!-- Add your saved figure here -->
<!-- ![Attention on an accurate caption](assets/attention_accurate.png) -->

### Finding 1 — Attention shifts on accurate captions

On a correctly-captioned image, attention *moves* between words: sharply localized on the man for "man", on the woman for "woman", down toward the seating area for "sitting". Function words ("and", "are") show diffuse attention — expected, since grammatical glue has no visual referent.

This is direct evidence the cross-attention mechanism learned meaningful image grounding.

### Finding 2 — Two distinct hallucination mechanisms

Attention maps revealed that not all hallucinations are the same failure:

<!-- ![Misattribution vs confabulation](assets/attention_hallucination.png) -->

**Misattribution** — real attention, wrong label.

> Image: a man drilling a hole in ice
> Generated: *"...one is holding a **fishing pole**..."*

Attention clearly shifts onto a real object in the frame (a sled-like object beside the man) at exactly the moment "fishing pole" is generated. The model attended to genuine visual evidence and named it incorrectly. This is a **perception/vocabulary** error.

**Confabulation** — no attention shift at all.

> Image: a man and baby in a yellow kayak
> Generated: *"girl in a yellow kayak is sitting on the water with her **dog**."*

Attention stays locked on the person's head/shoulder for *every* word, including "dog". No visual region drives that noun. The word comes from the language model's learned priors ("kayak... water... with her ___"), not the image. This is a **grounding** failure.

**Why the distinction matters:** these need different fixes. Misattribution is a recognition problem (more/better data). Confabulation is a grounding problem (stronger visual conditioning, contrastive objectives) — more data alone would not necessarily solve it.

---

## Robustness Analysis

Four corruption types, three severity levels each, evaluated on a fixed 200-image sample (same images across all conditions for comparability).

<!-- ![Robustness curves](assets/robustness_curves.png) -->

| Corruption | Severity | BLEU | ROUGE-L | METEOR |
|---|---|---|---|---|
| *None (baseline)* | — | 0.0348 | 0.2356 | 0.3064 |
| Occlusion | 0.1 | 0.0321 | 0.2167 | 0.2998 |
| Occlusion | 0.5 | 0.0230 | 0.1856 | 0.2549 |
| Occlusion | 0.8 | 0.0177 | 0.1728 | **0.2303** |
| Blur | 2 | 0.0285 | 0.2140 | 0.2920 |
| Blur | 5 | 0.0243 | 0.2005 | 0.2832 |
| Blur | 10 | 0.0189 | 0.1835 | 0.2455 |
| Rotation | 10° | 0.0308 | 0.2127 | 0.3062 |
| Rotation | 30° | 0.0284 | 0.2108 | 0.2883 |
| Rotation | 45° | 0.0251 | 0.2039 | **0.2845** |
| Color distortion | 0.3 | 0.0280 | 0.2147 | 0.2953 |
| Color distortion | 0.6 | 0.0288 | 0.2133 | 0.2952 |
| Color distortion | 0.9 | 0.0215 | 0.1865 | 0.2561 |

**Robustness ranking (most → least robust): rotation > color distortion > blur > occlusion**

**Rotation is the most robust** — METEOR drops only ~7% at 45°. This is architecturally consistent: rotation *reorients* information but deletes none of it, and ViT's patch-based global self-attention has weaker positional dependence than a CNN's local convolutions.

**Occlusion is the least robust** — METEOR drops ~25% at 80%. Unlike the other three, occlusion **genuinely deletes visual information** rather than distorting it. The model has less evidence, not noisier evidence.

**Color distortion shows a cliff, not a slope** — near-flat performance through severity 0.6, then a sharp drop at 0.9. A distinct failure profile worth noting: the model tolerates moderate lighting shifts, then breaks quickly past a threshold.

### Attention under occlusion

Running the attention visualization on a 50%-occluded image explains *why* occlusion is worst:

> **Ground truth:** *A couple and an infant, being held by the male, sitting next to a pond with a near by stroller.*
> **Generated (50% occluded):** *man wearing a black hat is standing in front of a tree with a bicycle in the background.*

Attention maps become **diffuse and low-contrast** — no sharp focus anywhere, unlike the concentrated hotspots seen on clean images. The model does not gracefully redirect attention to the still-visible region; it loses grounding entirely and falls back on generic, plausible-sounding scene furniture (a hat, a bicycle). This is confabulation triggered by information loss.

---

## Model Authorship Classifier

**Task:** given only a caption's text, predict which model wrote it — this model or SmolVLM.

| Component | Value |
|---|---|
| Model | `bert-base-uncased` + sequence classification head |
| Dataset | 2,428 captions (1,214 per model, balanced) |
| Split | 80/20 stratified |
| Training | 3 epochs, AdamW `lr=2e-5` |
| **Test accuracy** | **99.79%** (1 error out of 486) |

BERT was chosen over a GPT-style model deliberately: BERT is bidirectional, so every token attends to the full sentence in both directions. GPT models are causally masked (each token sees only earlier tokens) because that constraint is required for autoregressive generation — a constraint classification does not need.

**What this number does and does not mean.** 99.79% does *not* indicate either model captions well. The classifier learns **stylistic fingerprints**, not caption quality — a much easier task. This model produces terse, templated output from Flickr8k fine-tuning; SmolVLM produces verbose instruction-tuned descriptions ("In this image we can see... There is... There is..."). Two models with different architectures and training regimes are near-trivially separable.

**Error analysis — the single misclassification confirms this:**

> Text: *"a small child is standing on the floor in a room."*
> True: SmolVLM · Predicted: this model

The one caption the classifier got wrong is a SmolVLM output that happened to be unusually short and plain, dropping its characteristic verbose template. The classifier failed precisely when a caption deviated from its author's typical style — which validates that style, not content, is what it learned.

---

## Known Limitations

This is a demonstration and analysis project, not a production system.

- **Hallucination is present and unresolved.** The model invents objects (cameras, dogs, bicycles) that are not in the image. This is expected at this scale — ~7K training images and a 104M-parameter total model, versus hundreds of thousands to millions of image-text pairs for production captioning systems.
- **Encoder fine-tuning was attempted and abandoned.** Partially unfreezing ViT's last two transformer blocks with discriminative learning rates (`5e-6` encoder / `3e-5` decoder) plus light augmentation was tried to address the grounding failure directly. Training destabilized within the available compute budget, and the validated frozen-encoder model was kept instead. Documented as a negative result rather than omitted.
- **Weight decay is applied uniformly.** Standard practice excludes bias and LayerNorm parameters from weight decay via separate parameter groups; this implementation does not.
- **BLEU is low in absolute terms** (~0.036). BLEU-4 requires 4-gram overlap, and Flickr8k captions are often under 10 words, so exact 4-gram matches are rare by construction. Relative comparisons within this project are meaningful; the absolute value should not be read against machine-translation BLEU intuitions.
- **Attention visualization uses greedy decoding by default.** Beam search maintains multiple hypotheses that get pruned, making the attention-to-final-caption mapping ambiguous. A custom beam search implementation that carries per-beam attention history is included, but greedy remains the simpler, unambiguous default for interpretability.
- **Horizontal-flip augmentation** (in the abandoned run) would technically invalidate the small subset of Flickr8k captions that reference left/right orientation.

---

## Repository Structure

```
.
├── README.md
├── notebooks/
│   └── image_captioning.ipynb      # Full end-to-end pipeline
├── assets/
│   ├── attention_accurate.png
│   ├── attention_hallucination.png
│   ├── attention_occluded.png
│   └── robustness_curves.png
└── requirements.txt
```

---

## Setup

```bash
pip install torch transformers rouge-score evaluate nltk opencv-python-headless pandas tqdm scikit-learn
```

Dataset (Kaggle API token required):

```bash
kaggle datasets download -d adityajn105/flickr8k -p ./data --unzip
```

The notebook is written for Google Colab with a GPU runtime. Checkpoints are saved to Google Drive to survive session resets.

---

## Acknowledgements

- ViT encoder: [`WinKawaks/vit-small-patch16-224`](https://huggingface.co/WinKawaks/vit-small-patch16-224)
- Decoder: [`distilgpt2`](https://huggingface.co/distilgpt2)
- Benchmark model: [`HuggingFaceTB/SmolVLM-256M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
- Dataset: [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)
