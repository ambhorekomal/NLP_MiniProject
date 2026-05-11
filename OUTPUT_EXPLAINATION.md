# Load Report and Training Log Explanation

This file explains the output:

`0/400 [00:00<?, ? examples/s]`
`Loading weights:   0%|          | 0/199 [00:00<?, ?it/s]`
`BertForSequenceClassification LOAD REPORT from: bert-base-uncased`
`... UNEXPECTED / MISSING keys ...`
`Epoch 1 Loss: 0.5952598391071199`

## 1) `0/400 [00:00<?, ? examples/s]`

- This is a progress bar from dataset preprocessing/loading.
- `0/400` means it has not yet processed any of the 400 examples at that exact moment.
- `? examples/s` appears before speed is measured; this is normal right at startup.
- Not an error.

## 2) `Loading weights: 0% | 0/199`

- This means model checkpoint parameters are being read.
- `199` is the number of parameter tensors/chunks being loaded.
- `0%` is only the starting instant.
- Not an error.

## 3) `BertForSequenceClassification LOAD REPORT from: bert-base-uncased`

- You are loading a **base BERT pretrained checkpoint** into a **sequence-classification model**.
- `bert-base-uncased` was pretrained for general language modeling tasks, not directly for your final sentiment head.
- So some key mismatch is expected.

## 4) Why `UNEXPECTED` keys appear

Example keys:

- `cls.predictions.transform.dense.weight`
- `cls.seq_relationship.weight`
- `cls.predictions.bias`

Meaning:

- These belong to BERT pretraining heads (masked language modeling / next sentence prediction).
- Your current architecture (`BertForSequenceClassification`) does not need those heads.
- So the loader says they are **UNEXPECTED** and ignores them.

Is this bad?

- Usually **no**, this is normal when moving from pretraining checkpoint to downstream classification.

## 5) Why `MISSING` keys appear

Keys shown:

- `classifier.weight`
- `classifier.bias`

Meaning:

- These are the final sentiment classification layer parameters.
- They are not present in plain `bert-base-uncased` checkpoint.
- So Hugging Face initializes them randomly/newly.

Is this bad?

- Also **normal** for first-time fine-tuning.
- It means you must train on your task so this classifier learns sentiment boundaries.

## 6) Combined interpretation of UNEXPECTED + MISSING

This exact combination typically means:

- Base encoder weights loaded successfully from pretrained BERT.
- Task-specific head is newly created and needs training.
- Pretraining-only head weights were ignored.

So your model initialization behavior is correct for transfer learning.

## 7) `Epoch 1 Loss: 0.5952598391071199`

- This is average training loss after first epoch.
- In binary classification with cross-entropy, values around this range at early stage are common.
- Lower loss generally means model is fitting training data better.

How to judge if this is good:

- Compare with later epochs: if loss decreases (for example 0.59 -> 0.52 -> 0.45), learning is progressing.
- Also check validation/test metrics (accuracy, F1), because low training loss alone is not enough.

## 8) Is your run healthy?

From this log alone: **yes, looks healthy/expected**.

- No critical load failure is shown.
- Warnings are standard transfer-learning warnings.
- Epoch loss is finite and reasonable (not `nan`, not exploding).

## 9) What to do next

1. Train for more epochs (if compute allows), not only 1 epoch.
2. Monitor validation accuracy/F1 after each epoch.
3. Keep the best checkpoint by validation metric.
4. Compare with DistilBERT on same split for speed vs quality decision.

## 10) One-line summary

Your log indicates a **normal and correct BERT fine-tuning startup**: pretraining-only weights are ignored, classifier head is newly initialized, and training has started with a reasonable first-epoch loss.

---

## 11) DistilBERT: `DistilBertForSequenceClassification LOAD REPORT from: distilbert-base-uncased`

You are loading the **pretrained DistilBERT encoder** (`distilbert-base-uncased`) into **`DistilBertForSequenceClassification`**. That class adds a small **classification stack** on top of the encoder (in Hugging Face: a `pre_classifier` layer and a final `classifier` layer). The public checkpoint was trained for general representation learning / LM-style objectives, not for your IMDB head, so the loader will again report **UNEXPECTED** and **MISSING** keys. That is expected for transfer learning.

### UNEXPECTED keys (what they mean)

| Key | Simple meaning |
|-----|----------------|
| `vocab_transform.weight` / `.bias` | Parts of a **vocabulary projection** path tied to pretraining / distillation setup. They are **not** used by the sequence-classification forward pass, so the loader flags them as unexpected for this architecture. |
| `vocab_projector.bias` | Same family: **projection toward vocabulary space** from pretraining. Ignored when building the classifier model. |
| `vocab_layer_norm.weight` / `.bias` | **LayerNorm** around that pretraining-side projection path. Not needed for your classification head. |

**Bottom line:** treat **UNEXPECTED** here like BERT’s `cls.*` keys: **safe to ignore** when you *intentionally* load a pretrained base checkpoint into a **different task head**.

### MISSING keys (what they mean)

| Key | Simple meaning |
|-----|----------------|
| `pre_classifier.weight` / `pre_classifier.bias` | The **first linear layer** after the pooled sentence representation, specific to the classification model. Not stored in the raw pretrained checkpoint. **Randomly initialized** until you fine-tune. |
| `classifier.weight` / `classifier.bias` | The **final 2-way sentiment layer** (negative vs positive). Also **not** in the base checkpoint. **Randomly initialized** until you fine-tune. |

**Bottom line:** **MISSING** means “these layers are new for your task” — you **must** train on IMDB (as your notebook does) so they learn meaningful weights.

### Is DistilBERT’s report worse than BERT’s?

No. BERT showed UNEXPECTED pretraining-head keys and MISSING `classifier.*`. DistilBERT shows UNEXPECTED vocab/projection keys and MISSING **`pre_classifier` + `classifier`**. Different names, **same idea**: load encoder from pretraining, **drop** unused heads, **create** classification layers fresh.

---

## 12) Your training numbers (both models)

| Item | BERT | DistilBERT |
|------|------|------------|
| Epoch 1 training loss (average) | ~`0.5953` | ~`0.6102` |
| Test accuracy (your run) | **0.7725** | **0.79** |

**How to read this:**

- **Loss** is on the **training** set; slightly lower BERT loss here only means BERT’s parameters + batch produced a bit lower average error on those training batches for epoch 1. It does not guarantee better **test** accuracy.
- **Accuracy** is on the **held-out test** subset (400 reviews in your notebook). Here **DistilBERT scored higher** (~79% vs ~77.25%). On a small subset and a single epoch, **either model can win**; your result is valid and shows DistilBERT can match or beat BERT on accuracy while usually staying **faster/lighter** at inference.

---

## 13) Figure explanations (your output plots)

These match the charts from your notebook: model comparison bars, BERT confusion matrix, ROC curve, BERT training loss, **DistilBERT training loss**, and the **word cloud**.

### A) Model comparison bar chart (“Model Comparison”)

- **X-axis:** BERT vs DistilBERT.
- **Y-axis:** Score (0–1).
- **Blue bars:** Accuracy.
- **Orange bars:** F1 score.

**What it shows:**

- **DistilBERT** is slightly **higher** on both **accuracy** and **F1** in your run (roughly **~0.79** accuracy and **~0.82** F1 vs BERT **~0.77** accuracy and **~0.81** F1). So on this experiment, the smaller model did **not** pay a quality penalty; it did **slightly better** on this sample of data and training setup.
- **F1 > accuracy** for both models in the plot: F1 balances precision and recall; when it is a bit above accuracy, it often means the model is doing **reasonably well on both classes**, not only on the easy majority class.

**Takeaway for reports:** emphasize **efficiency vs quality**: DistilBERT can be **competitive or better** here, which is a strong result for deployment stories.

### B) Confusion matrix (heatmap, 2×2)

Rows = **actual** class, columns = **predicted** class (classes **0** and **1**). For your **BERT** run on **400** test examples, the counts work out as:

|  | Predicted 0 | Predicted 1 |
|--|------------|------------|
| **Actual 0** | **119** (true negatives) | **77** (false positives) |
| **Actual 1** | **14** (false negatives) | **190** (true positives) |

**Checks:**

- Total: \(119 + 77 + 14 + 190 = 400\).
- Accuracy: \((119 + 190) / 400 = 309/400 = \mathbf{0.7725}\), matching **`BERT Results: 0.7725`**.

**How to interpret:**

- **Strong on class 1:** many **true positives** (190) and few **false negatives** (14) → the model is **good at catching positive reviews**.
- **Weaker on class 0:** **77** negatives were called positive (**false positives**) → the model is **more confused when the true label is negative** (sometimes “pulled” toward positive). That pattern is common in sentiment data if positive wording appears often or the decision boundary skews.

**For your write-up:** mention **class imbalance in errors** (more errors when truth is 0) and that **F1 / per-class report** help beyond a single accuracy number.

### C) ROC curve (“ROC Curve”, AUC in legend)

- **X-axis:** False positive rate (how often negatives are wrongly called positive).
- **Y-axis:** True positive rate (recall on positives).
- **Solid curve:** your model at different probability thresholds.
- **Dashed diagonal:** random guessing (AUC ≈ 0.5).

**Your plot:** AUC ≈ **0.92**.

**Meaning:**

- The model **ranks** positive vs negative reviews well: at many thresholds it keeps **high true-positive rate** before **false positives** grow too fast.
- **0.92** is **strong** for a quick fine-tune on a small train slice; it aligns with decent separability even though accuracy is ~77–79% (accuracy uses one fixed 0.5 threshold; ROC summarizes **many** thresholds).

### D) Training loss plot (“Training Loss - BERT”)

- **Y-axis:** average training **loss** for an epoch.
- **X-axis:** epoch index (your code plots after epoch 1; Matplotlib may show the point near **0** depending on indexing).
- **Single point ~0.595:** matches **`Epoch 1 Loss: 0.5952598391071199`**.

**Meaning:**

- With **only one epoch**, you see **one** point — not yet a full “learning curve.”
- To show clear **downward** trend, run **more epochs** and plot `losses` per epoch; you should usually see loss decrease if learning rate and data are reasonable.

### E) Training loss plot (“Training Loss - DistilBERT”)

- **Same idea as the BERT loss plot**, but for **DistilBERT** after your notebook calls `plot_training_loss(distil_results["losses"], "DistilBERT")`.
- **Y-axis:** average **training loss** for the epoch (cross-entropy-style objective the model minimizes).
- **X-axis:** **Epoch** (with one epoch logged, the chart often shows a single point near index **0** depending on how the list is plotted).
- **Single point ≈ 0.61:** matches your log **`Epoch 1 Loss: 0.6101504720392681`**.

**How to read BERT vs DistilBERT loss (your numbers):**

- BERT epoch-1 loss ≈ **0.595**, DistilBERT ≈ **0.610** → on this run, **BERT’s training loss was slightly lower** after one pass. That only measures fit to the **training** batches, not guaranteed **test** accuracy.
- Your **test accuracy** was still **higher for DistilBERT** (0.79 vs 0.7725), which can happen with different architectures, regularization, and small data — another reason to rely on **held-out metrics**, not loss alone.

**What to say in a report:** one point is a **snapshot**; with **more epochs** you would expect a **curve** (ideally downward). Compare both models’ loss curves and test metrics together.

### F) Word cloud (“Word Cloud”)

This figure is **exploratory (EDA)**, not a model performance metric. The notebook builds it from the **first 200 training review texts** joined into one string, then `WordCloud(...).generate(text)`.

**What the sizes mean:**

- **Larger words** appear **more often** in that sample of raw text (frequency-based visualization).

**What your cloud is telling you:**

- **Domain:** very large **“movie”**, **“film”**, **“story”**, **“character”**, **“show”** → confirms **movie-review** language (consistent with **IMDB**).
- **Sentiment-related words** of mixed polarity appear (**“good”**, **“great”**, **“bad”**, **“never”**, etc.) → the corpus contains **both positive and negative** wording, which matches a **binary sentiment** task.
- **“br” showing up huge:** in IMDB text, **HTML line-break markup** often survives as the token **`br`** if reviews are not cleaned. The tokenizer may still learn around it, but for a **cleaner** word cloud you can **strip HTML tags** (or normalize whitespace) before `WordCloud`.

**Limitations (good to mention academically):**

- Frequency ≠ sentiment: common words can be **neutral** (“movie”, “one”).
- Only **200 reviews** are used → a **taste** of vocabulary, not the full dataset.
- Word clouds **do not** replace proper metrics (accuracy, F1, confusion matrix, ROC).

**One-line takeaway:** the word cloud supports that data is **on-domain** and **emotionally varied**; the prominent **“br”** hints at **HTML noise** and optional **preprocessing** improvement.

---

## 14) Tie it all together (viva-style answer)

**Load reports:** UNEXPECTED = weights from **pretraining** that this **task model** does not use; MISSING = **new head layers** (classifier / pre_classifier) that **must be learned** on IMDB. **Not errors.**

**Metrics:** BERT **0.7725** test accuracy; DistilBERT **0.79** on the same setup — DistilBERT **wins slightly** here.

**Plots:** Bars compare **accuracy and F1**; confusion matrix shows **where** mistakes happen (more **false positives** on true negatives for BERT); ROC **AUC ≈ 0.92** shows **good ranking** of classes; **BERT and DistilBERT** training-loss plots each show **one epoch** (~**0.595** vs ~**0.610**) and motivate **more epochs** for a real learning curve; the **word cloud** is **EDA** (movie-review vocabulary; watch for **`br`** / HTML artifacts).
