# Colab Sentiment Analyzer — Code Guide

This document explains every major part of `colab_sentiment_fast_full.py`: what it does, why it is written that way, and how the pieces connect. It matches the **Fast Colab** variant (smaller data, shorter sequences, manual PyTorch training loop, no Hugging Face `Trainer`).

---

## 1. Purpose of the script

The script implements an end-to-end pipeline:

1. Load the **IMDB** movie-review dataset (binary sentiment: negative / positive).
2. **Fine-tune** two pretrained transformers: **BERT** (`bert-base-uncased`) and **DistilBERT** (`distilbert-base-uncased`).
3. **Evaluate** each model with accuracy and F1 on a held-out subset.
4. Plot a **confusion matrix** for BERT on a sample of reviews.
5. Provide **attention-based token importance** (explainability) for arbitrary input text.
6. Launch a **Gradio** web UI so you can type text, pick a model, and see prediction + top tokens.

---

## 2. Environment and imports

### Install (run first in Colab)

```text
!pip install transformers datasets scikit-learn gradio matplotlib seaborn -q
```

- **transformers / datasets**: Hugging Face libraries for models, tokenizers, and `load_dataset`.
- **scikit-learn**: `accuracy_score`, `f1_score`, `confusion_matrix`.
- **gradio**: Browser UI.
- **matplotlib / seaborn**: Confusion matrix heatmap.

### Imports

- **torch**, **DataLoader**: Training loop and tensor handling.
- **traceback**: If something fails inside Gradio, the real error and stack trace are returned instead of a generic “Error”.

### Device

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

On Colab with **GPU** runtime, training and inference use CUDA; otherwise CPU (slower).

---

## 3. Data loading (fast settings)

```python
dataset = load_dataset("imdb")
dataset["train"] = dataset["train"].shuffle(seed=42).select(range(1000))
dataset["test"] = dataset["test"].shuffle(seed=42).select(range(400))
```

- **IMDB**: 50k reviews; labels are `0` (negative) and `1` (positive).
- **Subset**: 1000 train / 400 test examples for **faster** Colab runs. **Seed 42** makes splits reproducible.
- The **raw** `dataset["test"]` still has the column name **`label`** (used later for the confusion matrix on raw text).

---

## 4. `MAX_LEN = 128`

Each review is truncated or padded to **128 subword tokens**. Shorter sequences train faster and use less memory than the default 512, at some cost to handling very long reviews.

---

## 5. `inputs_to_device(batch, dev)`

Moves every tensor in a tokenizer batch dict to GPU/CPU:

```python
return {k: v.to(dev) for k, v in batch.items()}
```

This avoids subtle bugs where some tensors stay on CPU while the model is on GPU, and works reliably across Transformers versions (compared to relying only on `BatchEncoding.to(device)`).

---

## 6. Tokenization — `tokenize_data(model_name)`

Steps:

1. Load the correct **AutoTokenizer** for each model name (BERT vs DistilBERT vocabularies differ slightly in practice; each model is trained with its matching tokenizer).
2. **Map** over the dataset: `padding="max_length"`, `truncation=True`, `max_length=MAX_LEN`.
3. **Rename** `label` → **`labels`**: Hugging Face sequence-classification models compute **`loss`** only when the batch contains a **`labels`** key. If you keep `label`, `outputs.loss` is `None` and `loss.backward()` crashes.
4. **`set_format(type="torch", ...)`** so `DataLoader` yields PyTorch tensors for `input_ids`, `attention_mask`, and `labels`.

Returned: `(tokenizer, tokenized_dataset)` where `tokenized` has train/test splits ready for training.

---

## 7. Training — `train_model(model_name, ...)`

### Model construction

```python
AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    attn_implementation="eager",
)
```

- **`num_labels=2`**: Adds a classification head for binary sentiment.
- **`attn_implementation="eager"`**: Uses standard attention implementation so **`output_attentions=True`** returns real attention tensors. Some default backends (e.g. SDPA) may not expose attentions; without this, explainability can break or return `None` in the UI.

### Optimization

- **AdamW** optimizer, learning rate **2e-5** (typical for BERT-family fine-tuning).
- **Batch size 16**, **1 epoch** (fast Colab preset).
- Loop: `forward` → `loss.backward()` → `optimizer.step()` → `optimizer.zero_grad()`.

### Evaluation

After training, the model is set to **`eval()`**. For each **test** example, only `input_ids` and `attention_mask` are passed (no labels) to get logits; the predicted class is `argmax` over logits. Ground truth comes from **`sample["labels"]`**.

Metrics returned: **accuracy** and **macro F1** (via `f1_score` default for binary).

---

## 8. Training both models

The script calls `train_model` twice: once for BERT, once for DistilBERT. Each call **re-tokenizes** the same underlying text with that model’s tokenizer and trains a **separate** classifier. Printed lines:

```text
BERT: {'accuracy': ..., 'f1': ...}
DistilBERT: {'accuracy': ..., 'f1': ...}
```

---

## 9. Confusion matrix — `plot_confusion_matrix`

Uses the **raw** test split (`dataset["test"]`), **not** the tensor-formatted set, so it reads **`sample["text"]`** and **`sample["label"]`**. For each sample it tokenizes with the given model’s tokenizer, runs a forward pass, and compares predicted vs actual label. Builds a 2×2 **confusion matrix** and plots it with **seaborn.heatmap**. Only **150** examples are used (subset) to keep plotting quick.

---

## 10. Explainability — `explain_prediction`

1. Tokenize input text (same `MAX_LEN`).
2. Forward with **`output_attentions=True`**.
3. If **`outputs.attentions`** is `None`, return a placeholder token so the UI does not crash.
4. Take the **last transformer layer**’s attention: `outputs.attentions[-1]`.
5. Average over **heads** (`mean(dim=0)` on head dimension), then average over the query dimension to get one score per token position.
6. Align **token strings** from `convert_ids_to_tokens` with those scores (truncate to common length `m`).
7. Sort by score and return the **top 10** (subword) tokens.

**Note:** These scores are a **heuristic** visualization of attention mass, not a certified causal explanation. Subwords (e.g. `##ing`) appear as separate tokens.

---

## 11. Prediction — `predict`

Combines classification and explanation:

- Softmax on logits → **predicted class**, **confidence** = max probability.
- Human-readable label: **Positive** if class 1, else **Negative**.
- Calls **`explain_prediction`** for the token list.

---

## 12. Gradio UI — `app_fn` and `iface.launch`

- **Inputs**: text box + radio (**BERT** / **DistilBERT**).
- **Outputs**: prediction string (label + percent confidence) + multiline “Important Words” (token scores).
- **`try` / `except`**: On failure, returns a short error line and full **traceback** in the second output so debugging in Colab is possible.
- **`share=True`**: Gradio creates a **public URL** (useful when the browser is not on the same machine as the runtime, typical for Colab).

---

## 13. File map

| File | Role |
|------|------|
| `colab_sentiment_fast_full.py` | Full Colab script (paste after `pip install`). |
| `main.py` | Local variant using Hugging Face `Trainer` + saved models (optional). |

---

## 14. Common issues (recap)

| Symptom | Likely cause | Fix in this script |
|--------|----------------|---------------------|
| `loss` is `None` | Batch has `label` not `labels` | Rename column + `set_format` with `labels` |
| Gradio shows generic “Error” | Exception inside `app_fn` | `try`/`except` + traceback; fix root cause |
| No attention weights | SDPA / flash backend | `attn_implementation="eager"` |

---

## 15. Extending the project

- Increase **train/test sizes** and **epochs** for better accuracy (longer runtime).
- Set **`MAX_LEN`** to 256 or 512 for long reviews.
- Save models with **`model.save_pretrained(...)`** and reload for inference-only sessions.
- Replace attention heuristic with **integrated gradients** or **LIME** for stronger explainability (more code).

This guide should be enough to walk through a viva or to modify the notebook safely.
